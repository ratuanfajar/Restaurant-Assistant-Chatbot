"""
Reinforcement Learning Fine-Tuning untuk TSCP (Section 4.4).

Hanya fine-tune Decoder Stage 2 (response generator).
Decoder Stage 2 diperlakukan sebagai policy network π_Θ.

Reward:
  r(j) = +1 jika token yang di-generate adalah placeholder request slot yang diminta user
  r(j) = -0.1 untuk token lainnya

Update: REINFORCE (Policy Gradient) — Eq. 11
  ∇_Θ = (1/(m-m')) * Σ R(j) * ∇ log π_Θ(y_j)
"""

import os
import time
import torch
import torch.optim as optim

import src.config as config
from src.preprocess import tokenize, tokens_to_indices
from src.utils import (
    get_kt_from_bspan, parse_bspan, compute_rewards, compute_returns,
    indices_to_text,
)


def rl_finetune_one_sample(model, sample, word2idx, idx2word, database, device):
    """
    REINFORCE update untuk satu sample.
    
    Alur:
    1. Encode input
    2. Teacher-force Stage 1 (bspan) — kita tidak RL bspan
    3. Sample Stage 2 (response) secara autoregressive (bukan teacher forcing)
    4. Hitung reward dan return
    5. Hitung policy gradient loss
    
    Returns:
        loss: policy gradient loss (scalar tensor)
        reward_sum: total reward (float, untuk monitoring)
    """
    model.train()

    # --- Prepare input ---
    input_tokens = tokenize(sample["input"])
    input_indices = tokens_to_indices(input_tokens, word2idx)
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
    input_lengths = torch.tensor([len(input_indices)])

    bspan_tokens = tokenize(sample["target_bspan"])
    bspan_indices = tokens_to_indices(bspan_tokens, word2idx)

    sos_idx = word2idx[config.SOS_TOKEN]
    eos_idx = word2idx[config.EOS_TOKEN]

    bspan_input = torch.tensor([[sos_idx] + bspan_indices], dtype=torch.long, device=device)

    # --- 1. Encode ---
    encoder_outputs, encoder_hidden = model.encoder(input_tensor, input_lengths)

    # --- 2. Teacher-force Stage 1 (bspan) ---
    # Jalankan bspan melalui decoder1 GRU untuk dapat hidden states
    bspan_embedded = model.decoder1.embedding(bspan_input)
    bspan_outputs, bspan_final_hidden = model.decoder1.gru(bspan_embedded, encoder_hidden)

    # --- 3. Compute k_t ---
    bspan_text = sample["target_bspan"]
    kt = get_kt_from_bspan(bspan_text, database).unsqueeze(0).to(device)

    # --- 4. Sample Stage 2 (response) secara autoregressive ---
    # Parse requestable slots dari bspan untuk reward
    _, requestable = parse_bspan(bspan_text)

    hidden = bspan_final_hidden
    dec_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    generated_tokens = []
    log_probs_list = []

    for step in range(config.MAX_DECODE_LEN_RESPONSE):
        # Build extended vocab dari bspan (untuk CopyNet Stage 2)
        gen_log_prob, copy_score, hidden = model.decoder2.forward_step(
            dec_input, hidden, bspan_outputs, kt
        )

        # Simplified: gunakan hanya gen_log_prob untuk sampling RL
        # (copy mechanism tetap digunakan, tapi reward hanya dari gen tokens)
        probs = gen_log_prob.exp().squeeze(0)  # (vocab_size,)

        # Sample token (REINFORCE: sample, bukan argmax)
        dist = torch.distributions.Categorical(probs)
        sampled_idx = dist.sample()
        log_prob = dist.log_prob(sampled_idx)

        log_probs_list.append(log_prob)

        token_str = idx2word.get(sampled_idx.item(), config.UNK_TOKEN)
        generated_tokens.append(token_str)

        # Stop jika EOS
        if sampled_idx.item() == eos_idx:
            break

        # Next input
        dec_input = sampled_idx.unsqueeze(0).unsqueeze(0)

    if not generated_tokens:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    # --- 5. Hitung rewards & returns ---
    rewards = compute_rewards(generated_tokens, requestable)
    returns = compute_returns(rewards, gamma=config.RL_LAMBDA)

    # --- 6. Policy gradient loss ---
    # L = -1/(m-m') * Σ R(j) * log π(y_j)
    returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)
    log_probs_tensor = torch.stack(log_probs_list)

    # Normalize returns (baseline: mean return) untuk stabilitas
    if len(returns) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    policy_loss = -(log_probs_tensor * returns_tensor).mean()
    reward_sum = sum(rewards)

    return policy_loss, reward_sum


def train_rl(model, data, device="cpu"):
    """
    Main RL fine-tuning function.
    
    Hanya fine-tune Decoder Stage 2 (sesuai paper: policy network = response decoder).
    
    Args:
        model: TSCP model yang sudah di-pretrain supervised
        data: dict dari prepare_data()
        device: 'cpu' atau 'cuda'
        
    Returns:
        model: fine-tuned model
    """
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    database = data["database"]
    train_samples = data["train"]

    print(f"\n{'='*60}")
    print(f"REINFORCEMENT LEARNING FINE-TUNING")
    print(f"{'='*60}")
    print(f"Learning rate: {config.RL_LEARNING_RATE}")
    print(f"Lambda (decay): {config.RL_LAMBDA}")
    print(f"Reward: +{config.RL_REWARD_POS} (slot match), {config.RL_REWARD_NEG} (other)")
    print(f"Max epochs: {config.RL_EPOCHS}")

    # Optimizer — LR lebih kecil untuk RL (Section 5.2)
    optimizer = optim.Adam(model.decoder2.parameters(), lr=config.RL_LEARNING_RATE)

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    for epoch in range(1, config.RL_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0.0
        total_reward = 0.0

        # Shuffle samples
        import random
        indices = list(range(len(train_samples)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            sample = train_samples[idx]

            loss, reward_sum = rl_finetune_one_sample(
                model, sample, word2idx, idx2word, database, device
            )

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder2.parameters(), config.SL_CLIP_GRAD)
                optimizer.step()

            total_loss += loss.item()
            total_reward += reward_sum

        elapsed = time.time() - start_time
        avg_loss = total_loss / len(train_samples)
        avg_reward = total_reward / len(train_samples)

        print(
            f"RL Epoch {epoch:3d}/{config.RL_EPOCHS} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Reward: {avg_reward:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save checkpoint setiap epoch
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(config.SAVE_DIR, f"tscp_rl_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "word2idx": word2idx,
            }, checkpoint_path)

    # Save final RL model
    final_path = os.path.join(config.SAVE_DIR, "tscp_rl_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
    }, final_path)
    print(f"RL model saved to {final_path}")

    return model
