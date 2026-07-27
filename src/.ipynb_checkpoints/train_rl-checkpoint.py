"""
Reinforcement Learning Fine-Tuning untuk TSCP (Section 4.4).

Hanya fine-tune Decoder Stage 2 (response generator). Decoder Stage 2
diperlakukan sebagai policy network pi_Theta.

Desain (revisi — versi lama kolaps):
  - Reward sekuens: +1 untuk placeholder request slot yang diminta user
    (sekali per slot), -0.5 untuk placeholder yang tidak diminta (halusinasi),
    0 untuk token normal. Lihat utils.compute_rewards.
  - MIXED OBJECTIVE: loss = policy_loss + alpha * supervised_ce.
    Suku supervised (teacher-forced pada gold response) menjadi jangkar agar
    kelancaran bahasa hasil supervised tidak rusak (mencegah catastrophic
    forgetting dari REINFORCE murni).
  - Encoder + Decoder Stage 1 DIBEKUKAN (no_grad + detach): hanya decoder2
    yang di-update, sehingga graf autograd tidak menumpuk di encoder/decoder1.
  - Turn tanpa request slot DILEWATI (reward-nya netral, tidak informatif).
  - Checkpoint terbaik dipilih berdasarkan Success-F1 di dev set (metrik tugas),
    BUKAN rata-rata reward (yang justru memihak respons pendek/kolaps).

PENTING: Sampling dilakukan dari distribusi gabungan (generate + copy),
karena placeholder seperti NAME_SLOT/PHONE_SLOT umumnya dihasilkan lewat
copy mechanism dari bspan.
"""

import os
import time
import random

import torch
import torch.optim as optim

import src.config as config
from src.preprocessing import tokenize, tokens_to_indices
from src.model import _combine_copy_probs
from src.utils import (
    get_kt_from_bspan, parse_bspan, compute_rewards, compute_returns,
)


def _build_source_ext(tokens, word2idx, device):
    """Build extended vocab mapping untuk satu sample (dipakai untuk copy)."""
    vocab_size = len(word2idx)
    oov_tokens = []
    oov_map = {}
    ext_indices = []

    for token in tokens:
        if token in word2idx:
            ext_indices.append(word2idx[token])
        else:
            if token not in oov_map:
                oov_map[token] = vocab_size + len(oov_tokens)
                oov_tokens.append(token)
            ext_indices.append(oov_map[token])

    ext_vocab_size = vocab_size + len(oov_tokens)
    source_tensor = torch.tensor([ext_indices], dtype=torch.long, device=device)
    return ext_vocab_size, oov_tokens, source_tensor


def _supervised_anchor_loss(model, sample, bspan_outputs, bspan_final_hidden,
                            kt, oov_tokens, bspan_ext, ext_vocab_size,
                            word2idx, device):
    """
    Supervised CE (teacher forcing) pada gold response — jangkar mixed objective.
    Memakai bspan_outputs / bspan_final_hidden yang SUDAH detached, sehingga
    gradien hanya mengalir ke decoder2.
    """
    sos_idx = word2idx[config.SOS_TOKEN]
    eos_idx = word2idx[config.EOS_TOKEN]

    response_tokens = tokenize(sample["target_response"])
    response_indices = tokens_to_indices(response_tokens, word2idx)

    resp_input = torch.tensor([[sos_idx] + response_indices], dtype=torch.long, device=device)

    # Target dalam extended vocab (konsisten dengan copy dari bspan).
    resp_target_tokens = response_tokens + [config.EOS_TOKEN]
    resp_target_ext = model._target_to_ext(resp_target_tokens, word2idx, oov_tokens)
    resp_target_ext = torch.tensor([resp_target_ext], dtype=torch.long, device=device)

    resp_log_probs, _ = model.decoder2(
        resp_input, bspan_final_hidden, bspan_outputs, kt,
        bspan_ext, ext_vocab_size,
    )
    return model._compute_loss(resp_log_probs, resp_target_ext, ext_vocab_size)


def rl_finetune_one_sample(model, sample, word2idx, idx2word, database, device):
    """
    Satu langkah update RL (mixed objective) untuk satu sample.

    Returns:
        loss: total loss (policy + alpha*supervised), scalar tensor
        reward_sum: total reward rollout (float, untuk monitoring)
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
    vocab_size = len(word2idx)

    bspan_input = torch.tensor([[sos_idx] + bspan_indices], dtype=torch.long, device=device)

    bspan_text = sample["target_bspan"]
    _, requestable = parse_bspan(bspan_text)

    # --- Encoder + Stage 1 DIBEKUKAN (tidak di-RL) ---
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(input_tensor, input_lengths)
        bspan_embedded = model.decoder1.embedding(bspan_input)
        bspan_outputs, bspan_final_hidden = model.decoder1.gru(bspan_embedded, encoder_hidden)
    bspan_outputs = bspan_outputs.detach()
    bspan_final_hidden = bspan_final_hidden.detach()

    kt = get_kt_from_bspan(bspan_text, database).unsqueeze(0).to(device)

    # Extended vocab dari bspan tokens (untuk copy).
    full_bspan_tokens = [config.SOS_TOKEN] + bspan_tokens
    ext_vocab_size, oov_tokens, bspan_ext = _build_source_ext(
        full_bspan_tokens, word2idx, device
    )

    # --- Rollout: sampling Stage 2 secara autoregressive ---
    hidden = bspan_final_hidden
    dec_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    generated_tokens = []
    log_probs_list = []

    for _ in range(config.MAX_DECODE_LEN_RESPONSE):
        gen_logits, copy_score, hidden = model.decoder2.forward_step(
            dec_input, hidden, bspan_outputs, kt
        )
        log_probs = _combine_copy_probs(
            gen_logits, copy_score, bspan_ext, ext_vocab_size, vocab_size
        )  # (1, ext_vocab_size)
        probs = log_probs.exp().squeeze(0)

        dist = torch.distributions.Categorical(probs)
        sampled_idx = dist.sample()
        log_probs_list.append(dist.log_prob(sampled_idx))

        sampled_idx_val = sampled_idx.item()
        if sampled_idx_val < vocab_size:
            token_str = idx2word.get(sampled_idx_val, config.UNK_TOKEN)
        else:
            oov_offset = sampled_idx_val - vocab_size
            token_str = oov_tokens[oov_offset] if oov_offset < len(oov_tokens) else config.UNK_TOKEN
        generated_tokens.append(token_str)

        if sampled_idx_val == eos_idx:
            break

        next_idx = sampled_idx_val if sampled_idx_val < vocab_size else word2idx[config.UNK_TOKEN]
        dec_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    # --- Policy gradient loss ---
    rewards = compute_rewards(generated_tokens, requestable)
    returns = compute_returns(rewards, gamma=config.RL_LAMBDA)

    returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)
    log_probs_tensor = torch.stack(log_probs_list)

    if len(returns) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    policy_loss = -(log_probs_tensor * returns_tensor).mean()

    # --- Supervised anchor (mixed objective) ---
    sup_loss = _supervised_anchor_loss(
        model, sample, bspan_outputs, bspan_final_hidden, kt,
        oov_tokens, bspan_ext, ext_vocab_size, word2idx, device,
    )

    loss = policy_loss + config.RL_SUPERVISED_ALPHA * sup_loss
    return loss, sum(rewards)


def _dev_success_f1(model, data, device, max_samples):
    """Success-F1 di subset dev — dipakai untuk memilih checkpoint RL terbaik."""
    # Import lokal agar tidak circular saat modul di-load.
    from src.evaluate import generate_for_sample, compute_success_f1

    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    database = data["database"]
    dev_samples = data["dev"][:max_samples]

    model.eval()
    pred_responses, gold_bspans = [], []
    for sample in dev_samples:
        _, pred_resp, _ = generate_for_sample(
            model, sample, word2idx, idx2word, database, device
        )
        pred_responses.append(pred_resp)
        gold_bspans.append(sample["target_bspan"])

    return compute_success_f1(pred_responses, gold_bspans)


def train_rl(model, data, device="cpu"):
    """
    Main RL fine-tuning function.

    Hanya fine-tune Decoder Stage 2 (policy network = response decoder).
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
    print(f"Reward: +{config.RL_REWARD_POS} (slot match), "
          f"{config.RL_REWARD_HALLUCINATION} (slot halusinasi), "
          f"{config.RL_REWARD_NEG} (token normal)")
    print(f"Supervised anchor alpha: {config.RL_SUPERVISED_ALPHA}")
    print(f"Max epochs: {config.RL_EPOCHS}")

    # Hanya turn dengan request slot yang informatif untuk RL.
    rl_samples = [s for s in train_samples if parse_bspan(s["target_bspan"])[1]]
    print(f"Train samples dengan request slot: {len(rl_samples)} / {len(train_samples)}")

    optimizer = optim.Adam(model.decoder2.parameters(), lr=config.RL_LEARNING_RATE)

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Pilih checkpoint terbaik berdasarkan dev Success-F1 (bukan avg reward).
    best_f1 = _dev_success_f1(model, data, device, config.RL_DEV_EVAL_SAMPLES)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"Baseline dev Success-F1 (sebelum RL): {best_f1:.4f}")

    for epoch in range(1, config.RL_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0.0
        total_reward = 0.0

        indices = list(range(len(rl_samples)))
        random.shuffle(indices)

        for idx in indices:
            sample = rl_samples[idx]

            loss, reward_sum = rl_finetune_one_sample(
                model, sample, word2idx, idx2word, database, device
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder2.parameters(), config.SL_CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            total_reward += reward_sum

        n = max(len(rl_samples), 1)
        dev_f1 = _dev_success_f1(model, data, device, config.RL_DEV_EVAL_SAMPLES)
        elapsed = time.time() - start_time

        print(
            f"RL Epoch {epoch:3d}/{config.RL_EPOCHS} | "
            f"Avg Loss: {total_loss / n:.4f} | "
            f"Avg Reward: {total_reward / n:.4f} | "
            f"Dev Success-F1: {dev_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_path = os.path.join(config.SAVE_DIR, "tscp_rl_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "word2idx": word2idx,
                "dev_success_f1": best_f1,
            }, checkpoint_path)
            print(f"  → Best RL model saved (dev Success-F1={best_f1:.4f})")

    # Restore state dengan dev Success-F1 terbaik.
    model.load_state_dict(best_state)
    print(f"\nRestored RL model dengan best dev Success-F1 = {best_f1:.4f}")

    final_path = os.path.join(config.SAVE_DIR, "tscp_rl_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": word2idx,
    }, final_path)
    print(f"RL model saved to {final_path}")

    return model
