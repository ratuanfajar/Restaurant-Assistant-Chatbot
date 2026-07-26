"""
Supervised Learning Training Loop untuk TSCP.

Sesuai paper Section 5.2:
- Optimizer: Adam, LR = 0.003
- Early stopping pada dev set
- Teacher forcing di kedua decoder stages
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import src.config as config
from src.model import TSCP
from src.dataset import get_dataloader
from src.preprocessing import prepare_data, tokenize, tokens_to_indices
from src.utils import get_kt_from_bspan, parse_bspan


def compute_kt_for_batch(batch, database, word2idx):
    """
    Hitung k_t vector (3-dim one-hot) untuk setiap sample dalam batch.
    Menggunakan ground truth bspan untuk training.
    """
    kt_list = []
    for i in range(len(batch["bspan_tokens"])):
        bspan_tokens = batch["bspan_tokens"][i]
        bspan_text = " ".join(bspan_tokens)
        kt = get_kt_from_bspan(bspan_text, database)
        kt_list.append(kt)
    return torch.stack(kt_list)


def train_epoch(model, dataloader, optimizer, database, word2idx, device, clip_grad):
    """Satu epoch supervised training."""
    model.train()
    total_loss = 0.0
    total_bspan_loss = 0.0
    total_response_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        input_seq = batch["input_padded"].to(device)
        input_lengths = batch["input_lengths"]
        bspan_input = batch["bspan_input"].to(device)
        bspan_target = batch["bspan_target"].to(device)
        response_input = batch["response_input"].to(device)
        response_target = batch["response_target"].to(device)

        # Compute k_t
        kt_batch = compute_kt_for_batch(batch, database, word2idx).to(device)

        # Forward
        loss, bspan_loss, response_loss = model(
            input_seq, input_lengths,
            bspan_input, bspan_target,
            response_input, response_target,
            batch["input_tokens"], batch["bspan_tokens"],
            kt_batch, word2idx,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        total_bspan_loss += bspan_loss.item()
        total_response_loss += response_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_bspan = total_bspan_loss / max(num_batches, 1)
    avg_response = total_response_loss / max(num_batches, 1)
    return avg_loss, avg_bspan, avg_response


def evaluate_epoch(model, dataloader, database, word2idx, device):
    """Evaluasi loss pada dev set (untuk early stopping)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch["input_padded"].to(device)
            input_lengths = batch["input_lengths"]
            bspan_input = batch["bspan_input"].to(device)
            bspan_target = batch["bspan_target"].to(device)
            response_input = batch["response_input"].to(device)
            response_target = batch["response_target"].to(device)

            kt_batch = compute_kt_for_batch(batch, database, word2idx).to(device)

            loss, _, _ = model(
                input_seq, input_lengths,
                bspan_input, bspan_target,
                response_input, response_target,
                batch["input_tokens"], batch["bspan_tokens"],
                kt_batch, word2idx,
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_supervised(data, device="cpu"):
    """
    Main supervised training function.
    
    Args:
        data: dict dari prepare_data() — berisi train/dev/test samples, word2idx, database
        device: 'cpu' atau 'cuda'
        
    Returns:
        model: trained TSCP model
    """
    word2idx = data["word2idx"]
    database = data["database"]
    vocab_size = len(word2idx)

    print(f"\n{'='*60}")
    print(f"SUPERVISED TRAINING")
    print(f"{'='*60}")
    print(f"Vocab size: {vocab_size}")
    print(f"Device: {device}")
    print(f"Batch size: {config.SL_BATCH_SIZE}")
    print(f"Learning rate: {config.SL_LEARNING_RATE}")
    print(f"Hidden/Embed size: {config.HIDDEN_SIZE}/{config.EMBED_SIZE}")
    print(f"Max epochs: {config.SL_EPOCHS}")
    print(f"Early stopping patience: {config.SL_PATIENCE}")

    # DataLoaders
    train_loader = get_dataloader(data["train"], word2idx, config.SL_BATCH_SIZE, shuffle=True)
    dev_loader = get_dataloader(data["dev"], word2idx, config.SL_BATCH_SIZE, shuffle=False)

    # Model
    model = TSCP(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.SL_LEARNING_RATE)

    # Early stopping
    best_dev_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Checkpoint directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print(f"\nStarting training...\n")

    for epoch in range(1, config.SL_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_bspan, train_resp = train_epoch(
            model, train_loader, optimizer, database, word2idx, device, config.SL_CLIP_GRAD
        )

        # Evaluate on dev
        dev_loss = evaluate_epoch(model, dev_loader, database, word2idx, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:3d}/{config.SL_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} (B:{train_bspan:.4f} R:{train_resp:.4f}) | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping check
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save checkpoint
            checkpoint_path = os.path.join(config.SAVE_DIR, "tscp_supervised_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": best_dev_loss,
                "word2idx": word2idx,
                "vocab_size": vocab_size,
            }, checkpoint_path)
            print(f"  → Best model saved (dev_loss={best_dev_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.SL_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.SL_PATIENCE})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (dev_loss={best_dev_loss:.4f})")

    return model
