"""
Main entry point untuk TSCP Sequicity.

Usage:
  python run.py --mode train       # Supervised training
  python run.py --mode rl           # RL fine-tuning (setelah supervised)
  python run.py --mode eval         # Evaluasi pada test set
  python run.py --mode chat         # Interactive chatbot
  python run.py --mode all          # Train → RL → Eval (full pipeline)
"""

import argparse
import os
import sys
import torch

import src.config as config
from src.preprocessing import prepare_data
from src.train_supervised import train_supervised
from src.train_rl import train_rl
from src.evaluate import evaluate_model
from src.inference import interactive_chat, load_model_for_inference
from src.model import TSCP


def get_device():
    """Pilih device: CUDA jika tersedia, otherwise CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def main():
    parser = argparse.ArgumentParser(description="TSCP Sequicity — CamRest676")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "rl", "eval", "chat", "all"],
        help="Mode: train (supervised), rl (fine-tune), eval (test), chat (interactive), all (full pipeline)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path ke checkpoint untuk eval/chat/rl (default: auto-detect terbaru)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu' atau 'cuda' (default: auto-detect)",
    )
    args = parser.parse_args()

    device = args.device if args.device else get_device()

    # === Prepare Data ===
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    data = prepare_data()

    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    database = data["database"]
    vocab_size = len(word2idx)

    # === Mode: Train (Supervised) ===
    if args.mode in ("train", "all"):
        model = train_supervised(data, device=device)

        if args.mode == "train":
            print("\nSupervised training selesai.")
            return

    # === Mode: RL Fine-tuning ===
    if args.mode in ("rl", "all"):
        if args.mode == "rl":
            # Load dari checkpoint supervised
            ckpt_path = args.checkpoint or os.path.join(
                config.SAVE_DIR, "tscp_supervised_best.pt"
            )
            if not os.path.exists(ckpt_path):
                print(f"ERROR: Checkpoint tidak ditemukan: {ckpt_path}")
                print("Jalankan --mode train dulu, atau specify --checkpoint path.")
                sys.exit(1)
            model = load_model_for_inference(ckpt_path, word2idx, device)

        model = train_rl(model, data, device=device)

        if args.mode == "rl":
            print("\nRL fine-tuning selesai.")
            return

    # === Mode: Evaluate ===
    if args.mode in ("eval", "all"):
        if args.mode == "eval":
            # Load dari checkpoint
            ckpt_path = args.checkpoint
            if ckpt_path is None:
                # Coba RL final dulu, kalau ga ada pakai supervised
                rl_path = os.path.join(config.SAVE_DIR, "tscp_rl_final.pt")
                sl_path = os.path.join(config.SAVE_DIR, "tscp_supervised_best.pt")
                if os.path.exists(rl_path):
                    ckpt_path = rl_path
                elif os.path.exists(sl_path):
                    ckpt_path = sl_path
                else:
                    print("ERROR: Tidak ada checkpoint ditemukan. Jalankan training dulu.")
                    sys.exit(1)
            model = load_model_for_inference(ckpt_path, word2idx, device)

        results = evaluate_model(
            model, data["test"], word2idx, idx2word, database, device
        )

        if args.mode in ("eval", "all"):
            # Simpan hasil ke file
            results_path = os.path.join(config.SAVE_DIR, "eval_results.txt")
            with open(results_path, "w") as f:
                for metric, value in results.items():
                    f.write(f"{metric}: {value:.4f}\n")
            print(f"Results saved to {results_path}")

        if args.mode == "eval":
            return

    # === Mode: Chat ===
    if args.mode == "chat":
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            rl_path = os.path.join(config.SAVE_DIR, "tscp_rl_final.pt")
            sl_path = os.path.join(config.SAVE_DIR, "tscp_supervised_best.pt")
            if os.path.exists(rl_path):
                ckpt_path = rl_path
            elif os.path.exists(sl_path):
                ckpt_path = sl_path
            else:
                print("ERROR: Tidak ada checkpoint ditemukan. Jalankan training dulu.")
                sys.exit(1)
        model = load_model_for_inference(ckpt_path, word2idx, device)

        interactive_chat(model, word2idx, idx2word, database, device)

    if args.mode == "all":
        print("\n" + "=" * 60)
        print("FULL PIPELINE SELESAI")
        print("=" * 60)
        print(f"  1. Supervised training: DONE")
        print(f"  2. RL fine-tuning:      DONE")
        print(f"  3. Evaluation:          DONE")
        print(f"\nUntuk interactive chat:")
        print(f"  python run.py --mode chat")
        print("=" * 60)


if __name__ == "__main__":
    main()
