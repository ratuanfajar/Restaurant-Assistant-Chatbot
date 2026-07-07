#  Restaurant Assistant Chatbot

Model ini menggunakan satu arsitektur sequence-to-sequence (Two Stage CopyNet) untuk menangani dua tugas sekaligus: melacak kebutuhan user (belief tracking) dan menghasilkan respons alami, tanpa pipeline terpisah seperti sistem dialog tradisional.

---

## Struktur Folder

```
restorant-asisten/
├── data/
│   ├── CamRest676.json      # Dataset percakapan (676 dialog)
│   └── CamRestDB.json       # Knowledge base (110 restoran)
├── checkpoints/              # Model tersimpan hasil training
├── docs/                     # Dokumen planning dan catatan teknis
├── notebook/                  # Eksperimen/eksplorasi (Jupyter)
├── src/
│   ├── config.py              # Semua hyperparameter
│   ├── preprocess.py          # Load data, delexicalization, belief span
│   ├── dataset.py             # PyTorch Dataset & DataLoader
│   ├── model.py               # Encoder, Decoder Stage 1 & 2 (TSCP)
│   ├── utils.py               # KB search, beam search, reward, helper lain
│   ├── train_supervised.py    # Training supervised (cross entropy)
│   ├── train_rl.py            # Fine-tuning RL (REINFORCE)
│   ├── evaluate.py            # BLEU, Entity Match Rate, Success F1
│   ├── inference.py           # Mode chatbot interaktif
│   └── run.py                 # Entry point utama (CLI)
├── main.py
├── pyproject.toml
└── README.md
```

---

## Persiapan Environment

Proyek ini menggunakan `uv` sebagai package manager.

```bash
# Buat virtual environment
uv venv

# Aktifkan (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependency lain
uv sync

# Download data NLTK untuk BLEU score
uv run python -c "import nltk; nltk.download('punkt')"
```

Cek apakah GPU terdeteksi:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## Cara Menjalankan

Semua perintah dijalankan dari root folder project.

### Pipeline lengkap (training sampai evaluasi)

```bash
uv run python src/run.py --mode all --device cuda
```

### Bertahap

```bash
# 1. Supervised training
uv run python src/run.py --mode train --device cuda

# 2. Fine-tuning dengan reinforcement learning
uv run python src/run.py --mode rl --device cuda

# 3. Evaluasi pada test set
uv run python src/run.py --mode eval --device cuda

# 4. Coba chatbot secara interaktif
uv run python src/run.py --mode chat --device cuda
```

Kalau tidak punya GPU, ganti `--device cuda` menjadi `--device cpu`.

---

## Alur Kerja Model

1. **Encoder** membaca gabungan konteks dialog (belief span sebelumnya, respons sebelumnya, dan input user saat ini).
2. **Decoder Stage 1** menghasilkan belief span (B_t) -- ringkasan kebutuhan user seperti jenis makanan, area, dan kisaran harga.
3. Belief span dipakai untuk **mencari restoran** di knowledge base.
4. **Decoder Stage 2** menghasilkan respons berdasarkan belief span dan hasil pencarian, masih dalam bentuk placeholder (misal `NAME_SLOT`).
5. Placeholder diganti dengan data restoran asli sebelum ditampilkan ke user.

---

## Metrik Evaluasi
- BLEU 
- Entity Match Rate
- Success F1

Detail lengkap arsitektur dan penjelasan setiap komponen bisa dilihat di `docs/PLANNING_TSCP_REPLICATION.md`.
