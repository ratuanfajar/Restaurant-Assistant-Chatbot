# 🍽️ Streamlit Chatbot - TSCP (Two Stage CopyNet)

Chatbot interaktif berbasis web untuk rekomendasi restoran di Cambridge menggunakan arsitektur **TSCP (Two Stage CopyNet)** dari paper Sequicity.

## 📋 Prerequisites

Pastikan Anda sudah memiliki:
1. Model yang sudah di-training (checkpoint `.pt` files)
2. Data CamRest676 dan CamRestDB di folder `data/`
3. Checkpoint tersimpan di folder `checkpoints/`

## 🚀 Cara Menjalankan

### 1. Install Dependencies (jika belum)

```bash
pip install streamlit torch
```

### 2. Jalankan Streamlit App

```bash
cd /workspace
streamlit run chatbot_app.py
```

Aplikasi akan terbuka otomatis di browser Anda di alamat `http://localhost:8501`

## 📁 Struktur File

```
/workspace/
├── chatbot_app.py          # Main Streamlit app (BARU)
├── data/
│   ├── CamRest676.json    # Dataset dialog
│   └── CamRestDB.json     # Database restoran
├── checkpoints/           # Folder untuk model checkpoint
│   ├── tscp_supervised_best.pt
│   ├── tscp_rl_best.pt
│   └── tscp_rl_final.pt
├── src/
│   ├── preprocessing.py   # Modul preprocessing (eksp3)
│   ├── model.py           # TSCP model architecture
│   ├── utils.py           # Utility functions
│   ├── inference.py       # Inference helpers
│   └── config.py          # Configuration
└── notebook/
    ├── 01-preprocessing/eksp3.ipynb  # Preprocessing experiments
    ├── 02-supervised/eksp2.ipynb     # Supervised training
    └── 03-rl-tuning/rl-tuning.ipynb  # RL fine-tuning
```

## 🎯 Fitur Chatbot

### 1. **Multi-Turn Conversation**
- Mendukung percakapan multi-turn dengan context awareness
- Menyimpan history percakapan (belief span & response sebelumnya)

### 2. **Checkpoint Selection**
- Pilih checkpoint model dari sidebar (supervised atau RL fine-tuned)
- Support multiple checkpoint formats (.pt files)

### 3. **Beam Search Decoding**
- Konfigurasi beam size (1-10) untuk balancing quality & speed
- Default beam size = 5

### 4. **Debug Information**
- Expandable debug info untuk setiap response:
  - Predicted Belief Span
  - KB Matches (matching restaurants)
  - Delexicalized Response

### 5. **Session Management**
- Reset conversation button untuk mulai percakapan baru
- Session state persistence selama browser tidak di-refresh

## 🔧 Konfigurasi Model

Model ini telah melalui 3 tahap training:

| Tahap | Notebook | Deskripsi | Checkpoint |
|-------|----------|-----------|------------|
| 1 | `eksp3.ipynb` | Preprocessing & delexicalization | - |
| 2 | `eksp2.ipynb` | Supervised training | `tscp_supervised_best.pt` |
| 3 | `rl-tuning.ipynb` | RL fine-tuning | `tscp_rl_final.pt` |

## 💬 Contoh Percakapan

**User:** "I want a cheap Italian restaurant"  
**Bot:** "pizza hut is a cheap italian restaurant . would you like the address ?"

**User:** "Yes, and the phone number please"  
**Bot:** "the phone number is 01223 323361 . anything else ?"

## 🛠️ Troubleshooting

### Checkpoint tidak ditemukan
```
Error: Checkpoint tidak ditemukan: /workspace/checkpoints/tscp_rl_final.pt
```
**Solusi:** Pastikan Anda sudah training model dan checkpoint tersimpan di folder `checkpoints/`

### ModuleNotFoundError: No module named 'streamlit'
```bash
pip install streamlit
```

### CUDA out of memory
Edit `chatbot_app.py` dan ubah device menjadi CPU:
```python
device = torch.device("cpu")
```

## 📊 Arsitektur Model

```
User Input (U_t)
      ↓
┌─────────────────┐
│   Encoder GRU   │
└────────┬────────┘
         │ H^(x)
    ┌────┴────┐
    ↓         ↓
┌───────┐ ┌──────────┐
│Bspan  │ │ KB Search│
│Decoder│ └────┬─────┘
│(Stage1)      │ k_t
└───────┘      ↓
         ┌──────────┐
         │Response  │
         │Decoder   │
         │(Stage 2) │
         └────┬─────┘
              ↓
        Lexicalization
              ↓
        Final Response
```

## 📝 Customization

### Mengubah Beam Size Default
Edit line di `chatbot_app.py`:
```python
beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5)
# Ubah value=5 sesuai kebutuhan
```

### Menambah Custom Checkpoint
Cukup simpan file `.pt` di folder `checkpoints/`, akan otomatis terdeteksi.

### Mengubah Vocabulary Size
Edit parameter `max_vocab_size` di fungsi `load_vocabulary()`:
```python
word2idx, idx2word = build_vocabulary(temp_tokenized, max_vocab_size=800)
```

## 📄 License

Dibuat untuk tujuan eksperimen dan pembelajaran.

## 👨‍💻 Author

Built with ❤️ using Streamlit + PyTorch
