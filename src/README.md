# Panduan Penggunaan Modul


## Preprocessing (preprocessing.py)
Modul ini menangani seluruh pipeline data untuk dataset CamRest secara otomatis: load JSON → split → delexicalization → belief span → tokenisasi → vocab → indexing → DataLoader.

Semua proses berjalan begitu `CamRestDataModule` diinisialisasi — **tidak perlu preprocessing tambahan** dan **tidak ada file `.pt`** yang perlu di-load manual.

### 1. Import & Inisialisasi

```python
from preprocess import CamRestDataModule, SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

data = CamRestDataModule(data_dir="./data", batch_size=32, max_vocab_size=800)
```

Saat baris ini dijalankan, `data` otomatis menjalankan seluruh pipeline dan menyimpan hasilnya di memori — siap dipakai langsung.

### 2. Atribut yang Bisa Diakses

#### A. DataLoader (langsung masuk ke training loop)

```python
data.train_loader   # DataLoader untuk training (shuffle=True)
data.val_loader      # DataLoader untuk validasi (shuffle=False)
data.test_loader     # DataLoader untuk testing (shuffle=False)
```

Setiap iterasi `for batch in data.train_loader:` menghasilkan dictionary `batch` (struktur lengkap di bagian 3).

#### B. Vocabulary

```python
data.word2idx   # Dict: {"<pad>": 0, "<sos>": 1, ..., "the": 15, ...}  (size: 749)
data.idx2word   # Dict: {0: "<pad>", 1: "<sos>", ..., 15: "the", ...}
```

Kebutuhan untuk model:

| Kebutuhan | Cara ambil |
|---|---|
| Ukuran embedding layer & output layer | `len(data.word2idx)` |
| Index token `<pad>` (untuk `ignore_index` di loss function) | `data.word2idx[PAD_TOKEN]` → selalu `0` |

#### C. Konstanta (bisa di-*import* langsung dari file)

```python
from preprocess import SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from preprocess import INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE
from preprocess import SLOT_TOKENS, INFORMABLE_SLOTS, DB_FIELD_TO_SLOT
```

### 3. Struktur `batch` dari DataLoader

```python
batch = next(iter(data.train_loader))
```

`batch` adalah dictionary dengan 6 key:

| Key | Tipe | Shape | Keterangan |
|---|---|---|---|
| `input` | `torch.Tensor` (long) | `(batch_size, max_input_len)` | Encoder input: `B_{t-1} R_{t-1} U_t`. Sudah padded per-batch. |
| `bspan_in` | `torch.Tensor` (long) | `(batch_size, max_bspan_len)` | Decoder 1 input. Diawali `<sos>`, tanpa token terakhir. |
| `bspan_tgt` | `torch.Tensor` (long) | `(batch_size, max_bspan_len)` | Decoder 1 target. Tanpa `<sos>`, diakhiri `<eos>`. |
| `resp_in` | `torch.Tensor` (long) | `(batch_size, max_resp_len)` | Decoder 2 input. Diawali `<sos>`, tanpa token terakhir. |
| `resp_tgt` | `torch.Tensor` (long) | `(batch_size, max_resp_len)` | Decoder 2 target. Tanpa `<sos>`, diakhiri `<eos>`. |
| `input_oov` | `list[list[str]]` | Panjang luar = `batch_size` | Daftar kata OOV per sample. Dibutuhkan oleh mekanisme CopyNet. |

> **Catatan penting:** Panjang dimensi kedua (`max_input_len`, `max_bspan_len`, `max_resp_len`) berubah-ubah per batch karena *dynamic padding*. Ini normal dan sudah ditangani oleh `pad_sequence` di `collate_fn`.

### 4. Yang Perlu Disiapkan di Sisi Model

**Embedding Layer:**

```python
vocab_size = len(data.word2idx)  # 749
```

**Forward Pass (per batch):**

```python
encoder_input     = batch["input"]         # (B, L_enc)
bspan_decoder_in  = batch["bspan_in"]      # (B, L_bspan)
bspan_target      = batch["bspan_tgt"]     # (B, L_bspan)
resp_decoder_in   = batch["resp_in"]       # (B, L_resp)
resp_target       = batch["resp_tgt"]      # (B, L_resp)
oov_words         = batch["input_oov"]     # list[list[str]], panjang = B
```

**Loss Function:**

```python
pad_idx = data.word2idx[PAD_TOKEN]  # selalu 0
# Gunakan ignore_index=pad_idx agar loss tidak dihitung untuk token <pad>
```

**CopyNet (jika arsitektur mendukung):**

- `oov_words[i]` berisi daftar kata unik yang tidak ada di vocab untuk sample ke-`i` dalam batch.
- Index OOV dimulai dari `vocab_size` (749, 750, 751, ...).
- Model perlu memperluas output layer secara dinamis menjadi `vocab_size + max(len(oov_words[i]))` per batch.

### 5. Contoh Minimal Pengambilan Data

```python
from preprocess import CamRestDataModule

data = CamRestDataModule(data_dir="./data", batch_size=32)

vocab_size = len(data.word2idx)
pad_idx = data.word2idx["<pad>"]

for batch in data.train_loader:
    enc_in     = batch["input"]
    bspan_in   = batch["bspan_in"]
    bspan_tgt  = batch["bspan_tgt"]
    resp_in    = batch["resp_in"]
    resp_tgt   = batch["resp_tgt"]
    oov        = batch["input_oov"]

    break
```