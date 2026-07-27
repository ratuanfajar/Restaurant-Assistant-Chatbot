# Chatbot Asisten Restoran (TSCP)

Antarmuka chat berbasis Streamlit untuk model **Two Stage CopyNet (TSCP)** —
replikasi paper *Sequicity* (Lei et al., ACL 2018) pada dataset CamRest676.

## Menjalankan

```bash
uv run streamlit run app.py
# atau
streamlit run app.py
```

Lalu buka `http://localhost:8501`.

## Isi

| File | Peran |
|------|-------|
| `app.py` | UI Streamlit (riwayat chat, tombol reset, panel debug belief-span/KB) |
| `chatbot_engine.py` | Engine inferensi mandiri: arsitektur TSCP + tokenisasi + KB search + lexicalization + greedy decoding |
| `checkpoints/tscp_supervised_v2_best.pt` | Bobot model (supervised v2) — juga menyimpan `word2idx` |
| `data/CamRestDB.json` | Knowledge base 110 restoran untuk KB search & lexicalization |

## Alur per turn

```
input = B_{t-1} + R_{t-1} + U_t
  → Encoder (GRU)
  → Decoder Stage 1  → belief span B_t        (attention + copy dari input X)
  → KB search(B_t)   → k_t (no / exact / multiple match)
  → Decoder Stage 2  → response R_t (delex)    (attention + copy dari B_t, condition k_t)
  → lexicalize(R_t)  → jawaban final ke user
```

Percakapan bersifat multi-turn: `chatbot_engine.RestaurantAssistant` menyimpan
`B_{t-1}` dan `R_{t-1}` antar-giliran. Tombol **"Percakapan baru"** meng-reset state
(`B_0 = R_0 = kosong`).

## Catatan

- Model dilatih pada data berbahasa Inggris — berikan input dalam bahasa Inggris.
- Slot yang dikenali: masakan (mis. italian, chinese), area (centre/north/south/east/west),
  harga (cheap/moderate/expensive); permintaan alamat / nomor telepon / kode pos.
- **Decoding**: dipakai *greedy* (bukan beam search seperti paper Section 5.2). Ini
  disengaja — metrik checkpoint diukur dengan greedy, jadi output chatbot konsisten
  dengan angka evaluasi yang dilaporkan.
- Checkpoint memakai belief-span tag **`<Inf>`/`<Req>`** (kapital, ikut paper) dan vocab 613
  (pipeline v2). Engine tidak bergantung pada modul training — `word2idx` dibaca langsung
  dari checkpoint.
- **Pilihan checkpoint**: dipakai **supervised v2** karena responsnya lebih natural. Model
  **RL v2** (`tscp_rl_v2_best.pt`) menang di Success F1 tapi cenderung *reward-hacking*
  (mengulang placeholder slot), sehingga teksnya kurang enak dibaca untuk demo. Ganti path
  `CHECKPOINT` di `app.py` bila ingin memakai model RL.
