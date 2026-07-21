# Paper: Sequicity

## 1. Metadata
*   **Judul:** Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures
*   **Penulis:** Wenqiang Lei, Xisen Jin, Zhaochun Ren, Xiangnan He, Min-Yen Kan, Dawei Yin
*   **Konferensi:** ACL 2018 (Long Papers)
*   **Domain:** Task-Oriented Dialogue Systems (Sistem Dialog Berorientasi Tugas)

---

## 2. Latar Belakang & Masalah (The "Why")
Sistem dialog tradisional (seperti NDM) menggunakan desain **Pipeline** yang memiliki beberapa kelemahan fatal:
1.  **Kompleksitas & Kerapuhan:** Terdiri dari banyak modul terpisah (Intent classifier, Belief tracker, Policy, Response generator).
2.  **Gagal pada OOV (Out-of-Vocabulary):** *Belief tracker* menggunakan *multi-class classifier* dengan label yang sudah ditentukan sebelumnya (pre-defined). Jika user meminta nilai baru (misal: "makanan Zanzibarian"), sistem gagal total.
3.  **Delexicalization Input Tidak Skalabel:** Sistem lama mencoba mengganti nilai spesifik di *input user* dengan tag (misal: `<food>`). Karena keragaman bahasa manusia (lexical diversity), membuat kamus untuk ini sangat sulit dan tidak efisien.
4.  **Waktu Training Lama:** Membutuhkan waktu training yang sangat besar karena kompleksitas pipeline.

---

## 3. Solusi & Inovasi Utama (The "What")
Penulis mengusulkan **Sequicity**, sebuah framework holistik yang menggunakan **satu model Sequence-to-Sequence (Seq2Seq)** tunggal.

### Inovasi 1: Belief Spans (Bspan)
Alih-alih menggunakan classifier, sistem "menulis" status keyakinan dialog dalam bentuk teks terstruktur:
*   **Format:** `<Inf> nilai_slot_1 ; nilai_slot_2 </Inf> <Req> nama_slot_1 ; nama_slot_2 </Req>`
*   `<Inf>` (Informable): Berisi nilai/kendala user untuk pencarian database.
*   `<Req>` (Requestable): Berisi informasi yang diminta user untuk dijawab.

### Inovasi 2: Pergeseran Delexicalization
*   **Sistem Lama:** Mendelexicalisasi *Input User* (Gagal pada bahasa yang beragam).
*   **Sequicity:** Membiarkan input user apa adanya, dan **hanya mendelexicalisasi Respons Mesin ($R_t$)** saat training (mengganti nama restoran/alamat dengan `NAME_SLOT`, `ADDRESS_SLOT`). Saat inference, placeholder diganti dengan data asli dari Knowledge Base (Lexicalization).

### Inovasi 3: Asumsi Markov (Markov Assumption)
Model tidak perlu membaca seluruh riwayat percakapan. Pada *turn* $t$, input hanya membutuhkan:
$$X = B_{t-1} + R_{t-1} + U_t$$
*(Bspan sebelumnya + Respons sebelumnya + Ucapan user saat ini)*.

---

## 4. Arsitektur Model: TSCP (The "How")
Paper ini mengimplementasikan Sequicity menggunakan **Two Stage CopyNet (TSCP)**.

### Komponen Utama
1.  **Encoder (1 GRU):** Memproses input sequence $X$ menjadi *hidden states* $H^{(x)}$.
2.  **Decoder Stage 1 - Bspan GRU (1 GRU):** 
    *   Menghasilkan **Belief Span ($B_t$)**.
    *   **Copy Mechanism:** Menyalin kata langsung dari **Input ($X$)**.
3.  **Jembatan: Knowledge Base (KB) Search:**
    *   Decoding *pause* sejenak.
    *   Ekstrak nilai dari `<Inf>` di $B_t$, lalu query ke Database.
    *   Hasilkan vektor kondisi **$k_t$** (3 dimensi: *no match, exact match, multiple matches*).
4.  **Decoder Stage 2 - Response GRU (1 GRU):**
    *   Menghasilkan **Respons Mesin ($R_t$)**.
    *   *Initial hidden state* diambil dari *last hidden state* Decoder 1.
    *   **Copy Mechanism:** Menyalin kata dari **Bspan ($B_t$)** (bukan dari $X$, untuk menghemat ruang pencarian).
    *   **Conditioning:** Vektor $k_t$ di-*append* ke *embedding* input di setiap langkah.

---

## 5. Proses Training & Optimasi

### A. Supervised Learning (Pre-training)
*   **Loss Function:** Standard Cross-Entropy Loss.
*   **Teacher Forcing:** Digunakan saat training Decoder.

### B. Reinforcement Learning (Fine-Tuning)
Digunakan khusus untuk **Decoder Stage 2** agar model fokus menyelesaikan tugas (task completion), bukan hanya merangkai kalimat yang lancar.
*   **Policy Network:** Decoder Stage 2.
*   **Reward Function ($r(j)$):**
    *   **+1** jika model berhasil men-generate *placeholder* yang diminta user (misal: `<address>`).
    *   **-0.1** jika men-generate kata lain (untuk mencegah respons bertele-tele).
*   **Decay Parameter ($\lambda$):** 0.8.
*   **Algoritma:** Policy Gradient.

---

## 6. Eksperimen & Evaluasi

### Dataset
1.  **CamRest676:** Domain restoran. Split: **3:1:1** (Train: 408, Dev: 136, Test: 136).
2.  **KVRET:** Domain kalender, cuaca, POI. Split: **8:1:1** (Train: 2425, Dev: 302, Test: 302).

###  Metrik Evaluasi
1.  **BLEU:** Kualitas bahasa (fluency).
2.  **Entity Match Rate:** Keberhasilan menangkap semua constraint user untuk dicari di KB (Biner 0/1).
3.  **Success F1:** Keberhasilan menjawab semua *request* user (menyeimbangkan Recall dan Precision).
4.  **Training Time:** Waktu yang dibutuhkan hingga konvergen.

### Hasil Utama
*   TSCP mengungguli semua baseline (NDM, KVRN) di semua metrik.
*   **OOV Test:** TSCP tetap bekerja sangat baik bahkan jika 100% slot values adalah OOV, sementara pipeline (NDM) gagal total.
*   **Efisiensi:** Waktu training lebih cepat **satu order of magnitude** (misal: 7.3 menit vs 91.9 menit di CamRest676).
*   **Model Size:** Ukuran parameter jauh lebih kecil dan tidak membengkak saat jumlah slot values bertambah.

---

## 7. Detail Implementasi (Hyperparameters)
*   **Framework:** PyTorch.
*   **Hidden Size & Embedding Size ($d$):** 50.
*   **Vocabulary Size ($|V|$):** 800 (CamRest676), 1400 (KVRET).
*   **Optimizer:** Adam.
*   **Learning Rate:** 0.003 (Supervised), 0.0001 (RL).
*   **Decoding Strategy:** Beam Search dengan **beam size = 10**.
*   **Early Stopping:** Dilakukan pada Development set.

---

## 8. Penanganan Kasus Khusus (Discussions)
*   **User Berubah Pikiran (Inconsistent Requests):** Jika user mengganti constraint (misal: dari Jepang ke Prancis), Sequicity bisa mempelajarinya dari data. Penulis juga menambahkan **script post-processing** sederhana: jika ada dua nilai untuk slot yang sama di Bspan, hapus nilai lama dan simpan nilai dari *turn* terbaru.
*   **Lexicalization (Post-Processing):** Saat inference, setelah Decoder 2 menghasilkan respons berisi `NAME_SLOT`, sistem mengambil entitas yang cocok dari hasil KB Search dan mengganti string `NAME_SLOT` dengan nama restoran asli.