# Planning Replikasi TSCP (Two Stage CopyNet)

**Paper**: Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures (Lei et al., ACL 2018)

**Dataset**: CamRest676 (restaurant reservation domain, Cambridge)

---

## Referensi Section Paper

| Section | Isi | Persamaan |
|---------|-----|-----------|
| 3.1 | Encoder-Decoder Seq2Seq | Eq. 1-3 (Attention + Generate) |
| 4.1 | Belief Spans | Format `<Inf>...<Inf> <Req>...</Req>` |
| 4.2 | Sequicity Framework | Eq. 4a, 4b (Two-stage decoding) |
| 4.3 | Two Stage CopyNet | Eq. 5-9 (Copy mechanism + k_t conditioning) |
| 4.4 | Training | Eq. 10 (Cross Entropy), Eq. 11 (Policy Gradient) |
| 5.2 | Parameter Settings | Semua hyperparameter |

---

## FASE 1: Persiapan Data dan Knowledge Base

Data harus beres sebelum mulai coding model. Masalah data adalah sumber bug terbesar dalam replikasi paper dialog.

### 1.1 Dataset dan Knowledge Base

Dua file JSON yang dibutuhkan:

- **CamRest676.json** -- 676 dialogues berisi percakapan multi-turn dengan SLU annotations per turn (user utterance, system response, intent, slot-value pairs).
- **CamRestDB.json** -- 110 entri restoran. Setiap entri memiliki fields: `name`, `food`, `area`, `pricerange`, `address`, `phone`, `postcode`, `location`, `type`, `id`.

### 1.2 Delexicalization System Response

Di paper (Section 4.1 dan Figure 1), respons mesin (R_t) saat training tidak boleh berisi nama entitas asli. Nama restoran, alamat, dan informasi spesifik lainnya harus diganti dengan placeholder.

Mapping field database ke placeholder:

| Field DB | Placeholder |
|----------|-------------|
| name | NAME_SLOT |
| address | ADDRESS_SLOT |
| phone | PHONE_SLOT |
| postcode | POSTCODE_SLOT |

Yang perlu diperhatikan: hanya R_t (system response) yang dideleksikalisasi. User utterance (U_t) dan Belief Span (B_t) dibiarkan asli ("Italian", "cheap") tanpa penggantian apapun.

### 1.3 Konstruksi Belief Span (B_t)

Dari SLU annotations setiap turn, bangun belief span dengan format:

```
<inf> value1 ; value2 </inf> <req> slot1 ; slot2 </req>
```

Dua jenis slot yang diambil:

- **Informable slots**: ambil VALUES dari SLU dengan act="inform". Contoh: "italian", "cheap". Slot yang termasuk informable di CamRest676: `food`, `area`, `pricerange`.
- **Requestable slots**: ambil SLOT NAMES dari SLU dengan act="request". Contoh: "address", "phone". Slot yang termasuk requestable: `address`, `phone`, `postcode`, `food`, `area`, `pricerange`, `name`.

### 1.4 Format Input-Output Seq2Seq

Berdasarkan Eq. 4 di paper, setiap turn dialogue diformat menjadi pasangan input-target:

```
Input  (X):  B_{t-1} + R_{t-1} + U_t
Target (Y):  B_t (Stage 1) + R_t (Stage 2)
```

B_0 dan R_0 diinisialisasi sebagai **empty string** (string kosong), bukan token `<sos>`. Paper menyebut secara eksplisit: "B_0 and R_0 are initialized as empty sequences." Token `<sos>` hanya dipakai sebagai start token untuk decoder saat proses generate, bukan sebagai isi B_0/R_0. Jika B_0 diisi `<sos>`, encoder akan memproses token tambahan yang seharusnya tidak ada.

### 1.5 Split Data

Rasio 3:1:1 sesuai paper:

| Set | Jumlah Dialogues |
|-----|-----------------|
| Train | 408 |
| Dev | 136 |
| Test | 136 |

### 1.6 Build Vocabulary

Vocabulary dihitung hanya dari Training Set untuk menghindari data leakage. Kata-kata dari Dev dan Test Set tidak boleh masuk ke vocabulary; jika muncul saat inference, kata tersebut menjadi OOV dan ditangani oleh copy mechanism.

Ukuran vocabulary CamRest676 sekitar 800 token (sesuai paper |V| = 800).

Special tokens yang perlu didaftarkan: `<pad>`, `<sos>`, `<eos>`, `<unk>`, `<inf>`, `</inf>`, `<req>`, `</req>`, dan semua SLOT placeholder tokens.

---

## FASE 2: Arsitektur TSCP (PyTorch)

Tiga komponen utama yang perlu dibangun: Encoder, Decoder Stage 1, dan Decoder Stage 2.

### 2.1 Encoder (GRU)

Komponen:
- `nn.Embedding(vocab_size, embed_size=50, padding_idx=0)`
- `nn.GRU(embed_size=50, hidden_size=50, batch_first=True)`

Output yang dihasilkan:
- H^(x): hidden states seluruh timestep (dipakai oleh Stage 1 untuk attention dan copy)
- h_final: hidden state terakhir (dipakai sebagai initial hidden Stage 1)

### 2.2 Decoder Stage 1 -- Generate Belief Span (B_t)

GRU Decoder dengan CopyNet. Di dalam decoder ini terdapat dua mekanisme yang berbeda tujuan tetapi berjalan secara paralel pada setiap langkah decoding.

#### A. Attention untuk Generate Probability (Eq. 1-3)

Proses menghasilkan P_generate bukan sekadar "softmax dari hidden state ke vocabulary". Prosesnya melibatkan tiga langkah berurutan:

```
Langkah 1:  u_ij = v^T * tanh(W1 * h_i^(x) + W2 * h_j^(y))       [Eq. 1]
Langkah 2:  h_tilde = sum( softmax(u_ij) * h_i^(x) )              [Eq. 2]
Langkah 3:  P_gen = softmax( O * [h_tilde ; h_j^(y)] )            [Eq. 3]
```

Input ke output projection (matrix O) adalah **concatenation** dari context vector (h_tilde, hasil weighted sum dari encoder hiddens) dan decoder hidden state (h_j^(y)). Bukan hanya decoder hidden state saja. Jika context vector dilewati, model kehilangan informasi dari encoder dan performanya akan jauh di bawah paper.

Komponen PyTorch yang diperlukan:
- `W1 = nn.Linear(hidden_size, hidden_size)` -- proyeksi encoder hidden
- `W2 = nn.Linear(hidden_size, hidden_size)` -- proyeksi decoder hidden
- `v = nn.Linear(hidden_size, 1)` -- proyeksi ke scalar (attention score)
- `output_proj = nn.Linear(hidden_size * 2, vocab_size)` -- dimensi input hidden*2 karena concatenation [h_tilde ; h^(y)]

#### B. Copy Probability (Eq. 5-6)

Mekanisme untuk meng-copy token langsung dari input sequence X. Rumus berbeda dari attention di bagian A:

```
psi(x_i) = sigma(h_i^(x)^T * W_c) * h_j^(y)                      [Eq. 6]
P_copy(v) = (1/Z) * sum( exp(psi(x_i)) )  untuk semua x_i = v     [Eq. 5]
```

Komponen: `W_copy = nn.Linear(hidden_size, hidden_size)` -- copy projection matrix

#### C. Final Probability (CopyNet)

Kedua probabilitas dijumlahkan:

```
P_final(v) = P_generate(v) + P_copy(v),    v in V union X
```

Hal penting: ukuran output space bukan fixed `len(vocabulary)`, melainkan `len(vocabulary) + len(OOV_tokens_in_input)`. Setiap sample bisa memiliki jumlah OOV yang berbeda, sehingga dimensi output bersifat dinamis. Ini area paling rawan bug dalam implementasi CopyNet -- pastikan dimensi tensor di PyTorch menangani dynamic length ini dengan benar, terutama pada operasi `scatter_add_`.

Output Stage 1: token-token B_t beserta hidden states per-token (hidden states ini dibutuhkan Stage 2).

### 2.3 Decoder Stage 2 -- Generate Response (R_t)

Strukturnya serupa dengan Stage 1, tetapi ada empat perbedaan kunci.

#### A. Inisialisasi Hidden State

Hidden state awal GRU Decoder Stage 2 diambil dari **last hidden state Decoder Stage 1** -- bukan dari encoder.

#### B. Conditioning k_t (Eq. 9)

Hasil KB search direpresentasikan sebagai vektor 3-dimensi one-hot (k_t):
- `[1, 0, 0]` = no match (tidak ada restoran yang cocok)
- `[0, 1, 0]` = exact match (tepat 1 restoran ditemukan)
- `[0, 0, 1]` = multiple match (lebih dari 1 restoran cocok)

Vektor k_t di-concatenate ke embedding input pada setiap langkah decoding:

```
y'_j = [y_j ; k_t],    j in [m'+1, m]                              [Eq. 9]
```

Konsekuensinya, GRU input size di Stage 2 = `embed_size + 3`, bukan `embed_size` saja.

#### C. Sumber Attention dan Copy: Bspan, Bukan Encoder

Paper Section 4.3 menyebut secara eksplisit: "we have copy-attention mechanism on B_t instead of on X: treating all tokens of B_t as the candidate for copying and attention."

Ini berarti kedua mekanisme di Stage 2 -- baik attention untuk P_generate maupun copy untuk P_copy -- semuanya mengacu ke hidden states dari B_t, bukan encoder output H^(x). Perubahan sumber ini berlaku untuk attention (Eq. 7-8) dan juga untuk konteks yang dipakai menghitung P_generate.

#### D. Ringkasan Perbedaan Stage 1 vs Stage 2

| Aspek | Stage 1 | Stage 2 |
|-------|---------|---------|
| Attention P_gen | Ke encoder outputs H^(x) | Ke bspan hidden states |
| Copy P_copy | Dari input X (Eq. 5-6) | Dari B_t (Eq. 7-8) |
| Hidden init | Dari encoder final hidden | Dari Stage 1 final hidden |
| Input embedding | embed(y_j) | [embed(y_j) ; k_t] |
| GRU input dim | embed_size | embed_size + 3 |

---

## FASE 3: Supervised Learning

Model harus mampu menghasilkan kalimat yang masuk akal sebelum di-fine-tune dengan RL.

### 3.1 Training Flow

Encoder, Decoder Stage 1, dan Decoder Stage 2 dilatih bersama dalam satu forward pass per turn, menggunakan teacher forcing di kedua stage:

```
1. Encode X = B_{t-1} R_{t-1} U_t                    --> H^(x), h_final
2. Decode B_t dengan teacher forcing                  --> bspan_log_probs
   (masukkan ground truth B_t token per token)
3. Decode R_t dengan teacher forcing                  --> response_log_probs
   (masukkan ground truth R_t token per token)
4. Loss = CE(predicted_B_t, true_B_t) + CE(predicted_R_t, true_R_t)
5. Backprop sekali untuk seluruh model (encoder + decoder1 + decoder2)
```

Ketiga komponen di-optimize bersama melalui satu optimizer dan satu backward pass, bukan terpisah.

### 3.2 Loss Function

Cross entropy loss standar (Eq. 10) dihitung untuk token-token B_t dan R_t, lalu dijumlahkan. Perlu penanganan khusus karena CopyNet: target indices bisa berada di extended vocabulary (V union X), bukan hanya V.

### 3.3 Hyperparameters (Section 5.2)

| Parameter | Value |
|-----------|-------|
| Hidden size | 50 |
| Embedding size | 50 |
| Vocabulary size | ~800 |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Gradient clipping | 5.0 |
| Batch size | 32 |
| Max epochs | 100 |

### 3.4 Early Stopping

Paper Section 5.2 menyebut: "Early stopping is performed on developing set." Mekanismenya: monitor loss pada dev set setiap epoch, simpan model dengan dev loss terbaik, hentikan training jika tidak ada perbaikan selama N epoch berturut-turut (patience sekitar 10 epoch). Ini penting untuk mencegah overfitting, terutama karena CamRest676 hanya memiliki 408 training dialogues.

---

## FASE 4: Reinforcement Learning (Fine-Tuning)

Setelah model mampu menghasilkan kalimat yang masuk akal dari supervised training, RL digunakan untuk mendorong model agar lebih fokus pada penyelesaian tugas (task completion).

### 4.1 Setup Policy Network

Hanya Decoder Stage 2 yang diperlakukan sebagai policy network (pi_theta). Encoder dan Decoder Stage 1 tidak di-update selama RL. State didefinisikan sebagai hidden vector GRU, dan action adalah token yang dipilih (di-sample, bukan argmax).

### 4.2 Reward Function (Section 4.4)

Per-step reward:
- r(j) = +1.0 jika token yang di-generate adalah placeholder request slot yang memang diminta user (contoh: model men-generate `<address>` dan user memang minta alamat)
- r(j) = -0.1 untuk semua token lainnya

### 4.3 Discounted Return

```
R(j) = r(j) + lambda * r(j+1) + lambda^2 * r(j+2) + ... + lambda^(m-j+1) * r(m)
```

dengan lambda = 0.8 (decay parameter).

### 4.4 Policy Gradient Update (Eq. 11)

```
gradient = (1/(m-m')) * sum_{j=m'+1}^{m} R(j) * grad(log pi_theta(y_j))
```

### 4.5 Hyperparameters RL

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001 |
| Lambda (decay) | 0.8 |
| Beam size | 10 |

### 4.6 Stabilitas Training RL

Di awal RL, model mungkin tidak pernah men-generate placeholder yang benar (misalnya `<address>`) sehingga reward-nya -0.1 terus-menerus. Ini perilaku yang wajar. Untuk mencegah vanishing gradient, normalize returns dengan mengurangi mean dan membagi standard deviation sebagai baseline.

---

## FASE 5: Inference

### 5.1 Alur Per Turn

```
1. FORMAT INPUT
   Gabungkan: B_{t-1} + R_{t-1} + U_t
   (B_0 dan R_0 = empty string)

2. ENCODE
   Masukkan input ke Encoder --> H^(x), h_final

3. DECODE STAGE 1 (Bspan)
   Beam Search (beam_size=10) di Decoder Stage 1
   --> Generate B_t

4. KB SEARCH
   a. Parse B_t: ambil values di dalam <Inf>...</Inf>
   b. Query ke database restoran (match food/area/pricerange)
   c. Hitung jumlah match, tentukan k_t:
      - 0 match    --> k_t = [1, 0, 0]
      - 1 match    --> k_t = [0, 1, 0]
      - >1 match   --> k_t = [0, 0, 1]

5. DECODE STAGE 2 (Response)
   Beam Search di Decoder Stage 2
   Conditioned on: B_t hidden states + k_t
   --> Generate R_t (masih berisi placeholder)

6. LEXICALIZATION
   Ambil 1 restoran dari hasil KB Search.
   Ganti placeholder di R_t dengan value asli dari entri restoran tersebut.
   Contoh: NAME_SLOT --> "pizza hut city centre"

7. POST-PROCESSING: INCONSISTENT REQUESTS (Section 5.7)
   Jika user berganti pikiran (ada 2 nilai food di B_t),
   jalankan script post-processing:
   - Deteksi apakah 2 value merujuk ke slot yang sama
   - Simpan hanya value yang lebih baru (dari turn terakhir)

8. OUTPUT
   Tampilkan kalimat final ke user
```

### 5.2 Beam Search dengan CopyNet

Beam search untuk CopyNet lebih kompleks dari beam search standar karena output space-nya dinamis (V union X, bukan hanya V). Setiap hypothesis di beam perlu tracking terhadap extended vocabulary mapping. Token yang ada di input tetapi tidak ada di vocabulary (OOV) hanya bisa dihasilkan lewat copy mechanism, dan beam search harus mampu menangani kasus ini -- baik saat scoring maupun saat menentukan embedding input untuk langkah berikutnya (OOV tokens tidak memiliki embedding, sehingga perlu di-fallback ke `<unk>` embedding).

---

## FASE 6: Evaluasi

### 6.1 Metrik (Section 5)

**BLEU** -- mengukur kualitas bahasa R_t. Dihitung menggunakan corpus-level BLEU (misalnya `nltk.translate.bleu_score.corpus_bleu`), membandingkan R_t yang di-generate model dengan R_t ground truth dari dataset.

**Entity Match Rate** -- mengukur task completion dari sisi constraints. Bersifat binary per dialog: bernilai 1 jika B_t yang diprediksi model menghasilkan entitas yang sama dengan B_t ground truth saat keduanya di-query ke Knowledge Base, bernilai 0 jika tidak.

**Success F1** -- mengukur task completion dari sisi requests. Dihitung sebagai F1 score: apakah R_t yang dihasilkan model memuat semua request slot yang diminta user. Contoh: jika user meminta alamat dan nomor telepon, apakah response model memberikan keduanya. F1 digunakan (bukan accuracy/recall saja) untuk menyeimbangkan antara kelengkapan dan ketepatan.

### 6.2 Target Score (Table 2 Paper)

| Metrik | TSCP (Paper) |
|--------|-------------|
| Entity Match Rate | 0.927 |
| BLEU | 0.253 |
| Success F1 | 0.854 |
| Training Time | 7.3 min |

---

## Catatan Teknis

### CopyNet Dynamic Vocab Size

Di CopyNet, ukuran output softmax bukan `len(vocabulary)` tetap, melainkan `len(vocabulary) + len(OOV_tokens_in_input)`. Setiap sample memiliki jumlah OOV berbeda, sehingga dimensi output bersifat dinamis per batch. Implementasi di PyTorch memerlukan operasi `scatter_add_` untuk menggabungkan copy probability ke posisi token yang tepat dalam extended vocabulary -- ini area yang paling sering menimbulkan bug dimensi tensor.

### Pencegahan Data Leakage

Saat membangun vocabulary, hanya kata-kata dari Training Set yang boleh dihitung. Kata dari Dev/Test Set tidak boleh masuk ke vocabulary. Jika aturan ini dilanggar, skor OOV test akan palsu karena model "sudah tahu" kata-kata yang seharusnya out-of-vocabulary.

### Penempatan Token SOS dan EOS

- Decoder INPUT (teacher forcing): `<sos>` + target tokens
- Decoder TARGET (untuk hitung loss): target tokens + `<eos>`
- B_0 dan R_0 pada turn pertama: empty string, bukan `<sos>`

### Sumber Attention Stage 2

Decoder Stage 2 harus melakukan attention dan copy ke B_t hidden states, bukan ke encoder outputs H^(x). Ini perbedaan fundamental dari CopyNet standar dan merupakan kontribusi utama paper. Jika Stage 2 tetap mengacu ke encoder outputs, model kehilangan keuntungan dari belief span sebagai representasi terkompresi dan search space menjadi terlalu besar.

---

## Diagram Arsitektur

```
Input: B_{t-1} R_{t-1} U_t
    |
    v
+------------------+
|     ENCODER      |  GRU
|                  |  --> H^(x) [all hidden states]
|                  |  --> h_final [last hidden]
+--------+---------+
         |
         v
+--------------------------------------------------+
|  DECODER STAGE 1  --  Generate B_t                |
|                                                    |
|  GRU + 2 mekanisme paralel:                        |
|                                                    |
|  +-- Attention P_gen --+  +-- Copy P_copy --+     |
|  | Q: h_dec            |  | Eq. 5-6         |     |
|  | K,V: H^(x)         |  | Source: X        |     |
|  | --> context h_tilde |  | --> P_c(v)       |     |
|  | --> softmax(O*      |  |                  |     |
|  |     [h_tilde;h])    |  |                  |     |
|  | --> P_g(v)          |  |                  |     |
|  +---------------------+  +------------------+     |
|                                                    |
|  P_final(v) = P_g(v) + P_c(v),  v in V union X    |
|                                                    |
|  Output: B_t tokens + bspan hidden states          |
+--------+----------------------------------+--------+
         |                                  |
         |  Parse B_t --> Query KB --> k_t   |
         |  [no_match / exact / multiple]    |
         |                                  |
         v                                  v
+--------------------------------------------------+
|  DECODER STAGE 2  --  Generate R_t                |
|                                                    |
|  Init hidden = last hidden Stage 1                 |
|  GRU input = [embed(y_j) ; k_t]  (Eq. 9)         |
|                                                    |
|  +-- Attention P_gen --+  +-- Copy P_copy --+     |
|  | Q: h_dec            |  | Eq. 7-8         |     |
|  | K,V: B_t hiddens    |  | Source: B_t      |     |
|  | --> P_g(v)          |  | --> P_c(v)       |     |
|  +---------------------+  +------------------+     |
|                                                    |
|  P_final(v) = P_g(v) + P_c(v)                     |
|                                                    |
|  Output: R_t (delexicalized)                       |
+--------+-------------------------------------------+
         |
         v
    Lexicalize --> Final Response ke User
```
