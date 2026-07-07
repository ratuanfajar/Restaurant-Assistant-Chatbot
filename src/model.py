"""
Two Stage CopyNet (TSCP) — implementasi PyTorch dari paper Sequicity.

Arsitektur:
  1. Encoder (GRU): membaca X = B_{t-1} R_{t-1} U_t → H^(x)
  2. DecoderStage1 (GRU + CopyNet): generate B_t dari H^(x)
     - Attention + Generate probability (Eq. 1-3)
     - Copy probability dari input X (Eq. 5-6)
  3. DecoderStage2 (GRU + CopyNet): generate R_t dari B_t
     - Initial hidden = last hidden dari Stage 1
     - Attention + Generate probability dari B_t hidden states (bukan encoder)
     - Copy probability dari B_t (Eq. 7-8)
     - Conditioning pada k_t (Eq. 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config as config


class Encoder(nn.Module):
    """
    GRU Encoder: membaca input sequence X dan menghasilkan hidden states H^(x).
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.hidden_size = hidden_size

    def forward(self, input_seq, input_lengths=None):
        """
        Args:
            input_seq: (batch, seq_len) — padded token indices
            input_lengths: (batch,) — actual lengths (optional, untuk packing)
            
        Returns:
            outputs: (batch, seq_len, hidden_size) — all hidden states H^(x)
            hidden: (num_layers, batch, hidden_size) — final hidden state
        """
        embedded = self.embedding(input_seq)  # (batch, seq_len, embed_size)

        if input_lengths is not None:
            # Pack untuk efisiensi (skip padding tokens)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.gru(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.gru(embedded)

        return outputs, hidden


class DecoderStage1(nn.Module):
    """
    Decoder Stage 1: Generate belief span B_t.
    
    Menggunakan CopyNet dengan:
    - Attention ke encoder outputs H^(x) untuk generate probability (Eq. 1-3)
    - Copy mechanism dari input X (Eq. 5-6)
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # === Attention untuk Generate (Eq. 1-3) ===
        # u_ij = v^T * tanh(W1 * h_i^(x) + W2 * h_j^(y))
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)  # untuk encoder hidden
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)  # untuk decoder hidden
        self.v = nn.Linear(hidden_size, 1, bias=False)             # project ke scalar

        # Output projection: [context ; hidden] → vocab distribution (Eq. 3)
        # O ∈ R^{|V| x 2d} karena concat [h̃^(x) ; h^(y)]
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)

        # === Copy mechanism (Eq. 5-6) ===
        # ψ(x_i) = σ(h_i^(x)^T * W_c) * h_j^(y)
        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, encoder_outputs):
        """
        Satu langkah decoding.
        
        Args:
            dec_input: (batch, 1) — token index saat ini
            hidden: (num_layers, batch, hidden_size) — GRU hidden state
            encoder_outputs: (batch, src_len, hidden_size) — H^(x)
            
        Returns:
            gen_log_prob: (batch, vocab_size) — log prob generate
            copy_score: (batch, src_len) — raw copy scores (belum softmax)
            new_hidden: (num_layers, batch, hidden_size)
        """
        # --- GRU step ---
        embedded = self.embedding(dec_input)  # (batch, 1, embed_size)
        gru_out, new_hidden = self.gru(embedded, hidden)
        # gru_out: (batch, 1, hidden_size) = h_j^(y)
        h_dec = gru_out.squeeze(1)  # (batch, hidden_size)

        # --- Attention untuk Generate (Eq. 1-2) ---
        # W1 * h_i^(x): (batch, src_len, hidden_size)
        enc_proj = self.W1(encoder_outputs)
        # W2 * h_j^(y): (batch, hidden_size) → expand ke (batch, src_len, hidden_size)
        dec_proj = self.W2(h_dec).unsqueeze(1).expand_as(enc_proj)
        # u_ij = v^T * tanh(W1*h_i + W2*h_j)
        energy = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(2)  # (batch, src_len)
        attn_weights = F.softmax(energy, dim=1)  # (batch, src_len)

        # Context vector h̃_j^(x) = weighted sum of encoder hiddens (Eq. 2)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden_size)

        # Generate probability (Eq. 3): softmax(O * [h̃ ; h^(y)])
        concat = torch.cat([context, h_dec], dim=1)  # (batch, hidden*2)
        gen_logits = self.output_proj(concat)  # (batch, vocab_size)
        gen_log_prob = F.log_softmax(gen_logits, dim=1)

        # --- Copy scores (Eq. 5-6) ---
        # ψ(x_i) = σ(h_i^(x)^T * W_c) * h_j^(y)
        # h_i^(x) * W_c: (batch, src_len, hidden_size)
        copy_proj = torch.sigmoid(self.W_copy(encoder_outputs))  # (batch, src_len, hidden)
        # Dot product dengan h_dec: (batch, src_len)
        copy_score = torch.bmm(copy_proj, h_dec.unsqueeze(2)).squeeze(2)

        return gen_log_prob, copy_score, new_hidden

    def forward(self, dec_input_seq, hidden, encoder_outputs, source_indices_ext, ext_vocab_size):
        """
        Full forward pass untuk training (teacher forcing).
        
        Args:
            dec_input_seq: (batch, tgt_len) — decoder input (SOS + target tokens)
            hidden: initial hidden state dari encoder
            encoder_outputs: (batch, src_len, hidden_size)
            source_indices_ext: (batch, src_len) — extended vocab indices untuk input
            ext_vocab_size: int — |V| + |OOV|
            
        Returns:
            all_log_probs: (batch, tgt_len, ext_vocab_size) — combined log probabilities
            final_hidden: decoder final hidden state
        """
        batch_size, tgt_len = dec_input_seq.size()
        all_log_probs = []

        for t in range(tgt_len):
            dec_input = dec_input_seq[:, t].unsqueeze(1)  # (batch, 1)
            gen_log_prob, copy_score, hidden = self.forward_step(
                dec_input, hidden, encoder_outputs
            )

            # Combine generate + copy
            combined = self._combine_probs(
                gen_log_prob, copy_score, source_indices_ext, ext_vocab_size
            )
            all_log_probs.append(combined)

        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch, tgt_len, ext_vocab)
        return all_log_probs, hidden

    def _combine_probs(self, gen_log_prob, copy_score, source_indices, ext_vocab_size):
        """
        Gabungkan P_gen + P_copy ke extended vocabulary space.
        P(v) = P_gen(v) + P_copy(v),   v ∈ V ∪ X
        """
        batch_size = gen_log_prob.size(0)
        device = gen_log_prob.device

        # Gen prob → probability space
        gen_prob = gen_log_prob.exp()  # (batch, vocab_size)

        # Extend ke ext_vocab_size
        if ext_vocab_size > self.vocab_size:
            extra = torch.zeros(batch_size, ext_vocab_size - self.vocab_size, device=device)
            gen_prob = torch.cat([gen_prob, extra], dim=1)

        # Copy prob via softmax
        copy_prob = F.softmax(copy_score, dim=1)  # (batch, src_len)

        # Scatter add copy probs ke posisi yang tepat
        combined = gen_prob.clone()
        combined.scatter_add_(1, source_indices, copy_prob)

        # Kembali ke log space
        combined_log = torch.log(combined + 1e-12)
        return combined_log


class DecoderStage2(nn.Module):
    """
    Decoder Stage 2: Generate response R_t.
    
    Perbedaan dari Stage 1:
    - Initial hidden state = last hidden state dari Stage 1 (bukan encoder)
    - Attention dan Copy ke B_t hidden states (bukan encoder outputs X)
    - Input embedding di-concat dengan k_t (Eq. 9)
    """

    def __init__(self, vocab_size, embed_size, hidden_size, kb_size=3, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # GRU input size = embed_size + kb_size karena concat k_t (Eq. 9)
        self.gru = nn.GRU(
            embed_size + kb_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention untuk Generate (sama seperti Stage 1, tapi ke bspan hiddens)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)

        # Copy mechanism (Eq. 7-8) — dari bspan hiddens
        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, bspan_outputs, kt):
        """
        Satu langkah decoding Stage 2.
        
        Args:
            dec_input: (batch, 1)
            hidden: (num_layers, batch, hidden_size)
            bspan_outputs: (batch, bspan_len, hidden_size) — hidden states dari B_t
            kt: (batch, 3) — KB search result indicator
            
        Returns:
            gen_log_prob, copy_score, new_hidden
        """
        embedded = self.embedding(dec_input)  # (batch, 1, embed_size)

        # Concat k_t ke embedding (Eq. 9): y'_j = [y_j ; k_t]
        kt_expanded = kt.unsqueeze(1)  # (batch, 1, 3)
        gru_input = torch.cat([embedded, kt_expanded], dim=2)  # (batch, 1, embed+3)

        gru_out, new_hidden = self.gru(gru_input, hidden)
        h_dec = gru_out.squeeze(1)  # (batch, hidden_size)

        # Attention ke bspan outputs (bukan encoder!)
        bspan_proj = self.W1(bspan_outputs)
        dec_proj = self.W2(h_dec).unsqueeze(1).expand_as(bspan_proj)
        energy = self.v(torch.tanh(bspan_proj + dec_proj)).squeeze(2)
        attn_weights = F.softmax(energy, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), bspan_outputs).squeeze(1)

        concat = torch.cat([context, h_dec], dim=1)
        gen_logits = self.output_proj(concat)
        gen_log_prob = F.log_softmax(gen_logits, dim=1)

        # Copy dari bspan (Eq. 7-8)
        copy_proj = torch.sigmoid(self.W_copy(bspan_outputs))
        copy_score = torch.bmm(copy_proj, h_dec.unsqueeze(2)).squeeze(2)

        return gen_log_prob, copy_score, new_hidden

    def forward(self, dec_input_seq, hidden, bspan_outputs, kt,
                bspan_indices_ext, ext_vocab_size):
        """
        Full forward pass Stage 2 (teacher forcing).
        
        Args:
            dec_input_seq: (batch, tgt_len)
            hidden: initial hidden (dari Stage 1)
            bspan_outputs: (batch, bspan_len, hidden_size) — B_t hidden states
            kt: (batch, 3)
            bspan_indices_ext: (batch, bspan_len) — extended vocab indices untuk bspan
            ext_vocab_size: int
        """
        batch_size, tgt_len = dec_input_seq.size()
        all_log_probs = []

        for t in range(tgt_len):
            dec_input = dec_input_seq[:, t].unsqueeze(1)
            gen_log_prob, copy_score, hidden = self.forward_step(
                dec_input, hidden, bspan_outputs, kt
            )

            combined = self._combine_probs(
                gen_log_prob, copy_score, bspan_indices_ext, ext_vocab_size
            )
            all_log_probs.append(combined)

        all_log_probs = torch.stack(all_log_probs, dim=1)
        return all_log_probs, hidden

    def _combine_probs(self, gen_log_prob, copy_score, source_indices, ext_vocab_size):
        """Sama seperti Stage 1 _combine_probs."""
        batch_size = gen_log_prob.size(0)
        device = gen_log_prob.device

        gen_prob = gen_log_prob.exp()
        if ext_vocab_size > self.vocab_size:
            extra = torch.zeros(batch_size, ext_vocab_size - self.vocab_size, device=device)
            gen_prob = torch.cat([gen_prob, extra], dim=1)

        copy_prob = F.softmax(copy_score, dim=1)
        combined = gen_prob.clone()
        combined.scatter_add_(1, source_indices, copy_prob)
        combined_log = torch.log(combined + 1e-12)
        return combined_log


class TSCP(nn.Module):
    """
    Two Stage CopyNet — model utuh yang menggabungkan:
    Encoder + DecoderStage1 (bspan) + DecoderStage2 (response).
    
    Alur forward (training, teacher forcing):
      1. Encode X → H^(x), h_final
      2. Decode B_t (Stage 1) dengan teacher forcing → bspan_log_probs, bspan_hiddens
      3. Decode R_t (Stage 2) dengan teacher forcing → response_log_probs
         - Initial hidden = last hidden dari Stage 1
         - Attention & copy ke bspan hidden states
         - Conditioned pada k_t
    """

    def __init__(self, vocab_size, embed_size=config.EMBED_SIZE,
                 hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_GRU_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder1 = DecoderStage1(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder2 = DecoderStage2(vocab_size, embed_size, hidden_size,
                                       kb_size=config.KB_INDICATOR_SIZE,
                                       num_layers=num_layers, dropout=dropout)

    def forward(self, input_seq, input_lengths,
                bspan_input, bspan_target,
                response_input, response_target,
                input_tokens_batch, bspan_tokens_batch,
                kt_batch, word2idx):
        """
        Full forward pass untuk supervised training.
        
        Args:
            input_seq: (batch, src_len)
            input_lengths: (batch,)
            bspan_input: (batch, bspan_len) — decoder input (SOS + B_t)
            bspan_target: (batch, bspan_len) — decoder target (B_t + EOS)
            response_input: (batch, resp_len)
            response_target: (batch, resp_len)
            input_tokens_batch: list of list of str — raw tokens per sample
            bspan_tokens_batch: list of list of str
            kt_batch: (batch, 3) — KB indicator vectors
            word2idx: vocabulary mapping
            
        Returns:
            loss: scalar — total cross entropy loss
            bspan_loss: scalar — Stage 1 loss saja (untuk monitoring)
            response_loss: scalar — Stage 2 loss saja
        """
        device = input_seq.device
        batch_size = input_seq.size(0)

        # === 1. Encode ===
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        # === 2. Build extended vocab mappings (per-sample, karena CopyNet) ===
        # Stage 1: copy dari input X
        max_ext_vocab_s1 = self.vocab_size
        all_input_ext = []
        all_bspan_target_ext = []

        for i in range(batch_size):
            tokens = input_tokens_batch[i]
            ext_size, oov_tokens, input_ext_idx = self._build_copy_map(tokens, word2idx)
            max_ext_vocab_s1 = max(max_ext_vocab_s1, ext_size)

            # Pad input_ext_idx ke src_len
            src_len = input_seq.size(1)
            padded = input_ext_idx + [0] * (src_len - len(input_ext_idx))
            all_input_ext.append(padded[:src_len])

            # Convert bspan target ke extended indices
            bspan_tgt_tokens = self._indices_to_tokens(bspan_target[i], word2idx)
            bspan_tgt_ext = self._target_to_ext(bspan_tgt_tokens, word2idx, oov_tokens)
            bspan_len = bspan_target.size(1)
            bspan_tgt_ext = bspan_tgt_ext + [0] * (bspan_len - len(bspan_tgt_ext))
            all_bspan_target_ext.append(bspan_tgt_ext[:bspan_len])

        input_ext_tensor = torch.tensor(all_input_ext, dtype=torch.long, device=device)
        bspan_target_ext = torch.tensor(all_bspan_target_ext, dtype=torch.long, device=device)

        # === 3. Stage 1: Decode bspan ===
        bspan_log_probs, bspan_final_hidden = self.decoder1(
            bspan_input, encoder_hidden, encoder_outputs,
            input_ext_tensor, max_ext_vocab_s1
        )

        # Hitung Stage 1 loss
        bspan_loss = self._compute_loss(bspan_log_probs, bspan_target_ext, max_ext_vocab_s1)

        # === 4. Dapatkan bspan hidden states untuk Stage 2 ===
        # Re-run bspan melalui decoder1 GRU untuk mendapatkan per-token hidden states
        bspan_embedded = self.decoder1.embedding(bspan_input)
        bspan_outputs, _ = self.decoder1.gru(bspan_embedded, encoder_hidden)
        # bspan_outputs: (batch, bspan_len, hidden_size) — ini yang dipakai Stage 2

        # === 5. Stage 2: Build extended vocab dari bspan tokens ===
        max_ext_vocab_s2 = self.vocab_size
        all_bspan_ext = []
        all_resp_target_ext = []

        for i in range(batch_size):
            tokens = bspan_tokens_batch[i]
            ext_size, oov_tokens, bspan_ext_idx = self._build_copy_map(tokens, word2idx)
            max_ext_vocab_s2 = max(max_ext_vocab_s2, ext_size)

            bspan_len = bspan_input.size(1)
            padded = bspan_ext_idx + [0] * (bspan_len - len(bspan_ext_idx))
            all_bspan_ext.append(padded[:bspan_len])

            resp_tgt_tokens = self._indices_to_tokens(response_target[i], word2idx)
            resp_tgt_ext = self._target_to_ext(resp_tgt_tokens, word2idx, oov_tokens)
            resp_len = response_target.size(1)
            resp_tgt_ext = resp_tgt_ext + [0] * (resp_len - len(resp_tgt_ext))
            all_resp_target_ext.append(resp_tgt_ext[:resp_len])

        bspan_ext_tensor = torch.tensor(all_bspan_ext, dtype=torch.long, device=device)
        resp_target_ext = torch.tensor(all_resp_target_ext, dtype=torch.long, device=device)

        # === 6. Stage 2: Decode response ===
        response_log_probs, _ = self.decoder2(
            response_input, bspan_final_hidden, bspan_outputs, kt_batch,
            bspan_ext_tensor, max_ext_vocab_s2
        )

        response_loss = self._compute_loss(response_log_probs, resp_target_ext, max_ext_vocab_s2)

        # === Total loss ===
        total_loss = bspan_loss + response_loss

        return total_loss, bspan_loss, response_loss

    def _compute_loss(self, log_probs, targets, ext_vocab_size):
        """
        Hitung cross entropy loss untuk CopyNet output.
        
        log_probs: (batch, seq_len, ext_vocab_size)
        targets: (batch, seq_len) — extended vocab indices
        """
        batch_size, seq_len, _ = log_probs.size()

        # Flatten
        log_probs_flat = log_probs.view(-1, ext_vocab_size)  # (batch*seq, ext_vocab)
        targets_flat = targets.view(-1)  # (batch*seq,)

        # Mask padding (target=0 adalah PAD)
        non_pad_mask = targets_flat.ne(0)

        # Gather log probabilities di posisi target
        # Clamp target indices agar tidak melebihi ext_vocab_size
        targets_clamped = targets_flat.clamp(0, ext_vocab_size - 1)
        nll = -log_probs_flat.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)

        # Masked mean
        nll = nll * non_pad_mask.float()
        loss = nll.sum() / non_pad_mask.float().sum().clamp(min=1)

        return loss

    def _build_copy_map(self, tokens, word2idx):
        """Wrapper untuk build_copy_mapping dari utils."""
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

        ext_size = vocab_size + len(oov_tokens)
        return ext_size, oov_tokens, ext_indices

    def _target_to_ext(self, target_tokens, word2idx, oov_tokens):
        """Convert target tokens ke extended vocab indices."""
        vocab_size = len(word2idx)
        unk_idx = word2idx[config.UNK_TOKEN]
        oov_map = {tok: vocab_size + i for i, tok in enumerate(oov_tokens)}

        indices = []
        for token in target_tokens:
            if token in word2idx:
                indices.append(word2idx[token])
            elif token in oov_map:
                indices.append(oov_map[token])
            else:
                indices.append(unk_idx)
        return indices

    def _indices_to_tokens(self, index_tensor, word2idx):
        """Convert index tensor ke list of token strings."""
        idx2word = {v: k for k, v in word2idx.items()}
        tokens = []
        for idx in index_tensor:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            tokens.append(idx2word.get(idx_val, config.UNK_TOKEN))
        return tokens
