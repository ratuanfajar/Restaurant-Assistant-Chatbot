"""
Two Stage CopyNet (TSCP) — implementasi PyTorch dari paper Sequicity.

Arsitektur:
  1. Encoder (GRU): membaca X = B_{t-1} R_{t-1} U_t -> H^(x)
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
        embedded = self.embedding(input_seq)

        if input_lengths is not None:
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
    
    CopyNet dengan:
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

        # Attention untuk Generate (Eq. 1-3)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)

        # Copy mechanism (Eq. 5-6)
        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, encoder_outputs, src_mask=None):
        """
        Satu langkah decoding.

        Args:
            src_mask: (batch, src_len) bool — True untuk posisi valid, False untuk
                PAD. Kalau None (mis. decode batch=1), tidak ada masking.

        Returns:
            gen_logits: (batch, vocab_size) — raw logits (belum softmax)
            copy_score: (batch, src_len) — raw copy scores
            new_hidden: updated GRU hidden
        """
        embedded = self.embedding(dec_input)
        gru_out, new_hidden = self.gru(embedded, hidden)
        h_dec = gru_out.squeeze(1)

        # Attention (Eq. 1-2)
        enc_proj = self.W1(encoder_outputs)
        dec_proj = self.W2(h_dec).unsqueeze(1).expand_as(enc_proj)
        energy = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(2)
        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, -1e9)
        attn_weights = F.softmax(energy, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # Generate logits (Eq. 3) — return RAW LOGITS, bukan log_softmax
        concat = torch.cat([context, h_dec], dim=1)
        gen_logits = self.output_proj(concat)

        # Copy scores (Eq. 5-6)
        copy_proj = torch.sigmoid(self.W_copy(encoder_outputs))
        copy_score = torch.bmm(copy_proj, h_dec.unsqueeze(2)).squeeze(2)

        return gen_logits, copy_score, new_hidden

    def forward(self, dec_input_seq, hidden, encoder_outputs, source_indices_ext, ext_vocab_size):
        batch_size, tgt_len = dec_input_seq.size()
        # PAD di source = index 0; mask agar attention & copy tidak bocor ke PAD.
        src_mask = source_indices_ext != 0
        all_log_probs = []

        for t in range(tgt_len):
            dec_input = dec_input_seq[:, t].unsqueeze(1)
            gen_logits, copy_score, hidden = self.forward_step(
                dec_input, hidden, encoder_outputs, src_mask
            )
            combined = _combine_copy_probs(
                gen_logits, copy_score, source_indices_ext,
                ext_vocab_size, self.vocab_size, src_mask
            )
            all_log_probs.append(combined)

        all_log_probs = torch.stack(all_log_probs, dim=1)
        return all_log_probs, hidden


class DecoderStage2(nn.Module):
    """
    Decoder Stage 2: Generate response R_t.
    
    Perbedaan dari Stage 1:
    - Initial hidden state = last hidden state dari Stage 1
    - Attention dan Copy ke B_t hidden states (bukan encoder)
    - Input embedding di-concat dengan k_t (Eq. 9)
    """

    def __init__(self, vocab_size, embed_size, hidden_size, kb_size=3, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(
            embed_size + kb_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)

        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, bspan_outputs, kt, src_mask=None):
        """
        Args:
            src_mask: (batch, bspan_len) bool — True untuk posisi valid, False PAD.

        Returns:
            gen_logits: (batch, vocab_size) — raw logits
            copy_score: (batch, bspan_len) — raw copy scores
            new_hidden
        """
        embedded = self.embedding(dec_input)

        kt_expanded = kt.unsqueeze(1)
        gru_input = torch.cat([embedded, kt_expanded], dim=2)

        gru_out, new_hidden = self.gru(gru_input, hidden)
        h_dec = gru_out.squeeze(1)

        # Attention ke bspan
        bspan_proj = self.W1(bspan_outputs)
        dec_proj = self.W2(h_dec).unsqueeze(1).expand_as(bspan_proj)
        energy = self.v(torch.tanh(bspan_proj + dec_proj)).squeeze(2)
        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, -1e9)
        attn_weights = F.softmax(energy, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), bspan_outputs).squeeze(1)

        concat = torch.cat([context, h_dec], dim=1)
        gen_logits = self.output_proj(concat)

        # Copy dari bspan (Eq. 7-8)
        copy_proj = torch.sigmoid(self.W_copy(bspan_outputs))
        copy_score = torch.bmm(copy_proj, h_dec.unsqueeze(2)).squeeze(2)

        return gen_logits, copy_score, new_hidden

    def forward(self, dec_input_seq, hidden, bspan_outputs, kt,
                bspan_indices_ext, ext_vocab_size):
        batch_size, tgt_len = dec_input_seq.size()
        src_mask = bspan_indices_ext != 0
        all_log_probs = []

        for t in range(tgt_len):
            dec_input = dec_input_seq[:, t].unsqueeze(1)
            gen_logits, copy_score, hidden = self.forward_step(
                dec_input, hidden, bspan_outputs, kt, src_mask
            )
            combined = _combine_copy_probs(
                gen_logits, copy_score, bspan_indices_ext,
                ext_vocab_size, self.vocab_size, src_mask
            )
            all_log_probs.append(combined)

        all_log_probs = torch.stack(all_log_probs, dim=1)
        return all_log_probs, hidden


# ============================================================
# Shared: Combine Generate + Copy (FIX UTAMA)
# ============================================================

def _combine_copy_probs(gen_logits, copy_score, source_indices, ext_vocab_size, vocab_size, src_mask=None):
    """
    Gabungkan generate dan copy probability dengan normalisasi yang benar.
    
    Pendekatan: concat logits dari generate dan copy ke satu ruang,
    lalu softmax SEKALI di akhir agar total probability = 1.0.
    
    Ini menghindari masalah P_gen(sum=1) + P_copy(sum=1) = 2.0
    yang menyebabkan loss negatif.
    
    Args:
        gen_logits: (batch, vocab_size) — RAW logits (belum softmax)
        copy_score: (batch, src_len) — raw copy scores
        source_indices: (batch, src_len) — extended vocab indices untuk source
        ext_vocab_size: int
        vocab_size: int
        
    Returns:
        log_probs: (batch, ext_vocab_size) — normalized log probabilities
    """
    batch_size = gen_logits.size(0)
    device = gen_logits.device

    # Buat tensor extended vocab untuk menampung semua logits
    # Inisialisasi dengan nilai sangat negatif (agar token yang tidak ada = prob ~0)
    extended_logits = torch.full(
        (batch_size, ext_vocab_size), fill_value=-1e10, device=device
    )

    # Isi posisi vocabulary dengan generate logits
    extended_logits[:, :vocab_size] = gen_logits

    # Mask copy score di posisi PAD agar tidak menambah probabilitas ke index 0.
    if src_mask is not None:
        copy_score = copy_score.masked_fill(~src_mask, -1e10)

    # Tambahkan copy scores ke posisi yang sesuai (scatter_add supaya
    # token yang muncul berkali-kali di input terakumulasi)
    extended_logits.scatter_add_(1, source_indices, copy_score)

    # Satu kali log_softmax di akhir: total probability = 1.0
    log_probs = F.log_softmax(extended_logits, dim=1)

    return log_probs


class TSCP(nn.Module):
    """
    Two Stage CopyNet — model utuh.
    Encoder + DecoderStage1 (bspan) + DecoderStage2 (response).
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
        device = input_seq.device
        batch_size = input_seq.size(0)

        # Cache idx2word (dulu di-rebuild per sample di _indices_to_tokens — O(batch*|V|)).
        idx2word = getattr(self, "_idx2word_cache", None)
        if idx2word is None or len(idx2word) != len(word2idx):
            idx2word = {v: k for k, v in word2idx.items()}
            self._idx2word_cache = idx2word

        # === 1. Encode ===
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        # === 2. Build extended vocab Stage 1 ===
        max_ext_vocab_s1 = self.vocab_size
        all_input_ext = []
        all_bspan_target_ext = []

        for i in range(batch_size):
            tokens = input_tokens_batch[i]
            ext_size, oov_tokens, input_ext_idx = self._build_copy_map(tokens, word2idx)
            max_ext_vocab_s1 = max(max_ext_vocab_s1, ext_size)

            src_len = input_seq.size(1)
            padded = input_ext_idx + [0] * (src_len - len(input_ext_idx))
            all_input_ext.append(padded[:src_len])

            bspan_tgt_tokens = self._indices_to_tokens(bspan_target[i], idx2word)
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

        bspan_loss = self._compute_loss(bspan_log_probs, bspan_target_ext, max_ext_vocab_s1)

        # === 4. Bspan hidden states untuk Stage 2 ===
        bspan_embedded = self.decoder1.embedding(bspan_input)
        bspan_outputs, _ = self.decoder1.gru(bspan_embedded, encoder_hidden)

        # === 5. Build extended vocab Stage 2 ===
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

            resp_tgt_tokens = self._indices_to_tokens(response_target[i], idx2word)
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

        total_loss = bspan_loss + response_loss
        return total_loss, bspan_loss, response_loss

    def _compute_loss(self, log_probs, targets, ext_vocab_size):
        batch_size, seq_len, _ = log_probs.size()

        log_probs_flat = log_probs.view(-1, ext_vocab_size)
        targets_flat = targets.view(-1)

        non_pad_mask = targets_flat.ne(0)
        targets_clamped = targets_flat.clamp(0, ext_vocab_size - 1)
        nll = -log_probs_flat.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)

        nll = nll * non_pad_mask.float()
        loss = nll.sum() / non_pad_mask.float().sum().clamp(min=1)

        return loss

    def _build_copy_map(self, tokens, word2idx):
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

    def _indices_to_tokens(self, index_tensor, idx2word):
        tokens = []
        for idx in index_tensor:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            tokens.append(idx2word.get(idx_val, config.UNK_TOKEN))
        return tokens
