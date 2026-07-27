"""
Engine inferensi TSCP (Two Stage CopyNet) untuk chatbot restoran.

Alur per turn (mengikuti paper Sequicity, Lei et al. 2018):
    input  = B_{t-1} + R_{t-1} + U_t
    stage1 = decode belief span B_t          (attn + copy dari input X)
    KB     = search berdasar B_t -> k_t
    stage2 = decode response R_t             (attn + copy dari B_t, di-condition k_t)
    lexicalize R_t dengan entri KB terpilih
"""

import re
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Konstanta (harus cocok dengan konvensi checkpoint v2: tag kapital <Inf>/<Req>)
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE = "<Inf>", "</Inf>", "<Req>", "</Req>"
SLOT_TOKENS = ["NAME_SLOT", "ADDRESS_SLOT", "PHONE_SLOT", "POSTCODE_SLOT",
               "FOOD_SLOT", "AREA_SLOT", "PRICERANGE_SLOT"]
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                  INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE] + SLOT_TOKENS

INFORMABLE_SLOTS = ["food", "area", "pricerange"]
DB_FIELD_TO_SLOT = {"name": "NAME_SLOT", "address": "ADDRESS_SLOT", "phone": "PHONE_SLOT",
                    "postcode": "POSTCODE_SLOT", "food": "FOOD_SLOT", "area": "AREA_SLOT",
                    "pricerange": "PRICERANGE_SLOT"}

# Kata kunci "sinyal restoran": kalau input tak memuat satupun ini (dan bukan nilai slot
# dari DB), kemungkinan bukan permintaan restoran (mis. sapaan "good morning").
RESTAURANT_KEYWORDS = {
    "restaurant", "restaurants", "food", "foods", "eat", "eating", "dine", "dining",
    "cuisine", "serve", "serves", "serving", "menu", "meal", "lunch", "dinner", "breakfast",
    "takeaway", "takeout", "hungry", "place", "places", "table", "book", "booking",
    "reserve", "reservation", "recommend", "recommendation", "recommendations", "suggest",
    "looking", "find", "want", "need", "any", "options", "option", "something", "anything",
    "address", "phone", "number", "telephone", "postcode", "location", "call", "contact", "code",
    "cheap", "moderate", "moderately", "expensive", "price", "priced", "pricerange", "cost",
    "budget", "affordable", "north", "south", "east", "west", "centre", "center", "area", "near",
}

EMBED_SIZE = 50
HIDDEN_SIZE = 50
KB_INDICATOR_SIZE = 3
MAX_DECODE_LEN_BSPAN = 30
MAX_DECODE_LEN_RESPONSE = 60


# Arsitektur model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.hidden_size = hidden_size

    def forward(self, input_seq, input_lengths=None):
        embedded = self.embedding(input_seq)
        if input_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, hidden = self.gru(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Attn(nn.Module):
    """Attention Bahdanau: skor v^T tanh(W1 memory + W2 h_dec) -> context (Eq. 1-2)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def context(self, memory, h_dec, src_mask=None):
        e = self.v(torch.tanh(self.W1(memory) + self.W2(h_dec).unsqueeze(1))).squeeze(2)
        if src_mask is not None:
            e = e.masked_fill(~src_mask, -1e9)
        return torch.bmm(F.softmax(e, dim=1).unsqueeze(1), memory).squeeze(1)


class DecoderStage1(nn.Module):
    """Generate belief span: attention (Eq.1-3) + copy dari input X (Eq.5-6)."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = Attn(hidden_size)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)
        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, encoder_outputs, src_mask=None):
        gru_out, new_hidden = self.gru(self.embedding(dec_input), hidden)
        h_dec = gru_out.squeeze(1)
        context = self.attn.context(encoder_outputs, h_dec, src_mask)
        gen_logits = self.output_proj(torch.cat([context, h_dec], dim=1))
        copy_score = torch.bmm(torch.sigmoid(self.W_copy(encoder_outputs)), h_dec.unsqueeze(2)).squeeze(2)
        return gen_logits, copy_score, new_hidden


class DecoderStage2(nn.Module):
    """Generate response: init hidden dari stage1, attn+copy ke B_t, condition k_t (Eq.7-9)."""

    def __init__(self, vocab_size, embed_size, hidden_size, kb_size=3, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + kb_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = Attn(hidden_size)
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)
        self.W_copy = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_step(self, dec_input, hidden, bspan_outputs, kt, src_mask=None):
        gru_input = torch.cat([self.embedding(dec_input), kt.unsqueeze(1)], dim=2)
        gru_out, new_hidden = self.gru(gru_input, hidden)
        h_dec = gru_out.squeeze(1)
        context = self.attn.context(bspan_outputs, h_dec, src_mask)
        gen_logits = self.output_proj(torch.cat([context, h_dec], dim=1))
        copy_score = torch.bmm(torch.sigmoid(self.W_copy(bspan_outputs)), h_dec.unsqueeze(2)).squeeze(2)
        return gen_logits, copy_score, new_hidden


class TSCP(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder1 = DecoderStage1(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder2 = DecoderStage2(vocab_size, embed_size, hidden_size,
                                      kb_size=KB_INDICATOR_SIZE, num_layers=num_layers, dropout=dropout)


def combine_copy_probs(gen_logits, copy_score, source_indices, ext_vocab_size, vocab_size, src_mask=None):
    """P_final = P_generate + P_copy (softmax bersama [vocab ++ source]); token OOV lewat copy."""
    if src_mask is not None:
        copy_score = copy_score.masked_fill(~src_mask, -1e10)
    probs = F.softmax(torch.cat([gen_logits, copy_score], dim=1), dim=1)
    p_gen, p_copy = probs[:, :vocab_size], probs[:, vocab_size:]
    ext = torch.zeros((gen_logits.size(0), ext_vocab_size), device=gen_logits.device)
    ext[:, :vocab_size] = p_gen
    ext.scatter_add_(1, source_indices, p_copy)
    return torch.log(ext + 1e-12)


# Tokenisasi & indexing
def tokenize(text):
    """Pecah teks jadi token; lindungi token khusus, pisahkan tanda baca."""
    sorted_protected = sorted(SPECIAL_TOKENS, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in sorted_protected)
    segments = re.split(f"({pattern})", text)
    tokens = []
    for seg in segments:
        if seg in SPECIAL_TOKENS:
            tokens.append(seg)
        elif seg.strip():
            cleaned = re.sub(r"([.,!?;:'\"\(\)])", r" \1 ", seg)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            tokens.extend(cleaned.split())
    return tokens


def tokens_to_indices(tokens, word2idx):
    unk = word2idx[UNK_TOKEN]
    return [word2idx.get(t, unk) for t in tokens]


def build_source_ext(tokens, word2idx, device):
    """Bangun extended-vocab index untuk copy mechanism (OOV -> vocab_size + offset)."""
    vocab_size = len(word2idx)
    oov_tokens, oov_map, ext = [], {}, []
    for t in tokens:
        if t in word2idx:
            ext.append(word2idx[t])
        else:
            if t not in oov_map:
                oov_map[t] = vocab_size + len(oov_tokens)
                oov_tokens.append(t)
            ext.append(oov_map[t])
    ext_vocab_size = vocab_size + len(oov_tokens)
    return ext_vocab_size, oov_tokens, torch.tensor([ext], dtype=torch.long, device=device)


# Knowledge base: parse bspan, search, lexicalize
def parse_bspan(bspan_text):
    inf = re.search(r"<inf>\s*(.*?)\s*</inf>", bspan_text, re.I)
    req = re.search(r"<req>\s*(.*?)\s*</req>", bspan_text, re.I)
    informable = [v.strip() for v in inf.group(1).split(";") if v.strip()] if inf and inf.group(1).strip() else []
    requestable = [v.strip() for v in req.group(1).split(";") if v.strip()] if req and req.group(1).strip() else []
    return informable, requestable


def search_kb(bspan_text, database):
    """
    Cari restoran yang cocok dgn semua informable value.
    Return (matches_untuk_lexicalize, kt one-hot, status).

    - status "no_constraint": belum ada constraint -> k_t multiple, TAPI jangan pilih
      restoran spesifik (matches kosong) agar tidak melexicalize entri acak.
    - status "no_match": ada constraint tapi 0 hasil -> k_t [1,0,0], matches kosong.
    - status "exact"/"multiple": ada hasil -> lexicalize dari matches[0] (paper: ambil 1 entri).
    """
    informable, _ = parse_bspan(bspan_text)
    if not informable:
        return [], torch.tensor([0., 0., 1.]), "no_constraint"
    matches = [e for e in database
               if all(any(str(e.get(slot, "")).lower() == v.lower() for slot in INFORMABLE_SLOTS)
                      for v in informable)]
    n = len(matches)
    if n == 0:
        return [], torch.tensor([1., 0., 0.]), "no_match"
    elif n == 1:
        return matches, torch.tensor([0., 1., 0.]), "exact"
    return matches, torch.tensor([0., 0., 1.]), "multiple"


# Fallback bila placeholder tidak bisa dilexicalize (mis. no_match / field kosong),
# supaya token mentah seperti NAME_SLOT tidak pernah bocor ke user.
_SLOT_FALLBACK = {
    "NAME_SLOT": "the restaurant", "ADDRESS_SLOT": "the address",
    "PHONE_SLOT": "the phone number", "POSTCODE_SLOT": "the postcode",
    "FOOD_SLOT": "that type of", "AREA_SLOT": "that area",
    "PRICERANGE_SLOT": "that price range",
}


def strip_leftover_slots(text):
    for slot, repl in _SLOT_FALLBACK.items():
        text = text.replace(slot, repl)
    return text


def resolve_inconsistent_bspan(bspan_text, database):
    """Jika ada 2 informable value untuk slot yg sama, simpan yg terakhir (Section 5.7)."""
    informable, requestable = parse_bspan(bspan_text)
    if len(informable) <= 1:
        return bspan_text

    # petakan value -> slot berdasar DB
    value_to_slot = {}
    for e in database:
        for slot in INFORMABLE_SLOTS:
            val = str(e.get(slot, "")).lower().strip()
            if val:
                value_to_slot[val] = slot

    slot_to_value = {}
    order = []
    for v in informable:
        slot = value_to_slot.get(v.lower(), v.lower())  # value tak dikenal jadi slot unik
        if slot not in slot_to_value:
            order.append(slot)
        slot_to_value[slot] = v  # timpa -> nilai terakhir menang
    resolved = [slot_to_value[s] for s in order]

    return (f"{INF_OPEN} {' ; '.join(resolved)} {INF_CLOSE} "
            f"{REQ_OPEN} {' ; '.join(requestable)} {REQ_CLOSE}")


def lexicalize_response(response, kb_matches):
    """Ganti placeholder (NAME_SLOT dll) dengan nilai asli dari 1 entri KB."""
    if not kb_matches:
        return response
    entry = kb_matches[0]
    out = response
    for field, slot in DB_FIELD_TO_SLOT.items():
        val = str(entry.get(field, "")).strip()
        if val:
            out = out.replace(slot, val)
    return out

# Greedy decoding
def _greedy(forward_step_fn, source_ext, ext_vocab_size, oov_tokens,
            word2idx, idx2word, init_hidden, max_len, device, extra=None):
    vocab_size = len(word2idx)
    sos, eos = word2idx[SOS_TOKEN], word2idx[EOS_TOKEN]
    unk = word2idx[UNK_TOKEN]
    hidden = init_hidden
    dec_input = torch.tensor([[sos]], dtype=torch.long, device=device)
    out_tokens = []
    for _ in range(max_len):
        gen_logits, copy_score, hidden = forward_step_fn(dec_input, hidden, extra)
        log_probs = combine_copy_probs(gen_logits, copy_score, source_ext, ext_vocab_size, vocab_size)
        best = log_probs.squeeze(0).argmax().item()
        if best == eos:
            break
        if best < vocab_size:
            token = idx2word.get(best, UNK_TOKEN)
        else:
            off = best - vocab_size
            token = oov_tokens[off] if off < len(oov_tokens) else UNK_TOKEN
        out_tokens.append(token)
        nxt = best if best < vocab_size else unk
        dec_input = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return out_tokens

# Chatbot
class RestaurantAssistant:
    def __init__(self, checkpoint_path, db_path, device="cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.word2idx = ckpt["word2idx"]
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.model = TSCP(len(self.word2idx)).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        with open(db_path, "r", encoding="utf-8") as f:
            self.database = json.load(f)
        # kosakata "domain" untuk deteksi input di luar domain: semua kata vocab
        # (kata bahasa Inggris dari data) + token nilai slot dari DB (mis. seafood).
        self.known_domain = set(self.word2idx)
        for e in self.database:
            for slot in INFORMABLE_SLOTS:
                self.known_domain.update(str(e.get(slot, "")).lower().split())
        # sinyal restoran = nilai slot informable dari DB + kata kunci restoran
        self.restaurant_signal = set(RESTAURANT_KEYWORDS)
        for e in self.database:
            for slot in INFORMABLE_SLOTS:
                self.restaurant_signal.update(str(e.get(slot, "")).lower().split())
        self.reset()

    def reset(self):
        """Mulai percakapan baru (B_0 = R_0 = kosong)."""
        self.prev_bspan = ""
        self.prev_response = ""

    def _fallback(self, message, status):
        """Jawaban terkontrol (tanpa menjalankan model / mengubah state)."""
        return {"bspan": self.prev_bspan, "response": message, "response_delex": "",
                "num_matches": 0, "kt": [0., 0., 1.], "kb_match": None, "status": status}

    @torch.no_grad()
    def respond(self, user_utterance):
        w2i, i2w, device = self.word2idx, self.idx2word, self.device
        model = self.model

        # 0. guard input kosong/whitespace: jangan jalankan model & jangan ubah state
        user_clean = user_utterance.lower().strip()
        if not user_clean:
            return self._fallback(
                "Please tell me what kind of restaurant you're looking for — "
                "a cuisine, an area, or a price range.", "empty_input")

        # 0b. guard out-of-domain: bila tak ada satu pun kata yang dikenali (mis. input
        # non-Inggris/gibberish seperti "assalamualaikum"), model tak punya sinyal ->
        # beri jawaban jujur, jangan mengarang respons restoran.
        content = [t for t in tokenize(user_clean)
                   if t not in SPECIAL_TOKENS and any(ch.isalnum() for ch in t)]
        if content and not any(t in self.known_domain for t in content):
            return self._fallback(
                "Sorry, I can only help you find restaurants, and I understand English only. "
                "Try telling me a cuisine, area, or price range.", "out_of_domain")

        # 0c. guard tanpa sinyal restoran: input bahasa Inggris tapi bukan permintaan
        # restoran (mis. "good morning", "how are you") -> sapa/klarifikasi, bukan mengarang.
        if content and not any(t in self.restaurant_signal for t in content):
            return self._fallback(
                "Hi! I can help you find a restaurant. Tell me a cuisine, an area "
                "(centre / north / south / east / west), or a price range "
                "(cheap / moderate / expensive).", "no_restaurant_signal")

        # 1. format input B_{t-1} R_{t-1} U_t
        parts = [p for p in (self.prev_bspan, self.prev_response) if p]
        parts.append(user_clean)
        input_text = " ".join(parts)
        input_tokens = tokenize(input_text)

        # 2. encode
        input_tensor = torch.tensor([tokens_to_indices(input_tokens, w2i)], dtype=torch.long, device=device)
        input_lengths = torch.tensor([input_tensor.size(1)])
        encoder_outputs, encoder_hidden = model.encoder(input_tensor, input_lengths)

        # 3. decode belief span (stage 1) - copy source = input X
        ext_size, oov, src_ext = build_source_ext(input_tokens, w2i, device)
        bspan_tokens = _greedy(
            lambda di, h, _e: model.decoder1.forward_step(di, h, encoder_outputs),
            src_ext, ext_size, oov, w2i, i2w, encoder_hidden,
            MAX_DECODE_LEN_BSPAN, device)
        bspan_text = " ".join(bspan_tokens)

        # 4. post-process + KB search
        bspan_text = resolve_inconsistent_bspan(bspan_text, self.database)
        kb_matches, kt, kb_status = search_kb(bspan_text, self.database)
        kt = kt.to(device)

        # 5. bspan hidden states untuk stage 2
        bspan_core = tokenize(bspan_text)
        full_bspan_tokens = [SOS_TOKEN] + bspan_core
        bspan_in = torch.tensor([tokens_to_indices(full_bspan_tokens, w2i)], dtype=torch.long, device=device)
        bspan_embedded = model.decoder1.embedding(bspan_in)
        bspan_outputs, bspan_hidden = model.decoder1.gru(bspan_embedded, encoder_hidden)

        # 6. decode response (stage 2) - copy source = B_t, condition k_t
        bext_size, boov, bsrc_ext = build_source_ext(full_bspan_tokens, w2i, device)
        kt_row = kt.unsqueeze(0)
        response_tokens = _greedy(
            lambda di, h, _e: model.decoder2.forward_step(di, h, bspan_outputs, kt_row),
            bsrc_ext, bext_size, boov, w2i, i2w, bspan_hidden,
            MAX_DECODE_LEN_RESPONSE, device)
        response_delex = " ".join(response_tokens)

        # 7. lexicalize (hanya bila ada entri terpilih) + bersihkan placeholder sisa
        response_final = lexicalize_response(response_delex, kb_matches)
        response_final = strip_leftover_slots(response_final)

        # 8. update state (konteks turn berikutnya pakai response DELEX, spt training)
        self.prev_bspan = bspan_text
        self.prev_response = response_delex

        return {
            "bspan": bspan_text,
            "response": response_final,
            "response_delex": response_delex,
            "num_matches": len(kb_matches),
            "kt": kt.tolist(),
            "kb_match": kb_matches[0] if kb_matches else None,
            "status": kb_status,
        }
