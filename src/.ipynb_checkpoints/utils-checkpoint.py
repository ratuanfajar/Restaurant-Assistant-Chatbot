"""
Utility functions:
- Knowledge Base search (query DB berdasarkan bspan)
- Bspan parsing (ekstrak informable/requestable values)
- Beam search untuk inference (menangani CopyNet dynamic vocab)
- Lexicalization (ganti placeholder → value asli)
- Reward computation untuk RL
"""

import torch
import re
import src.config as config


# ============================================================
# Bspan Parsing
# ============================================================

def parse_bspan(bspan_text):
    """
    Parse belief span string menjadi informable values dan requestable slots.
    
    Input:  "<inf> italian ; cheap </inf> <req> address ; phone </req>"
    Output: (["italian", "cheap"], ["address", "phone"])
    """
    informable = []
    requestable = []

    # Extract informable values
    inf_match = re.search(r"<inf>\s*(.*?)\s*</inf>", bspan_text)
    if inf_match:
        inf_str = inf_match.group(1).strip()
        if inf_str:
            informable = [v.strip() for v in inf_str.split(";") if v.strip()]

    # Extract requestable slots
    req_match = re.search(r"<req>\s*(.*?)\s*</req>", bspan_text)
    if req_match:
        req_str = req_match.group(1).strip()
        if req_str:
            requestable = [v.strip() for v in req_str.split(";") if v.strip()]

    return informable, requestable


# ============================================================
# Knowledge Base Search
# ============================================================

def search_kb(bspan_text, database):
    """
    Query knowledge base berdasarkan informable values dari bspan.
    
    Returns:
        matches: list of matching DB entries
        kt: tensor 3-dim one-hot [no_match, exact_match, multiple_match]
    """
    informable, _ = parse_bspan(bspan_text)

    if not informable:
        # Tidak ada constraint → semua restoran match (multiple)
        return database, torch.tensor([0.0, 0.0, 1.0])

    matches = []
    for entry in database:
        match = True
        for value in informable:
            # Cek apakah value ada di salah satu informable field
            found = False
            for slot in config.INFORMABLE_SLOTS:
                if entry.get(slot, "").lower() == value.lower():
                    found = True
                    break
            if not found:
                match = False
                break
        if match:
            matches.append(entry)

    n = len(matches)
    if n == 0:
        kt = torch.tensor([1.0, 0.0, 0.0])  # no match
    elif n == 1:
        kt = torch.tensor([0.0, 1.0, 0.0])  # exact match
    else:
        kt = torch.tensor([0.0, 0.0, 1.0])  # multiple match

    return matches, kt


def get_kt_from_bspan(bspan_text, database):
    """Wrapper yang hanya return kt vector."""
    _, kt = search_kb(bspan_text, database)
    return kt


# ============================================================
# Lexicalization (Post-processing saat inference)
# ============================================================

def lexicalize_response(response, kb_matches):
    """
    Ganti placeholder di response dengan value asli dari KB match.
    
    Misal: "NAME_SLOT is a cheap restaurant" 
         → "Pizza Hut is a cheap restaurant"
    """
    if not kb_matches:
        return response

    entity = kb_matches[0]  # Ambil entity pertama

    for field, placeholder in config.DB_FIELD_TO_SLOT.items():
        value = entity.get(field, "")
        if value:
            response = response.replace(placeholder.lower(), value.lower())
            response = response.replace(placeholder, value)

    return response


# ============================================================
# Token/Index Conversion Helpers (untuk CopyNet)
# ============================================================

def build_copy_mapping(input_tokens, word2idx):
    """
    Bangun mapping untuk CopyNet: posisi token di input yang bisa di-copy.
    
    CopyNet memperluas output space dari V ke V ∪ X.
    Token yang ada di input tapi tidak di vocab (OOV) diberi
    index sementara di luar vocabulary.
    
    Returns:
        extended_vocab_size: |V| + jumlah OOV tokens unik di input
        oov_tokens: list of OOV token strings
        input_extended_indices: input indices dalam extended vocab
            (vocab tokens pakai idx asli, OOV pakai |V| + offset)
    """
    vocab_size = len(word2idx)
    oov_tokens = []
    oov_map = {}  # OOV token → extended idx

    input_extended_indices = []
    for token in input_tokens:
        if token in word2idx:
            input_extended_indices.append(word2idx[token])
        else:
            if token not in oov_map:
                oov_map[token] = vocab_size + len(oov_tokens)
                oov_tokens.append(token)
            input_extended_indices.append(oov_map[token])

    extended_vocab_size = vocab_size + len(oov_tokens)
    return extended_vocab_size, oov_tokens, input_extended_indices


def target_to_extended_indices(target_tokens, word2idx, oov_tokens):
    """
    Convert target tokens ke extended indices (untuk menghitung loss dengan CopyNet).
    OOV tokens yang juga ada di input (dan bisa di-copy) pakai extended idx.
    """
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


# ============================================================
# Beam Search
# ============================================================

class BeamHypothesis:
    """Satu hypothesis dalam beam search."""
    def __init__(self, tokens, token_indices, log_prob, hidden):
        self.tokens = tokens
        self.token_indices = token_indices
        self.log_prob = log_prob
        self.hidden = hidden

    @property
    def score(self):
        """Normalized log probability (length penalty)."""
        return self.log_prob / max(len(self.tokens), 1)


def beam_search_decode(decoder, encoder_outputs, initial_hidden, 
                       word2idx, idx2word, source_tokens,
                       beam_size, max_len, kt=None,
                       copy_source_outputs=None, copy_source_tokens=None):
    """
    Beam search decoding untuk CopyNet decoder.
    
    Args:
        decoder: DecoderStage1 atau DecoderStage2
        encoder_outputs: hidden states dari encoder (atau bspan hidden untuk stage 2)
        initial_hidden: hidden state awal decoder
        word2idx, idx2word: vocabulary mappings
        source_tokens: token list dari source (input atau bspan) untuk copy
        beam_size: beam width
        max_len: max decode length
        kt: knowledge base indicator (hanya untuk stage 2)
        copy_source_outputs: hidden states untuk copy attention
        copy_source_tokens: tokens untuk copy reference
        
    Returns:
        best_tokens: list of decoded token strings
    """
    device = encoder_outputs.device
    sos_idx = word2idx[config.SOS_TOKEN]
    eos_idx = word2idx[config.EOS_TOKEN]
    vocab_size = len(word2idx)

    # Build extended vocab mapping
    ext_size, oov_tokens, source_ext_indices = build_copy_mapping(
        source_tokens, word2idx
    )
    source_ext_tensor = torch.tensor(source_ext_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Tokens untuk copy reference
    ref_tokens = copy_source_tokens if copy_source_tokens is not None else source_tokens
    ref_outputs = copy_source_outputs if copy_source_outputs is not None else encoder_outputs

    # Initialize beams
    initial_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    beams = [BeamHypothesis(
        tokens=[],
        token_indices=[sos_idx],
        log_prob=0.0,
        hidden=initial_hidden
    )]

    completed = []

    for step in range(max_len):
        all_candidates = []

        for beam in beams:
            if beam.token_indices[-1] == eos_idx:
                completed.append(beam)
                continue

            # Prepare input
            last_idx = beam.token_indices[-1]
            # Jika OOV (extended idx >= vocab_size), pakai UNK sebagai embedding input
            if last_idx >= vocab_size:
                embed_idx = word2idx[config.UNK_TOKEN]
            else:
                embed_idx = last_idx

            dec_input = torch.tensor([[embed_idx]], dtype=torch.long, device=device)
            hidden = beam.hidden

            # Forward satu step
            if kt is not None:
                # Stage 2 decoder
                gen_logprob, copy_score, new_hidden = decoder.forward_step(
                    dec_input, hidden, ref_outputs, kt.unsqueeze(0)
                )
            else:
                # Stage 1 decoder
                gen_logprob, copy_score, new_hidden = decoder.forward_step(
                    dec_input, hidden, encoder_outputs
                )

            # Gabungkan generate + copy probabilities (di log space)
            # gen_logprob: (1, vocab_size)
            # copy_score: (1, source_len)
            log_probs = compute_combined_logprob(
                gen_logprob, copy_score, source_ext_tensor, ext_size, vocab_size
            )
            log_probs = log_probs.squeeze(0)  # (ext_vocab_size,)

            # Top-k
            topk_logprobs, topk_indices = log_probs.topk(beam_size)

            for k in range(beam_size):
                token_idx = topk_indices[k].item()
                token_logprob = topk_logprobs[k].item()

                # Determine token string
                if token_idx < vocab_size:
                    token_str = idx2word.get(token_idx, config.UNK_TOKEN)
                else:
                    oov_offset = token_idx - vocab_size
                    if oov_offset < len(oov_tokens):
                        token_str = oov_tokens[oov_offset]
                    else:
                        token_str = config.UNK_TOKEN

                new_beam = BeamHypothesis(
                    tokens=beam.tokens + [token_str],
                    token_indices=beam.token_indices + [token_idx],
                    log_prob=beam.log_prob + token_logprob,
                    hidden=new_hidden
                )
                all_candidates.append(new_beam)

        if not all_candidates:
            break

        # Select top beams
        all_candidates.sort(key=lambda b: b.score, reverse=True)
        beams = all_candidates[:beam_size]

        # Early stop jika semua beams selesai
        if all(b.token_indices[-1] == eos_idx for b in beams):
            completed.extend(beams)
            break

    # Pilih hypothesis terbaik
    if completed:
        completed.sort(key=lambda b: b.score, reverse=True)
        return completed[0].tokens
    elif beams:
        beams.sort(key=lambda b: b.score, reverse=True)
        return beams[0].tokens
    else:
        return []


def compute_combined_logprob(gen_logprob, copy_score, source_indices, 
                              ext_vocab_size, vocab_size):
    """
    Gabungkan generate probability dan copy probability.
    P(v) = P_gen(v) + P_copy(v)
    
    Args:
        gen_logprob: (batch, vocab_size) — log probabilities dari generator
        copy_score: (batch, source_len) — raw scores untuk copy
        source_indices: (batch, source_len) — extended vocab indices dari source
        ext_vocab_size: ukuran extended vocabulary (V + OOV)
        vocab_size: ukuran vocabulary asli
    """
    batch_size = gen_logprob.size(0)
    device = gen_logprob.device

    # Convert gen dari log-space ke probability space
    gen_prob = gen_logprob.exp()  # (batch, vocab_size)

    # Extend gen_prob ke extended vocab size (tambah kolom 0 untuk OOV)
    if ext_vocab_size > vocab_size:
        extra = torch.zeros(batch_size, ext_vocab_size - vocab_size, device=device)
        gen_prob = torch.cat([gen_prob, extra], dim=1)  # (batch, ext_vocab_size)

    # Copy probabilities: softmax dari copy scores
    copy_prob = torch.softmax(copy_score, dim=-1)  # (batch, source_len)

    # Scatter-add copy probs ke extended vocab positions
    combined = gen_prob.clone()
    combined.scatter_add_(1, source_indices, copy_prob)

    # Convert kembali ke log-space (dengan epsilon untuk stabilitas)
    combined_log = torch.log(combined + 1e-12)

    return combined_log


# ============================================================
# Reward Computation (untuk RL — Section 4.4)
# ============================================================

def compute_rewards(generated_tokens, requestable_slots_requested):
    """
    Hitung per-step reward untuk RL fine-tuning (Section 4.4).

    Skema reward (revisi — desain lama menghukum SETIAP token normal -0.1
    sehingga policy kolaps jadi output slot telanjang / respons kosong):

      r(j) = +RL_REWARD_POS          → token adalah placeholder slot yang diminta
                                        user (hanya diberi SEKALI per slot, agar
                                        model tidak spam token slot yang sama)
      r(j) = RL_REWARD_HALLUCINATION → token placeholder slot yang TIDAK diminta
                                        (halusinasi slot)
      r(j) = RL_REWARD_NEG (=0)      → token lainnya → netral, tidak dihukum

    Reward normal token dibuat netral supaya kelancaran bahasa dijaga oleh
    supervised anchor (mixed objective), bukan dirusak oleh RL.

    Args:
        generated_tokens: list of token strings dari response
        requestable_slots_requested: list of slot names yang diminta user

    Returns:
        rewards: list of float, satu per token
    """
    # Mapping dari requested slot name ke placeholder token
    slot_to_placeholder = {
        "address": "ADDRESS_SLOT",
        "phone": "PHONE_SLOT",
        "postcode": "POSTCODE_SLOT",
        "food": "FOOD_SLOT",
        "area": "AREA_SLOT",
        "pricerange": "PRICERANGE_SLOT",
        "name": "NAME_SLOT",
    }

    all_placeholders = {s.lower() for s in config.SLOT_TOKENS}

    expected_placeholders = set()
    for slot in requestable_slots_requested:
        ph = slot_to_placeholder.get(slot.lower())
        if ph:
            expected_placeholders.add(ph.lower())

    rewarded = set()  # slot yang sudah diberi reward positif
    rewards = []
    for token in generated_tokens:
        tok = token.lower()
        if tok in expected_placeholders and tok not in rewarded:
            rewards.append(config.RL_REWARD_POS)
            rewarded.add(tok)
        elif tok in all_placeholders and tok not in expected_placeholders:
            rewards.append(config.RL_REWARD_HALLUCINATION)
        else:
            rewards.append(config.RL_REWARD_NEG)

    return rewards


def compute_returns(rewards, gamma=config.RL_LAMBDA):
    """
    Hitung discounted returns dari rewards.
    R(j) = r(j) + gamma * r(j+1) + gamma^2 * r(j+2) + ...
    """
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


# ============================================================
# Handling Inconsistent User Requests (Section 5.7)
# ============================================================

def resolve_inconsistent_bspan(bspan_text, database):
    """
    Post-processing untuk bspan: jika ada dua value untuk slot yang sama
    (misal dua food types), hanya simpan yang terbaru (terakhir).
    Sesuai Section 5.7 paper.
    """
    informable, requestable = parse_bspan(bspan_text)

    # Cek apakah ada duplikat per slot type
    slot_assignments = {}
    for value in informable:
        # Cari slot mana yang cocok
        for entry in database:
            for slot in config.INFORMABLE_SLOTS:
                if entry.get(slot, "").lower() == value.lower():
                    slot_assignments[slot] = value
                    break

    # Rebuild: ambil value terakhir per slot
    deduped_values = list(slot_assignments.values())

    inf_str = " ; ".join(deduped_values) if deduped_values else ""
    req_str = " ; ".join(requestable) if requestable else ""

    return f"{config.INF_OPEN} {inf_str} {config.INF_CLOSE} {config.REQ_OPEN} {req_str} {config.REQ_CLOSE}"


# ============================================================
# Decode indices ke text
# ============================================================

def indices_to_text(indices, idx2word, skip_special=True):
    """Convert list of indices ke readable text."""
    tokens = []
    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        word = idx2word.get(idx, config.UNK_TOKEN)
        if skip_special and word in {config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN}:
            continue
        tokens.append(word)
    return " ".join(tokens)
