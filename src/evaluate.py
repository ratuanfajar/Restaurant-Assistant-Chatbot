"""
Evaluasi TSCP pada test set — 3 metrik sesuai paper Section 5:

1. BLEU: kualitas bahasa (nltk.translate.bleu_score)
2. Entity Match Rate: apakah bspan menghasilkan semua constraints benar
3. Success F1: apakah response memuat semua request slots yang diminta user
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import src.config as config
from src.preprocess import tokenize, tokens_to_indices
from src.model import _combine_copy_probs
from src.utils import (
    parse_bspan, search_kb, get_kt_from_bspan,
    lexicalize_response, indices_to_text, resolve_inconsistent_bspan,
    build_copy_mapping,
)


def _build_source_ext_tensor(tokens, word2idx, device):
    """
    Build extended vocab mapping untuk satu sample.
    Returns: ext_vocab_size, oov_tokens, source_indices_tensor (1, src_len)
    """
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

    ext_vocab_size = vocab_size + len(oov_tokens)
    source_tensor = torch.tensor([ext_indices], dtype=torch.long, device=device)
    return ext_vocab_size, oov_tokens, source_tensor


def greedy_decode_bspan(model, encoder_outputs, encoder_hidden, word2idx, idx2word,
                         input_tokens, device):
    """
    Greedy decoding Stage 1 (bspan), menggunakan _combine_copy_probs
    yang sama dengan training.
    """
    sos_idx = word2idx[config.SOS_TOKEN]
    eos_idx = word2idx[config.EOS_TOKEN]
    vocab_size = len(word2idx)

    # Build extended vocab dari input tokens
    ext_vocab_size, oov_tokens, source_ext = _build_source_ext_tensor(
        input_tokens, word2idx, device
    )

    hidden = encoder_hidden
    dec_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    generated_tokens = []

    for _ in range(config.MAX_DECODE_LEN_BSPAN):
        gen_logits, copy_score, hidden = model.decoder1.forward_step(
            dec_input, hidden, encoder_outputs
        )

        # Gabungkan gen + copy dengan fungsi yang sama seperti training
        log_probs = _combine_copy_probs(
            gen_logits, copy_score, source_ext, ext_vocab_size, vocab_size
        )

        best_idx = log_probs.squeeze(0).argmax().item()

        if best_idx == eos_idx:
            break

        # Tentukan token string
        if best_idx < vocab_size:
            token = idx2word.get(best_idx, config.UNK_TOKEN)
        else:
            oov_offset = best_idx - vocab_size
            if oov_offset < len(oov_tokens):
                token = oov_tokens[oov_offset]
            else:
                token = config.UNK_TOKEN

        generated_tokens.append(token)

        # Untuk token berikutnya: jika OOV, pakai UNK sebagai embedding input
        next_idx = best_idx if best_idx < vocab_size else word2idx[config.UNK_TOKEN]
        dec_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return generated_tokens


def greedy_decode_response(model, bspan_outputs, bspan_final_hidden,
                            kt, word2idx, idx2word, bspan_tokens, device):
    """
    Greedy decoding Stage 2 (response), menggunakan _combine_copy_probs
    yang sama dengan training.
    """
    sos_idx = word2idx[config.SOS_TOKEN]
    eos_idx = word2idx[config.EOS_TOKEN]
    vocab_size = len(word2idx)

    # Build extended vocab dari bspan tokens
    ext_vocab_size, oov_tokens, bspan_ext = _build_source_ext_tensor(
        bspan_tokens, word2idx, device
    )

    hidden = bspan_final_hidden
    dec_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    kt_tensor = kt.unsqueeze(0).to(device)

    generated_tokens = []

    for _ in range(config.MAX_DECODE_LEN_RESPONSE):
        gen_logits, copy_score, hidden = model.decoder2.forward_step(
            dec_input, hidden, bspan_outputs, kt_tensor
        )

        log_probs = _combine_copy_probs(
            gen_logits, copy_score, bspan_ext, ext_vocab_size, vocab_size
        )

        best_idx = log_probs.squeeze(0).argmax().item()

        if best_idx == eos_idx:
            break

        if best_idx < vocab_size:
            token = idx2word.get(best_idx, config.UNK_TOKEN)
        else:
            oov_offset = best_idx - vocab_size
            if oov_offset < len(oov_tokens):
                token = oov_tokens[oov_offset]
            else:
                token = config.UNK_TOKEN

        generated_tokens.append(token)

        next_idx = best_idx if best_idx < vocab_size else word2idx[config.UNK_TOKEN]
        dec_input = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return generated_tokens


def generate_for_sample(model, sample, word2idx, idx2word, database, device):
    """Generate bspan + response untuk satu test sample."""
    model.eval()

    with torch.no_grad():
        input_tokens = tokenize(sample["input"])
        input_indices = tokens_to_indices(input_tokens, word2idx)
        input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
        input_lengths = torch.tensor([len(input_indices)])

        # 1. Encode
        encoder_outputs, encoder_hidden = model.encoder(input_tensor, input_lengths)

        # 2. Decode bspan (Stage 1)
        pred_bspan_tokens = greedy_decode_bspan(
            model, encoder_outputs, encoder_hidden, word2idx, idx2word,
            input_tokens, device
        )
        pred_bspan_text = " ".join(pred_bspan_tokens)

        # Post-processing
        pred_bspan_text = resolve_inconsistent_bspan(pred_bspan_text, database)
        pred_bspan_tokens = tokenize(pred_bspan_text)

        # 3. KB search
        kb_matches, kt = search_kb(pred_bspan_text, database)

        # 4. Bspan hidden states untuk Stage 2
        # Jalankan bspan tokens melalui decoder1 GRU untuk dapat hidden states
        bspan_indices = tokens_to_indices(pred_bspan_tokens, word2idx)
        sos_idx = word2idx[config.SOS_TOKEN]
        bspan_input_tensor = torch.tensor(
            [[sos_idx] + bspan_indices], dtype=torch.long, device=device
        )

        bspan_embedded = model.decoder1.embedding(bspan_input_tensor)
        bspan_outputs, bspan_final_hidden = model.decoder1.gru(
            bspan_embedded, encoder_hidden
        )

        # 5. Decode response (Stage 2)
        # Gabungkan SOS token bspan dengan tokens bspan untuk referensi copy
        full_bspan_tokens = [config.SOS_TOKEN] + pred_bspan_tokens

        pred_response_tokens = greedy_decode_response(
            model, bspan_outputs, bspan_final_hidden,
            kt, word2idx, idx2word, full_bspan_tokens, device
        )
        pred_response_text = " ".join(pred_response_tokens)

    return pred_bspan_text, pred_response_text, kb_matches


# ============================================================
# Metrik Evaluasi
# ============================================================

def compute_bleu(references, hypotheses):
    refs = [[ref] for ref in references]
    smooth = SmoothingFunction().method1

    # Filter: skip jika hypothesis kosong
    filtered_refs = []
    filtered_hyps = []
    for ref, hyp in zip(refs, hypotheses):
        if len(hyp) > 0:
            filtered_refs.append(ref)
            filtered_hyps.append(hyp)

    if not filtered_hyps:
        return 0.0

    score = corpus_bleu(filtered_refs, filtered_hyps, smoothing_function=smooth)
    return score


def compute_entity_match_rate(pred_bspans, gold_bspans, database):
    matches = 0
    total = 0

    for pred_b, gold_b in zip(pred_bspans, gold_bspans):
        pred_inf, _ = parse_bspan(pred_b)
        gold_inf, _ = parse_bspan(gold_b)

        if set(v.lower() for v in pred_inf) == set(v.lower() for v in gold_inf):
            matches += 1
        total += 1

    return matches / max(total, 1)


def compute_success_f1(pred_responses, gold_bspans):
    slot_to_placeholder = {
        "address": "address_slot",
        "phone": "phone_slot",
        "postcode": "postcode_slot",
        "food": "food_slot",
        "area": "area_slot",
        "pricerange": "pricerange_slot",
        "name": "name_slot",
    }

    all_precision = []
    all_recall = []

    for pred_resp, gold_b in zip(pred_responses, gold_bspans):
        _, requestable = parse_bspan(gold_b)

        if not requestable:
            continue

        expected = set()
        for req in requestable:
            ph = slot_to_placeholder.get(req.lower())
            if ph:
                expected.add(ph)

        if not expected:
            continue

        pred_lower = pred_resp.lower()
        found = set()
        for ph in expected:
            if ph in pred_lower:
                found.add(ph)

        all_placeholders_in_response = set()
        for ph in slot_to_placeholder.values():
            if ph in pred_lower:
                all_placeholders_in_response.add(ph)

        recall = len(found) / len(expected) if expected else 0.0

        if all_placeholders_in_response:
            correct_in_response = found.intersection(all_placeholders_in_response)
            precision = len(correct_in_response) / len(all_placeholders_in_response)
        else:
            precision = 0.0

        all_precision.append(precision)
        all_recall.append(recall)

    if not all_precision:
        return 0.0

    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)

    if avg_precision + avg_recall == 0:
        return 0.0

    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    return f1


# ============================================================
# Full Evaluation Pipeline
# ============================================================

def evaluate_model(model, test_samples, word2idx, idx2word, database, device="cpu"):
    print(f"\n{'='*60}")
    print(f"EVALUATING ON TEST SET ({len(test_samples)} samples)")
    print(f"{'='*60}")

    model.eval()

    all_pred_bspans = []
    all_gold_bspans = []
    all_pred_responses = []
    all_gold_responses = []

    for i, sample in enumerate(test_samples):
        pred_bspan, pred_response, kb_matches = generate_for_sample(
            model, sample, word2idx, idx2word, database, device
        )

        all_pred_bspans.append(pred_bspan)
        all_gold_bspans.append(sample["target_bspan"])
        all_pred_responses.append(pred_response)
        all_gold_responses.append(sample["target_response"])

        if i < 5:
            print(f"\n--- Sample {i} ---")
            print(f"  Input:          {sample['input'][:80]}...")
            print(f"  Gold Bspan:     {sample['target_bspan']}")
            print(f"  Pred Bspan:     {pred_bspan}")
            print(f"  Gold Response:  {sample['target_response'][:80]}...")
            print(f"  Pred Response:  {pred_response[:80]}...")

    ref_tokenized = [tokenize(r) for r in all_gold_responses]
    hyp_tokenized = [tokenize(r) for r in all_pred_responses]
    bleu = compute_bleu(ref_tokenized, hyp_tokenized)

    entity_match = compute_entity_match_rate(all_pred_bspans, all_gold_bspans, database)
    success_f1 = compute_success_f1(all_pred_responses, all_gold_bspans)

    results = {
        "BLEU": bleu,
        "Entity_Match_Rate": entity_match,
        "Success_F1": success_f1,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  BLEU:              {bleu:.4f}")
    print(f"  Entity Match Rate: {entity_match:.4f}")
    print(f"  Success F1:        {success_f1:.4f}")
    print(f"{'='*60}")

    return results
