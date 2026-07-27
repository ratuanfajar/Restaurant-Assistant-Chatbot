"""Arsitektur TSCP v2 + inference & evaluasi.

Dipakai bersama oleh notebook supervised (eksp3) dan RL fine-tuning.
Struktur layer identik dengan yang melatih `tscp_supervised_v2_best.pt`.
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.preprocessing_v2 as pp


def combine_copy_probs(gen_logits, copy_score, source_indices, ext_size, vocab_size, src_mask=None):
    # P_final = P_generate + P_copy: softmax bersama atas [vocab ++ posisi source],
    # probabilitas copy disebar ke index extended-vocab (token OOV hanya lewat copy).
    if src_mask is not None:
        copy_score = copy_score.masked_fill(~src_mask, -1e10)
    probs = F.softmax(torch.cat([gen_logits, copy_score], dim=1), dim=1)
    p_gen, p_copy = probs[:, :vocab_size], probs[:, vocab_size:]
    ext = torch.zeros((gen_logits.size(0), ext_size), device=gen_logits.device)
    ext[:, :vocab_size] = p_gen
    ext.scatter_add_(1, source_indices, p_copy)
    return torch.log(ext + 1e-12)


class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.gru = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, seq, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(seq), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hidden = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, hidden


class _Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def context(self, memory, h_dec, src_mask):
        e = self.v(torch.tanh(self.W1(memory) + self.W2(h_dec).unsqueeze(1))).squeeze(2)
        if src_mask is not None:
            e = e.masked_fill(~src_mask, -1e9)
        return torch.bmm(F.softmax(e, dim=1).unsqueeze(1), memory).squeeze(1)


class DecoderStage1(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.vocab_size = vocab
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.attn = _Attn(hidden)
        self.output_proj = nn.Linear(hidden * 2, vocab)
        self.W_copy = nn.Linear(hidden, hidden, bias=False)

    def step(self, dec_input, hidden, memory, src_mask=None):
        out, hidden = self.gru(self.embedding(dec_input), hidden)
        h = out.squeeze(1)
        ctx = self.attn.context(memory, h, src_mask)
        gen = self.output_proj(torch.cat([ctx, h], dim=1))
        copy = torch.bmm(torch.sigmoid(self.W_copy(memory)), h.unsqueeze(2)).squeeze(2)
        return gen, copy, hidden

    def decode(self, seq, hidden, memory, ext_source, ext_size):
        mask = ext_source != 0
        outs = []
        for t in range(seq.size(1)):
            gen, copy, hidden = self.step(seq[:, t:t + 1], hidden, memory, mask)
            outs.append(combine_copy_probs(gen, copy, ext_source, ext_size, self.vocab_size, mask))
        return torch.stack(outs, dim=1), hidden


class DecoderStage2(nn.Module):
    def __init__(self, vocab, embed, hidden, kb):
        super().__init__()
        self.vocab_size = vocab
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.gru = nn.GRU(embed + kb, hidden, batch_first=True)
        self.attn = _Attn(hidden)
        self.output_proj = nn.Linear(hidden * 2, vocab)
        self.W_copy = nn.Linear(hidden, hidden, bias=False)

    def step(self, dec_input, hidden, memory, kt, src_mask=None):
        gin = torch.cat([self.embedding(dec_input), kt.unsqueeze(1)], dim=2)
        out, hidden = self.gru(gin, hidden)
        h = out.squeeze(1)
        ctx = self.attn.context(memory, h, src_mask)
        gen = self.output_proj(torch.cat([ctx, h], dim=1))
        copy = torch.bmm(torch.sigmoid(self.W_copy(memory)), h.unsqueeze(2)).squeeze(2)
        return gen, copy, hidden

    def decode(self, seq, hidden, memory, kt, ext_source, ext_size):
        mask = ext_source != 0
        outs = []
        for t in range(seq.size(1)):
            gen, copy, hidden = self.step(seq[:, t:t + 1], hidden, memory, kt, mask)
            outs.append(combine_copy_probs(gen, copy, ext_source, ext_size, self.vocab_size, mask))
        return torch.stack(outs, dim=1), hidden


def build_copy_map(tokens, word2idx):
    vs = len(word2idx)
    oov, oov_map, ext = [], {}, []
    for t in tokens:
        if t in word2idx:
            ext.append(word2idx[t])
        else:
            if t not in oov_map:
                oov_map[t] = vs + len(oov)
                oov.append(t)
            ext.append(oov_map[t])
    return ext, oov_map, vs + len(oov)


def target_to_ext(tokens, word2idx, oov_map):
    unk = word2idx[pp.UNK_TOKEN]
    return [word2idx[t] if t in word2idx else oov_map.get(t, unk) for t in tokens]


def _pad_rows(rows, width, device):
    return torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long, device=device)


class TSCP(nn.Module):
    def __init__(self, vocab, embed=50, hidden=50, kb=3):
        super().__init__()
        self.vocab_size = vocab
        self.encoder = Encoder(vocab, embed, hidden)
        self.decoder1 = DecoderStage1(vocab, embed, hidden)
        self.decoder2 = DecoderStage2(vocab, embed, hidden, kb)

    @staticmethod
    def _loss(logp, target, ext_size):
        lp, tg = logp.reshape(-1, ext_size), target.reshape(-1)
        mask = tg.ne(0)
        nll = -lp.gather(1, tg.clamp(0, ext_size - 1).unsqueeze(1)).squeeze(1) * mask.float()
        return nll.sum() / mask.float().sum().clamp(min=1)

    def forward(self, batch, word2idx, kt):
        dev = batch["input"].device
        enc_out, enc_h = self.encoder(batch["input"], batch["input_lengths"])
        B = batch["input"].size(0)
        Lx, Lb, Lr = batch["input"].size(1), batch["bspan_in"].size(1), batch["resp_in"].size(1)

        src_rows, btgt_rows, ext1 = [], [], self.vocab_size
        for i in range(B):
            ext, omap, esz = build_copy_map(batch["input_tokens"][i], word2idx)
            ext1 = max(ext1, esz)
            src_rows.append(ext)
            btgt_rows.append(target_to_ext(batch["bspan_tgt_tokens"][i], word2idx, omap))
        bspan_logp, bspan_hidden = self.decoder1.decode(
            batch["bspan_in"], enc_h, enc_out, _pad_rows(src_rows, Lx, dev), ext1)
        loss_b = self._loss(bspan_logp, _pad_rows(btgt_rows, Lb, dev), ext1)

        bspan_out, _ = self.decoder1.gru(self.decoder1.embedding(batch["bspan_in"]), enc_h)
        bsrc_rows, rtgt_rows, ext2 = [], [], self.vocab_size
        for i in range(B):
            ext, omap, esz = build_copy_map(batch["bspan_in_tokens"][i], word2idx)
            ext2 = max(ext2, esz)
            bsrc_rows.append(ext)
            rtgt_rows.append(target_to_ext(batch["resp_tgt_tokens"][i], word2idx, omap))
        resp_logp, _ = self.decoder2.decode(
            batch["resp_in"], bspan_hidden, bspan_out, kt, _pad_rows(bsrc_rows, Lb, dev), ext2)
        loss_r = self._loss(resp_logp, _pad_rows(rtgt_rows, Lr, dev), ext2)
        return loss_b + loss_r, loss_b, loss_r


# ------------------------- Inference & evaluasi -------------------------

@torch.no_grad()
def _greedy(decoder, hidden, memory, src, ext_size, inv_oov, word2idx, idx2word, kt=None, max_len=50):
    dev = src.device
    mask = src != 0
    vocab = len(word2idx)
    di = torch.tensor([[word2idx[pp.SOS_TOKEN]]], device=dev)
    out = []
    for _ in range(max_len):
        if kt is None:
            gen, copy, hidden = decoder.step(di, hidden, memory, mask)
        else:
            gen, copy, hidden = decoder.step(di, hidden, memory, kt, mask)
        idx = combine_copy_probs(gen, copy, src, ext_size, vocab, mask).argmax(1).item()
        if idx == word2idx[pp.EOS_TOKEN]:
            break
        out.append(idx2word[idx] if idx < vocab else inv_oov.get(idx, pp.UNK_TOKEN))
        di = torch.tensor([[idx if idx < vocab else word2idx[pp.UNK_TOKEN]]], device=dev)
    return out


@torch.no_grad()
def generate(model, sample, word2idx, idx2word, database, device="cpu"):
    model.eval()
    it = pp.tokenize(sample["input"])
    ids = torch.tensor([pp.tokens_to_ids(it, word2idx)], device=device)
    enc_out, enc_h = model.encoder(ids, torch.tensor([len(it)]))

    ext, omap, esz = build_copy_map(it, word2idx)
    src = torch.tensor([ext], device=device)
    pb = _greedy(model.decoder1, enc_h, enc_out, src, esz, {v: k for k, v in omap.items()}, word2idx, idx2word)
    pb_text = " ".join(pb)

    bspan_tokens = [pp.SOS_TOKEN] + pb
    bids = torch.tensor([pp.tokens_to_ids(bspan_tokens, word2idx)], device=device)
    bspan_out, bspan_h = model.decoder1.gru(model.decoder1.embedding(bids), enc_h)
    bext, bomap, besz = build_copy_map(bspan_tokens, word2idx)
    bsrc = torch.tensor([bext], device=device)
    kt = pp.get_kt_from_bspan(pb_text, database).unsqueeze(0).to(device)
    pr = _greedy(model.decoder2, bspan_h, bspan_out, bsrc, besz,
                 {v: k for k, v in bomap.items()}, word2idx, idx2word, kt=kt)
    return pb_text, " ".join(pr)


SLOT_PLACEHOLDER = {"address": "address_slot", "phone": "phone_slot", "postcode": "postcode_slot",
                    "food": "food_slot", "area": "area_slot", "pricerange": "pricerange_slot",
                    "name": "name_slot"}


def corpus_bleu(references, hypotheses, max_n=4):
    """Corpus BLEU-4 (1 referensi/hipotesis) dengan smoothing sederhana; tanpa dependensi."""
    p_num, p_den = [0] * max_n, [0] * max_n
    c_len = r_len = 0
    for ref, hyp in zip(references, hypotheses):
        c_len += len(hyp)
        r_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ng = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            ref_ng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            p_num[n - 1] += sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            p_den[n - 1] += sum(hyp_ng.values())
    if c_len == 0:
        return 0.0
    precisions = [(num / den if num else 1e-9 / den) if den else 1e-9 for num, den in zip(p_num, p_den)]
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def entity_match(pred_bspans, gold_bspans):
    ok = sum(set(v.lower() for v in pp.parse_bspan(p)[0]) == set(v.lower() for v in pp.parse_bspan(g)[0])
             for p, g in zip(pred_bspans, gold_bspans))
    return ok / max(len(gold_bspans), 1)


def success_f1(pred_responses, gold_bspans):
    precs, recs = [], []
    for resp, g in zip(pred_responses, gold_bspans):
        expected = {SLOT_PLACEHOLDER[r.lower()] for r in pp.parse_bspan(g)[1] if r.lower() in SLOT_PLACEHOLDER}
        if not expected:
            continue
        low = resp.lower()
        found = {p for p in expected if p in low}
        present = {p for p in SLOT_PLACEHOLDER.values() if p in low}
        recs.append(len(found) / len(expected))
        precs.append(len(found & present) / len(present) if present else 0.0)
    if not precs:
        return 0.0
    P, R = sum(precs) / len(precs), sum(recs) / len(recs)
    return 2 * P * R / (P + R) if P + R else 0.0


@torch.no_grad()
def evaluate(model, samples, word2idx, idx2word, database, device="cpu", show=0):
    model.eval()
    pred_b, gold_b, pred_r, refs, hyps = [], [], [], [], []
    for i, s in enumerate(samples):
        pb, pr = generate(model, s, word2idx, idx2word, database, device)
        pred_b.append(pb); gold_b.append(s["target_bspan"]); pred_r.append(pr)
        refs.append(pp.tokenize(s["target_response"])); hyps.append(pp.tokenize(pr))
        if i < show:
            print(f"[{i}] gold B: {s['target_bspan']}\n    pred B: {pb}\n    pred R: {pr}")
    pairs = [(r, h) for r, h in zip(refs, hyps) if h]
    bleu = corpus_bleu([r for r, _ in pairs], [h for _, h in pairs]) if pairs else 0.0
    return {"BLEU": bleu, "EntityMatch": entity_match(pred_b, gold_b), "SuccessF1": success_f1(pred_r, gold_b)}
