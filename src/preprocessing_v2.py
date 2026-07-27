"""Preprocessing v2 CamRest676 untuk TSCP.

Perbedaan dari v1: delexicalization word-boundary + vocab hygiene (min_freq=2),
sehingga token langka menjadi OOV dan melatih copy-mechanism.
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

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

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
SEED = 42

_PUNCT = re.compile(r"([.,!?;:'\"()])")
_PROTECT = re.compile("(" + "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")")


def find_data_dir():
    p = Path.cwd()
    while not (p / "data").exists() and p != p.parent:
        p = p.parent
    return p / "data"


def load_raw(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "CamRest676.json", encoding="utf-8") as f:
        dialogues = json.load(f)
    with open(data_dir / "CamRestDB.json", encoding="utf-8") as f:
        database = json.load(f)
    return dialogues, database


def split_dialogues(dialogues, seed=SEED):
    random.seed(seed)
    idx = list(range(len(dialogues)))
    random.shuffle(idx)
    n = len(dialogues)
    a, b = int(n * TRAIN_RATIO), int(n * TRAIN_RATIO) + int(n * VAL_RATIO)
    pick = lambda sl: [dialogues[i] for i in sl]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def collect_slot_values(database):
    pairs, seen = [], set()
    for entry in database:
        for field, slot in DB_FIELD_TO_SLOT.items():
            val = str(entry.get(field, "")).lower().strip()
            if val and (val, slot) not in seen:
                seen.add((val, slot))
                pairs.append((val, slot))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def delexicalize(response, slot_value_pairs):
    delex = response.lower()
    for val, slot in slot_value_pairs:
        delex = re.sub(rf"\b{re.escape(val)}\b", slot, delex)
    return delex


def construct_bspan(slu_annotations):
    informable, requestable = [], []
    for slu in slu_annotations:
        act = slu["act"]
        for pair in slu["slots"]:
            if act == "inform":
                name, value = pair[0], pair[1]
                if value != "dontcare" and name in INFORMABLE_SLOTS and value.lower() not in informable:
                    informable.append(value.lower())
            elif act == "request":
                req = pair[1] if pair[0] == "slot" else pair[0]
                if req.lower() not in requestable:
                    requestable.append(req.lower())
    return f"{INF_OPEN} {' ; '.join(informable)} {INF_CLOSE} {REQ_OPEN} {' ; '.join(requestable)} {REQ_CLOSE}"


def process_dialogues(dialogues, slot_value_pairs):
    samples = []
    for dialogue in dialogues:
        prev_bspan, prev_response = "", ""
        for turn in dialogue["dial"]:
            user = turn["usr"]["transcript"].lower().strip()
            bspan = construct_bspan(turn["usr"]["slu"])
            response = delexicalize(turn["sys"]["sent"].lower().strip(), slot_value_pairs)
            context = [p for p in (prev_bspan, prev_response) if p] + [user]
            samples.append({"input": " ".join(context), "target_bspan": bspan, "target_response": response})
            prev_bspan, prev_response = bspan, response
    return samples


def tokenize(text):
    tokens = []
    for seg in _PROTECT.split(text):
        if seg in SPECIAL_TOKENS:
            tokens.append(seg)
        elif seg.strip():
            tokens.extend(_PUNCT.sub(r" \1 ", seg).split())
    return tokens


def build_vocab(train_samples, min_freq=2, max_vocab=800):
    counter = Counter()
    for s in train_samples:
        counter.update(tokenize(s["input"]))
        counter.update(tokenize(s["target_bspan"]))
        counter.update(tokenize(s["target_response"]))
    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if word in word2idx or freq < min_freq:
            continue
        if len(word2idx) >= max_vocab:
            break
        word2idx[word] = len(word2idx)
    return word2idx, {i: w for w, i in word2idx.items()}


def tokens_to_ids(tokens, word2idx):
    unk = word2idx[UNK_TOKEN]
    return [word2idx.get(t, unk) for t in tokens]


def parse_bspan(text):
    inf = re.search(r"<Inf>\s*(.*?)\s*</Inf>", text, re.I)
    req = re.search(r"<Req>\s*(.*?)\s*</Req>", text, re.I)
    grab = lambda m: [x.strip() for x in m.group(1).split(";") if x.strip()] if m and m.group(1).strip() else []
    return grab(inf), grab(req)


def get_kt_from_bspan(bspan_text, database):
    informable, _ = parse_bspan(bspan_text)
    if not informable:
        return torch.tensor([0., 0., 1.])
    n = sum(1 for e in database
            if all(any(str(e.get(s, "")).lower() == v.lower() for s in INFORMABLE_SLOTS) for v in informable))
    if n == 0:
        return torch.tensor([1., 0., 0.])
    if n == 1:
        return torch.tensor([0., 1., 0.])
    return torch.tensor([0., 0., 1.])


class CamRestDataset(Dataset):
    def __init__(self, samples, word2idx):
        self.rows = []
        for s in samples:
            it = tokenize(s["input"])
            bt = [SOS_TOKEN] + tokenize(s["target_bspan"]) + [EOS_TOKEN]
            rt = [SOS_TOKEN] + tokenize(s["target_response"]) + [EOS_TOKEN]
            self.rows.append({
                "input_tokens": it,
                "bspan_in_tokens": bt[:-1],
                "bspan_tgt_tokens": bt[1:],
                "resp_tgt_tokens": rt[1:],
                "input_ids": tokens_to_ids(it, word2idx),
                "bspan_in_ids": tokens_to_ids(bt[:-1], word2idx),
                "resp_in_ids": tokens_to_ids(rt[:-1], word2idx),
                "bspan_text": s["target_bspan"],
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _pad(seqs, pad=0):
    m = max(len(x) for x in seqs)
    return torch.tensor([x + [pad] * (m - len(x)) for x in seqs], dtype=torch.long)


def collate(batch):
    return {
        "input": _pad([b["input_ids"] for b in batch]),
        "input_lengths": torch.tensor([len(b["input_ids"]) for b in batch]),
        "bspan_in": _pad([b["bspan_in_ids"] for b in batch]),
        "resp_in": _pad([b["resp_in_ids"] for b in batch]),
        "input_tokens": [b["input_tokens"] for b in batch],
        "bspan_in_tokens": [b["bspan_in_tokens"] for b in batch],
        "bspan_tgt_tokens": [b["bspan_tgt_tokens"] for b in batch],
        "resp_tgt_tokens": [b["resp_tgt_tokens"] for b in batch],
        "bspan_text": [b["bspan_text"] for b in batch],
    }


def get_dataloader(samples, word2idx, batch_size=32, shuffle=True):
    return DataLoader(CamRestDataset(samples, word2idx), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)


def prepare_data(data_dir=None, min_freq=2, max_vocab=800, verbose=True):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    dialogues, database = load_raw(data_dir)
    tr, va, te = split_dialogues(dialogues)
    pairs = collect_slot_values(database)
    train = process_dialogues(tr, pairs)
    val = process_dialogues(va, pairs)
    test = process_dialogues(te, pairs)
    word2idx, idx2word = build_vocab(train, min_freq, max_vocab)
    if verbose:
        print(f"Dialog -> train {len(tr)} | val {len(va)} | test {len(te)}")
        print(f"Sample -> train {len(train)} | val {len(val)} | test {len(test)}")
        print(f"Vocab  -> {len(word2idx)} (min_freq={min_freq}, max={max_vocab})")
    return {"train": train, "val": val, "test": test,
            "word2idx": word2idx, "idx2word": idx2word, "database": database}
