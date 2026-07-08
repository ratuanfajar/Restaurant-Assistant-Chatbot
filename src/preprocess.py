"""
Preprocessing pipeline untuk CamRest676:
1. Load raw JSON
2. Delexicalize system responses (R_t)
3. Construct belief spans (B_t) dari SLU annotations
4. Format input-output pairs per turn sesuai Eq. 4 paper
5. Build vocabulary (hanya dari training set)
6. Split data 3:1:1
"""

import json
import re
import random
from collections import Counter

import src.config as config


def load_raw_data():
    """Load CamRest676.json dan CamRestDB.json."""
    with open(config.CAMREST_PATH, "r") as f:
        dialogues = json.load(f)
    with open(config.DB_PATH, "r") as f:
        database = json.load(f)
    return dialogues, database


def split_data(dialogues, seed=42):
    """
    Split dialogues menjadi train/dev/test dengan rasio 3:1:1.
    Shuffle dengan seed tetap untuk reproducibility.
    """
    random.seed(seed)
    indices = list(range(len(dialogues)))
    random.shuffle(indices)

    n = len(dialogues)
    n_train = int(n * config.TRAIN_RATIO)
    n_dev = int(n * config.DEV_RATIO)

    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    train = [dialogues[i] for i in train_idx]
    dev = [dialogues[i] for i in dev_idx]
    test = [dialogues[i] for i in test_idx]

    print(f"Data split -> Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
    return train, dev, test


def delexicalize_response(response, entity=None, database=None):
    """
    Delexicalize system response: ganti nama restoran, alamat, dll.
    dengan placeholder (NAME_SLOT, ADDRESS_SLOT, ...).
    
    Hanya R_t yang dideleksikalisasi, bukan U_t atau B_t.
    """
    delex = response.lower()

    if database is None:
        return delex

    # Kumpulkan nilai informable unik (food/area/pricerange) untuk di-delex.
    # Ini nilai pendek & umum (mis. "italian", "centre", "cheap") sehingga
    # diganti dengan word-boundary agar tidak merusak kata lain.
    food_values, area_values, price_values = set(), set(), set()

    # Cari semua entity di database yang mungkin muncul di response
    for entry in database:
        name = entry.get("name", "").lower()
        address = entry.get("address", "").lower()
        phone = entry.get("phone", "").lower()
        postcode = entry.get("postcode", "").lower()

        # Ganti yang paling panjang dulu (name, address) untuk menghindari partial match
        if name and name in delex:
            delex = delex.replace(name, "NAME_SLOT")
        if address and address in delex:
            delex = delex.replace(address, "ADDRESS_SLOT")
        if phone and phone in delex:
            delex = delex.replace(phone, "PHONE_SLOT")
        if postcode and postcode in delex:
            delex = delex.replace(postcode, "POSTCODE_SLOT")

        if entry.get("food"):
            food_values.add(entry["food"].lower())
        if entry.get("area"):
            area_values.add(entry["area"].lower())
        if entry.get("pricerange"):
            price_values.add(entry["pricerange"].lower())

    # Delex food/area/pricerange (nilai terpanjang dulu, word-boundary).
    for value in sorted(food_values, key=len, reverse=True):
        delex = re.sub(rf"\b{re.escape(value)}\b", "FOOD_SLOT", delex)
    for value in sorted(area_values, key=len, reverse=True):
        delex = re.sub(rf"\b{re.escape(value)}\b", "AREA_SLOT", delex)
    for value in sorted(price_values, key=len, reverse=True):
        delex = re.sub(rf"\b{re.escape(value)}\b", "PRICERANGE_SLOT", delex)

    return delex


def construct_bspan(slu_annotations):
    """
    Bangun belief span dari SLU annotation per turn.
    Format: <inf> value1 ; value2 </inf> <req> slot1 ; slot2 </req>
    
    - Informable slots: ambil VALUES (misal: "italian", "cheap")
    - Requestable slots: ambil SLOT NAMES (misal: "address", "phone")
    """
    informable_values = []
    requestable_names = []

    for slu in slu_annotations:
        act = slu["act"]
        for slot_pair in slu["slots"]:
            if act == "inform":
                slot_name = slot_pair[0]
                slot_value = slot_pair[1]
                # Skip 'dontcare' dan 'slot' (meta key)
                if slot_value != "dontcare" and slot_name in config.INFORMABLE_SLOTS:
                    if slot_value.lower() not in informable_values:
                        informable_values.append(slot_value.lower())
            elif act == "request":
                # Format request: ['slot', 'address'] -> ambil 'address'
                req_name = slot_pair[1] if slot_pair[0] == "slot" else slot_pair[0]
                if req_name.lower() not in requestable_names:
                    requestable_names.append(req_name.lower())

    inf_str = " ; ".join(informable_values) if informable_values else ""
    req_str = " ; ".join(requestable_names) if requestable_names else ""

    bspan = f"{config.INF_OPEN} {inf_str} {config.INF_CLOSE} {config.REQ_OPEN} {req_str} {config.REQ_CLOSE}"
    return bspan


def process_dialogue(dialogue, database):
    """
    Proses satu dialogue menjadi list of (input, target_bspan, target_response) per turn.
    
    Sesuai Eq. 4:
      Input  X = B_{t-1} R_{t-1} U_t
      Target Y = B_t (stage 1) + R_t (stage 2)
      
    B_0 dan R_0 di-initialize sebagai empty string (bukan <sos>).
    """
    turns = dialogue["dial"]
    processed_turns = []

    prev_bspan = ""
    prev_response = ""

    for t, turn in enumerate(turns):
        user_utt = turn["usr"]["transcript"].lower().strip()
        sys_response = turn["sys"]["sent"].lower().strip()

        # Construct B_t dari SLU annotations
        bspan = construct_bspan(turn["usr"]["slu"])

        # Delexicalize R_t
        delex_response = delexicalize_response(sys_response, database=database)

        # Format input: B_{t-1} + R_{t-1} + U_t
        input_parts = []
        if prev_bspan:
            input_parts.append(prev_bspan)
        if prev_response:
            input_parts.append(prev_response)
        input_parts.append(user_utt)
        input_seq = " ".join(input_parts)

        processed_turns.append({
            "input": input_seq,
            "target_bspan": bspan,
            "target_response": delex_response,
            "raw_user": user_utt,
            "raw_system": sys_response,
            "dialogue_id": dialogue.get("dialogue_id", ""),
            "turn": t,
        })

        # Update prev untuk turn berikutnya
        prev_bspan = bspan
        prev_response = delex_response

    return processed_turns


def process_all_dialogues(dialogues, database):
    """Proses semua dialogues menjadi flat list of turn-level samples."""
    all_samples = []
    for dial in dialogues:
        samples = process_dialogue(dial, database)
        all_samples.extend(samples)
    return all_samples


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    # Pisahkan punctuation dari kata
    text = re.sub(r"([.,!?;:'\"\(\)])", r" \1 ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocabulary(train_samples, min_freq=1):
    """
    Build vocabulary HANYA dari training set (anti data leakage).
    
    Returns:
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
    """
    counter = Counter()

    for sample in train_samples:
        counter.update(tokenize(sample["input"]))
        counter.update(tokenize(sample["target_bspan"]))
        counter.update(tokenize(sample["target_response"]))

    # Filter by min_freq
    vocab_words = [w for w, c in counter.items() if c >= min_freq]

    # Build mappings — special tokens first
    word2idx = {}
    for i, token in enumerate(config.SPECIAL_TOKENS):
        word2idx[token] = i

    idx = len(config.SPECIAL_TOKENS)
    for word in sorted(vocab_words):
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1

    idx2word = {v: k for k, v in word2idx.items()}

    print(f"Vocabulary size: {len(word2idx)} (|V| di paper: ~800)")
    return word2idx, idx2word


def tokens_to_indices(tokens, word2idx):
    """Convert list of tokens ke list of indices, UNK untuk unknown."""
    unk_idx = word2idx[config.UNK_TOKEN]
    return [word2idx.get(t, unk_idx) for t in tokens]


def prepare_data():
    """
    Main preprocessing pipeline:
    1. Load data
    2. Split 3:1:1
    3. Process dialogues (bspan, delex)
    4. Build vocab dari train
    5. Return semua data yang siap dipakai
    """
    dialogues, database = load_raw_data()
    train_dial, dev_dial, test_dial = split_data(dialogues)

    train_samples = process_all_dialogues(train_dial, database)
    dev_samples = process_all_dialogues(dev_dial, database)
    test_samples = process_all_dialogues(test_dial, database)

    print(f"Samples -> Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}")

    word2idx, idx2word = build_vocabulary(train_samples)

    return {
        "train": train_samples,
        "dev": dev_samples,
        "test": test_samples,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "database": database,
    }


if __name__ == "__main__":
    data = prepare_data()
    # Print beberapa contoh
    for i in range(3):
        s = data["train"][i]
        print(f"\n=== Sample {i} (Dial {s['dialogue_id']}, Turn {s['turn']}) ===")
        print(f"INPUT:    {s['input'][:120]}...")
        print(f"BSPAN:    {s['target_bspan']}")
        print(f"RESPONSE: {s['target_response'][:120]}...")
