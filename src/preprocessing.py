import json
import re
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Konfigurasi Global

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE = "<Inf>", "</Inf>", "<Req>", "</Req>"

SLOT_TOKENS = [
    "NAME_SLOT", "ADDRESS_SLOT", "PHONE_SLOT", "POSTCODE_SLOT",
    "FOOD_SLOT", "AREA_SLOT", "PRICERANGE_SLOT"
]

SPECIAL_TOKENS = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE
] + SLOT_TOKENS

INFORMABLE_SLOTS = ["food", "area", "pricerange"]

DB_FIELD_TO_SLOT = {
    "name": "NAME_SLOT", "address": "ADDRESS_SLOT", "phone": "PHONE_SLOT",
    "postcode": "POSTCODE_SLOT", "food": "FOOD_SLOT", "area": "AREA_SLOT",
    "pricerange": "PRICERANGE_SLOT"
}

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
SEED = 42
MAX_VOCAB_SIZE = 800


# Helpers

def load_raw_data(data_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Memuat file JSON dialog dan database restoran dari direktori yang diberikan."""
    with open(data_dir / "CamRest676.json", "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    with open(data_dir / "CamRestDB.json", "r", encoding="utf-8") as f:
        database = json.load(f)
    return dialogues, database


def split_dialogues(dialogues: List[Dict], seed: int = SEED) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Membagi dialog menjadi train, val, dan test dengan rasio 3:1:1 secara reproducible."""
    random.seed(seed)
    indices = list(range(len(dialogues)))
    random.shuffle(indices)
    
    n = len(dialogues)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train = [dialogues[i] for i in indices[:n_train]]
    val = [dialogues[i] for i in indices[n_train:n_train + n_val]]
    test = [dialogues[i] for i in indices[n_train + n_val:]]
    return train, val, test


def precompute_db_slots(database: List[Dict]) -> Dict[str, List[str]]:
    """
    Mengumpulkan semua nilai unik dari database untuk setiap slot.
    Nilai diurutkan berdasarkan panjang string (terpanjang dulu) untuk mencegah 
    bug substring saat proses delexicalization (misal: 'pizza hut' tertukar dengan 'pizza hut city centre').
    """
    slot_values = {slot: set() for slot in DB_FIELD_TO_SLOT.values()}
    for entry in database:
        for field, slot in DB_FIELD_TO_SLOT.items():
            val = entry.get(field, "").lower().strip()
            if val:
                slot_values[slot].add(val)
                
    return {slot: sorted(vals, key=len, reverse=True) for slot, vals in slot_values.items()}


def delexicalize_response(response: str, db_slot_values: Dict[str, List[str]]) -> str:
    """
    Mengganti nilai entitas spesifik dalam response sistem dengan placeholder slot generik.
    Slot spesifik (nama, alamat, dll) menggunakan replace biasa, sedangkan slot umum 
    (makanan, area, harga) menggunakan regex word-boundary agar tidak merusak kata lain.
    """
    delex = response.lower()
    
    specific_slots = ["NAME_SLOT", "ADDRESS_SLOT", "PHONE_SLOT", "POSTCODE_SLOT"]
    for slot in specific_slots:
        for val in db_slot_values[slot]:
            delex = delex.replace(val, slot)
            
    general_slots = ["FOOD_SLOT", "AREA_SLOT", "PRICERANGE_SLOT"]
    for slot in general_slots:
        for val in db_slot_values[slot]:
            delex = re.sub(rf"\b{re.escape(val)}\b", slot, delex)
            
    return delex


def construct_bspan(slu_annotations: List[Dict]) -> str:
    """
    Membentuk belief span dari anotasi SLU user.
    Mengembalikan string berformat: <Inf> val1 ; val2 </Inf> <Req> req1 ; req2 </Req>
    """
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


def process_dialogues(dialogues: List[Dict], db_slot_values: Dict[str, List[str]]) -> List[Dict]:
    """
    Memformat setiap turn percakapan menjadi sampel independen.
    Input digabung dari bspan dan response turn sebelumnya, serta utterance user saat ini.
    """
    processed_samples = []
    
    for dialogue in dialogues:
        prev_bspan, prev_response = "", ""
        for turn_idx, turn in enumerate(dialogue["dial"]):
            user_utt = turn["usr"]["transcript"].lower().strip()
            bspan = construct_bspan(turn["usr"]["slu"])
            response = delexicalize_response(turn["sys"]["sent"].lower().strip(), db_slot_values)
            
            context_parts = [p for p in (prev_bspan, prev_response) if p]
            full_input = " ".join(context_parts + [user_utt])
            
            processed_samples.append({
                "input": full_input,
                "target_bspan": bspan,
                "target_response": response,
            })
            
            prev_bspan, prev_response = bspan, response
            
    return processed_samples


def tokenize(text: str, protected_tokens: List[str]) -> List[str]:
    """
    Memecah teks menjadi token. Token khusus (seperti <Inf>, NAME_SLOT) dilindungi agar tidak terpecah.
    Tanda baca dipisahkan dari kata dasarnya.
    """
    sorted_protected = sorted(protected_tokens, key=len, reverse=True)
    pattern_protected = "|".join(re.escape(tok) for tok in sorted_protected)
    segments = re.split(f"({pattern_protected})", text)
    
    result_tokens = []
    for segment in segments:
        if segment in protected_tokens:
            result_tokens.append(segment)
        elif segment.strip():
            cleaned = re.sub(r"([.,!?;:'\"\(\)])", r" \1 ", segment)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            result_tokens.extend(cleaned.split())
            
    return result_tokens


def build_vocabulary(tokenized_samples: List[Dict], max_vocab_size: int = MAX_VOCAB_SIZE) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Membangun kamus kata ke indeks (word2idx) khusus dari data training.
    Kata diurutkan berdasarkan frekuensi (tertinggi), lalu alfabetis. Dibatasi hingga max_vocab_size.
    """
    counter = Counter()
    for s in tokenized_samples:
        counter.update(s["tokens_input"])
        counter.update(s["tokens_bspan"])
        counter.update(s["tokens_response"])

    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    for word, _ in sorted_words:
        if len(word2idx) >= max_vocab_size:
            break
        word2idx.setdefault(word, len(word2idx))
        
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def tokens_to_indices_copynet(tokens: List[str], word2idx: Dict[str, int]) -> Tuple[List[int], List[str]]:
    """
    Mengonversi token menjadi indeks numerik. 
    Jika ada kata OOV (Out-Of-Vocabulary), kata tersebut diberi indeks sementara 
    (mulai dari vocab_size) dan kata aslinya disimpan dalam list oov_words untuk mekanisme CopyNet.
    """
    vocab_size = len(word2idx)
    indices, oov_words, oov_map = [], [], {}
    
    for t in tokens:
        if t in word2idx:
            indices.append(word2idx[t])
        else:
            if t not in oov_map:
                oov_map[t] = vocab_size + len(oov_words)
                oov_words.append(t)
            indices.append(oov_map[t])
            
    return indices, oov_words


# Pytorch Dataset & DataLoader

class CamRestDataset(Dataset):
    """Dataset PyTorch yang memetakan sampel teks menjadi tensor indeks dengan format teacher-forcing."""
    
    def __init__(self, samples: List[Dict], word2idx: Dict[str, int]):
        self.data = []
        for s in samples:
            tokens_input = tokenize(s["input"], SPECIAL_TOKENS)
            tokens_bspan = [SOS_TOKEN] + tokenize(s["target_bspan"], SPECIAL_TOKENS) + [EOS_TOKEN]
            tokens_response = [SOS_TOKEN] + tokenize(s["target_response"], SPECIAL_TOKENS) + [EOS_TOKEN]

            input_idx, input_oov = tokens_to_indices_copynet(tokens_input, word2idx)
            bspan_idx, _ = tokens_to_indices_copynet(tokens_bspan, word2idx)
            resp_idx, _ = tokens_to_indices_copynet(tokens_response, word2idx)

            self.data.append({
                "input_idx": input_idx,
                "input_oov": input_oov,  
                "bspan_in": bspan_idx[:-1],  
                "bspan_tgt": bspan_idx[1:],  
                "resp_in": resp_idx[:-1],
                "resp_tgt": resp_idx[1:],
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Melakukan dynamic padding per batch. 
    Sequence hanya dipadding sepanjang sequence terpanjang di dalam batch tersebut, 
    menghemat komputasi GPU/CPU secara signifikan dibanding global padding.
    """
    pad_idx = 0 
    
    inputs = [torch.tensor(item["input_idx"], dtype=torch.long) for item in batch]
    bspan_ins = [torch.tensor(item["bspan_in"], dtype=torch.long) for item in batch]
    bspan_tgts = [torch.tensor(item["bspan_tgt"], dtype=torch.long) for item in batch]
    resp_ins = [torch.tensor(item["resp_in"], dtype=torch.long) for item in batch]
    resp_tgts = [torch.tensor(item["resp_tgt"], dtype=torch.long) for item in batch]
    oov_lists = [item["input_oov"] for item in batch] 

    return {
        "input": pad_sequence(inputs, batch_first=True, padding_value=pad_idx),
        "bspan_in": pad_sequence(bspan_ins, batch_first=True, padding_value=pad_idx),
        "bspan_tgt": pad_sequence(bspan_tgts, batch_first=True, padding_value=pad_idx),
        "resp_in": pad_sequence(resp_ins, batch_first=True, padding_value=pad_idx),
        "resp_tgt": pad_sequence(resp_tgts, batch_first=True, padding_value=pad_idx),
        "input_oov": oov_lists, 
    }


# Main Wrapper Class

class CamRestDataModule:
    """
    Class orkestrator untuk seluruh pipeline preprocessing.
    Menginisialisasi class ini akan otomatis memuat data, memproses, membangun vocab, 
    dan menyiapkan PyTorch DataLoader yang siap pakai untuk training.
    """
    def __init__(self, data_dir: str, batch_size: int = 32, max_vocab_size: int = MAX_VOCAB_SIZE):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        
        self.word2idx = None
        self.idx2word = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._prepare_data()

    def _prepare_data(self):
        print("[CamRestDataModule] Loading and processing raw data...")
        dialogues, database = load_raw_data(self.data_dir)
        train_dial, val_dial, test_dial = split_dialogues(dialogues)
        
        db_slot_values = precompute_db_slots(database)
        train_samples = process_dialogues(train_dial, db_slot_values)
        val_samples = process_dialogues(val_dial, db_slot_values)
        test_samples = process_dialogues(test_dial, db_slot_values)
        
        print("[CamRestDataModule] Building vocabulary...")
        temp_tokenized = [{
            "tokens_input": tokenize(s["input"], SPECIAL_TOKENS),
            "tokens_bspan": tokenize(s["target_bspan"], SPECIAL_TOKENS),
            "tokens_response": tokenize(s["target_response"], SPECIAL_TOKENS)
        } for s in train_samples]
        
        self.word2idx, self.idx2word = build_vocabulary(temp_tokenized, self.max_vocab_size)
        print(f"[CamRestDataModule] Vocab size: {len(self.word2idx)}")

        print("[CamRestDataModule] Creating PyTorch DataLoaders...")
        train_dataset = CamRestDataset(train_samples, self.word2idx)
        val_dataset = CamRestDataset(val_samples, self.word2idx)
        test_dataset = CamRestDataset(test_samples, self.word2idx)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        print("[CamRestDataModule] Preprocessing Complete")