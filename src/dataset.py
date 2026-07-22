"""
PyTorch Dataset & DataLoader untuk TSCP.

Setiap sample terdiri dari:
- input_seq:  token indices untuk B_{t-1} R_{t-1} U_t
- bspan_seq:  token indices untuk B_t (target stage 1)
- response_seq: token indices untuk R_t (target stage 2)
- input_tokens: raw token list (untuk CopyNet — perlu tahu token asli di input)
- bspan_tokens: raw token list (untuk CopyNet stage 2)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import src.config as config
from src.preprocessing import tokenize, tokens_to_indices


class SequicityDataset(Dataset):
    def __init__(self, samples, word2idx):
        """
        Args:
            samples: list of dict dari preprocess.process_all_dialogues()
            word2idx: vocabulary mapping
        """
        self.samples = samples
        self.word2idx = word2idx
        self.pad_idx = word2idx[config.PAD_TOKEN]
        self.sos_idx = word2idx[config.SOS_TOKEN]
        self.eos_idx = word2idx[config.EOS_TOKEN]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize
        input_tokens = tokenize(sample["input"])
        bspan_tokens = tokenize(sample["target_bspan"])
        response_tokens = tokenize(sample["target_response"])

        # Convert to indices
        input_indices = tokens_to_indices(input_tokens, self.word2idx)
        bspan_indices = tokens_to_indices(bspan_tokens, self.word2idx)
        response_indices = tokens_to_indices(response_tokens, self.word2idx)

        # Tambah SOS/EOS untuk decoder targets
        # Decoder input: <sos> + tokens
        # Decoder target: tokens + <eos>
        bspan_input = [self.sos_idx] + bspan_indices
        bspan_target = bspan_indices + [self.eos_idx]

        response_input = [self.sos_idx] + response_indices
        response_target = response_indices + [self.eos_idx]

        return {
            "input_indices": torch.tensor(input_indices, dtype=torch.long),
            "input_tokens": input_tokens,  # raw tokens untuk CopyNet
            "bspan_input": torch.tensor(bspan_input, dtype=torch.long),
            "bspan_target": torch.tensor(bspan_target, dtype=torch.long),
            "bspan_tokens": bspan_tokens,  # raw tokens untuk Stage 2 CopyNet
            "response_input": torch.tensor(response_input, dtype=torch.long),
            "response_target": torch.tensor(response_target, dtype=torch.long),
            "raw_input": sample["input"],
            "raw_bspan": sample["target_bspan"],
            "raw_response": sample["target_response"],
        }


def collate_fn(batch):
    """
    Custom collate: pad sequences ke panjang max dalam batch.
    
    CopyNet membutuhkan dynamic output space (V ∪ X),
    jadi kita juga perlu menyimpan mapping token → posisi di input
    untuk setiap sample.
    """
    pad_idx = batch[0]["input_indices"].new_tensor(0)  # PAD = 0

    # Pad semua sequence
    input_padded = pad_sequence(
        [b["input_indices"] for b in batch], batch_first=True, padding_value=0
    )
    bspan_input_padded = pad_sequence(
        [b["bspan_input"] for b in batch], batch_first=True, padding_value=0
    )
    bspan_target_padded = pad_sequence(
        [b["bspan_target"] for b in batch], batch_first=True, padding_value=0
    )
    response_input_padded = pad_sequence(
        [b["response_input"] for b in batch], batch_first=True, padding_value=0
    )
    response_target_padded = pad_sequence(
        [b["response_target"] for b in batch], batch_first=True, padding_value=0
    )

    # Lengths (untuk packing jika diperlukan)
    input_lengths = torch.tensor([len(b["input_indices"]) for b in batch])
    bspan_lengths = torch.tensor([len(b["bspan_input"]) for b in batch])
    response_lengths = torch.tensor([len(b["response_input"]) for b in batch])

    # Raw tokens (list of lists — tidak bisa di-tensor, simpan as-is)
    input_tokens_batch = [b["input_tokens"] for b in batch]
    bspan_tokens_batch = [b["bspan_tokens"] for b in batch]

    return {
        "input_padded": input_padded,
        "input_lengths": input_lengths,
        "input_tokens": input_tokens_batch,
        "bspan_input": bspan_input_padded,
        "bspan_target": bspan_target_padded,
        "bspan_lengths": bspan_lengths,
        "bspan_tokens": bspan_tokens_batch,
        "response_input": response_input_padded,
        "response_target": response_target_padded,
        "response_lengths": response_lengths,
    }


def get_dataloader(samples, word2idx, batch_size, shuffle=True):
    """Buat DataLoader dari samples."""
    dataset = SequicityDataset(samples, word2idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )
