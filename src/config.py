"""
Hyperparameters untuk TSCP (Two Stage CopyNet) — sesuai paper Sequicity Section 5.2.
"""

import os

# === Paths ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CAMREST_PATH = os.path.join(DATA_DIR, "CamRest676.json")
DB_PATH = os.path.join(DATA_DIR, "CamRestDB.json")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# === Data Split (3:1:1 sesuai paper) ===
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
TEST_RATIO = 0.2

# === Model Architecture (Section 5.2) ===
EMBED_SIZE = 50
HIDDEN_SIZE = 50
NUM_GRU_LAYERS = 1
DROPOUT = 0.0

# === Knowledge Base Search Result Vector ===
KB_INDICATOR_SIZE = 3  # [no_match, exact_match, multiple_match]

# === Special Tokens ===
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Bspan delimiters
INF_OPEN = "<inf>"
INF_CLOSE = "</inf>"
REQ_OPEN = "<req>"
REQ_CLOSE = "</req>"

# Delexicalization placeholders
SLOT_TOKENS = [
    "NAME_SLOT",
    "ADDRESS_SLOT",
    "PHONE_SLOT",
    "POSTCODE_SLOT",
    "FOOD_SLOT",
    "AREA_SLOT",
    "PRICERANGE_SLOT",
]

# Semua special tokens
SPECIAL_TOKENS = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    INF_OPEN, INF_CLOSE, REQ_OPEN, REQ_CLOSE,
] + SLOT_TOKENS

# === Training — Supervised (Section 5.2) ===
SL_LEARNING_RATE = 0.003
SL_EPOCHS = 100
SL_BATCH_SIZE = 32
SL_CLIP_GRAD = 5.0
SL_PATIENCE = 10  # early stopping patience

# === Training — Reinforcement Learning (Section 5.2) ===
RL_LEARNING_RATE = 0.0001
RL_EPOCHS = 50
RL_BATCH_SIZE = 1  # RL biasanya per-sample
RL_LAMBDA = 0.8  # decay parameter untuk return
RL_REWARD_POS = 1.0
RL_REWARD_NEG = -0.1

# === Inference ===
BEAM_SIZE = 10
MAX_DECODE_LEN_BSPAN = 30
MAX_DECODE_LEN_RESPONSE = 60

# === Informable & Requestable slots (CamRest676) ===
INFORMABLE_SLOTS = ["food", "area", "pricerange"]
REQUESTABLE_SLOTS = ["address", "phone", "postcode", "food", "area", "pricerange", "name"]

# Mapping dari DB field ke placeholder token
DB_FIELD_TO_SLOT = {
    "name": "NAME_SLOT",
    "address": "ADDRESS_SLOT",
    "phone": "PHONE_SLOT",
    "postcode": "POSTCODE_SLOT",
    "food": "FOOD_SLOT",
    "area": "AREA_SLOT",
    "pricerange": "PRICERANGE_SLOT",
}
