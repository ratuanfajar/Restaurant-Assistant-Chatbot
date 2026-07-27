"""
Streamlit Chatbot Interface untuk TSCP (Two Stage CopyNet) / Sequicity.

Chatbot ini menggunakan model yang telah di-training dengan:
- Preprocessing dari notebook eksp3
- Supervised training dari notebook eksp2  
- RL fine-tuning dari notebook rl tuning

Checkpoint model tersimpan di folder checkpoints/
"""

import streamlit as st
import torch
import json
import os
from pathlib import Path

# Import modul dari src
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import (
    tokenize, build_vocabulary, process_dialogues, 
    precompute_db_slots, split_dialogues, load_raw_data,
    SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
)
from src.model import TSCP
from src.utils import (
    search_kb, get_kt_from_bspan, lexicalize_response,
    resolve_inconsistent_bspan, parse_bspan, build_copy_mapping,
    beam_search_decode
)
import src.config as config


# ============================================================
# Helper Functions
# ============================================================

def tokens_to_indices(tokens, word2idx):
    """Convert tokens to indices (tanpa OOV mapping untuk inference sederhana)."""
    unk_idx = word2idx.get(UNK_TOKEN, 0)
    indices = []
    for t in tokens:
        if t in word2idx:
            indices.append(word2idx[t])
        else:
            indices.append(unk_idx)
    return indices


@st.cache_resource
def load_database():
    """Load database restoran dari file JSON."""
    db_path = Path(__file__).parent / "data" / "CamRestDB.json"
    with open(db_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    return database


@st.cache_resource
def load_vocabulary():
    """
    Load vocabulary dari data training.
    Untuk production, sebaiknya vocabulary disimpan terpisah.
    Di sini kita rebuild dari data training.
    """
    data_dir = Path(__file__).parent / "data"
    dialogues, database = load_raw_data(data_dir)
    train_dial, _, _ = split_dialogues(dialogues)
    
    db_slot_values = precompute_db_slots(database)
    train_samples = process_dialogues(train_dial, db_slot_values)
    
    # Tokenize samples untuk build vocab
    temp_tokenized = [{
        "tokens_input": tokenize(s["input"], SPECIAL_TOKENS),
        "tokens_bspan": tokenize(s["target_bspan"], SPECIAL_TOKENS),
        "tokens_response": tokenize(s["target_response"], SPECIAL_TOKENS)
    } for s in train_samples]
    
    word2idx, idx2word = build_vocabulary(temp_tokenized, max_vocab_size=800)
    return word2idx, idx2word


@st.cache_resource
def load_model(checkpoint_name="tscp_rl_final.pt"):
    """
    Load trained model dari checkpoint.
    
    Args:
        checkpoint_name: nama file checkpoint di folder checkpoints/
        
    Returns:
        model: TSCP model yang sudah di-load
        word2idx, idx2word: vocabulary mappings
        database: knowledge base
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load database
    database = load_database()
    
    # Load checkpoint
    checkpoint_path = Path(__file__).parent / "checkpoints" / checkpoint_name
    
    if not checkpoint_path.exists():
        st.error(f"Checkpoint tidak ditemukan: {checkpoint_path}")
        st.info("Pastikan Anda sudah training model dan checkpoint tersimpan di folder checkpoints/")
        return None, None, None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load vocabulary dari checkpoint jika ada, otherwise rebuild dari data
    if 'word2idx' in checkpoint:
        word2idx = checkpoint['word2idx']
        idx2word = {v: k for k, v in word2idx.items()}
        st.info(f"Vocabulary loaded from checkpoint: {len(word2idx)} tokens")
    else:
        word2idx, idx2word = load_vocabulary()
        st.info(f"Vocabulary rebuilt from data: {len(word2idx)} tokens")
    
    vocab_size = len(word2idx)
    
    # Initialize model
    model = TSCP(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, word2idx, idx2word, database, device


def generate_response(model, user_input, prev_bspan, prev_response, 
                      word2idx, idx2word, database, device,
                      beam_size=5):
    """
    Generate response untuk satu turn percakapan.
    
    Alur:
    1. Format input: B_{t-1} + R_{t-1} + U_t
    2. Encode input
    3. Decode bspan (Stage 1)
    4. KB Search berdasarkan bspan
    5. Decode response (Stage 2)
    6. Lexicalization
    
    Returns:
        response: str - response final (sudah di-lexicalize)
        pred_bspan_text: str - predicted belief span
        kb_matches: list - matching restaurants dari KB
    """
    # === 1. Format input ===
    input_parts = []
    if prev_bspan:
        input_parts.append(prev_bspan)
    if prev_response:
        input_parts.append(prev_response)
    input_parts.append(user_input.lower().strip())
    full_input = " ".join(input_parts)
    
    # === 2. Tokenize & Encode ===
    input_tokens = tokenize(full_input, SPECIAL_TOKENS)
    input_indices = tokens_to_indices(input_tokens, word2idx)
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
    input_lengths = torch.tensor([len(input_indices)])
    
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(input_tensor, input_lengths)
        
        # === 3. Decode Bspan (Stage 1) dengan Beam Search ===
        pred_bspan_tokens = beam_search_decode(
            decoder=model.decoder1,
            encoder_outputs=encoder_outputs,
            initial_hidden=encoder_hidden,
            word2idx=word2idx,
            idx2word=idx2word,
            source_tokens=input_tokens,
            beam_size=beam_size,
            max_len=config.MAX_DECODE_LEN_BSPAN,
            kt=None,
            copy_source_outputs=encoder_outputs,
            copy_source_tokens=input_tokens
        )
        
        pred_bspan_text = " ".join(pred_bspan_tokens)
        
        # Post-processing: resolve inconsistent bspan
        pred_bspan_text = resolve_inconsistent_bspan(pred_bspan_text, database)
        pred_bspan_tokens = tokenize(pred_bspan_text, SPECIAL_TOKENS)
        
        # === 4. KB Search ===
        kb_matches, kt = search_kb(pred_bspan_text, database)
        informable, requestable = parse_bspan(pred_bspan_text)
        
        # === 5. Dapatkan bspan hidden states untuk Stage 2 ===
        bspan_indices = tokens_to_indices(pred_bspan_tokens, word2idx)
        sos_idx = word2idx[SOS_TOKEN]
        bspan_input = torch.tensor([[sos_idx] + bspan_indices], dtype=torch.long, device=device)
        
        bspan_embedded = model.decoder1.embedding(bspan_input)
        bspan_outputs, bspan_final_hidden = model.decoder1.gru(bspan_embedded, encoder_hidden)
        
        # === 6. Decode Response (Stage 2) dengan Beam Search ===
        pred_response_tokens = beam_search_decode(
            decoder=model.decoder2,
            encoder_outputs=bspan_outputs,
            initial_hidden=bspan_final_hidden,
            word2idx=word2idx,
            idx2word=idx2word,
            source_tokens=pred_bspan_tokens,
            beam_size=beam_size,
            max_len=config.MAX_DECODE_LEN_RESPONSE,
            kt=kt,
            copy_source_outputs=bspan_outputs,
            copy_source_tokens=pred_bspan_tokens
        )
        
        pred_response_text = " ".join(pred_response_tokens)
    
    # === 7. Lexicalization ===
    final_response = lexicalize_response(pred_response_text, kb_matches)
    
    return final_response, pred_bspan_text, kb_matches


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Sequicity Chatbot - CamRest676",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Sequicity Restaurant Chatbot")
st.markdown("""
Chatbot untuk rekomendasi restoran di Cambridge menggunakan arsitektur **TSCP (Two Stage CopyNet)**.
Model ini telah melalui tahap:
1. **Preprocessing** (eksp3) - Data preparation dan delexicalization
2. **Supervised Training** (eksp2) - Training awal dengan teacher forcing
3. **RL Fine-tuning** (rl tuning) - Optimasi dengan reinforcement learning
""")

# Sidebar untuk konfigurasi
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Pilih checkpoint
    checkpoint_folder = Path(__file__).parent / "checkpoints"
    available_checkpoints = []
    if checkpoint_folder.exists():
        available_checkpoints = [f.name for f in checkpoint_folder.glob("*.pt")]
    
    if available_checkpoints:
        selected_checkpoint = st.selectbox(
            "Pilih Checkpoint Model",
            available_checkpoints,
            index=len(available_checkpoints) - 1  # Default ke yang terakhir
        )
    else:
        selected_checkpoint = None
        st.warning("Tidak ada checkpoint ditemukan!")
    
    beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("**Informasi Model:**")
    st.markdown("- Architecture: TSCP (Two Stage CopyNet)")
    st.markdown("- Dataset: CamRest676")
    st.markdown("- Slots: food, area, pricerange")
    
    st.markdown("---")
    st.markdown("**Contoh Pertanyaan:**")
    st.markdown("- \"I want a cheap Italian restaurant\"")
    st.markdown("- \"What about the phone number?\"")
    st.markdown("- \"Is there a Chinese restaurant in the north area?\"")

# Session state untuk menyimpan history percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.prev_bspan = ""
    st.session_state.prev_response = ""
    st.session_state.turn = 0

# Load model
if selected_checkpoint:
    with st.spinner("Loading model..."):
        model, word2idx, idx2word, database, device = load_model(selected_checkpoint)
    
    if model is not None:
        st.success("✅ Model berhasil dimuat!")
        
        # Tombol reset percakapan
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### 💬 Mulai Percakapan")
        with col2:
            if st.button("🔄 Reset"):
                st.session_state.messages = []
                st.session_state.prev_bspan = ""
                st.session_state.prev_response = ""
                st.session_state.turn = 0
                st.rerun()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ketik pertanyaan tentang restoran..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, bspan, kb_matches = generate_response(
                        model=model,
                        user_input=prompt,
                        prev_bspan=st.session_state.prev_bspan,
                        prev_response=st.session_state.prev_response,
                        word2idx=word2idx,
                        idx2word=idx2word,
                        database=database,
                        device=device,
                        beam_size=beam_size
                    )
                    
                    st.markdown(response)
                    
                    # Debug info (expandable)
                    with st.expander("🔍 Debug Info"):
                        st.write("**Belief Span:**")
                        st.code(bspan)
                        st.write("**KB Matches:**")
                        if kb_matches:
                            for match in kb_matches[:3]:  # Show max 3 matches
                                st.json(match)
                        else:
                            st.write("No matches found")
                        st.write("**Delexicalized Response:**")
                        # Get delex response from model output before lexicalization
                        st.code(response)
            
            # Update session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.prev_bspan = bspan
            st.session_state.prev_response = response  # Store delexicalized version
            st.session_state.turn += 1
            
            st.rerun()
    else:
        st.error("Gagal memuat model. Pastikan checkpoint tersedia.")
else:
    st.warning("⚠️ Silakan pilih checkpoint model dari sidebar untuk memulai.")
    st.info("""
    ### Cara Menggunakan:
    1. Pastikan Anda sudah melakukan training model
    2. Checkpoint harus tersimpan di folder `checkpoints/`
    3. Pilih checkpoint dari sidebar
    4. Mulai percakapan dengan mengetik pertanyaan tentang restoran
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with Streamlit | TSCP Architecture | CamRest676 Dataset
</div>
""", unsafe_allow_html=True)
