"""
Interactive Chatbot Inference — mode testing Sequicity/TSCP.

Alur per turn (sesuai Fase 5 planning):
1. Encode: masukkan history + U_t ke Encoder
2. Decode 1 (Bspan): generate B_t
3. KB Search: parse B_t, query database, tentukan k_t
4. Decode 2 (Response): generate R_t (masih berisi placeholder)
5. Lexicalization: ganti placeholder dengan value asli dari KB
6. Post-processing: handle inconsistent bspan (Section 5.7)
"""

import torch
import os

import src.config as config
from src.model import TSCP
from src.preprocessing import tokenize, tokens_to_indices
from src.evaluate import greedy_decode_bspan, greedy_decode_response
from src.utils import (
    search_kb, get_kt_from_bspan, lexicalize_response,
    resolve_inconsistent_bspan, parse_bspan,
)


class SequicityChatbot:
    """Interactive chatbot wrapper untuk TSCP."""

    def __init__(self, model, word2idx, idx2word, database, device="cpu"):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.database = database
        self.device = device

        self.model.eval()

        # Dialogue state
        self.prev_bspan = ""
        self.prev_response = ""
        self.turn = 0

    def reset(self):
        """Reset dialogue state untuk percakapan baru."""
        self.prev_bspan = ""
        self.prev_response = ""
        self.turn = 0
        print("\n[System] Dialogue reset. Mulai percakapan baru.\n")

    def respond(self, user_input):
        """
        Proses satu turn user input dan return response.
        
        Args:
            user_input: str — utterance dari user
            
        Returns:
            response: str — response final (sudah di-lexicalize)
        """
        user_input = user_input.lower().strip()

        # === 1. Format input: B_{t-1} + R_{t-1} + U_t ===
        input_parts = []
        if self.prev_bspan:
            input_parts.append(self.prev_bspan)
        if self.prev_response:
            input_parts.append(self.prev_response)
        input_parts.append(user_input)
        full_input = " ".join(input_parts)

        # === 2. Encode ===
        input_tokens = tokenize(full_input)
        input_indices = tokens_to_indices(input_tokens, self.word2idx)
        input_tensor = torch.tensor([input_indices], dtype=torch.long, device=self.device)
        input_lengths = torch.tensor([len(input_indices)])

        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.model.encoder(
                input_tensor, input_lengths
            )

            # === 3. Decode Bspan (Stage 1) ===
            pred_bspan_tokens = greedy_decode_bspan(
                self.model, encoder_outputs, encoder_hidden,
                self.word2idx, self.idx2word, input_tokens, self.device
            )
            pred_bspan_text = " ".join(pred_bspan_tokens)

            # Post-processing: resolve inconsistent bspan
            pred_bspan_text = resolve_inconsistent_bspan(pred_bspan_text, self.database)
            pred_bspan_tokens = tokenize(pred_bspan_text)

            # === 4. KB Search ===
            kb_matches, kt = search_kb(pred_bspan_text, self.database)
            informable, requestable = parse_bspan(pred_bspan_text)

            # === 5. Dapatkan bspan hidden states ===
            bspan_indices = tokens_to_indices(pred_bspan_tokens, self.word2idx)
            sos_idx = self.word2idx[config.SOS_TOKEN]
            bspan_input = torch.tensor(
                [[sos_idx] + bspan_indices], dtype=torch.long, device=self.device
            )

            bspan_embedded = self.model.decoder1.embedding(bspan_input)
            bspan_outputs, bspan_final_hidden = self.model.decoder1.gru(
                bspan_embedded, encoder_hidden
            )

            # === 6. Decode Response (Stage 2) ===
            pred_response_tokens = greedy_decode_response(
                self.model, bspan_outputs, bspan_final_hidden,
                kt, self.word2idx, self.idx2word, pred_bspan_tokens, self.device
            )
            pred_response_text = " ".join(pred_response_tokens)

        # === 7. Lexicalization: ganti placeholder → value asli ===
        final_response = lexicalize_response(pred_response_text, kb_matches)

        # === 8. Update state untuk turn berikutnya ===
        self.prev_bspan = pred_bspan_text
        self.prev_response = pred_response_text  # Simpan versi delexicalized
        self.turn += 1

        # Debug info
        print(f"  [Debug] Bspan: {pred_bspan_text}")
        print(f"  [Debug] KB matches: {len(kb_matches)}")
        print(f"  [Debug] k_t: {kt.tolist()}")
        print(f"  [Debug] Delex response: {pred_response_text}")

        return final_response


def load_model_for_inference(checkpoint_path, word2idx, device="cpu"):
    """Load trained model dari checkpoint."""
    vocab_size = len(word2idx)
    model = TSCP(vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model


def interactive_chat(model, word2idx, idx2word, database, device="cpu"):
    """
    Mode chatbot interaktif di terminal.
    Ketik 'quit' untuk keluar, 'reset' untuk mulai percakapan baru.
    """
    chatbot = SequicityChatbot(model, word2idx, idx2word, database, device)

    print("\n" + "=" * 60)
    print("SEQUICITY CHATBOT — CamRest676 Restaurant Booking")
    print("=" * 60)
    print("Ketik pertanyaan tentang restoran di Cambridge.")
    print("Contoh: 'I want a cheap Italian restaurant'")
    print("Ketik 'reset' untuk mulai ulang, 'quit' untuk keluar.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            chatbot.reset()
            continue

        response = chatbot.respond(user_input)
        print(f"Bot: {response}\n")
