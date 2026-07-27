"""
Streamlit chatbot untuk asisten reservasi restoran (CamRest676).
Model: Two Stage CopyNet (TSCP) hasil replikasi paper Sequicity (Lei et al., 2018),
setelah RL fine-tuning (checkpoint tscp_rl_best.pt).

Jalankan:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from chatbot_engine import RestaurantAssistant

ROOT = Path(__file__).resolve().parent
# v2 supervised dipakai: respons lebih natural drpd RL v2 yang reward-hacking
# (mengulang placeholder). Ganti ke "tscp_rl_v2_best.pt" bila ingin model RL.
CHECKPOINT = ROOT / "checkpoints" / "tscp_supervised_v2_best.pt"
DB_PATH = ROOT / "data" / "CamRestDB.json"

st.set_page_config(page_title="Asisten Restoran Cambridge", page_icon="🍽️", layout="centered")


@st.cache_resource(show_spinner="Memuat model TSCP...")
def load_bot():
    return RestaurantAssistant(str(CHECKPOINT), str(DB_PATH), device="cpu")


def new_conversation():
    bot = load_bot()
    bot.reset()
    st.session_state.messages = []
    st.session_state.last_debug = None


# --- Sidebar ---
with st.sidebar:
    st.header("🍽️ Asisten Restoran")
    st.caption(
        "Chatbot task-oriented berbasis **Two Stage CopyNet (TSCP)** — "
        "replikasi paper *Sequicity* (Lei et al., ACL 2018), dataset CamRest676."
    )
    st.markdown(
        "**Cara pakai:** minta rekomendasi restoran berdasarkan jenis masakan, "
        "area (centre/north/south/east/west), atau kisaran harga "
        "(cheap/moderate/expensive), lalu tanyakan alamat / nomor telepon / kode pos."
    )
    if st.button("🔄 Percakapan baru", use_container_width=True):
        new_conversation()
        st.rerun()

    show_debug = st.toggle("Tampilkan detail model (belief span / KB)", value=True)

    st.divider()
    st.subheader("Contoh")
    examples = [
        "i'm looking for a cheap restaurant in the centre",
        "do you have any italian food?",
        "what is the address and phone number?",
    ]
    for ex in examples:
        st.markdown(f"- _{ex}_")


# --- State init ---
if "messages" not in st.session_state:
    new_conversation()

bot = load_bot()

st.title("Asisten Reservasi Restoran")
st.caption("Tanya dalam bahasa Inggris (model dilatih pada dataset CamRest676 yang berbahasa Inggris).")

# --- Render riwayat chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# --- Input ---
if prompt := st.chat_input("Contoh: i want an expensive restaurant in the south"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Berpikir..."):
            result = bot.respond(prompt)
        reply = result["response"].strip() or "_(maaf, saya tidak dapat menghasilkan respons)_"
        st.markdown(reply)

        if show_debug:
            with st.expander("🔍 Detail model (turn ini)"):
                st.markdown(f"**Belief span (B_t):** `{result['bspan']}`")
                status_label = {
                    "no_constraint": "belum ada constraint",
                    "no_match": "no match", "exact": "exact match",
                    "multiple": "multiple match", "empty_input": "input kosong",
                }
                st.markdown(
                    f"**KB search (k_t):** `{result['kt']}` "
                    f"→ {status_label.get(result.get('status'), '-')} "
                    f"({result['num_matches']} restoran cocok)"
                )
                st.markdown(f"**Response (delex):** `{result['response_delex']}`")
                if result.get("kb_match"):
                    m = result["kb_match"]
                    st.markdown(
                        f"**Restoran terpilih:** {m.get('name','-')} — "
                        f"{m.get('food','-')}, {m.get('area','-')}, {m.get('pricerange','-')}"
                    )

    st.session_state.messages.append({"role": "assistant", "content": reply})
