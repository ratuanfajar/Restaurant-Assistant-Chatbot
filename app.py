"""Streamlit chat UI for the restaurant assistant."""

from pathlib import Path

import streamlit as st

from src.chatbot_engine import RestaurantAssistant

ROOT = Path(__file__).resolve().parent

# Supervised v2 gives more natural replies than the RL model (which repeats slots).
CHECKPOINT = ROOT / "checkpoints" / "tscp_supervised_v2_best.pt"
DB_PATH = ROOT / "data" / "CamRestDB.json"

STATUS_LABEL = {
    "no_constraint": "no constraint yet", "no_match": "no match",
    "exact": "exact match", "multiple": "multiple matches",
    "empty_input": "empty input", "out_of_domain": "out of domain",
}
_SKIP_DETAILS = {"empty_input", "out_of_domain"}

st.set_page_config(page_title="Restaurant Assistant", layout="centered")


@st.cache_resource(show_spinner="Loading the assistant...")
def load_bot():
    return RestaurantAssistant(str(CHECKPOINT), str(DB_PATH), device="cpu")


def new_conversation():
    load_bot().reset()
    st.session_state.messages = []


def render_details(result):
    """Per-turn pipeline output: belief span, KB search, delexicalized response, KB entry."""
    with st.expander("Processing details"):
        st.markdown(f"**Belief span (B_t):** `{result['bspan']}`")
        st.markdown(
            f"**KB search (k_t):** `{result['kt']}` -> "
            f"{STATUS_LABEL.get(result.get('status'), '-')} "
            f"({result['num_matches']} restaurants matched)"
        )
        st.markdown(f"**Response (delexicalized):** `{result['response_delex']}`")
        if result.get("kb_match"):
            m = result["kb_match"]
            st.markdown(
                f"**Selected restaurant:** {m.get('name', '-')} — "
                f"{m.get('food', '-')}, {m.get('area', '-')}, {m.get('pricerange', '-')}"
            )


# Sidebar: how to use + examples + debug toggle
with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "Ask for a restaurant by **cuisine**, **area** "
        "(centre / north / south / east / west), or **price** "
        "(cheap / moderate / expensive). Then ask for its **address**, "
        "**phone number**, or **postcode**."
    )
    st.info("This assistant understands **English only**.")

    st.markdown("**Examples**")
    for ex in [
        "I'm looking for a cheap restaurant in the centre",
        "Do you have any Italian food?",
        "What is the address and phone number?",
    ]:
        st.markdown(f"- _{ex}_")

    st.divider()
    show_details = st.toggle("Show processing details", value=True)
    if st.button("New conversation", use_container_width=True):
        new_conversation()
        st.rerun()


# State
if "messages" not in st.session_state:
    new_conversation()

bot = load_bot()

st.title("Restaurant Assistant")

# Conversation
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I can help you find a restaurant. Tell me what you're looking for — "
            "a cuisine, an area, or a price range. Please chat in English."
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        result = msg.get("result")
        if show_details and result and result.get("status") not in _SKIP_DETAILS:
            render_details(result)

if prompt := st.chat_input("Message the assistant (in English)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = bot.respond(prompt)
        reply = result["response"].strip() or "_Sorry, I couldn't generate a response._"
        st.markdown(reply)
        if show_details and result.get("status") not in _SKIP_DETAILS:
            render_details(result)

    st.session_state.messages.append({"role": "assistant", "content": reply, "result": result})
