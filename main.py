from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from model import ChatbotModel, LOW_CONFIDENCE_MESSAGE


APP_TITLE = "NLP Chatbot"


def get_examples() -> List[str]:
    return [
        "Hi",
        "What time is it?",
        "What is today's date?",
        "Tell me a joke",
        "What are the college working hours?",
        "What is machine learning?",
        "How do I prepare for placements?",
        "Where is the college located?",
    ]


@st.cache_resource(show_spinner=False)
def load_model() -> ChatbotModel:
    intents_path = Path(__file__).with_name("intents.json")
    return ChatbotModel(intents_path=intents_path, confidence_threshold=0.5)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm an NLP chatbot. Ask me a question, or pick an example from the sidebar.",
            }
        ]


def render_sidebar() -> Dict[str, str]:
    st.sidebar.title("Chatbot")

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. What would you like to ask?"}
        ]

    st.sidebar.markdown("### Example questions")
    example = st.sidebar.radio(
        "Click one to send",
        options=get_examples(),
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### Debug")
    show_debug = st.sidebar.toggle("Show matching details", value=False)

    st.sidebar.markdown("### Notes")
    st.sidebar.caption(
        "This is an offline ML chatbot trained on `intents.json` using TF-IDF + Logistic Regression."
    )

    return {"example": example, "show_debug": str(show_debug)}


def _inject_css() -> None:
    # Light, modern styling without fighting Streamlit too much.
    st.markdown(
        """
<style>
  .block-container { max-width: 900px; padding-top: 1.2rem; }
  .small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }

  /* ChatGPT-like message layout */
  .chat-row { display: flex; gap: 10px; align-items: flex-end; margin: 10px 0; }
  .chat-row.user { justify-content: flex-end; }
  .chat-row.assistant { justify-content: flex-start; }

  .chat-avatar {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex: 0 0 28px;
    border: 1px solid rgba(49, 51, 63, 0.12);
    background: rgba(49, 51, 63, 0.04);
  }
  .chat-avatar.user { background: rgba(59, 130, 246, 0.12); border-color: rgba(59, 130, 246, 0.25); }
  .chat-avatar.assistant { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.25); }

  .chat-bubble {
    max-width: 75%;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid rgba(49, 51, 63, 0.12);
    background: rgba(255, 255, 255, 0.7);
    line-height: 1.35;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  .chat-bubble.user {
    background: rgba(59, 130, 246, 0.14);
    border-color: rgba(59, 130, 246, 0.25);
  }
  .chat-bubble.assistant {
    background: rgba(16, 185, 129, 0.10);
    border-color: rgba(16, 185, 129, 0.22);
  }

  [data-testid="stChatInput"] textarea { border-radius: 12px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_message(role: str, content: str) -> None:
    safe_role = "assistant" if role != "user" else "user"
    avatar = "🤖" if safe_role == "assistant" else "🙂"

    if safe_role == "user":
        html = f"""
<div class="chat-row user">
  <div class="chat-bubble user">{content}</div>
  <div class="chat-avatar user" title="You">{avatar}</div>
</div>
"""
    else:
        html = f"""
<div class="chat-row assistant">
  <div class="chat-avatar assistant" title="Bot">{avatar}</div>
  <div class="chat-bubble assistant">{content}</div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
    _inject_css()

    st.title(APP_TITLE)
    st.caption("Confidence-scored intent recognition (TF-IDF + Logistic Regression).")

    model = load_model()
    init_state()
    sidebar = render_sidebar()
    show_debug = sidebar["show_debug"] == "True"

    # Render history
    for msg in st.session_state.messages:
        _render_message(msg["role"], msg["content"])

    # Input
    prompt = st.chat_input("Type your message…")

    # Sidebar example -> send behavior
    example = sidebar["example"]
    if "last_example" not in st.session_state:
        st.session_state.last_example = ""
    if prompt is None and example and example != st.session_state.last_example:
        if st.session_state.last_example != "":
            prompt = example
        st.session_state.last_example = example

    if not prompt:
        return

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_message("user", prompt)

    # Bot response
    with st.spinner("Thinking..."):
        reply, details = model.get_response(prompt)

    _render_message("assistant", reply)

    # Confidence display (always show, interview-friendly)
    conf_pct = int(round(details.effective_confidence * 100))
    if reply == LOW_CONFIDENCE_MESSAGE:
        st.markdown(
            f"<div class='small-muted'>Confidence: {conf_pct}% (low)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='small-muted'>Confidence: {conf_pct}%</div>",
            unsafe_allow_html=True,
        )

    if show_debug:
        st.caption(
            f"Intent: `{details.intent}` · "
            f"Effective: `{details.effective_confidence:.3f}` · "
            f"Proba: `{details.confidence:.3f}` · "
            f"Similarity: `{details.similarity:.3f}`"
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

