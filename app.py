"""Chat UI for the RAG agent.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import uuid

import streamlit as st

from rag_agent.agent import SEARCH_TOOL, RAGAgent, list_free_models, make_client
from rag_agent.config import KB_DIR, MAX_SEARCHES, MODE, MODEL, TOP_K
from rag_agent.retriever import BM25Retriever

st.set_page_config(page_title="Helix Cloud Helpdesk", page_icon="💬", layout="wide")


@st.cache_resource
def get_retriever() -> BM25Retriever:
    return BM25Retriever.from_dir(KB_DIR)


@st.cache_data(ttl=3600, show_spinner="Loading free models from OpenRouter…")
def get_free_models() -> list[dict]:
    return list_free_models()


# ---------------------------------------------------------------------------
# Sidebar: settings + tools
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    api_key = os.getenv("OPENROUTER_API_KEY") or st.text_input(
        "OpenRouter API key", type="password", help="Get a free key at openrouter.ai/keys, or set it in .env"
    )

    model_ids = [MODEL]
    tool_support: dict[str, bool] = {}
    try:
        models = get_free_models()
        tool_support = {m["id"]: m["tools"] for m in models}
        model_ids = list(dict.fromkeys([MODEL, *tool_support]))
    except Exception as e:  # offline or OpenRouter unreachable: keep the configured model
        st.caption(f"Couldn't load the model list ({e.__class__.__name__}); using MODEL from .env.")
    model = st.selectbox(
        "Free model",
        model_ids,
        format_func=lambda m: f"{m}  🔧" if tool_support.get(m) else m,
        help="🔧 = supports tool calling (needed for agent mode)",
    )
    mode = st.radio(
        "Mode",
        ["agent", "simple"],
        index=0 if MODE == "agent" else 1,
        help="agent: the model decides when and what to search.\n\nsimple: search with the question first, then answer.",
    )
    top_k = st.slider("Chunks per search (top_k)", 1, 8, TOP_K)
    max_searches = st.slider("Max searches per question", 1, 5, MAX_SEARCHES, disabled=mode == "simple")

    st.divider()
    st.header("Tools the agent can use")
    fn = SEARCH_TOOL["function"]
    st.markdown(f"**`{fn['name']}`**  \n{fn['description']}")
    st.caption("Input: " + ", ".join(f"`{k}` ({v['type']})" for k, v in fn["parameters"]["properties"].items()))
    retriever = get_retriever()
    docs = sorted({c.doc_id for c in retriever.chunks})
    with st.expander(f"Knowledge base: {len(docs)} documents, {len(retriever.chunks)} chunks"):
        st.markdown("\n".join(f"- `{d}`" for d in docs))

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"ui-{uuid.uuid4()}"  # new conversation = new Langfuse session
        st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.title("💬 Helix Cloud Helpdesk")
st.caption("Ask about company policies: PTO, travel, laptops, on-call pay, parental leave and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # One Langfuse session per chat conversation (one trace per turn).
    st.session_state.session_id = f"ui-{uuid.uuid4()}"


def render_details(details: dict) -> None:
    calls = details.get("tool_calls") or []
    label = f"🔧 {len(calls)} tool call{'s' if len(calls) != 1 else ''} · {details['model']} · {details['mode']} mode"
    with st.expander(label):
        for i, call in enumerate(calls, 1):
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = call["arguments"]
            st.markdown(f"**{i}. `{call['tool']}`**")
            st.json(args, expanded=True)
            st.text_area("Result", call["result"], height=160, key=f"r-{details['id']}-{i}", disabled=True)
        if not calls:
            st.write("The model answered without calling a tool.")
        st.caption(
            f"Sources: {', '.join(details['sources']) or 'none'} · "
            f"tokens in {details['input_tokens']} / out {details['output_tokens']}"
        )


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("details"):
            render_details(m["details"])

question = st.chat_input("Ask a question…") or st.session_state.pop("pending", None)

if not st.session_state.messages and not question:
    st.markdown("**Try one of these:**")
    examples = [
        "How many PTO days do I get per year?",
        "What is the hotel cap in London, and can my conference ticket come from my learning budget?",
        "I lost my laptop on a trip. Who do I tell and how quickly?",
        "What is the 401(k) match?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

if question:
    if not api_key:
        st.error("Add your OpenRouter API key in the sidebar (or OPENROUTER_API_KEY in .env).")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            try:
                agent = RAGAgent(client=make_client(api_key), retriever=retriever, model=model,
                                 mode=mode, top_k=top_k, max_searches=max_searches)
                result = agent.run(question, history=history,
                                   session_id=st.session_state.session_id, tags=["ui"])
            except Exception as e:
                st.error(f"{e.__class__.__name__}: {e}")
                st.session_state.messages.pop()
                st.stop()
        answer = result.answer or "_(empty answer)_"
        st.markdown(answer)
        details = {
            "id": len(st.session_state.messages),
            "model": model,
            "mode": result.mode,
            "tool_calls": result.tool_calls,
            "sources": result.sources,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        render_details(details)
    st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
