"""Helix Cloud helpdesk RAG agent (free OpenRouter model).

    python main.py "How many PTO days do I get?"   # ask one question
    python main.py                                 # interactive chat
    python main.py --models                        # list free OpenRouter models
"""

from __future__ import annotations

import sys
import uuid

from rag_agent.agent import RAGAgent, flush_traces, list_free_models


def ask(agent: RAGAgent, question: str, history: list[dict[str, str]] | None = None,
        session_id: str | None = None) -> str:
    result = agent.run(question, history=history, session_id=session_id, tags=["cli"])
    print(f"\n{result.answer}\n")
    print(f"  model: {agent.model} ({result.mode} mode)")
    print(f"  sources: {', '.join(result.sources) or 'none'}")
    print(f"  searches: {result.searches}")
    print(f"  tokens: in={result.input_tokens} out={result.output_tokens}\n")
    return result.answer


def show_models() -> None:
    models = list_free_models()
    print(f"{'model id':<60}{'context':>10}  tools")
    for m in models:
        print(f"{m['id']:<60}{m.get('context_length') or '-':>10}  {'yes' if m['tools'] else 'no'}")
    print(f"\n{len(models)} free models. Set MODEL=<id> in .env (prefer one with tools=yes).")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--models"]:
        show_models()
        return

    agent = RAGAgent()
    try:
        if args:
            ask(agent, " ".join(args))
        else:
            chat(agent)
    finally:
        flush_traces()  # short-lived process: send buffered Langfuse traces before exit


def chat(agent: RAGAgent) -> None:
    """Interactive chat with memory; all turns share one Langfuse session."""
    session_id = f"cli-{uuid.uuid4()}"
    history: list[dict[str, str]] = []

    print(f"Helix Cloud helpdesk ({agent.model}) — ask a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = ask(agent, question, history=history, session_id=session_id)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
