"""Helix Cloud helpdesk RAG agent (free OpenRouter model).

    python main.py "How many PTO days do I get?"   # ask one question
    python main.py                                 # interactive chat
    python main.py --models                        # list free OpenRouter models
    python main.py --check-tracing                 # test the Langfuse connection
"""

from __future__ import annotations

import sys
import uuid

from rag_agent.agent import RAGAgent, flush_traces, list_free_models
from rag_agent.tracing import check_tracing, trace_url, tracing_status


def ask(agent: RAGAgent, question: str, history: list[dict[str, str]] | None = None,
        session_id: str | None = None) -> str:
    result = agent.run(question, history=history, session_id=session_id, tags=["cli"])
    print(f"\n{result.answer}\n")
    print(f"  model: {agent.model} ({result.mode} mode)")
    print(f"  sources: {', '.join(result.sources) or 'none'}")
    print(f"  searches: {result.searches}")
    print(f"  tokens: in={result.input_tokens} out={result.output_tokens}")
    if result.trace_id:
        print(f"  trace: {trace_url(result.trace_id) or result.trace_id}")
    print()
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
    if args == ["--check-tracing"]:
        status = tracing_status()
        print(status.summary())
        print(f"  .env file:   {status.env_file}\n  secret key:  {status.secret_key}\n"
              f"  SDK version: langfuse {status.sdk_version}\n")
        ok, message = check_tracing()
        print(("OK: " if ok else "PROBLEM: ") + message)
        sys.exit(0 if ok else 1)

    print(tracing_status().summary())
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
