"""Helix Cloud helpdesk RAG agent (free OpenRouter model).

    python main.py "How many PTO days do I get?"   # ask one question
    python main.py                                 # interactive chat
    python main.py --models                        # list free OpenRouter models
"""

from __future__ import annotations

import sys

from rag_agent.agent import RAGAgent, list_free_models


def ask(agent: RAGAgent, question: str) -> None:
    result = agent.run(question)
    print(f"\n{result.answer}\n")
    print(f"  model: {agent.model} ({result.mode} mode)")
    print(f"  sources: {', '.join(result.sources) or 'none'}")
    print(f"  searches: {result.searches}")
    print(f"  tokens: in={result.input_tokens} out={result.output_tokens}\n")


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
    if args:
        ask(agent, " ".join(args))
        return

    print(f"Helix Cloud helpdesk ({agent.model}) — ask a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(agent, question)


if __name__ == "__main__":
    main()
