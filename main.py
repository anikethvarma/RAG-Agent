"""Helix Cloud helpdesk RAG agent.

    python main.py "How many PTO days do I get?"   # ask one question
    python main.py                                 # interactive chat
"""

from __future__ import annotations

import sys

from rag_agent.agent import RAGAgent


def ask(agent: RAGAgent, question: str) -> None:
    result = agent.run(question)
    print(f"\n{result.answer}\n")
    print(f"  sources: {', '.join(result.sources) or 'none'}")
    print(f"  searches: {result.searches}")
    print(f"  tokens: in={result.input_tokens} out={result.output_tokens}\n")


def main() -> None:
    agent = RAGAgent()
    if len(sys.argv) > 1:
        ask(agent, " ".join(sys.argv[1:]))
        return

    print("Helix Cloud helpdesk — ask a question (empty line or Ctrl-D to quit).")
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
