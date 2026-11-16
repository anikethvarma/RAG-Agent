"""The RAG agent: Claude + a `search_knowledge_base` tool in a tool-use loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from .config import (
    EFFORT,
    ENABLE_FALLBACKS,
    FALLBACK_BETA,
    KB_DIR,
    MAX_SEARCHES,
    MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)
from .retriever import BM25Retriever, Chunk

MAX_STEPS = 6

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Keyword search over Helix Cloud's internal policy documents (HR, travel, IT, "
        "security, benefits). Returns the most relevant document sections with their doc id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords, e.g. 'hotel cap per night London'.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class AgentResult:
    question: str
    answer: str
    retrieved: list[Chunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.retrieved))


class RAGAgent:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        retriever: BM25Retriever | None = None,
        model: str = MODEL,
        top_k: int = TOP_K,
        max_searches: int = MAX_SEARCHES,
    ):
        self.client = client or anthropic.Anthropic()
        self.retriever = retriever or BM25Retriever.from_dir(KB_DIR)
        self.model = model
        self.top_k = top_k
        self.max_searches = max_searches

    def _call_claude(self, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
            output_config={"effort": EFFORT},
        )
        if ENABLE_FALLBACKS:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        return self.client.beta.messages.create(**kwargs)

    def _run_search(self, tool_input: Any, result: AgentResult) -> tuple[str, bool]:
        query = tool_input.get("query") if isinstance(tool_input, dict) else None
        if not isinstance(query, str) or not query.strip():
            return "Error: `query` must be a non-empty string.", True
        if len(result.searches) >= self.max_searches:
            return "Search limit reached. Answer now with the information you already have.", True

        hits = self.retriever.search(query, top_k=self.top_k)
        result.searches.append(query)
        seen = {c.chunk_id for c in result.retrieved}
        result.retrieved.extend(c for c, _ in hits if c.chunk_id not in seen)
        return "\n\n---\n\n".join(c.render() for c, _ in hits), False

    def run(self, question: str) -> AgentResult:
        result = AgentResult(question=question, answer="")
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        for _ in range(MAX_STEPS):
            response = self._call_claude(messages)
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens

            if response.stop_reason == "refusal":
                result.answer = "Sorry, I can't help with that request."
                return result

            # Append the full content (thinking blocks included) unchanged.
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
                return result

            tool_results = []
            for tu in tool_uses:
                content, is_error = self._run_search(tu.input, result)
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tu.id, "content": content}
                if is_error:
                    block["is_error"] = True
                tool_results.append(block)
            messages.append({"role": "user", "content": tool_results})

        result.answer = "(no answer: step limit reached)"
        return result
