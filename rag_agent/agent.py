"""The RAG agent: Claude + a `search_knowledge_base` tool in a manual tool-use loop.

A manual loop (instead of the SDK tool runner) is used on purpose so every
Claude call and every retrieval becomes its own Langfuse observation:

    rag-agent (agent)
    ├── claude-call #1 (generation)   model, tokens, stop_reason
    ├── search_knowledge_base (retriever)  query -> chunk ids + BM25 scores
    ├── claude-call #2 (generation)
    └── ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic

from .config import (
    AGENT_EFFORT,
    AGENT_MODEL,
    ENABLE_FALLBACKS,
    FALLBACK_BETA,
    KB_DIR,
    VARIANTS,
    AgentVariant,
)
from .retriever import BM25Retriever, Chunk
from .tracing import observation, to_jsonable

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
    variant: str
    retrieved: list[Chunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.retrieved))

    def to_output(self) -> dict[str, Any]:
        """The task output handed to Langfuse evaluators (and shown in the UI)."""
        return {
            "answer": self.answer,
            "retrieved_chunk_ids": [c.chunk_id for c in self.retrieved],
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "contexts": [c.render() for c in self.retrieved],
            "searches": self.searches,
        }


class RAGAgent:
    def __init__(
        self,
        variant: str | AgentVariant = "baseline",
        client: anthropic.Anthropic | None = None,
        retriever: BM25Retriever | None = None,
        model: str = AGENT_MODEL,
    ):
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        self.client = client or anthropic.Anthropic()
        self.retriever = retriever or BM25Retriever.from_dir(KB_DIR)
        self.model = model

    # -- Claude call -------------------------------------------------------
    def _call_claude(self, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=self.variant.system_prompt,
            tools=[SEARCH_TOOL],
            messages=messages,
            output_config={"effort": AGENT_EFFORT},
        )
        if ENABLE_FALLBACKS:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        return self.client.beta.messages.create(**kwargs)

    # -- Tool --------------------------------------------------------------
    def _run_search(self, tool_input: Any, result: AgentResult) -> tuple[str, bool]:
        query = tool_input.get("query") if isinstance(tool_input, dict) else None
        with observation(
            "search_knowledge_base",
            as_type="retriever",
            input={"query": query, "top_k": self.variant.top_k},
        ) as span:
            if not isinstance(query, str) or not query.strip():
                span.update(level="ERROR", status_message="missing query")
                return "Error: `query` must be a non-empty string.", True
            if len(result.searches) >= self.variant.max_searches:
                msg = "Search budget exhausted. Answer now with the information you already have."
                span.update(output=msg, level="WARNING")
                return msg, True

            hits = self.retriever.search(query, top_k=self.variant.top_k)
            result.searches.append(query)
            seen = {c.chunk_id for c in result.retrieved}
            result.retrieved.extend(c for c, _ in hits if c.chunk_id not in seen)
            span.update(
                output=[{"chunk_id": c.chunk_id, "score": round(s, 3), "text": c.text} for c, s in hits]
            )
            return "\n\n---\n\n".join(c.render() for c, _ in hits), False

    # -- Loop --------------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        result = AgentResult(question=question, answer="", variant=self.variant.name)
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        with observation(
            "rag-agent",
            as_type="agent",
            input={"question": question},
            metadata={"variant": self.variant.name, "top_k": self.variant.top_k,
                      "max_searches": self.variant.max_searches, "model": self.model},
        ) as agent_span:
            for step in range(1, MAX_STEPS + 1):
                with observation(
                    f"claude-call-{step}",
                    as_type="generation",
                    model=self.model,
                    input=to_jsonable(messages),
                    model_parameters={"effort": AGENT_EFFORT, "max_tokens": 16000},
                ) as gen:
                    response = self._call_claude(messages)
                    gen.update(
                        output=to_jsonable(response.content),
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                        metadata={"stop_reason": response.stop_reason, "served_by": response.model},
                    )
                result.input_tokens += response.usage.input_tokens
                result.output_tokens += response.usage.output_tokens
                result.stop_reason = response.stop_reason

                if response.stop_reason == "refusal":
                    result.answer = "Sorry, I can't help with that request."
                    break

                # Always append the full content (thinking blocks included) unchanged.
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "pause_turn":
                    continue

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
                    break

                tool_results = []
                for tu in tool_uses:
                    content, is_error = self._run_search(tu.input, result)
                    block = {"type": "tool_result", "tool_use_id": tu.id, "content": content}
                    if is_error:
                        block["is_error"] = True
                    tool_results.append(block)
                messages.append({"role": "user", "content": tool_results})
            else:
                result.answer = result.answer or "(no answer: step limit reached)"

            agent_span.update(output=result.to_output())
        return result
