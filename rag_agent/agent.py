"""The RAG agent: a free OpenRouter model + a `search_knowledge_base` tool.

OpenRouter exposes an OpenAI-compatible API, so the `openai` SDK is used with
OpenRouter's base URL.

Langfuse tracing (one trace per question / chat turn):

    answer-question            agent      input: user question, output: answer
    ├── generate-response      generation model, messages + tools in, reply + reasoning out, tokens, cost
    ├── search-knowledge-base  retriever  input: query, output: retrieved chunks
    ├── generate-response      generation
    └── ...

Chat turns share a `session_id`, so the Sessions view shows whole conversations.
Tracing is a no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import openai

from .config import (  # noqa: I001  (must come before the Langfuse imports: loads .env)
    KB_DIR,
    MAX_SEARCHES,
    MODE,
    MODEL,
    OPENROUTER_BASE_URL,
    SIMPLE_PROMPT,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
)
from .retriever import BM25Retriever, Chunk

from langfuse import get_client, propagate_attributes  # noqa: E402
from openai import OpenAI  # noqa: E402

# Generations are recorded explicitly (see RAGAgent._complete) rather than with the
# `langfuse.openai` drop-in, because the drop-in drops the `reasoning` field that
# OpenRouter returns for thinking models, and Langfuse recommends capturing it.

GENERATION_NAME = "generate-response"

MAX_STEPS = 6

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Keyword search over Helix Cloud's internal policy documents (HR, travel, IT, "
            "security, benefits). Returns the most relevant document sections with their doc id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, e.g. 'hotel cap per night London'.",
                }
            },
            "required": ["query"],
        },
    },
}


def make_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Create a free key at openrouter.ai and put it in .env")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        max_retries=3,  # free models are rate-limited; retry 429s with backoff
        default_headers={"X-Title": "RAG Agent"},
    )


def flush_traces() -> None:
    """Send buffered traces. Call before a short-lived process (the CLI) exits."""
    get_client().flush()


@dataclass
class AgentResult:
    question: str
    answer: str
    mode: str = MODE
    retrieved: list[Chunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    # Every tool call the model made: {"tool", "arguments", "result"} (shown in the UI)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    trace_id: str | None = None  # Langfuse trace of this answer (None when tracing is off)

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.retrieved))


def _assistant_message(msg: Any) -> dict[str, Any]:
    """The model's reply in OpenAI message format, including reasoning when the model returns it."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning") or extra.get("reasoning_content")
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _tools_unsupported(err: openai.APIStatusError) -> bool:
    msg = str(err).lower()
    return err.status_code in (400, 404) and ("tool" in msg or "function" in msg)


class RAGAgent:
    def __init__(
        self,
        client: OpenAI | None = None,
        retriever: BM25Retriever | None = None,
        model: str = MODEL,
        mode: str = MODE,
        top_k: int = TOP_K,
        max_searches: int = MAX_SEARCHES,
    ):
        self.client = client or make_client()
        self.retriever = retriever or BM25Retriever.from_dir(KB_DIR)
        self.model = model
        self.mode = mode
        self.top_k = top_k
        self.max_searches = max_searches

    # -- helpers -----------------------------------------------------------
    def _complete(self, messages: list[dict[str, Any]], result: AgentResult, tools: bool):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            extra_body={"usage": {"include": True}},  # ask OpenRouter to return usage + cost
        )
        if tools:
            kwargs.update(tools=[SEARCH_TOOL], tool_choice="auto")

        with get_client().start_as_current_observation(
            as_type="generation",
            name=GENERATION_NAME,
            model=self.model,
            # Same shape the Langfuse OpenAI integration uses, so the UI renders a chat.
            input={"messages": messages, "tools": [SEARCH_TOOL]} if tools else messages,
            model_parameters={"temperature": TEMPERATURE, **({"tool_choice": "auto"} if tools else {})},
        ) as generation:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as err:
                generation.update(level="ERROR", status_message=f"{err.__class__.__name__}: {err}")
                raise
            if not response.choices:
                generation.update(level="ERROR", status_message="model returned no choices")
                raise RuntimeError(f"Model returned no choices: {response}")

            msg = response.choices[0].message
            usage = response.usage
            prompt_tokens = (usage.prompt_tokens or 0) if usage else 0
            completion_tokens = (usage.completion_tokens or 0) if usage else 0
            result.input_tokens += prompt_tokens
            result.output_tokens += completion_tokens

            cost = getattr(usage, "cost", None) if usage else None  # OpenRouter usage accounting (USD)
            generation.update(
                output=_assistant_message(msg),
                usage_details={"input": prompt_tokens, "output": completion_tokens} if usage else None,
                cost_details={"total": float(cost)} if isinstance(cost, (int, float)) else None,
                metadata={"finish_reason": response.choices[0].finish_reason,
                          "provider": getattr(response, "provider", None)},
            )
            return msg

    def _search(self, query: str, result: AgentResult) -> str:
        with get_client().start_as_current_observation(
            as_type="retriever",
            name="search-knowledge-base",
            input={"query": query},
            metadata={"top_k": self.top_k, "retriever": "bm25", "kb_dir": KB_DIR.name},
        ) as span:
            hits = self.retriever.search(query, top_k=self.top_k)
            result.searches.append(query)
            seen = {c.chunk_id for c in result.retrieved}
            result.retrieved.extend(c for c, _ in hits if c.chunk_id not in seen)
            span.update(
                output=[
                    {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": round(score, 3),
                     "title": c.title, "text": c.text}
                    for c, score in hits
                ],
                metadata={"num_results": len(hits), "doc_ids": sorted({c.doc_id for c, _ in hits})},
            )
            return "\n\n---\n\n".join(c.render() for c, _ in hits)

    def _run_tool(self, arguments: str | None, result: AgentResult) -> str:
        output = self._run_tool_inner(arguments, result)
        result.tool_calls.append({"tool": "search_knowledge_base", "arguments": arguments, "result": output})
        return output

    def _run_tool_inner(self, arguments: str | None, result: AgentResult) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return self._rejected_tool_call(
                arguments, "Error: tool arguments were not valid JSON. Call the tool again with {\"query\": \"...\"}.")
        query = args.get("query") if isinstance(args, dict) else None
        if not isinstance(query, str) or not query.strip():
            return self._rejected_tool_call(arguments, "Error: `query` must be a non-empty string.")
        if len(result.searches) >= self.max_searches:
            return self._rejected_tool_call(
                arguments, "Search limit reached. Answer now with the information you already have.")
        return self._search(query, result)

    @staticmethod
    def _rejected_tool_call(arguments: str | None, message: str) -> str:
        """Record a tool call that never reached the retriever, so it is visible in the trace."""
        with get_client().start_as_current_observation(
            as_type="tool", name="search-knowledge-base-rejected", input={"arguments": arguments},
        ) as span:
            span.update(output=message, level="WARNING", status_message=message)
        return message

    # -- modes -------------------------------------------------------------
    def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AgentResult:
        """Answer `question`; one Langfuse trace per call.

        history:    earlier turns as [{"role": "user"|"assistant", "content": str}]
        session_id: groups the turns of one conversation in Langfuse's Sessions view
        user_id:    who asked (per-user filtering and cost)
        tags:       known-upfront dimensions, e.g. ["ui"] or ["cli"]
        """
        history = [{"role": h["role"], "content": h["content"]} for h in (history or [])]
        requested_mode = self.mode
        with get_client().start_as_current_observation(
            as_type="agent",
            name="answer-question",
            input=question,  # only the user's message, not the raw function arguments
            metadata={"model": self.model, "mode": self.mode, "top_k": self.top_k,
                      "max_searches": self.max_searches, "history_turns": len(history)},
        ) as root:
            with propagate_attributes(
                trace_name="answer-question",
                session_id=session_id,
                user_id=user_id,
                tags=tags,
            ):
                result = self._dispatch(question, history)
            result.trace_id = getattr(root, "trace_id", None)
            root.update(
                output=result.answer,
                metadata={"mode": result.mode, "fell_back_to_simple": result.mode != requested_mode,
                          "searches": result.searches, "sources": result.sources,
                          "input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
            )
        return result

    def _dispatch(self, question: str, history: list[dict[str, str]]) -> AgentResult:
        if self.mode == "simple":
            return self._run_simple(question, history)
        try:
            return self._run_agent(question, history)
        except openai.APIStatusError as err:
            if not _tools_unsupported(err):
                raise
            print(f"[{self.model} does not support tool calling here; using simple mode]")
            self.mode = "simple"
            return self._run_simple(question, history)

    def _run_simple(self, question: str, history: list[dict[str, str]]) -> AgentResult:
        """Classic RAG: retrieve with the question, then answer in one call."""
        result = AgentResult(question=question, answer="", mode="simple")
        context = self._search(question, result)
        result.tool_calls.append({"tool": "search_knowledge_base (automatic, simple mode)",
                                  "arguments": json.dumps({"query": question}), "result": context})
        messages = [
            {"role": "system", "content": SIMPLE_PROMPT.format(context=context)},
            *history,
            {"role": "user", "content": question},
        ]
        result.answer = (self._complete(messages, result, tools=False).content or "").strip()
        return result

    def _run_agent(self, question: str, history: list[dict[str, str]]) -> AgentResult:
        """Agentic RAG: the model decides when and what to search."""
        result = AgentResult(question=question, answer="", mode="agent")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": question},
        ]
        for _ in range(MAX_STEPS):
            msg = self._complete(messages, result, tools=True)
            if not msg.tool_calls:
                result.answer = (msg.content or "").strip()
                return result

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                if tc.function.name == "search_knowledge_base":
                    content = self._run_tool(tc.function.arguments, result)
                else:
                    content = f"Error: unknown tool {tc.function.name!r}."
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        result.answer = "(no answer: step limit reached)"
        return result


def list_free_models() -> list[dict[str, Any]]:
    """Free OpenRouter models, tool-capable first. The models endpoint needs no key."""
    import urllib.request

    with urllib.request.urlopen(f"{OPENROUTER_BASE_URL}/models", timeout=30) as resp:
        data = json.load(resp)["data"]
    free = [
        m for m in data
        if m["id"].endswith(":free")
        or (str(m.get("pricing", {}).get("prompt")) == "0" and str(m.get("pricing", {}).get("completion")) == "0")
    ]
    for m in free:
        m["tools"] = "tools" in (m.get("supported_parameters") or [])
    return sorted(free, key=lambda m: (not m["tools"], m["id"]))
