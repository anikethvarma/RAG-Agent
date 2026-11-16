"""The RAG agent: a free OpenRouter model + a `search_knowledge_base` tool.

OpenRouter exposes an OpenAI-compatible API, so the `openai` SDK is used with
OpenRouter's base URL.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from .config import (
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


def make_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Create a free key at openrouter.ai and put it in .env")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        max_retries=3,  # free models are rate-limited; retry 429s with backoff
        default_headers={"X-Title": "RAG Agent"},
    )


@dataclass
class AgentResult:
    question: str
    answer: str
    mode: str = MODE
    retrieved: list[Chunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.retrieved))


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
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=TEMPERATURE)
        if tools:
            kwargs.update(tools=[SEARCH_TOOL], tool_choice="auto")
        response = self.client.chat.completions.create(**kwargs)
        if response.usage:
            result.input_tokens += response.usage.prompt_tokens or 0
            result.output_tokens += response.usage.completion_tokens or 0
        if not response.choices:
            raise RuntimeError(f"Model returned no choices: {response}")
        return response.choices[0].message

    def _search(self, query: str, result: AgentResult) -> str:
        hits = self.retriever.search(query, top_k=self.top_k)
        result.searches.append(query)
        seen = {c.chunk_id for c in result.retrieved}
        result.retrieved.extend(c for c, _ in hits if c.chunk_id not in seen)
        return "\n\n---\n\n".join(c.render() for c, _ in hits)

    def _run_tool(self, arguments: str | None, result: AgentResult) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Error: tool arguments were not valid JSON. Call the tool again with {\"query\": \"...\"}."
        query = args.get("query") if isinstance(args, dict) else None
        if not isinstance(query, str) or not query.strip():
            return "Error: `query` must be a non-empty string."
        if len(result.searches) >= self.max_searches:
            return "Search limit reached. Answer now with the information you already have."
        return self._search(query, result)

    # -- modes -------------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        if self.mode == "simple":
            return self._run_simple(question)
        try:
            return self._run_agent(question)
        except openai.APIStatusError as err:
            if not _tools_unsupported(err):
                raise
            print(f"[{self.model} does not support tool calling here; using simple mode]")
            self.mode = "simple"
            return self._run_simple(question)

    def _run_simple(self, question: str) -> AgentResult:
        """Classic RAG: retrieve with the question, then answer in one call."""
        result = AgentResult(question=question, answer="", mode="simple")
        context = self._search(question, result)
        messages = [
            {"role": "system", "content": SIMPLE_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        result.answer = (self._complete(messages, result, tools=False).content or "").strip()
        return result

    def _run_agent(self, question: str) -> AgentResult:
        """Agentic RAG: the model decides when and what to search."""
        result = AgentResult(question=question, answer="", mode="agent")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
