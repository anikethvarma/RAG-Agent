"""Offline tests: a fake OpenAI-compatible client stands in for OpenRouter, so
these run without an API key or network.  `python -m pytest -q`"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import openai

from rag_agent.agent import AgentResult, RAGAgent
from rag_agent.config import KB_DIR
from rag_agent.retriever import BM25Retriever


def _resp(content=None, tool_calls=None):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=100, completion_tokens=20))


class FakeCompletions:
    """Agent mode: first call -> tool call, second -> answer. Can also reject tools."""

    def __init__(self, query: str, answer: str, reject_tools: bool = False):
        self.query, self.answer, self.reject_tools = query, answer, reject_tools
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs and self.reject_tools:
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise openai.NotFoundError(
                "No endpoints found that support tool use.",
                response=httpx.Response(404, request=req), body=None,
            )
        if "tools" in kwargs and len(self.calls) == 1:
            tc = NS(id="call_1", type="function",
                    function=NS(name="search_knowledge_base", arguments=json.dumps({"query": self.query})))
            return _resp(tool_calls=[tc])
        return _resp(content=self.answer)


def fake_client(query: str = "", answer: str = "", reject_tools: bool = False):
    return NS(chat=NS(completions=FakeCompletions(query, answer, reject_tools)))


def test_retriever_finds_current_travel_policy():
    r = BM25Retriever.from_dir(KB_DIR)
    hits = r.search("hotel cap per night London", top_k=3)
    assert hits[0][0].doc_id == "travel_expense_2025"


def test_agent_mode_tool_loop():
    client = fake_client("PTO days per year accrue", "You get 22 days [pto_policy].")
    res = RAGAgent(client=client, mode="agent").run("How many PTO days do I get?")
    assert res.answer == "You get 22 days [pto_policy]."
    assert res.mode == "agent"
    assert res.searches == ["PTO days per year accrue"]
    assert res.sources[0] == "pto_policy"
    msgs = client.chat.completions.calls[1]["messages"]
    assert msgs[2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": msgs[3]["content"]}


def test_simple_mode():
    client = fake_client(answer="$320/night [travel_expense_2025].")
    res = RAGAgent(client=client, mode="simple").run("What is the hotel cap in London?")
    assert res.mode == "simple" and res.answer.startswith("$320")
    call = client.chat.completions.calls[0]
    assert "tools" not in call and "travel_expense_2025" in call["messages"][0]["content"]


def test_falls_back_to_simple_when_tools_unsupported():
    client = fake_client(answer="22 days [pto_policy].", reject_tools=True)
    agent = RAGAgent(client=client, mode="agent")
    res = agent.run("How many PTO days do I get?")
    assert res.mode == "simple" and res.answer.startswith("22 days")
    assert agent.mode == "simple"  # remembered for the next question


def test_bad_tool_arguments_and_search_limit():
    agent = RAGAgent(client=fake_client(), max_searches=1)
    result = AgentResult("q", "")
    assert "not valid JSON" in agent._run_tool("{oops", result)
    assert "travel" in agent._run_tool('{"query": "hotel"}', result)
    assert "limit" in agent._run_tool('{"query": "per diem"}', result)


def test_history_and_tool_calls_recorded():
    client = fake_client("hotel London", "$320/night [travel_expense_2025].")
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    res = RAGAgent(client=client, mode="agent").run("Hotel cap in London?", history=history)
    # the list keeps growing after the call, so check the first four entries
    roles = [m["role"] for m in client.chat.completions.calls[0]["messages"]][:4]
    assert roles == ["system", "user", "assistant", "user"]
    assert res.tool_calls[0]["tool"] == "search_knowledge_base"
    assert "travel_expense_2025" in res.tool_calls[0]["result"]


def test_assistant_message_keeps_reasoning_and_tool_calls():
    from rag_agent.agent import _assistant_message

    tc = NS(id="call_1", type="function", function=NS(name="search_knowledge_base", arguments='{"query": "x"}'))
    msg = NS(content=None, tool_calls=[tc], model_extra={"reasoning": "Need to search first."})
    out = _assistant_message(msg)
    assert out["reasoning"] == "Need to search first."
    assert out["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'


def test_env_file_loads_utf16_and_bom(tmp_path, monkeypatch):
    from rag_agent.config import _load_env_file

    for name, data in {
        "utf16": "TEST_ENV_KEY_A=one\r\n".encode("utf-16"),
        "bom": "﻿TEST_ENV_KEY_B=two\n".encode("utf-8"),
    }.items():
        path = tmp_path / f"{name}.env"
        path.write_bytes(data)
        _load_env_file(path)
    import os

    assert os.environ.pop("TEST_ENV_KEY_A") == "one"
    assert os.environ.pop("TEST_ENV_KEY_B") == "two"
