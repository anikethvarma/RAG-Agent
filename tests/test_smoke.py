"""Offline tests: a fake Anthropic client stands in for Claude, so these run
without an API key.  `python -m pytest -q`"""

from __future__ import annotations

from types import SimpleNamespace as NS

from rag_agent.agent import AgentResult, RAGAgent
from rag_agent.config import KB_DIR
from rag_agent.retriever import BM25Retriever


class FakeMessages:
    """First call -> tool_use(search), second call -> final text answer."""

    def __init__(self, query: str, answer: str):
        self.query, self.answer = query, answer
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = NS(input_tokens=100, output_tokens=20)
        if len(self.calls) == 1:
            block = NS(type="tool_use", id="tu_1", name="search_knowledge_base", input={"query": self.query})
            return NS(content=[block], stop_reason="tool_use", usage=usage)
        return NS(content=[NS(type="text", text=self.answer)], stop_reason="end_turn", usage=usage)


def fake_client(query: str, answer: str):
    return NS(beta=NS(messages=FakeMessages(query, answer)))


def test_retriever_finds_current_travel_policy():
    r = BM25Retriever.from_dir(KB_DIR)
    hits = r.search("hotel cap per night London", top_k=3)
    assert hits[0][0].doc_id == "travel_expense_2025"


def test_agent_loop():
    client = fake_client("PTO days per year accrue", "You get 22 days [pto_policy].")
    res = RAGAgent(client=client).run("How many PTO days do I get?")
    assert res.answer == "You get 22 days [pto_policy]."
    assert res.searches == ["PTO days per year accrue"]
    assert res.sources[0] == "pto_policy"
    # messages list is shared and keeps growing: user, assistant, tool_result, assistant
    msgs = client.beta.messages.calls[1]["messages"]
    assert msgs[2]["content"][0]["tool_use_id"] == "tu_1"


def test_search_limit_enforced():
    agent = RAGAgent(client=fake_client("x", "y"), max_searches=1)
    result = AgentResult("q", "")
    _, err = agent._run_search({"query": "hotel"}, result)
    assert not err
    out, err = agent._run_search({"query": "per diem"}, result)
    assert err and "limit" in out
