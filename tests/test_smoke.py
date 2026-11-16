"""Offline smoke tests: a fake Anthropic client stands in for Claude, so these
run without API keys or a Langfuse server.  `python -m pytest -q`"""

from __future__ import annotations

from types import SimpleNamespace as NS

from rag_agent import metrics
from rag_agent.agent import AgentResult, RAGAgent
from rag_agent.config import KB_DIR
from rag_agent.metrics import Judgment, rag_score_composite, retrieval_evaluator, scores_from_judgment
from rag_agent.retriever import BM25Retriever


def _usage():
    return NS(input_tokens=100, output_tokens=20)


class FakeMessages:
    """First call -> tool_use(search), second call -> final text answer."""

    def __init__(self, query: str, answer: str, judgment: Judgment | None = None):
        self.query, self.answer, self.judgment = query, answer, judgment
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            block = NS(type="tool_use", id="tu_1", name="search_knowledge_base", input={"query": self.query})
            block.model_dump = lambda **_: {"type": "tool_use", "id": "tu_1", "input": {"query": self.query}}
            return NS(content=[block], stop_reason="tool_use", usage=_usage(), model=kwargs["model"])
        text = NS(type="text", text=self.answer)
        text.model_dump = lambda **_: {"type": "text", "text": self.answer}
        return NS(content=[text], stop_reason="end_turn", usage=_usage(), model=kwargs["model"])

    def parse(self, **kwargs):
        return NS(parsed_output=self.judgment, stop_reason="end_turn")


def fake_client(**kw):
    return NS(beta=NS(messages=FakeMessages(**kw)))


def test_retriever_finds_current_travel_policy():
    r = BM25Retriever.from_dir(KB_DIR)
    hits = r.search("hotel cap per night London", top_k=3)
    assert hits[0][0].doc_id == "travel_expense_2025"


def test_agent_loop_and_retrieval_metrics():
    client = fake_client(query="PTO days per year accrue", answer="You get 22 days [pto_policy].")
    agent = RAGAgent("baseline", client=client)
    res = agent.run("How many PTO days do I get?")
    assert res.answer.startswith("You get 22 days")
    assert res.searches == ["PTO days per year accrue"]
    assert len(res.retrieved) == 3
    # tool_result was sent back with the matching id
    # (the agent keeps appending to the same list, so index explicitly: user, assistant, tool_result)
    msgs = client.beta.messages.calls[1]["messages"]
    assert msgs[2]["content"][0]["tool_use_id"] == "tu_1"

    evals = retrieval_evaluator(
        input={"question": "q"},
        output=res.to_output(),
        expected_output={"gold_doc_ids": ["pto_policy", "learning_budget"]},
    )
    by = {e.name: e.value for e in evals}
    assert by["context_recall"] == 0.5  # found 1 of 2 gold docs


def test_search_budget_enforced():
    client = fake_client(query="hotel", answer="ok")
    agent = RAGAgent("low_recall", client=client)
    agent_result = AgentResult("q", "", "low_recall")
    out, err = agent._run_search({"query": "hotel"}, agent_result)
    assert not err and len(agent_result.retrieved) == 1
    out, err = agent._run_search({"query": "per diem"}, agent_result)
    assert err and "budget" in out


def test_judge_scores_are_fractions():
    j = Judgment(
        claims=[
            {"claim": "cap is $260", "supported": True},
            {"claim": "cap is $200", "supported": False},
            {"claim": "per diem $75", "supported": True},
        ],
        facts=[
            {"fact": "a", "verdict": "covered"},
            {"fact": "b", "verdict": "partially_covered"},
            {"fact": "c", "verdict": "missing"},
            {"fact": "d", "verdict": "contradicted"},
        ],
        relevance=4,
        relevance_reason="on point",
    )
    by = {e.name: e.value for e in scores_from_judgment(j, has_key_facts=True)}
    assert by == {"faithfulness": 0.667, "answer_correctness": 0.375, "answer_relevance": 0.75}

    comp = rag_score_composite(input=None, output=None, expected_output=None, metadata=None,
                               evaluations=scores_from_judgment(j, True))
    assert comp.name == "rag_score" and 0 < comp.value < 1


def test_llm_judge_evaluator_with_fake_client():
    j = Judgment(claims=[], facts=[{"fact": "not covered", "verdict": "covered"}], relevance=5, relevance_reason="x")
    metrics.set_judge_client(fake_client(query="", answer="", judgment=j))
    evals = metrics.llm_judge_evaluator(
        input={"question": "401k match?"},
        output={"answer": "The documents don't cover this.", "contexts": []},
        expected_output={"key_facts": ["not covered"]},
    )
    by = {e.name: e.value for e in evals}
    assert by == {"faithfulness": 1.0, "answer_correctness": 1.0, "answer_relevance": 1.0}
