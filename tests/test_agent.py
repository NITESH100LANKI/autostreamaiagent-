"""
tests/test_agent.py
-------------------
Pytest test suite for the AutoStream AI Agent.

Test coverage
-------------
1.  Intent classifier — rule‑based (greeting, pricing, lead).
2.  Intent classifier — edge cases (empty, mixed-intent).
3.  Email validator utility.
4.  Lead collector state machine (name → email → platform).
5.  Invalid email handling in lead collector.
6.  Tool executor — fires only when all fields valid.
7.  Tool executor — skips duplicate capture.
8.  RAG retriever — returns a string (graceful without API key).
9.  Response generator — lead prompts (deterministic, no LLM needed).
10. Full graph compilation check.
11. Full conversation flow via compiled graph + MemorySaver (mocked tool).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import agent.nodes as nodes_module
from agent.graph import create_agent_graph
from agent.nodes import intent_classifier, lead_collector, response_generator, tool_executor
from agent.tools import validate_email


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _state(**kwargs) -> dict:
    """Build a minimal AgentState for testing."""
    defaults = {
        "messages": [],
        "intent": "greeting",
        "intent_confidence": 0.9,
        "lead_active": False,
        "lead_name": "",
        "lead_email": "",
        "lead_platform": "",
        "lead_captured": False,
        "retrieved_info": "",
        "leads_count": 0,
    }
    defaults.update(kwargs)
    return defaults


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _ai(text: str) -> AIMessage:
    return AIMessage(content=text)


# ════════════════════════════════════════════════════════════════════════════
# 1. Intent classifier — happy path
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text,expected_intent", [
    ("Hello there", "greeting"),
    ("Hi!", "greeting"),
    ("Good morning", "greeting"),
    ("What is the price of your plan?", "pricing"),
    ("How much does the Pro plan cost?", "pricing"),
    ("Tell me about your features", "pricing"),
    ("I want to try pro", "lead"),
    ("I'd like to subscribe", "lead"),
    ("How do I sign up?", "lead"),
    ("I want to get started", "lead"),
])
def test_intent_classifier_rule_based(text, expected_intent):
    state = _state(messages=[_human(text)])
    result = intent_classifier(state)
    assert result["intent"] == expected_intent
    assert 0 < result["intent_confidence"] <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# 2. Intent classifier — edge cases
# ════════════════════════════════════════════════════════════════════════════
def test_intent_classifier_empty_messages():
    result = intent_classifier(_state(messages=[]))
    assert result["intent"] == "greeting"


def test_intent_classifier_confidence_in_bounds():
    state = _state(messages=[_human("What is the basic plan price?")])
    result = intent_classifier(state)
    assert 0.0 <= result["intent_confidence"] <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# 3. Email validator
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("email,valid", [
    ("alice@example.com", True),
    ("bob.smith+tag@domain.co.uk", True),
    ("user@company.io", True),
    ("notanemail", False),
    ("missing@", False),
    ("@nodomain.com", False),
    ("spaces in@email.com", False),
    ("", False),
])
def test_validate_email(email, valid):
    assert validate_email(email) is valid


# ════════════════════════════════════════════════════════════════════════════
# 4. Lead collector — sequential field capture
# ════════════════════════════════════════════════════════════════════════════
def test_lead_collector_captures_name():
    """Bot's last AI message asked for name → user's answer should be stored."""
    state = _state(
        intent="lead",
        messages=[
            _human("I want to try Pro"),
            _ai("What is your name?"),
            _human("Alice"),
        ],
    )
    result = lead_collector(state)
    assert result["lead_name"] == "Alice"
    assert result["lead_email"] == ""
    assert result["lead_platform"] == ""


def test_lead_collector_captures_email():
    state = _state(
        intent="lead",
        lead_name="Alice",
        messages=[
            _human("I want to try Pro"),
            _ai("What is your name?"),
            _human("Alice"),
            _ai("Thanks, Alice! What is your email address?"),
            _human("alice@example.com"),
        ],
    )
    result = lead_collector(state)
    assert result["lead_name"] == "Alice"
    assert result["lead_email"] == "alice@example.com"


def test_lead_collector_captures_platform():
    state = _state(
        intent="lead",
        lead_name="Alice",
        lead_email="alice@example.com",
        messages=[
            _ai("Which platform do you primarily create content for?"),
            _human("YouTube"),
        ],
    )
    result = lead_collector(state)
    assert result["lead_platform"] == "YouTube"


# ════════════════════════════════════════════════════════════════════════════
# 5. Invalid email handling
# ════════════════════════════════════════════════════════════════════════════
def test_lead_collector_rejects_invalid_email():
    state = _state(
        intent="lead",
        lead_name="Alice",
        messages=[
            _ai("What is your email address?"),
            _human("not-an-email"),
        ],
    )
    result = lead_collector(state)
    # Should store the sentinel, NOT the bad email
    assert result["lead_email"] == "__invalid__"


def test_response_generator_asks_again_on_invalid_email():
    state = _state(
        intent="lead",
        lead_name="Alice",
        lead_email="__invalid__",
    )
    result = response_generator(state)
    reply = result["messages"][-1].content.lower()
    assert "valid" in reply or "email" in reply


# ════════════════════════════════════════════════════════════════════════════
# 6. Tool executor — fires only when all fields are valid
# ════════════════════════════════════════════════════════════════════════════
def test_tool_executor_fires_when_complete(monkeypatch):
    fired = []
    monkeypatch.setattr(nodes_module, "mock_lead_capture",
                        lambda n, e, p: fired.append((n, e, p)) or True)

    state = _state(lead_name="Alice", lead_email="alice@example.com", lead_platform="YouTube")
    result = tool_executor(state)

    assert len(fired) == 1
    assert fired[0] == ("Alice", "alice@example.com", "YouTube")
    assert result["lead_captured"] is True
    assert result["leads_count"] == 1


def test_tool_executor_skips_incomplete_data(monkeypatch):
    fired = []
    monkeypatch.setattr(nodes_module, "mock_lead_capture",
                        lambda n, e, p: fired.append(True) or True)

    state = _state(lead_name="Alice", lead_email="", lead_platform="YouTube")
    tool_executor(state)
    assert len(fired) == 0


# ════════════════════════════════════════════════════════════════════════════
# 7. Tool executor — no duplicate capture
# ════════════════════════════════════════════════════════════════════════════
def test_tool_executor_no_duplicate(monkeypatch):
    fired = []
    monkeypatch.setattr(nodes_module, "mock_lead_capture",
                        lambda n, e, p: fired.append(True) or True)

    state = _state(
        lead_name="Alice", lead_email="alice@example.com",
        lead_platform="YouTube", lead_captured=True,
    )
    tool_executor(state)
    assert len(fired) == 0          # already captured → skip


# ════════════════════════════════════════════════════════════════════════════
# 8. RAG retriever — graceful without API key
# ════════════════════════════════════════════════════════════════════════════
def test_rag_retriever_returns_string(monkeypatch):
    # Patch _get_vectorstore to return None (simulates no API key)
    monkeypatch.setattr(nodes_module, "_get_vectorstore", lambda: None)
    state = _state(
        intent="pricing",
        messages=[_human("What is the Pro plan price?")],
    )
    from agent.nodes import rag_retriever
    result = rag_retriever(state)
    assert isinstance(result["retrieved_info"], str)
    assert len(result["retrieved_info"]) > 0


# ════════════════════════════════════════════════════════════════════════════
# 9. Response generator — lead prompts (deterministic, no LLM)
# ════════════════════════════════════════════════════════════════════════════
def test_response_generator_asks_name_first():
    state = _state(intent="lead", lead_name="", lead_email="", lead_platform="")
    result = response_generator(state)
    text = result["messages"][-1].content.lower()
    assert "name" in text


def test_response_generator_asks_email_after_name():
    state = _state(intent="lead", lead_name="Bob", lead_email="", lead_platform="")
    result = response_generator(state)
    text = result["messages"][-1].content.lower()
    assert "email" in text


def test_response_generator_asks_platform_after_email():
    state = _state(intent="lead", lead_name="Bob",
                   lead_email="bob@example.com", lead_platform="")
    result = response_generator(state)
    text = result["messages"][-1].content.lower()
    assert "platform" in text or "content" in text or "youtube" in text


def test_response_generator_confirms_after_capture():
    state = _state(
        intent="lead",
        lead_name="Bob", lead_email="bob@example.com",
        lead_platform="TikTok", lead_captured=True,
    )
    result = response_generator(state)
    text = result["messages"][-1].content.lower()
    assert "set" in text or "team" in text or "welcome" in text


# ════════════════════════════════════════════════════════════════════════════
# 10. Graph compilation sanity check
# ════════════════════════════════════════════════════════════════════════════
def test_graph_compiles_successfully():
    graph = create_agent_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    node_keys = list(compiled.nodes.keys())
    for expected in ["IntentClassifier", "RAGRetriever", "LeadCollector",
                     "ToolExecutor", "ResponseGenerator"]:
        assert expected in node_keys


# ════════════════════════════════════════════════════════════════════════════
# 11. Full conversation flow via compiled graph
# ════════════════════════════════════════════════════════════════════════════
def test_full_lead_capture_flow(monkeypatch):
    """
    End-to-end: user expresses interest → bot collects name / email / platform
    → mock_lead_capture is called exactly once.

    The invoke() helper mirrors main.py: it carries ALL non-message state
    fields (lead_name, lead_email, lead_platform, lead_captured, …) forward
    between turns so the graph router's `lead_in_progress` check works.
    """
    captured = []
    monkeypatch.setattr(
        nodes_module, "mock_lead_capture",
        lambda n, e, p: captured.append((n, e, p)) or True,
    )

    graph = create_agent_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-flow-001"}}

    # All non-message state fields that must persist across turns
    NON_MSG_KEYS = [
        "intent", "intent_confidence",
        "lead_active",
        "lead_name", "lead_email", "lead_platform",
        "lead_captured", "retrieved_info", "leads_count",
    ]

    base = {
        "intent": "", "intent_confidence": 0.0,
        "lead_active": False,
        "lead_name": "", "lead_email": "", "lead_platform": "",
        "lead_captured": False, "retrieved_info": "", "leads_count": 0,
    }

    def invoke(text: str) -> dict:
        nonlocal base
        result = compiled.invoke(
            {**base, "messages": [HumanMessage(content=text)]}, config
        )
        # Carry every non-message field forward so routing sees current state
        for key in NON_MSG_KEYS:
            if key in result:
                base[key] = result[key]
        return result

    # Step 1 — trigger lead intent
    r1 = invoke("I want to subscribe to the Pro plan")
    assert "name" in r1["messages"][-1].content.lower(), \
        f"Expected name prompt, got: {r1['messages'][-1].content!r}"

    # Step 2 — provide name (lead_name is now carried in base → lead_in_progress=True)
    r2 = invoke("Alice")
    assert "email" in r2["messages"][-1].content.lower(), \
        f"Expected email prompt, got: {r2['messages'][-1].content!r}"

    # Step 3 — provide valid email
    r3 = invoke("alice@example.com")
    text3 = r3["messages"][-1].content.lower()
    assert any(w in text3 for w in ("platform", "content", "youtube", "tiktok", "set", "instagram")), \
        f"Expected platform prompt or confirmation, got: {r3['messages'][-1].content!r}"

    # Step 4 — provide platform; tool should fire now
    invoke("YouTube")

    assert len(captured) == 1, f"Expected 1 capture, got {len(captured)}"
    assert captured[0] == ("Alice", "alice@example.com", "YouTube")
