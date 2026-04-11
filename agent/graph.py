"""
agent/graph.py
--------------
Builds and returns the compiled LangGraph StateGraph for the AutoStream agent.

Graph topology
--------------
START
  └─► IntentClassifier
        ├─(greeting)──────────────────────────► ResponseGenerator ─► END
        ├─(pricing)──► RAGRetriever ──────────► ResponseGenerator ─► END
        └─(lead)────► LeadCollector ─► ToolExecutor ─► ResponseGenerator ─► END
"""

import logging

from langgraph.graph import END, START, StateGraph

from agent.memory import AgentState
from agent.nodes import (
    intent_classifier,
    lead_collector,
    rag_retriever,
    response_generator,
    tool_executor,
)

logger = logging.getLogger(__name__)


# ── Routing function (called after IntentClassifier) ───────────────────────
def _route_by_intent(state: AgentState) -> str:
    """
    Return the name of the next node based on the classified intent.

    Uses ``lead_active`` (set by LeadCollector on the PREVIOUS turn) to keep
    an in-progress lead flow alive even when the user's raw message (e.g. just
    their name or email) doesn't match any lead keyword.
    """
    intent = state.get("intent", "greeting")
    lead_active = state.get("lead_active", False)
    lead_captured = state.get("lead_captured", False)

    # Keep routing to LeadCollector while the sub-flow is active and incomplete
    if (intent == "lead" or lead_active) and not lead_captured:
        logger.debug("Routing → LeadCollector (intent=%s, lead_active=%s)", intent, lead_active)
        return "LeadCollector"

    if intent == "pricing":
        logger.debug("Routing → RAGRetriever")
        return "RAGRetriever"

    logger.debug("Routing → ResponseGenerator (intent=%s)", intent)
    return "ResponseGenerator"


# ── Graph factory ──────────────────────────────────────────────────────────
def create_agent_graph() -> StateGraph:
    """
    Construct the AutoStream LangGraph StateGraph.

    Returns the *uncompiled* graph so callers can attach their own
    checkpointer (e.g. MemorySaver for CLI, SqliteSaver for persistence).
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────
    graph.add_node("IntentClassifier", intent_classifier)
    graph.add_node("RAGRetriever", rag_retriever)
    graph.add_node("LeadCollector", lead_collector)
    graph.add_node("ToolExecutor", tool_executor)
    graph.add_node("ResponseGenerator", response_generator)

    # ── Entry edge ──────────────────────────────────────────────────────
    graph.add_edge(START, "IntentClassifier")

    # ── Conditional fan-out from IntentClassifier ───────────────────────
    graph.add_conditional_edges(
        "IntentClassifier",
        _route_by_intent,
        {
            "RAGRetriever": "RAGRetriever",
            "LeadCollector": "LeadCollector",
            "ResponseGenerator": "ResponseGenerator",
        },
    )

    # ── Downstream edges ────────────────────────────────────────────────
    graph.add_edge("RAGRetriever", "ResponseGenerator")
    graph.add_edge("LeadCollector", "ToolExecutor")
    graph.add_edge("ToolExecutor", "ResponseGenerator")
    graph.add_edge("ResponseGenerator", END)

    logger.info("AutoStream agent graph created.")
    return graph
