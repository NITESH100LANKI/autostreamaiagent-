"""
agent/memory.py
---------------
Defines the AgentState TypedDict used across all LangGraph nodes.

Fields
------
messages        : Full conversation history (auto-merged via operator.add).
intent          : Classified intent for the current turn.
intent_confidence: 0.0–1.0 confidence score from the classifier.
lead_name       : Collected lead name (persisted across turns).
lead_email      : Collected lead email (persisted across turns).
lead_platform   : Collected platform (e.g. YouTube, Instagram).
lead_captured   : Flag – True once mock_lead_capture() has been called.
retrieved_info  : Raw text retrieved from the vector store.
leads_count     : Session-level analytics counter.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # ── Conversation history (LangGraph merges additions automatically) ──
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Routing & classification ──
    intent: str                  # "greeting" | "pricing" | "lead"
    intent_confidence: float     # 0.0–1.0

    # ── Lead capture sub-flow ──
    lead_active: bool            # True once lead flow has been triggered
    lead_name: str
    lead_email: str
    lead_platform: str
    lead_captured: bool          # True once tool has fired

    # ── RAG ──
    retrieved_info: str          # Joined text from top-k docs

    # ── Analytics ──
    leads_count: int             # Cumulative leads captured this session
