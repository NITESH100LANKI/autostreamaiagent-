"""
main.py
-------
CLI entry point for the AutoStream AI Agent.

Features
--------
- Multi-turn memory via LangGraph MemorySaver (persisted per thread_id).
- Special commands: /help, /reset, /stats, /quit.
- Session analytics: tracks lead capture count.
- Graceful error handling on every turn.

Usage
-----
    # Set API key (required for LLM + RAG features)
    $env:GOOGLE_API_KEY = "AIza..."

    python main.py
"""

import logging
import os
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_agent_graph

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,             # Updated to INFO for transparency
    format="%(asctime)s %(levelname)-8s %(name)s -> %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout               # Crucial for PowerShell compatibility
)
logger = logging.getLogger(__name__)


# ── UI helpers ─────────────────────────────────────────────────────────────
BANNER = """
+------------------------------------------------------+
|        AutoStream AI Agent  -  CLI Interface         |
|  Type /help for commands  |  /quit to exit           |
+------------------------------------------------------+"""

HELP_TEXT = """\
  /help    — Show this help message
  /reset   — Start a fresh conversation (new thread)
  /stats   — Show session analytics
  /quit    — Exit the chat"""


def _print_bot(msg: str) -> None:
    print(f"\n  Bot: {msg}\n")


def _print_system(msg: str) -> None:
    print(f"\n  [INFO] {msg}\n")


# ── Session state factory ──────────────────────────────────────────────────
def _fresh_base_state() -> dict:
    """Return a clean non-message state dict for a new conversation thread."""
    return {
        "intent": "",
        "intent_confidence": 0.0,
        "lead_active": False,
        "lead_name": "",
        "lead_email": "",
        "lead_platform": "",
        "lead_captured": False,
        "retrieved_info": "",
        "leads_count": 0,
    }


# ── Main loop ──────────────────────────────────────────────────────────────
def main() -> None:
    # ── API key check ──────────────────────────────────────────────────
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "\n⚠️  WARNING: GOOGLE_API_KEY is not set.\n"
            "   Intent Gemini fallback and RAG embeddings are disabled.\n"
            "   Rule-based intent + static RAG fallback will be used instead.\n",
            file=sys.stdout,
        )

    # ── Build agent ────────────────────────────────────────────────────
    graph = create_agent_graph()
    memory = MemorySaver()
    compiled_agent = graph.compile(checkpointer=memory)

    # ── Session variables ──────────────────────────────────────────────
    session_total_leads = 0
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    base_state = _fresh_base_state()

    def make_config() -> dict:
        return {"configurable": {"thread_id": thread_id}}

    print(BANNER)

    while True:
        try:
            raw = input("  You: ").strip()

            # ── Empty input ────────────────────────────────────────────
            if not raw:
                continue

            # ── Special commands ───────────────────────────────────────
            cmd = raw.lower()

            if cmd in ("/quit", "/exit", "quit", "exit", "q"):
                _print_bot("Thanks for chatting with AutoStream! Have a great day. 👋")
                _print_system(f"Session ended. Total leads captured: {session_total_leads}")
                break

            if cmd == "/help":
                print(HELP_TEXT)
                continue

            if cmd == "/stats":
                _print_system(
                    f"Session analytics → Leads captured: {session_total_leads} | "
                    f"Thread: {thread_id}"
                )
                continue

            if cmd == "/reset":
                # Start a brand-new thread → fresh memory slice
                thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                base_state = _fresh_base_state()
                _print_system("Conversation reset. Starting fresh! 🔄")
                continue

            # ── Normal turn ────────────────────────────────────────────
            turn_input = {**base_state, "messages": [HumanMessage(content=raw)]}
            result = compiled_agent.invoke(turn_input, make_config())

            # Persist lead fields so next turn's state machine continues correctly.
            base_state["lead_active"]   = result.get("lead_active") or False
            base_state["lead_name"]     = result.get("lead_name") or ""
            base_state["lead_email"]    = result.get("lead_email") or ""
            base_state["lead_platform"] = result.get("lead_platform") or ""
            base_state["lead_captured"] = result.get("lead_captured") or False
            base_state["leads_count"]   = result.get("leads_count") or 0

            # ── Track analytics ────────────────────────────────────────
            new_count = base_state["leads_count"]
            if new_count > session_total_leads:
                session_total_leads = new_count
                _print_system(f"🎯 Lead #{session_total_leads} recorded this session!")

            # ── Print reply ────────────────────────────────────────────
            reply_msgs = result.get("messages", [])
            if reply_msgs:
                _print_bot(reply_msgs[-1].content)

        except KeyboardInterrupt:
            print()
            _print_bot("Session interrupted. Goodbye! 👋")
            break

        except Exception as exc:
            logger.exception("Unhandled error in main loop.")
            print(f"\n  ❌  Error: {exc}\n  Please try again or type /reset.\n")


if __name__ == "__main__":
    main()
