import streamlit as st
import uuid
import os
import time
import logging
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.graph import create_agent_graph

# ── Sync Secrets (Required for Cloud Stability, Safe for Local) ────────────
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # Fallback to os.environ locally

# ── Page Configuration & Polish ───────────────────────────────────────────
st.set_page_config(page_title="AutoStream AI Agent", page_icon="🤖", layout="centered")

# Header Section
st.title("🤖 AutoStream AI Agent")
st.markdown("#### *Fast • Reliable • Zero Hallucination*")
st.caption("Pro-grade Sales & Support Intelligence")

# ── 1. CACHE GRAPH (STABLE) ──────────────────────────────────────────────
@st.cache_resource
def get_graph():
    """Build and compile the graph once and reuse across sessions."""
    uncompiled_graph = create_agent_graph()
    memory = MemorySaver()
    return uncompiled_graph.compile(checkpointer=memory)

graph = get_graph()

# ── 2. SESSION STATE MEMORY ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"

if "base_state" not in st.session_state:
    st.session_state.base_state = {
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

# ── 3. CHAT INTERFACE ─────────────────────────────────────────────────────
st.divider()

# Rendering History (Lazy & Minimal)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── 4. AGENT EXECUTION (INPUT DRIVEN ONLY) ────────────────────────────────
if prompt := st.chat_input("How can I help you?"):
    # User Input Entry
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare turn input
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    turn_input = {**st.session_state.base_state, "messages": [HumanMessage(content=prompt)]}
    
    # Assistant Logic
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Thinking..."):
                # INVOKE GRAPH (Single call per input)
                result = graph.invoke(turn_input, config)
                
                # Optimized State Sync
                st.session_state.base_state.update({
                    k: result[k] for k in st.session_state.base_state if k in result
                })
                
                # Typing Effect & Result Display
                if result.get("messages"):
                    bot_msg = result["messages"][-1].content
                    
                    # Simulated Typing Stream for UX Polish
                    full_response = ""
                    for word in bot_msg.split(' '):
                        full_response += word + " "
                        placeholder.write(full_response + "▌")
                        time.sleep(0.015) # Faster, smoother pace
                    
                    placeholder.write(bot_msg) # Final clean output without cursor
                    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        except Exception as e:
            logging.error(f"Execution Error: {e}")
            error_text = "API limit reached. Please try again later."
            placeholder.warning(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

# ── 5. FOOTER & SIDEBAR ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8em;'>"
    "Built with LangGraph + Gemini | stable-v1.2"
    "</div>", 
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Session Summary")
    st.metric("Leads Captured", st.session_state.base_state["leads_count"])
    st.info("Agent strictly uses retrieved knowledge for pricing.")
    
    if st.button("Reset Chat Session"):
        st.session_state.messages = []
        st.session_state.thread_id = f"web-refreshed-{uuid.uuid4().hex[:8]}"
        st.session_state.base_state = {
            "intent": "", "intent_confidence": 0.0, "lead_active": False,
            "lead_name": "", "lead_email": "", "lead_platform": "",
            "lead_captured": False, "retrieved_info": "", "leads_count": 0,
        }
        st.rerun()
