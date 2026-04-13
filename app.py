import streamlit as st
import uuid
import os
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

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(page_title="AutoStream AI Agent", page_icon="🤖")
st.title("🤖 AutoStream AI Agent")
st.caption("Sales & Support | Fixed Stable Version")

# ── Initialize Stable State ──────────────────────────────────────────────
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

# ── Load Agent (Simple Wrapper) ──────────────────────────────────────────
@st.cache_resource
def get_agent():
    graph = create_agent_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent = get_agent()

# ── Render Chat History ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Logic ────────────────────────────────────────────────────────────
if prompt := st.chat_input("How can I help you today?"):
    # 1. User Entry
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Agent Configuration
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    turn_input = {**st.session_state.base_state, "messages": [HumanMessage(content=prompt)]}
    
    # 3. Agent Execution
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = agent.invoke(turn_input, config)
                
                # Update background state (Persist lead info, etc.)
                for key in st.session_state.base_state.keys():
                    if key in result:
                        st.session_state.base_state[key] = result[key]
                
                # Display final response
                if result.get("messages"):
                    bot_msg = result["messages"][-1].content
                    st.markdown(bot_msg)
                    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        except Exception as e:
            # APPROVED ERROR HANDLING
            logging.error(f"API Error: {e}")
            error_text = "API limit reached. Please try again later."
            st.warning(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Summary")
    st.metric("Leads Captured", st.session_state.base_state["leads_count"])
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.rerun()
