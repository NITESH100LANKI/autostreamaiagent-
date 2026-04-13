import streamlit as st
import uuid
import os
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.graph import create_agent_graph

# ── Simple UI Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="AutoStream AI Agent", page_icon="🤖")
st.title("🤖 AutoStream AI Agent")

# ── Session State ──────────────────────────────────────────────────────────
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

# ── Build Agent (Simple) ──────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    graph = create_agent_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent = load_agent()

# ── Chat Loop ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    # User message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run Agent
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    turn_input = {**st.session_state.base_state, "messages": [HumanMessage(content=prompt)]}
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(turn_input, config)
            
            # Update background state
            for key in st.session_state.base_state.keys():
                if key in result:
                    st.session_state.base_state[key] = result[key]
            
            # Show response
            if result.get("messages"):
                bot_msg = result["messages"][-1].content
                st.markdown(bot_msg)
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})

# ── Sidebar Stats ─────────────────────────────────────────────────────────
with st.sidebar:
    st.metric("Leads Captured", st.session_state.base_state["leads_count"])
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.base_state["leads_count"] = 0
        st.rerun()
