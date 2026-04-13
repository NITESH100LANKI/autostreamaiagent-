import streamlit as st
import uuid
import os
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.graph import create_agent_graph

import logging

# ── Sync Secrets with Environment (Critical for Cloud Deployment) ──────────
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants & Helpers ───────────────────────────────────────────────────
TITLE = "AutoStream AI Agent"
ICON = "🤖"

def _fresh_base_state() -> dict:
    """Return a clean state dict for a new conversation thread."""
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

# ── Session State Initialization ──────────────────────────────────────────
@st.cache_resource
def get_compiled_agent():
    """Build and compile the agent graph. Cached as a resource."""
    logger.info("Compiling agent graph...")
    graph = create_agent_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

def init_session_state():
    """Initialize only the essential UI-related state first."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        
    if "base_state" not in st.session_state:
        st.session_state.base_state = _fresh_base_state()

def ensure_agent():
    """Ensure the agent is compiled and available in session state."""
    if "compiled_agent" not in st.session_state:
        with st.spinner("Initializing AI Brain (first time may take a minute)..."):
            st.session_state.compiled_agent = get_compiled_agent()

def reset_conversation():
    st.session_state.messages = []
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.base_state = _fresh_base_state()
    st.toast("Conversation reset! 🔄")

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered")

# ── CSS for ChatGPT-like appearance ──────────────────────────────────────
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ── UI Header ─────────────────────────────────────────────────────────────
st.title(f"{ICON} {TITLE}")
st.caption("AutoStream Sales & Support Agent | Powered by LangGraph & Gemini")

# ── Chat Logic ────────────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Info")
    st.write(f"**Thread ID:** `{st.session_state.get('thread_id', 'N/A')}`")
    
    leads_captured = st.session_state.get("base_state", {}).get("leads_count", 0)
    st.metric("Leads Captured", leads_captured)
    
    st.divider()
    if st.button("Reset Chat", use_container_width=True, type="primary"):
        reset_conversation()
        st.rerun()

# ── Chat Logic ────────────────────────────────────────────────────────────
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Message AutoStream..."):
    # 0. Ensure agent is ready (slow step)
    ensure_agent()
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Prepare config
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # 3. Prepare input state (merge base_state + history)
    turn_input = {
        **st.session_state.base_state, 
        "messages": [HumanMessage(content=prompt)]
    }

    # 4. Invoke agent with Streaming
    try:
        with st.chat_message("assistant"):
            # We'll use a placeholder for chunks and a container for the stream
            response_placeholder = st.empty()
            full_response = ""
            
            # Helper to run async generator in sync streamlit
            import asyncio

            async def get_agent_tokens():
                has_streamed_any = False
                async for event in st.session_state.compiled_agent.astream_events(turn_input, config, version="v2"):
                    # 1. Capture tokens from ResponseGenerator
                    if event["event"] == "on_llm_stream":
                        node = event.get("metadata", {}).get("langgraph_node")
                        # Only stream tokens from the actual response generator
                        if node == "ResponseGenerator":
                            chunk = event["data"]["chunk"]
                            token = chunk.content if hasattr(chunk, "content") else str(chunk)
                            if token:
                                has_streamed_any = True
                                yield token
                
                # 2. If no tokens were streamed (e.g. Lead Collector rule-based msg),
                # we fetch the final state message.
                if not has_streamed_any:
                    final_state = st.session_state.compiled_agent.get_state(config)
                    messages = final_state.values.get("messages", [])
                    if messages:
                        yield messages[-1].content

            # Define the generator for st.write_stream
            def stream_wrapper():
                loop = asyncio.new_event_loop()
                async_gen = get_agent_tokens()
                while True:
                    try:
                        token = loop.run_until_complete(async_gen.__anext__())
                        yield token
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        logger.error(f"Streaming error chunk: {e}")
                        break
                loop.close()

            # Execute the stream
            full_response = st.write_stream(stream_wrapper)
            
            # 5. Final State Sync (Update lead fields, etc.)
            # After stream ends, we pull the final computed state
            final_snapshot = st.session_state.compiled_agent.get_state(config)
            result = final_snapshot.values
            
            # Update persistent fields in base_state
            for key in st.session_state.base_state.keys():
                if key in result:
                    st.session_state.base_state[key] = result[key]
            
            # Add to history
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.warning("Agent produced no response.")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Agent error: {error_msg}")
        
        # Check if it's a known API error
        is_api_error = any(kw in error_msg.lower() for kw in ["quota", "limit", "429", "invalid api key", "api_key"])
        
        if is_api_error:
             st.error("⚠️ API Issue: Please check your GOOGLE_API_KEY and quota.")
             st.info("Make sure you have set the GOOGLE_API_KEY environment variable before running the app.")
        else:
             st.error(f"🤖 Agent encountered an error: {error_msg}")
             st.info("This might happen if the agent's internal logic fails. Try 'Reset Chat' in the sidebar.")
        
        # Diagnostics
        if not os.environ.get("GOOGLE_API_KEY"):
            st.warning("Warning: GOOGLE_API_KEY environment variable is not set.")
