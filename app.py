"""
Step 7: Add memory + a Streamlit UI to the agent from Step 6.

Run it with: streamlit run app.py
(NOT python3 app.py -- Streamlit apps are run differently, see below)
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from ingest import load_existing_vector_store, retrieve

@st.cache_resource
def load_setup():
    """Runs once and caches the result -- won't reload on every message."""
    vs = load_existing_vector_store()
    model = ChatGroq(model="llama-3.1-8b-instant")
    return vs, model

vector_store, llm = load_setup()

PROMPT_TEMPLATE = """Answer the question using only this context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""


class State(TypedDict):
    question: str
    route: str
    docs: str
    answer: str


def router_node(state: State) -> State:
    classify_prompt = f"""Classify this question into exactly one word: "notes" or "calculator".
Use "calculator" only if it's a pure math question.

Question: {state['question']}

Answer with exactly one word: notes or calculator"""
    response = llm.invoke(classify_prompt)
    decision = response.content.strip().lower()
    state["route"] = "calculator" if "calculator" in decision else "notes"
    return state


def route_decision(state: State) -> Literal["retrieve", "calculator"]:
    return "calculator" if state["route"] == "calculator" else "retrieve"


def retrieve_node(state: State) -> State:
    chunks = retrieve(vector_store, state["question"], k=3)
    state["docs"] = "\n\n".join(chunk.page_content for chunk in chunks)
    return state


def generate_node(state: State) -> State:
    prompt = PROMPT_TEMPLATE.format(context=state["docs"], question=state["question"])
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


def calculator_node(state: State) -> State:
    prompt = f"Answer this math question directly and briefly: {state['question']}"
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# --- Build the graph, same as Step 6 ---
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("calculator", calculator_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "calculator": "calculator"})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
graph.add_edge("calculator", END)

# --- NEW: add a memory checkpointer when compiling ---
memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)


# =========================================================
# Streamlit UI
# =========================================================
st.title("📚 My Study Agent")
st.caption("Ask questions about your notes, or give it a math question.")

# Keep chat history visible on the page across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# A fixed conversation ID so the checkpointer knows which memory to load
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session-1"

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box at the bottom
question = st.chat_input("Ask something...")

if question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run the graph, passing the thread_id so it remembers past turns
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = app_graph.invoke({"question": question}, config=config)

    # Show the answer
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"Routed to: {result['route']}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
