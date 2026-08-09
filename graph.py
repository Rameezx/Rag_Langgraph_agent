"""
Step 5: Wrap the RAG pipeline in a minimal LangGraph.
Same retrieve -> generate flow as rag_chat.py, just structured as a graph.

Run it directly to test: python3 graph.py
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ingest import load_existing_vector_store, retrieve

# --- Setup (same as rag_chat.py) ---
vector_store = load_existing_vector_store()
llm = ChatGroq(model="llama-3.1-8b-instant")

PROMPT_TEMPLATE = """Answer the question using only this context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""


# --- 1. Define the State: the clipboard passed between boxes ---
class State(TypedDict):
    question: str
    docs: str
    answer: str


# --- 2. Define the nodes: one function per box ---
def retrieve_node(state: State) -> State:
    """Box 1: takes the question, fills in 'docs' on the clipboard."""
    chunks = retrieve(vector_store, state["question"], k=3)
    state["docs"] = "\n\n".join(chunk.page_content for chunk in chunks)
    return state


def generate_node(state: State) -> State:
    """Box 2: takes 'docs' + 'question', fills in 'answer' on the clipboard."""
    prompt = PROMPT_TEMPLATE.format(context=state["docs"], question=state["question"])
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# --- 3. Build the graph: connect the boxes with arrows ---
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")        # where the flow starts
graph.add_edge("retrieve", "generate")   # arrow: retrieve -> generate
graph.add_edge("generate", END)          # arrow: generate -> done

app = graph.compile()


# --- 4. Run it ---
if __name__ == "__main__":
    test_question = "What is prompt engineering?"  # change to match your PDF

    print(f"Question: {test_question}\n")
    result = app.invoke({"question": test_question})
    print(f"Answer: {result['answer']}")
