"""
Step 6: Make the graph agentic by adding a router node.
Instead of always going retrieve -> generate, the LLM first decides
which path the question needs.

Run it directly to test: python3 agent.py
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ingest import load_existing_vector_store, retrieve

vector_store = load_existing_vector_store()
llm = ChatGroq(model="llama-3.1-8b-instant")

PROMPT_TEMPLATE = """Answer the question using only this context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""


# --- 1. State: same clipboard as before, plus a 'route' field ---
class State(TypedDict):
    question: str
    route: str
    docs: str
    answer: str


# --- 2. Router node: the NEW box. Decides which path to take. ---
def router_node(state: State) -> State:
    """Asks the LLM to classify the question, stores the decision on the clipboard."""
    classify_prompt = f"""Classify this question into exactly one word: "notes" or "calculator".
Use "calculator" only if it's a pure math question (e.g. "what is 12 * 8").
Otherwise use "notes".

Question: {state['question']}

Answer with exactly one word: notes or calculator"""

    response = llm.invoke(classify_prompt)
    decision = response.content.strip().lower()
    state["route"] = "calculator" if "calculator" in decision else "notes"
    return state


# --- 3. The function that reads the router's decision and picks the arrow ---
def route_decision(state: State) -> Literal["retrieve", "calculator"]:
    """LangGraph calls this to decide which node to go to next."""
    return "calculator" if state["route"] == "calculator" else "retrieve"


# --- 4. Existing nodes from Step 5 ---
def retrieve_node(state: State) -> State:
    chunks = retrieve(vector_store, state["question"], k=3)
    state["docs"] = "\n\n".join(chunk.page_content for chunk in chunks)
    return state


def generate_node(state: State) -> State:
    prompt = PROMPT_TEMPLATE.format(context=state["docs"], question=state["question"])
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# --- 5. New calculator node: a second possible path ---
def calculator_node(state: State) -> State:
    """Handles pure math questions directly, no retrieval needed."""
    prompt = f"Answer this math question directly and briefly: {state['question']}"
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# --- 6. Build the graph with a fork in it ---
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("calculator", calculator_node)

graph.set_entry_point("router")

# This is the fork: after "router", check route_decision() to pick the next box
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",     # if route_decision returns "retrieve" -> go to retrieve node
        "calculator": "calculator", # if route_decision returns "calculator" -> go to calculator node
    },
)

graph.add_edge("retrieve", "generate")  # notes path continues to generate
graph.add_edge("generate", END)
graph.add_edge("calculator", END)       # calculator path ends directly, no need for generate

app = graph.compile()


# --- 7. Run it with a few different questions to see the routing in action ---
if __name__ == "__main__":
    test_questions = [
        "What is prompt engineering?",  # should route to "notes"
        "What is 45 * 12?",             # should route to "calculator"
    ]

    for q in test_questions:
        print(f"Question: {q}")
        result = app.invoke({"question": q})
        print(f"Routed to: {result['route']}")
        print(f"Answer: {result['answer']}\n")
