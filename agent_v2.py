"""
Step 8: Iterate on the agent — add a web search tool as a third route.
Same pattern as adding the calculator in Step 6, just one more branch.

Run it directly to test: python3 agent_v2.py
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from ingest import load_existing_vector_store, retrieve

vector_store = load_existing_vector_store()
llm = ChatGroq(model="llama-3.1-8b-instant")
search_tool = TavilySearchResults(max_results=3)  # NEW: the web search tool

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
    # UPDATED: router now knows about three options instead of two
    classify_prompt = f"""Classify this question into exactly one word: "notes", "calculator", or "web".
- "calculator": pure math questions (e.g. "what is 12 * 8")
- "web": anything needing current/real-time info (news, weather, sports, recent events)
- "notes": everything else, assumed answerable from personal study notes

Question: {state['question']}

Answer with exactly one word: notes, calculator, or web"""

    response = llm.invoke(classify_prompt)
    decision = response.content.strip().lower()
    if "calculator" in decision:
        state["route"] = "calculator"
    elif "web" in decision:
        state["route"] = "web"
    else:
        state["route"] = "notes"
    return state


def route_decision(state: State) -> Literal["retrieve", "calculator", "web"]:
    # translate the router's word ("notes") into the actual node name ("retrieve")
    if state["route"] == "notes":
        return "retrieve"
    return state["route"]  # "calculator" or "web" already match node names


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


def web_search_node(state: State) -> State:
    """NEW node: searches the web, then asks the LLM to summarize the results."""
    results = search_tool.invoke(state["question"])
    combined = "\n\n".join(r["content"] for r in results)
    prompt = f"Using this web search info, answer briefly:\n\n{combined}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("calculator", calculator_node)
graph.add_node("web", web_search_node)  # NEW node registered

graph.set_entry_point("router")

# UPDATED: conditional edges now have a third destination
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",
        "calculator": "calculator",
        "web": "web",
    },
)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
graph.add_edge("calculator", END)
graph.add_edge("web", END)  # NEW edge

app = graph.compile()


if __name__ == "__main__":
    test_questions = [
        "What is prompt engineering?",   # -> notes
        "What is 45 * 12?",              # -> calculator
        "What's the latest AI news today?",  # -> web
    ]

    for q in test_questions:
        print(f"Question: {q}")
        result = app.invoke({"question": q})
        print(f"Routed to: {result['route']}")
        print(f"Answer: {result['answer']}\n")
