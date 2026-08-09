"""
Step 4: Connect retrieval (from ingest.py) to the LLM.
This turns your standalone retriever into an actual RAG chatbot.

Run it directly to test: python rag_chat.py
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from ingest import load_existing_vector_store, retrieve

# Reuse the vector store you already built with ingest.py
# (make sure you've run `python ingest.py` at least once before this)
vector_store = load_existing_vector_store()

llm = ChatGroq(model="llama-3.1-8b-instant")

PROMPT_TEMPLATE = """Answer the question using only this context. If the context doesn't contain the answer, say you don't know — do not make anything up.

Context:
{context}

Question: {question}
"""


def ask(question: str, k: int = 3) -> str:
    # 1. Retrieve the relevant chunks
    chunks = retrieve(vector_store, question, k=k)
    context = "\n\n".join(chunk.page_content for chunk in chunks)

    # 2. Fill in the prompt template
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 3. Send to the LLM
    response = llm.invoke(prompt)

    # 4. Return the answer
    return response.content


if __name__ == "__main__":
    # Change this to a question you already know the answer to from your PDF
    test_question = "What is prompt engineering?"

    print(f"Question: {test_question}\n")
    answer = ask(test_question)
    print(f"Answer: {answer}")
