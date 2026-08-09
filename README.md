# 📚 Study Agent — RAG-Powered Agentic AI Assistant

A personal study assistant built with a Retrieval-Augmented Generation (RAG) pipeline and an agentic architecture using LangGraph. The agent reads your notes/PDFs and answers questions from them, while also being able to route math questions to a calculator and current-events questions to a live web search — deciding which tool to use on its own.

## What it does

- **Ask questions about your own notes** — answers are grounded in your actual uploaded documents, not generic AI knowledge
- **Handles math questions directly** — routes pure calculations away from document retrieval
- **Searches the web for current information** — routes time-sensitive questions (news, current events) to a live search tool
- **Remembers conversation context** — supports natural follow-up questions within a session
- **Runs as a chat interface in the browser** — built with Streamlit, no terminal required to use it

## How it works

The agent is built as a graph (using LangGraph) rather than a fixed pipeline:

1. **Router node** — classifies each incoming question into one of three categories: notes, calculator, or web
2. **Retrieve node** — for notes questions, searches a local vector database (ChromaDB) of embedded document chunks for the most relevant context
3. **Generate node** — combines retrieved context with the question and sends it to the LLM
4. **Calculator node** — handles math questions directly
5. **Web search node** — queries a live search API (Tavily) for current information

This routing is decided by the LLM itself at runtime, not hardcoded — that's what makes it an agent rather than a simple RAG chatbot.

## Tech stack

- **LLM**: Groq API (Llama 3.1)
- **Orchestration**: LangChain + LangGraph
- **Vector store**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Web search**: Tavily API
- **UI**: Streamlit
- **Language**: Python

## Project structure

```
study-agent/
├── data/              # Source PDFs/notes for the knowledge base
├── ingest.py          # Loads, chunks, embeds, and stores documents in ChromaDB
├── rag_chat.py        # Basic RAG chatbot (retrieval + generation, no graph)
├── graph.py           # Minimal LangGraph wrapper around the RAG pipeline
├── agent.py           # Agentic version with router + calculator tool
├── agent_v2.py        # Extended agent with a third route: live web search
├── app.py             # Streamlit chat UI with memory
└── requirements.txt
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

4. Add PDFs to the `data/` folder, then build the vector store:
   ```
   python3 ingest.py
   ```

5. Run the chat app:
   ```
   streamlit run app.py
   ```

## Status

This is an actively evolving learning project, built step by step as a practical implementation of RAG and agentic AI concepts. Planned next steps include adding a self-reflection/answer-verification node and evaluating retrieval quality with RAGAS.

## Author

Rameez Aftab
