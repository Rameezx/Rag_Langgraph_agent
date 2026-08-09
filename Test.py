import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

print("API key loaded:", bool(os.getenv("GROQ_API_KEY")))

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

response = llm.invoke("Say hello in one sentence")

print(response.content)