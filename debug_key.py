import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

if key is None:
    print("PROBLEM: GROQ_API_KEY was not found.")
else:
    print(f"Key found. Length: {len(key)} characters")
    print(f"First 8 chars: {key[:8]}")
    print(f"Last 8 chars: {key[-8:]}")
