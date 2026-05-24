"""
Tests that both API keys work by making one tiny call to each provider.
Prints checkmarks for success, full error messages for failure.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env from current directory

# --- Test OpenAI (embeddings) ---
print("Testing OpenAI...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="hello"
    )
    vector_length = len(response.data[0].embedding)
    print(f"OpenAI ✓  (got back a {vector_length}-dim vector)")
except Exception as e:
    print(f"OpenAI ✗  {type(e).__name__}: {e}")

# --- Test Anthropic (LLM) ---
print("\nTesting Anthropic...")
try:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say hi in 3 words"}]
    )
    reply = response.content[0].text
    print(f"Anthropic ✓  (Claude said: '{reply}')")
except Exception as e:
    print(f"Anthropic ✗  {type(e).__name__}: {e}")
