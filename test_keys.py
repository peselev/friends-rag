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

# --- Test Cohere (reranker) ---
# Two keys power the "higher accuracy" toggle's degradation chain: the trial key
# is tried first, the prod key is the fallback. Both are OPTIONAL — if neither is
# set the toggle just serves the fast result. We test each independently so one
# missing/bad key doesn't mask the other.
def test_cohere_key(label, env_var):
    key = os.getenv(env_var)
    if not key:
        print(f"Cohere ({label}) –  {env_var} not set (toggle will fall back)")
        return
    try:
        import cohere
        client = cohere.ClientV2(key)
        response = client.rerank(
            model="rerank-v3.5",
            query="Who forgets Rachel's name?",
            documents=["Ross says Rachel at the altar.", "Joey orders a sandwich."],
            top_n=1,
        )
        top = response.results[0]
        print(f"Cohere ({label}) ✓  (top doc #{top.index}, score {top.relevance_score:.3f})")
    except Exception as e:
        print(f"Cohere ({label}) ✗  {type(e).__name__}: {e}")

print("\nTesting Cohere...")
test_cohere_key("trial", "COHERE_API_KEY")
test_cohere_key("prod", "COHERE_API_KEY_PROD")