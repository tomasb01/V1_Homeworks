"""Step 1: verify that we can reach the OpenAI API with the key from .env."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or api_key.startswith("<"):
    raise SystemExit("OPENAI_API_KEY is missing. Copy .env.example to .env and fill in your key.")

client = OpenAI(api_key=api_key)
model = os.environ.get("OPENAI_MODEL", "gpt-6-luna")

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with one word: ready"}],
    reasoning_effort="none",
)
print(f"Model {model} says: {response.choices[0].message.content}")
