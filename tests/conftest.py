import os

# main.py refuses to start without a key. The tests never call the real API, so any value will do.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
