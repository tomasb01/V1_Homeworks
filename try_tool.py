"""Call a single tool by hand, without the LLM.

Usage:
    uv run try_tool.py <tool_name> '<json arguments>'

Example:
    uv run try_tool.py web_search '{"query": "Apple MacBook Pro M5 price", "country": "germany"}'
"""

import json
import sys

from dotenv import load_dotenv

from tools import execute_tool

load_dotenv()

if len(sys.argv) != 3:
    raise SystemExit(__doc__)

result = execute_tool(sys.argv[1], sys.argv[2])
print(json.dumps(result, indent=2, ensure_ascii=False))
