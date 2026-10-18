"""Tool calling loop: the LLM decides which tools to call, Python runs them."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is missing. Copy .env.example to .env and fill in your key.")

# The SDK itself retries short outages (network, 429, 5xx) with growing pauses; we only set the limits.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=3, timeout=60)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
MAX_STEPS = 10  # safety limit against an endless tool loop

SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Use a tool when you need current data, "
    "an image check or a currency conversion; otherwise answer directly. "
    "Search results are only snippets: before you state a price, open the product page with read_page "
    "and confirm the price for the exact variant the user wants (e.g. the color). "
    "Convert currency only for the final price you chose. "
    "A price comparison site is a good enough source: use the lowest listed price for the variant, "
    "do not hunt for the exact merchant. Check an image at most twice. "
    "As soon as you have the price, the color check and the conversion, stop searching and answer. "
    "Only state facts (price, seller, color) that appeared in a tool result, and say what you could not verify. "
    "Do not add citation markers. "
    "Answer in the language of the user."
)

# Sent before the last allowed step, when tools are switched off.
FINAL_STEP_NOTE = (
    "You have no tool steps left. Answer now with what you found so far. "
    "Clearly say what you could not verify."
)


class LLMError(Exception):
    """The OpenAI call failed even after the SDK's retries. The message is meant for the user."""


def ask_llm(messages: list, tool_choice: str):
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,  # "auto": the LLM decides whether to call a tool
            reasoning_effort="none",  # Chat Completions + tools needs reasoning off
        )
    except openai.AuthenticationError:
        raise LLMError("OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file.")
    except openai.RateLimitError:
        raise LLMError("OpenAI rate limit or quota exceeded. Wait a moment or check your billing and limits.")
    except openai.APITimeoutError:
        raise LLMError("OpenAI did not answer in time. Try again.")
    except openai.APIConnectionError:
        raise LLMError("Cannot reach OpenAI. Check your internet connection.")
    except openai.APIStatusError as e:
        raise LLMError(f"OpenAI returned an error (HTTP {e.status_code}): {e.message}")


def run_tool(call) -> dict:
    return execute_tool(call.function.name, call.function.arguments)


def run(user_query: str) -> tuple[str, list[dict]]:
    """Return the final answer and a record of every tool call: [{"name": ..., "error": bool}]."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    calls: list[dict] = []

    for step in range(1, MAX_STEPS + 1):
        # On the last step tools are switched off, so the LLM must give a final answer.
        last_step = step == MAX_STEPS
        if last_step:
            print(f"[step {step}] tool steps used up, asking for a final answer")
            messages.append({"role": "system", "content": FINAL_STEP_NOTE})

        response = ask_llm(messages, tool_choice="none" if last_step else "auto")
        message = response.choices[0].message

        # No tool requested -> this is the final answer.
        if not message.tool_calls:
            return message.content, calls

        # Keep the LLM's request in the history, exactly as it asked for it.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )

        # Run all requested tools at the same time (they are independent and mostly wait for the network).
        started = time.perf_counter()
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(run_tool, message.tool_calls))  # same order as the requests
        elapsed = time.perf_counter() - started

        # Print and send back the results, matched to their tool_call_id.
        print(f"[step {step}] {len(results)} tool call(s) in parallel, {elapsed:.1f}s")
        for call, result in zip(message.tool_calls, results):
            calls.append({"name": call.function.name, "error": "error" in result})
            print(f"[step {step}] -> {call.function.name}({call.function.arguments[:150]})")
            print(f"[step {step}] <- {json.dumps(result, ensure_ascii=False)[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return f"Stopped after {MAX_STEPS} steps without a final answer.", calls


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Your question: ")
    try:
        answer, _ = run(query)
    except LLMError as e:
        raise SystemExit(f"\nError: {e}")
    print("\n=== Answer ===")
    print(answer)
