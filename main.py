"""Tool calling loop: the LLM decides which tools to call, Python runs them."""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from openai import OpenAI

from agent_logging import setup_logging
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

log = logging.getLogger("agent.main")

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
    "When you search for a product, pass the user's wishes to web_search as `requirements`: short, checkable "
    "statements such as 'The product is black', 'The offer is for new goods', 'The shop is in Germany'. "
    "Be precise: write 'The chip is the base Apple M5, not M5 Pro or M5 Max', not just 'The chip is Apple M5'. "
    "The results come back ranked; 'contradicts' and 'not_confirmed' show what "
    "you still need to check on the page with read_page. "
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


def _readable(error: openai.OpenAIError) -> str:
    if isinstance(error, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
    if isinstance(error, openai.RateLimitError):
        return "OpenAI rate limit or quota exceeded. Wait a moment or check your billing and limits."
    if isinstance(error, openai.APITimeoutError):  # must come before APIConnectionError, its parent class
        return "OpenAI did not answer in time. Try again."
    if isinstance(error, openai.APIConnectionError):
        return "Cannot reach OpenAI. Check your internet connection."
    if isinstance(error, openai.APIStatusError):
        return f"OpenAI returned an error (HTTP {error.status_code}): {error.message}"
    return f"OpenAI call failed: {error}"


def ask_llm(messages: list, tool_choice: str):
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,  # "auto": the LLM decides whether to call a tool
            reasoning_effort="none",  # Chat Completions + tools needs reasoning off
        )
    except openai.OpenAIError as e:
        log.error("OpenAI call failed: %s: %s", type(e).__name__, e)
        raise LLMError(_readable(e)) from e


def run_tool(call) -> dict:
    return execute_tool(call.function.name, call.function.arguments)


def run(user_query: str) -> tuple[str, list[dict]]:
    """Return the final answer and a record of every tool call: [{"name": ..., "error": bool}]."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    calls: list[dict] = []
    log.debug("Query: %s", user_query)

    for step in range(1, MAX_STEPS + 1):
        # On the last step tools are switched off, so the LLM must give a final answer.
        last_step = step == MAX_STEPS
        if last_step:
            log.info("[step %d] tool steps used up, asking for a final answer", step)
            messages.append({"role": "system", "content": FINAL_STEP_NOTE})

        tool_choice = "none" if last_step else "auto"
        llm_started = time.perf_counter()
        response = ask_llm(messages, tool_choice=tool_choice)
        usage = getattr(response, "usage", None)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", "?")
        log.debug(
            "[step %d] LLM call took %.1fs (tool_choice=%s, history %d chars; "
            "tokens in/out: %s/%s, %s of the input cached)",
            step,
            time.perf_counter() - llm_started,
            tool_choice,
            len(json.dumps(messages, ensure_ascii=False)),
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            cached,
        )
        message = response.choices[0].message

        # No tool requested -> this is the final answer.
        if not message.tool_calls:
            log.debug("[step %d] final answer: %s", step, message.content)
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

        # Log and send back the results, matched to their tool_call_id.
        log.info("[step %d] %d tool call(s) in parallel, %.1fs", step, len(results), elapsed)
        for call, result in zip(message.tool_calls, results):
            calls.append({"name": call.function.name, "error": "error" in result})
            content = json.dumps(result, ensure_ascii=False)
            log.info("[step %d] -> %s(%s)", step, call.function.name, call.function.arguments[:150])
            log.info("[step %d] <- %s", step, content[:200])
            log.debug("[step %d] full result of %s (%s): %s", step, call.function.name, call.id, content)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    return f"Stopped after {MAX_STEPS} steps without a final answer.", calls


if __name__ == "__main__":
    setup_logging()
    query = " ".join(sys.argv[1:]) or input("Your question: ")
    try:
        answer, _ = run(query)
    except LLMError as e:
        raise SystemExit(f"\nError: {e}")
    print("\n=== Answer ===")
    print(answer)
