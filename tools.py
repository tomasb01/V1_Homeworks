"""Tool implementations and their schemas. The main app only calls execute_tool()."""

import json
import os

import requests
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# ---------- convert_currency ----------

class ConvertCurrencyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0, description="Amount of money to convert, e.g. 1199.")
    from_currency: str = Field(description="ISO 4217 code of the source currency, e.g. 'EUR', 'GBP', 'USD'.")
    to_currency: str = Field(description="ISO 4217 code of the target currency, e.g. 'CZK'.")

    @field_validator("from_currency", "to_currency")
    @classmethod
    def _check_code(cls, v: str) -> str:
        code = v.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"'{v}' is not a 3-letter currency code like 'EUR'.")
        return code


def convert_currency(args: ConvertCurrencyArgs) -> dict:
    src, dst = args.from_currency, args.to_currency
    if src == dst:
        return {"amount": args.amount, "from": src, "to": dst, "rate": 1.0, "result": args.amount, "date": None}

    response = requests.get(
        "https://api.frankfurter.dev/v1/latest",
        params={"base": src, "symbols": dst},
        timeout=10,
    )
    if response.status_code == 404:
        raise ValueError(f"Unsupported currency code: '{src}' or '{dst}' (rates are from the ECB).")
    response.raise_for_status()

    body = response.json()
    rate = body["rates"][dst]
    return {
        "amount": args.amount,
        "from": src,
        "to": dst,
        "rate": rate,
        "result": round(args.amount * rate, 2),
        "date": body["date"],
    }


# ---------- web_search ----------

class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=300, description="Search query, e.g. 'Apple MacBook Pro M5 black price'.")
    country: str | None = Field(
        default=None,
        description="Optional: boost results from this country. Lowercase English name, e.g. 'germany', 'united kingdom'.",
    )
    max_results: int = Field(default=5, ge=1, le=10, description="How many results to return (1-10).")


def web_search(args: WebSearchArgs) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    payload = {
        "query": args.query,
        "max_results": args.max_results,
        "include_images": True,
    }
    if args.country:
        payload["country"] = args.country.strip().lower()

    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Tavily error {response.status_code}: {response.text[:300]}")

    body = response.json()

    # Only the URLs go back to the LLM; check_image looks at the actual image.
    images = [img["url"] if isinstance(img, dict) else img for img in body.get("images", [])[:5]]

    return {
        "query": args.query,
        "results": [
            {"title": r["title"], "url": r["url"], "content": r["content"][:500]}  # keep token usage small
            for r in body.get("results", [])
        ],
        "images": images,
    }


# ---------- read_page ----------

def _check_http_url(v: str) -> str:
    if not v.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return v


class ReadPageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="URL of the page to read, e.g. a product page from web_search results.")
    query: str = Field(
        min_length=3,
        max_length=200,
        description="What to look for on the page, e.g. 'price of the black 16GB 512GB variant'. "
        "Only the most relevant parts of the page are returned.",
    )

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        return _check_http_url(v)


def read_page(args: ReadPageArgs) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    response = requests.post(
        "https://api.tavily.com/extract",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"urls": args.url, "query": args.query, "chunks_per_source": 5, "include_images": True},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"Tavily error {response.status_code}: {response.text[:300]}")

    body = response.json()
    if body.get("failed_results"):
        raise RuntimeError(f"Could not read the page: {body['failed_results'][0].get('error', 'unknown error')}")

    page = body["results"][0]
    return {
        "url": args.url,
        "content": page["raw_content"][:6000],  # keep token usage small
        "images": page.get("images", [])[:5],
    }


# ---------- check_image ----------

class CheckImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_url: str = Field(description="Public http(s) URL of the image to inspect.")
    question: str = Field(
        min_length=3,
        max_length=300,
        description="Question about the image, e.g. 'Is the laptop in the picture black?'.",
    )

    @field_validator("image_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        return _check_http_url(v)


def check_image(args: CheckImageArgs) -> dict:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=30)
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{args.question}\nAnswer in at most two short sentences."},
                    {"type": "image_url", "image_url": {"url": args.image_url}},
                ],
            }
        ],
        reasoning_effort="none",
    )
    return {"image_url": args.image_url, "answer": response.choices[0].message.content}


# ---------- registry ----------

_TOOLS = {
    "read_page": (
        "Open a web page and return the parts relevant to your query (e.g. price and variant of a product), "
        "plus image URLs found on the page. Use it to confirm details that search snippets do not show.",
        ReadPageArgs,
        read_page,
    ),
    "check_image": (
        "Look at an image from a URL and answer a question about it, "
        "e.g. to verify the color or content of a product photo.",
        CheckImageArgs,
        check_image,
    ),
    "web_search": (
        "Search the web. Returns result titles, URLs, text snippets (which may contain prices) "
        "and image URLs with descriptions.",
        WebSearchArgs,
        web_search,
    ),
    "convert_currency": (
        "Convert an amount of money between currencies using current exchange rates.",
        ConvertCurrencyArgs,
        convert_currency,
    ),
}

# Schemas in the format expected by the OpenAI `tools` parameter.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": model.model_json_schema()},
    }
    for name, (description, model, _) in _TOOLS.items()
]


def execute_tool(name: str, raw_arguments: str) -> dict:
    """Parse, validate and run a tool. Never raises: errors come back as {"error": ...}."""
    if name not in _TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(_TOOLS)}"}
    _, model, func = _TOOLS[name]
    try:
        args = model.model_validate(json.loads(raw_arguments or "{}"))
    except json.JSONDecodeError as e:
        return {"error": f"Arguments are not valid JSON: {e}"}
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return {"error": f"Invalid arguments: {problems}"}
    try:
        return func(args)
    except Exception as e:  # tool failure must reach the LLM, not crash the app
        return {"error": f"{type(e).__name__}: {e}"}
