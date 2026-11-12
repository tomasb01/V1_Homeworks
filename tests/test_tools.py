"""Offline tests of the tools: no internet, no API keys needed (all HTTP and OpenAI calls are faked)."""

import json
from types import SimpleNamespace

import pytest
import requests

import tools


def call(name: str, **arguments) -> dict:
    return tools.execute_tool(name, json.dumps(arguments))


class FakeResponse:
    """Stands in for requests.Response."""

    def __init__(self, status_code: int = 200, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Any HTTP call that a test did not fake on purpose fails loudly."""

    def blocked(*args, **kwargs):
        raise AssertionError("unexpected network call in a unit test")

    monkeypatch.setattr(tools.requests, "get", blocked)
    monkeypatch.setattr(tools.requests, "post", blocked)


# ---------- execute_tool: parsing and dispatch ----------

def test_schemas_describe_all_four_tools():
    names = {schema["function"]["name"] for schema in tools.TOOL_SCHEMAS}
    assert names == {"web_search", "read_page", "check_image", "convert_currency"}
    for schema in tools.TOOL_SCHEMAS:
        parameters = schema["function"]["parameters"]
        assert schema["type"] == "function"
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False  # unknown arguments are rejected
        assert parameters["required"]


def test_unknown_tool_returns_error():
    assert "Unknown tool" in tools.execute_tool("make_coffee", "{}")["error"]


def test_invalid_json_returns_error():
    assert "not valid JSON" in tools.execute_tool("convert_currency", "not json")["error"]


def test_empty_arguments_are_treated_as_empty_object():
    assert "Field required" in tools.execute_tool("web_search", "")["error"]


# ---------- convert_currency ----------

def test_convert_currency_converts_and_normalizes_codes(monkeypatch):
    monkeypatch.setattr(
        tools.requests, "get", lambda *a, **k: FakeResponse(body={"rates": {"CZK": 24.0}, "date": "2026-10-16"})
    )
    result = call("convert_currency", amount=100, from_currency="eur", to_currency="czk")
    assert result == {"amount": 100.0, "from": "EUR", "to": "CZK", "rate": 24.0, "result": 2400.0, "date": "2026-10-16"}


def test_convert_currency_same_currency_needs_no_request():
    result = call("convert_currency", amount=50, from_currency="CZK", to_currency="CZK")
    assert result["result"] == 50  # the blocked network would have turned this into an error


def test_convert_currency_unknown_currency_is_reported(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(status_code=404))
    assert "Unsupported currency" in call("convert_currency", amount=1, from_currency="XXX", to_currency="CZK")["error"]


def test_convert_currency_network_failure_is_reported(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("no internet")

    monkeypatch.setattr(tools.requests, "get", fail)
    assert "no internet" in call("convert_currency", amount=1, from_currency="EUR", to_currency="CZK")["error"]


@pytest.mark.parametrize(
    "arguments, expected_message",
    [
        ({"amount": -5, "from_currency": "EUR", "to_currency": "CZK"}, "greater than 0"),
        ({"amount": 5, "from_currency": "EURO", "to_currency": "CZK"}, "3-letter"),
        ({"amount": 5, "from_currency": "EUR"}, "Field required"),
        ({"amount": 5, "from_currency": "EUR", "to_currency": "CZK", "extra": 1}, "Extra inputs"),
        ({"amount": "abc", "from_currency": "EUR", "to_currency": "CZK"}, "valid number"),
    ],
)
def test_convert_currency_rejects_invalid_arguments(arguments, expected_message):
    assert expected_message in call("convert_currency", **arguments)["error"]


# ---------- web_search ----------

def test_web_search_needs_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "TAVILY_API_KEY" in call("web_search", query="apple m5")["error"]


@pytest.mark.parametrize("arguments", [{"query": "hi"}, {"query": "apple m5", "max_results": 50}])
def test_web_search_rejects_invalid_arguments(arguments):
    assert "Invalid arguments" in call("web_search", **arguments)["error"]


def test_web_search_shapes_the_result(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, payload=json)
        return FakeResponse(
            body={
                "results": [{"title": "T", "url": "https://shop.example", "content": "x" * 900}],
                "images": [{"url": f"https://img.example/{i}.jpg", "description": "d"} for i in range(8)],
            }
        )

    monkeypatch.setattr(tools.requests, "post", fake_post)
    result = call("web_search", query="apple m5", country="Germany")

    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert sent["payload"]["country"] == "germany"  # normalized to lower case
    assert len(result["results"][0]["content"]) == tools.SNIPPET_CHARS  # snippet is shortened
    assert result["images"] == [f"https://img.example/{i}.jpg" for i in range(tools.MAX_IMAGES)]  # a few plain URLs


REQUIREMENTS = ["The product is black", "The offer is for new goods"]


def tavily_results(count: int) -> dict:
    return {"results": [{"title": f"offer {i}", "url": f"https://shop.example/{i}", "content": "text"} for i in range(count)]}


def test_web_search_without_jev_ignores_requirements_and_says_so(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(json)
        return FakeResponse(body=tavily_results(5))

    monkeypatch.setattr(tools.requests, "post", fake_post)
    result = call("web_search", query="apple m5", requirements=REQUIREMENTS)

    assert sent["max_results"] == 5  # no reason to fetch more candidates
    assert len(result["results"]) == 5 and "match" not in result["results"][0]
    assert "Jev filter is not configured" in result["note"]


def test_web_search_with_jev_fetches_more_and_returns_only_the_best(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    sent = {}

    def fake_post(url, headers, json, timeout):
        if "tavily" in url:
            sent.update(json)
            return FakeResponse(body=tavily_results(10))
        # Jev: the higher the offer number, the better it matches
        met = int(json["state"]["title"].split()[-1]) / 10
        probabilities = {"met": met, "contradicted": 1 - met, "not_mentioned": 0.0}
        answers = {f"r{i}": {"probabilities": probabilities} for i in range(len(json["questions"]))}
        return FakeResponse(body={"answers": answers})

    monkeypatch.setattr(tools.requests, "post", fake_post)
    result = call("web_search", query="apple m5", max_results=3, requirements=REQUIREMENTS)

    assert sent["max_results"] == tools.JEV_CANDIDATES
    assert [r["title"] for r in result["results"]] == ["offer 9", "offer 8", "offer 7"]
    assert result["results"][0]["match"] == 0.9
    assert "note" not in result


@pytest.mark.parametrize(
    "requirements",
    [["a"], [f"requirement number {i}" for i in range(7)], []],
    ids=["too short", "more than 6", "empty list is fine"],
)
def test_web_search_validates_requirements(monkeypatch, requirements):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(body=tavily_results(2)))
    result = call("web_search", query="apple m5", requirements=requirements)
    assert ("error" in result) == (requirements != [])


def test_web_search_reports_tavily_errors(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(status_code=401, text="bad key"))
    error = call("web_search", query="apple m5")["error"]
    assert "401" in error and "bad key" in error


# ---------- read_page ----------

def test_read_page_rejects_url_without_scheme():
    assert "http" in call("read_page", url="geizhals.de", query="price")["error"]


def test_read_page_limits_content_and_images(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    page = {"raw_content": "y" * 9000, "images": [f"https://img.example/{i}.png" for i in range(15)]}
    monkeypatch.setattr(
        tools.requests, "post", lambda *a, **k: FakeResponse(body={"results": [page], "failed_results": []})
    )
    result = call("read_page", url="https://shop.example/p", query="price of the black variant")
    assert len(result["content"]) == tools.PAGE_CHARS
    assert len(result["images"]) == tools.MAX_IMAGES


def test_read_page_reports_pages_that_cannot_be_read(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    body = {"results": [], "failed_results": [{"url": "https://x.example", "error": "Failed to fetch url"}]}
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(body=body))
    assert "Failed to fetch url" in call("read_page", url="https://x.example", query="price")["error"]


# ---------- check_image ----------

class FakeOpenAI:
    """Stands in for openai.OpenAI; remembers how it was used."""

    constructor_args: dict = {}
    request: dict = {}
    failure: Exception | None = None

    def __init__(self, **kwargs):
        FakeOpenAI.constructor_args = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeOpenAI.request = kwargs
        if FakeOpenAI.failure:
            raise FakeOpenAI.failure
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Yes, it is black."))])


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.failure = None
    monkeypatch.setattr(tools, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def test_check_image_rejects_non_http_url():
    assert "http" in call("check_image", image_url="ftp://x/y.png", question="What is it?")["error"]


def test_check_image_sends_url_and_question_to_vision_model(fake_openai):
    result = call("check_image", image_url="https://img.example/a.jpg", question="Is the laptop black?")

    assert result == {"image_url": "https://img.example/a.jpg", "answer": "Yes, it is black."}
    assert fake_openai.constructor_args["timeout"] == 30
    content = fake_openai.request["messages"][0]["content"]
    assert "Is the laptop black?" in content[0]["text"]
    assert content[1]["image_url"]["url"] == "https://img.example/a.jpg"


def test_check_image_reports_openai_failures(fake_openai):
    fake_openai.failure = TimeoutError("Request timed out.")
    assert "timed out" in call("check_image", image_url="https://img.example/a.jpg", question="Is it black?")["error"]
