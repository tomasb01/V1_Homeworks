"""Offline tests of the tool calling loop: the LLM is replaced by a script of prepared replies."""

import json
import threading
from types import SimpleNamespace

import openai
import pytest

import main

httpx2 = pytest.importorskip("httpx2")  # HTTP library used by the openai package, needed to build its exceptions


def tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def reply(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """The part of an OpenAI response that main.py reads."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedLLM:
    """Replaces main.ask_llm: returns prepared replies in order and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, messages, tool_choice):
        self.requests.append({"messages": list(messages), "tool_choice": tool_choice})
        return self.replies[len(self.requests) - 1]


def tool_messages(request: dict) -> list[dict]:
    return [m for m in request["messages"] if m["role"] == "tool"]


# ---------- the loop ----------

def test_answer_without_tool(monkeypatch):
    llm = ScriptedLLM(reply("Paříž"))
    monkeypatch.setattr(main, "ask_llm", llm)

    answer, calls = main.run("Jaké je hlavní město Francie?")

    assert (answer, calls) == ("Paříž", [])
    assert llm.requests[0]["tool_choice"] == "auto"
    assert [m["role"] for m in llm.requests[0]["messages"]] == ["system", "user"]


def test_tool_result_goes_back_to_the_llm(monkeypatch):
    llm = ScriptedLLM(
        reply(tool_calls=[tool_call("c1", "convert_currency", {"amount": 1199})]),
        reply("To je 29 163 Kč."),
    )
    monkeypatch.setattr(main, "ask_llm", llm)
    monkeypatch.setattr(main, "execute_tool", lambda name, args: {"result": 29163.28})

    answer, calls = main.run("Kolik je 1199 EUR v korunách?")

    assert answer == "To je 29 163 Kč."
    assert calls == [{"name": "convert_currency", "error": False}]
    second_request = llm.requests[1]["messages"]
    assert second_request[-2]["role"] == "assistant"  # the LLM's own request stays in the history
    assert second_request[-2]["tool_calls"][0]["id"] == "c1"
    assert second_request[-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"result": 29163.28})}


def test_tool_error_is_sent_back_and_marked(monkeypatch):
    llm = ScriptedLLM(
        reply(tool_calls=[tool_call("c1", "check_image", {"image_url": "https://x.example/a.jpg"})]),
        reply("Obrázek nejde stáhnout."),
    )
    monkeypatch.setattr(main, "ask_llm", llm)
    monkeypatch.setattr(main, "execute_tool", lambda name, args: {"error": "404 not found"})

    answer, calls = main.run("Je tohle černé?")

    assert answer == "Obrázek nejde stáhnout."
    assert calls == [{"name": "check_image", "error": True}]
    assert "404 not found" in tool_messages(llm.requests[1])[0]["content"]  # the LLM gets to react to the error


def test_several_tools_in_one_round_run_in_parallel_and_keep_their_ids(monkeypatch):
    llm = ScriptedLLM(
        reply(tool_calls=[tool_call("c1", "web_search", {}), tool_call("c2", "convert_currency", {})]),
        reply("hotovo"),
    )
    monkeypatch.setattr(main, "ask_llm", llm)

    # Both tools wait for each other. If they ran one after another, the first would time out.
    both_running = threading.Barrier(2, timeout=5)

    def fake_execute_tool(name, arguments):
        both_running.wait()
        return {"tool": name}

    monkeypatch.setattr(main, "execute_tool", fake_execute_tool)

    answer, calls = main.run("dotaz")

    assert answer == "hotovo"
    results = tool_messages(llm.requests[1])
    assert [(m["tool_call_id"], json.loads(m["content"])) for m in results] == [
        ("c1", {"tool": "web_search"}),
        ("c2", {"tool": "convert_currency"}),
    ]
    assert [c["name"] for c in calls] == ["web_search", "convert_currency"]


def test_consecutive_rounds_build_on_each_other(monkeypatch):
    llm = ScriptedLLM(
        reply(tool_calls=[tool_call("c1", "web_search", {})]),
        reply(tool_calls=[tool_call("c2", "convert_currency", {})]),
        reply("hotovo"),
    )
    monkeypatch.setattr(main, "ask_llm", llm)
    monkeypatch.setattr(main, "execute_tool", lambda name, args: {"tool": name})

    answer, calls = main.run("dotaz")

    assert answer == "hotovo"
    assert [c["name"] for c in calls] == ["web_search", "convert_currency"]
    assert len(tool_messages(llm.requests[2])) == 2  # the last request carries both results


def test_last_step_switches_tools_off_and_forces_an_answer(monkeypatch):
    monkeypatch.setattr(main, "MAX_STEPS", 3)
    requests_seen = []

    def stubborn_llm(messages, tool_choice):
        """Asks for a tool again and again, until tools are switched off."""
        requests_seen.append({"messages": list(messages), "tool_choice": tool_choice})
        if tool_choice == "none":
            return reply("Našel jsem jen část, zbytek nelze ověřit.")
        return reply(tool_calls=[tool_call(f"c{len(requests_seen)}", "web_search", {})])

    monkeypatch.setattr(main, "ask_llm", stubborn_llm)
    monkeypatch.setattr(main, "execute_tool", lambda name, args: {"ok": True})

    answer, calls = main.run("dotaz")

    assert answer == "Našel jsem jen část, zbytek nelze ověřit."
    assert [r["tool_choice"] for r in requests_seen] == ["auto", "auto", "none"]
    assert requests_seen[-1]["messages"][-1] == {"role": "system", "content": main.FINAL_STEP_NOTE}
    assert len(calls) == 2


# ---------- failures of the OpenAI call itself ----------

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(error_class, status_code: int):
    return error_class("failed", response=httpx2.Response(status_code, request=REQUEST), body=None)


@pytest.mark.parametrize(
    "failure, expected_message",
    [
        (status_error(openai.AuthenticationError, 401), "API key"),
        (status_error(openai.RateLimitError, 429), "rate limit"),
        (openai.APITimeoutError(request=REQUEST), "in time"),
        (openai.APIConnectionError(request=REQUEST), "internet"),
        (status_error(openai.InternalServerError, 500), "HTTP 500"),
    ],
)
def test_openai_failures_become_readable_errors(monkeypatch, failure, expected_message):
    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(main.client.chat.completions, "create", fail)

    with pytest.raises(main.LLMError, match=expected_message):
        main.ask_llm([{"role": "user", "content": "ahoj"}], tool_choice="auto")
