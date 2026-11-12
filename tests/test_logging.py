"""Tests of the logging: what goes to the console, what goes to the file, and what the app logs on the way."""

import json
import logging
from types import SimpleNamespace

import pytest
import requests

import agent_logging
import jev_filter
import main
import tools


@pytest.fixture
def configured_logging(tmp_path, capsys):  # capsys first: the console handler binds to sys.stdout when created
    """Calls the real setup_logging() with a temporary file and cleans up afterwards."""
    log_file = tmp_path / "agent.log"
    agent_logging.setup_logging(log_file)
    yield log_file
    logger = logging.getLogger(agent_logging.LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True


def flush_file_handlers():
    for handler in logging.getLogger(agent_logging.LOGGER_NAME).handlers:
        handler.flush()


# ---------- setup_logging ----------

def test_console_shows_info_and_file_shows_everything(configured_logging, capsys):
    logging.getLogger("agent.main").info("visible progress")
    logging.getLogger("agent.main").debug("technical detail")
    flush_file_handlers()

    console = capsys.readouterr().out
    assert "visible progress" in console and "technical detail" not in console
    assert console.strip() == "visible progress"  # plain message, no timestamp or level

    in_file = configured_logging.read_text(encoding="utf-8")
    assert "visible progress" in in_file and "technical detail" in in_file
    assert "DEBUG" in in_file and "agent.main" in in_file and "MainThread" in in_file  # level, logger and thread


def test_tool_warnings_stay_out_of_the_console_but_reach_the_file(configured_logging, capsys):
    logging.getLogger("agent.tools").warning("web_search failed after 0.10s: boom")
    logging.getLogger("agent.jev").warning("Jev could not rate something")
    flush_file_handlers()

    console = capsys.readouterr().out
    assert "boom" not in console  # already visible in the [step N] output
    assert "Jev could not rate something" in console  # nothing else would tell the user about it
    assert "boom" in configured_logging.read_text(encoding="utf-8")


def test_calling_setup_twice_does_not_duplicate_lines(configured_logging, capsys):
    agent_logging.setup_logging(configured_logging)
    logging.getLogger("agent.main").info("only once")
    assert capsys.readouterr().out.count("only once") == 1


# ---------- what the app logs ----------

def test_failed_tool_is_logged_as_a_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="agent")
    tools.execute_tool("convert_currency", "not json")

    warning = next(r for r in caplog.records if r.name == "agent.tools" and r.levelno == logging.WARNING)
    assert "convert_currency failed" in warning.getMessage() and "not valid JSON" in warning.getMessage()


def test_successful_tool_is_logged_with_its_arguments_and_duration(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="agent")
    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            status_code=200, ok=True, json=lambda: {"rates": {"CZK": 24.0}, "date": "2026-10-16"}, raise_for_status=lambda: None
        ),
    )
    tools.execute_tool("convert_currency", json.dumps({"amount": 10, "from_currency": "EUR", "to_currency": "CZK"}))

    messages = [r.getMessage() for r in caplog.records if r.name == "agent.tools"]
    assert any("called with" in m and '"amount": 10' in m for m in messages)
    assert any("convert_currency succeeded in" in m for m in messages)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_jev_failure_is_logged_instead_of_swallowed(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="agent")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    def fail(*args, **kwargs):
        raise requests.ConnectionError("Jev is down")

    monkeypatch.setattr(jev_filter.requests, "post", fail)
    result = {"title": "Some offer", "url": "https://shop.example", "content": "text"}
    jev_filter.rank_results([result], ["The product is black"], top_n=1)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Jev could not rate 'Some offer'" in w and "Jev is down" in w for w in warnings)
    assert any("could not rate any of the 1 results" in w for w in warnings)


def test_run_logs_steps_results_and_the_final_answer(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="agent")
    replies = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(id="c1", function=SimpleNamespace(name="web_search", arguments='{"query": "m5"}'))
                            ],
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15),
            ),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hotovo.", tool_calls=None))]),
        ]
    )
    monkeypatch.setattr(main, "ask_llm", lambda messages, tool_choice: next(replies))
    monkeypatch.setattr(main, "execute_tool", lambda name, args: {"results": ["x" * 500]})

    main.run("Najdi M5")

    by_level = lambda level: [r.getMessage() for r in caplog.records if r.name == "agent.main" and r.levelno == level]
    info, debug = by_level(logging.INFO), by_level(logging.DEBUG)

    assert "[step 1] 1 tool call(s) in parallel" in info[0]
    assert info[1] == '[step 1] -> web_search({"query": "m5"})'
    assert info[2].startswith('[step 1] <- {"results": ["xxx') and len(info[2]) < 230  # shortened for the console
    assert any("full result of web_search (c1)" in m and "x" * 500 in m for m in debug)  # complete in the file
    assert any("tokens in/out: 120/15" in m for m in debug)
    assert any("final answer: Hotovo." in m for m in debug)
    assert "Query: Najdi M5" in debug[0]


def test_openai_failure_is_logged_with_its_type(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="agent")

    def fail(**kwargs):
        raise TimeoutError("no answer")  # not an OpenAI error: must not be swallowed or relabelled

    monkeypatch.setattr(main.client.chat.completions, "create", fail)
    with pytest.raises(TimeoutError):
        main.ask_llm([{"role": "user", "content": "hi"}], tool_choice="auto")
    assert not [r for r in caplog.records if r.name == "agent.main"]
