"""Offline tests of the Jev filter: the OpenRouter call is faked, so no key and no internet are needed."""

import pytest
import requests

import jev_filter

REQUIREMENTS = ["The product is black", "The offer is for new goods"]


class FakeResponse:
    def __init__(self, body: dict | None = None, ok: bool = True, status_code: int = 200):
        self._body, self.ok, self.status_code, self.text = body or {}, ok, status_code, "error text"

    def json(self) -> dict:
        return self._body


def jev_body(*probabilities: dict) -> dict:
    """What Jev answers: one choice answer per requirement."""
    return {
        "answers": {
            f"r{i}": {"choice": max(p, key=p.get), "confidence": 0.9, "probabilities": p}
            for i, p in enumerate(probabilities)
        }
    }


MET = {"met": 1.0, "contradicted": 0.0, "not_mentioned": 0.0}
CONTRADICTED = {"met": 0.0, "contradicted": 1.0, "not_mentioned": 0.0}
SILENT = {"met": 0.0, "contradicted": 0.0, "not_mentioned": 1.0}


def result(title: str) -> dict:
    return {"title": title, "url": f"https://shop.example/{title}", "content": "text"}


@pytest.fixture
def jev(monkeypatch):
    """Fake Jev: the answer depends on the title of the result it is asked about."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    answers = {"perfect": (MET, MET), "silent": (MET, SILENT), "wrong": (CONTRADICTED, MET)}
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "payload": json})
        title = json["state"]["title"]
        if title == "broken":
            raise requests.ConnectionError("Jev is down")
        return FakeResponse(jev_body(*answers[title]))

    monkeypatch.setattr(jev_filter.requests, "post", fake_post)
    return calls


def test_is_available_depends_on_the_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    assert jev_filter.is_available()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert not jev_filter.is_available()


def test_results_are_ranked_and_flagged(jev):
    ranked = jev_filter.rank_results([result("wrong"), result("silent"), result("perfect")], REQUIREMENTS, top_n=3)

    assert [r["title"] for r in ranked] == ["perfect", "silent", "wrong"]
    perfect, silent, wrong = ranked
    assert perfect["match"] == 1.0 and perfect["contradicts"] == [] and perfect["not_confirmed"] == []
    assert silent["match"] == 0.75  # (1 + 0.5) / 2: silence is only half a point
    assert silent["not_confirmed"] == ["The offer is for new goods"]
    assert silent["contradicts"] == []  # silence is not a contradiction
    assert wrong["match"] == 0.5  # (0 + 1) / 2
    assert wrong["contradicts"] == ["The product is black"]


def test_only_the_best_results_are_returned(jev):
    ranked = jev_filter.rank_results([result("wrong"), result("perfect"), result("silent")], REQUIREMENTS, top_n=2)
    assert [r["title"] for r in ranked] == ["perfect", "silent"]


def test_one_call_per_result_with_one_question_per_requirement(jev):
    jev_filter.rank_results([result("perfect"), result("silent")], REQUIREMENTS, top_n=2)

    assert len(jev) == 2
    call = jev[0]
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["payload"]["model"] == jev_filter.JEV_MODEL
    assert set(call["payload"]["state"]) == {"title", "url", "text"}
    questions = call["payload"]["questions"]
    assert list(questions) == ["r0", "r1"]
    assert all(q["type"] == "choice" and set(q["criteria"]) == {"met", "contradicted", "not_mentioned"} for q in questions.values())
    assert "The product is black" in questions["r0"]["instructions"]


def test_a_result_that_could_not_be_rated_is_kept_at_the_end(jev):
    ranked = jev_filter.rank_results([result("broken"), result("wrong"), result("perfect")], REQUIREMENTS, top_n=3)

    assert [r["title"] for r in ranked] == ["perfect", "wrong", "broken"]
    assert "match" not in ranked[-1]  # no score invented for it


def test_if_jev_fails_completely_results_stay_as_they_are(jev):
    results = [result("broken"), result("broken"), result("broken")]
    assert jev_filter.rank_results(results, REQUIREMENTS, top_n=2) == results[:2]


def test_http_error_from_jev_counts_as_a_failed_rating(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(jev_filter.requests, "post", lambda *a, **k: FakeResponse(ok=False, status_code=402))
    results = [result("a"), result("b")]
    assert jev_filter.rank_results(results, REQUIREMENTS, top_n=5) == results
