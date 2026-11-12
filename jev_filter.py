"""Optional fast filter: Jev (via OpenRouter) rates how well each search result matches the requirements.

Jev does not write text. For every result we send it the result as a "state" and, for each requirement,
a question with three possible answers. It replies with probabilities. Output tokens are free, so asking
several questions in one call costs the same as asking one.

A search snippet often says nothing about a requirement (for example whether goods are new). That is
not the same as contradicting it, so "not mentioned" is its own answer and is not punished as hard.

The filter is optional: without OPENROUTER_API_KEY, or when Jev fails, results stay unranked.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger("agent.jev")

JEV_URL = os.environ.get("JEV_URL", "https://openrouter.ai/api/alpha/decisions")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
NOT_MENTIONED_VALUE = 0.5  # a requirement the text does not cover counts as "half met"
FLAG_AT = 0.5  # probability from which a requirement is reported as contradicted / not confirmed


def is_available() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _requirement_question(requirement: str) -> dict:
    return {
        "type": "choice",
        "instructions": f"Does this search result match the requirement: {requirement}",
        "criteria": {
            "met": "The title or text clearly states, or strongly implies, that the requirement is met.",
            "contradicted": "The title or text says the opposite of the requirement.",
            "not_mentioned": "The title and text say nothing about the requirement.",
        },
    }


def _rate(result: dict, requirements: list[str]) -> list[dict]:
    """One Jev call for one search result: probabilities of met / contradicted / not_mentioned per requirement."""
    response = requests.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}", "Content-Type": "application/json"},
        json={
            "model": JEV_MODEL,
            "state": {"title": result["title"], "url": result["url"], "text": result["content"]},
            "questions": {f"r{i}": _requirement_question(text) for i, text in enumerate(requirements)},
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Jev call failed ({response.status_code}): {response.text[:200]}")
    answers = response.json()["answers"]
    return [answers[f"r{i}"]["probabilities"] for i in range(len(requirements))]


def _try_rate(result: dict, requirements: list[str]) -> list[dict] | None:
    try:
        return _rate(result, requirements)
    except Exception as e:  # one failed rating must not break the search, but it must not go unnoticed either
        log.warning("Jev could not rate '%s': %s: %s", result["title"][:60], type(e).__name__, e)
        return None


def rank_results(results: list[dict], requirements: list[str], top_n: int) -> list[dict]:
    """Return the top_n results that match the requirements best, each with its match score.

    Every returned result gets "match" (0..1), "contradicts" (requirements the text speaks against) and
    "not_confirmed" (requirements the text does not cover). Results Jev failed to rate come last, without
    a score. If Jev could not rate any result, the first top_n results are returned unchanged.
    """
    with ThreadPoolExecutor() as pool:  # the ratings are independent, so they run at the same time
        ratings = list(pool.map(lambda r: _try_rate(r, requirements), results))

    if all(rating is None for rating in ratings):
        log.warning("Jev could not rate any of the %d results, keeping the original order", len(results))
        return results[:top_n]

    rated, unrated = [], []
    for result, rating in zip(results, ratings):
        if rating is None:
            unrated.append(result)  # Jev failed for this one: keep it, but behind all rated results
            continue
        values = [p.get("met", 0) + NOT_MENTIONED_VALUE * p.get("not_mentioned", 0) for p in rating]
        rated.append(
            {
                **result,
                "match": round(sum(values) / len(values), 2),
                "contradicts": [req for req, p in zip(requirements, rating) if p.get("contradicted", 0) >= FLAG_AT],
                "not_confirmed": [req for req, p in zip(requirements, rating) if p.get("not_mentioned", 0) >= FLAG_AT],
            }
        )
    ranking = sorted(rated, key=lambda r: r["match"], reverse=True) + unrated
    for position, r in enumerate(ranking, start=1):
        log.debug(
            "Jev #%d match=%s contradicts=%s not_confirmed=%s | %s",
            position, r.get("match", "n/a"), r.get("contradicts"), r.get("not_confirmed"), r["title"][:70],
        )
    return ranking[:top_n]
