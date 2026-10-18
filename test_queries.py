"""Five demo queries, one for each scenario required by the homework.

Run: uv run test_queries.py        (all five)
     uv run test_queries.py 4      (only query number 4)

The LLM is not deterministic, so each check only looks at WHICH tools were used, not at the exact text.
"""

import sys

from main import LLMError, run

BROKEN_IMAGE = "https://example.com/this-image-does-not-exist.jpg"

TESTS = [
    {
        "title": "1. Answer without a tool",
        "query": "Jaké je hlavní město Francie?",
        "expected": "The LLM answers directly, no tool is called.",
        "check": lambda calls: len(calls) == 0,
    },
    {
        "title": "2. One tool (convert_currency)",
        "query": "Kolik je 1199 EUR v korunách?",
        "expected": "Calls convert_currency once and answers with the converted amount.",
        "check": lambda calls: [c["name"] for c in calls] == ["convert_currency"],
    },
    {
        "title": "3. A different tool (web_search)",
        "query": "Vyhledej na internetu, kdy byl představen čip Apple M5.",
        "expected": "Calls web_search (and possibly read_page), but no currency conversion.",
        "check": lambda calls: any(c["name"] == "web_search" for c in calls)
        and not any(c["name"] == "convert_currency" for c in calls),
    },
    {
        "title": "4. Several tools for one query",
        "query": "Najdi nejlevnější černý Apple MacBook M5 v Německu, ověř podle obrázku, "
        "že je černý, a cenu převeď do CZK.",
        "expected": "Uses web_search, check_image and convert_currency (read_page optional).",
        "check": lambda calls: {"web_search", "check_image", "convert_currency"} <= {c["name"] for c in calls},
    },
    {
        "title": "5. Tool gets bad input / fails",
        "query": f"Je na tomto obrázku černý notebook? {BROKEN_IMAGE}",
        "expected": "check_image fails (image cannot be downloaded); the LLM reports the problem "
        "instead of crashing or inventing an answer.",
        "check": lambda calls: any(c["name"] == "check_image" and c["error"] for c in calls),
    },
]


def main() -> None:
    selected = [TESTS[int(sys.argv[1]) - 1]] if len(sys.argv) > 1 else TESTS
    passed = 0
    for test in selected:
        print(f"\n{'=' * 70}\n{test['title']}\nQuery:    {test['query']}\nExpected: {test['expected']}\n{'-' * 70}")
        try:
            answer, calls = run(test["query"])
        except LLMError as e:
            print(f"\nResult: FAIL (LLM error: {e})")
            continue
        ok = test["check"](calls) and bool(answer) and not answer.startswith("Stopped after")
        passed += ok
        print(f"\nAnswer: {answer}")
        print(f"Tools used: {[c['name'] + (' (error)' if c['error'] else '') for c in calls] or 'none'}")
        print(f"Result: {'PASS' if ok else 'FAIL'}")
    print(f"\n{'=' * 70}\n{passed}/{len(selected)} passed")


if __name__ == "__main__":
    main()
