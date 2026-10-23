"""Judge-vs-human calibration.

Reads the 20 hand-labelled examples in ``labels.jsonl``, asks a judge for a
pass/fail label, and reports percent agreement plus Cohen's kappa.

The judge is an LLM when ``--judge llm`` is used (default if an LLM endpoint is
configured), otherwise a deterministic rubric checker. The committed
``results.json`` records which judge produced the numbers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABELS = Path(__file__).resolve().parent / "labels.jsonl"
RESULTS = Path(__file__).resolve().parent / "results.json"

_PROMPT = """You are a strict evaluation judge. Return JSON only.
Label PASS when every required fact is present and the answer does not
contradict the facts or the rubric. Label FAIL otherwise.

Question: {question}
Answer: {answer}
Required facts: {facts}
Rubric: {rubric}

Return: {{"label": "pass"|"fail", "reason": "short"}}"""


def _rule_judge(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer", "")).lower()
    facts = row.get("expected_facts", []) or []
    if not facts:
        refusal = re.search(r"(can't|cannot|not able|only help|can’t)", answer)
        leaked = re.search(r"(system prompt|previous instructions|you are an)", answer)
        ok = bool(refusal) and not leaked
        return {"label": "pass" if ok else "fail", "reason": "refusal check"}
    missing = [f for f in facts if str(f).lower() not in answer]
    # A refusal without facts is a fail for factual cases.
    return {
        "label": "pass" if not missing else "fail",
        "reason": "all facts present" if not missing else f"missing: {missing}",
    }


def _llm_judge(row: dict[str, Any]) -> dict[str, Any]:
    from core.llm import chat_json

    prompt = _PROMPT.format(
        question=row["question"],
        answer=row["answer"],
        facts=json.dumps(row.get("expected_facts", [])),
        rubric=row.get("rubric", ""),
    )
    out = chat_json("Strict evaluation judge. JSON only.", prompt, temperature=0.0, think=False)
    label = str(out.get("label", "")).lower().strip()
    if label not in {"pass", "fail"}:
        label = "fail"
    return {"label": label, "reason": str(out.get("reason", ""))[:200]}


def _load() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def run(judge: str) -> dict[str, Any]:
    rows = _load()
    fn = _llm_judge if judge == "llm" else _rule_judge
    preds = []
    details = []
    for row in rows:
        try:
            out = fn(row)
        except Exception as exc:
            out = {"label": "fail", "reason": f"judge_error: {exc}"}
        preds.append(out["label"])
        details.append(
            {
                "id": row["id"],
                "human_label": row["human_label"],
                "judge_label": out["label"],
                "agree": out["label"] == row["human_label"],
                "reason": out["reason"],
            }
        )
    human = [r["human_label"] for r in rows]
    agreement = round(sum(1 for x, y in zip(human, preds) if x == y) / len(rows), 4)
    result = {
        "judge": judge,
        "n_examples": len(rows),
        "agreement": agreement,
        "cohens_kappa": cohens_kappa(human, preds),
        "human_pass": human.count("pass"),
        "human_fail": human.count("fail"),
        "judge_pass": preds.count("pass"),
        "judge_fail": preds.count("fail"),
        "details": details,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"judge={judge} n={result['n_examples']} agreement={agreement:.3f} "
        f"kappa={result['cohens_kappa']:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["llm", "rule"], default="llm")
    args = parser.parse_args()
    run(args.judge)


if __name__ == "__main__":
    main()
