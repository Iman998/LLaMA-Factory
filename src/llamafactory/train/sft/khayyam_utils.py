# khayyam_utils.py
from __future__ import annotations
import re
from collections import defaultdict
from typing import Sequence, Mapping, Any

# ─── Static prompt ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You will receive one question followed by four options, only one of which is correct.
Your task is to give the necessary explanation for the answer and, at the end, state
the correct option in the same format shown in the example.

Example

Sample input:
Question: What is the capital of Iran?
1) Tehran
2) Mashhad
3) Isfahan
4) Tabriz

Sample output:
Answer: The capital of Iran is Tehran.
Option ****1**** is correct.
""".strip()

# ─── Regex helper ─────────────────────────────────────────────────────────────
_PATTERNS: list[re.Pattern] = [
    re.compile(r"\*{4}\s*([1-4])\s*\*{4}"),     # ****2****
    re.compile(r"Option\s+\*?([1-4])\*?"),       # Option 2  or Option **2**
    re.compile(r"[پپ]اسخ[:：]?\s*([1-4])\)"),    # پاسخ: 2)
    re.compile(r"\b([1-4])\s*\)"),              # bare "2)"
]

def extract_between_asterisks(text: str) -> str | None:
    """
    Return the chosen option (1-4) or None if not found.
    Tries multiple patterns in order of strictness.
    """
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None

# ─── Prompt builder (used only for preparing the Khayyam dataset) ─────────────
def build_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question] + [f"{i+1}) {c}" for i, c in enumerate(choices)]
    return "\n".join(lines)



# ─── Accuracy metric (unchanged) ──────────────────────────────────────────────
def khayyam_choice_metrics(
    preds_txt : Sequence[str],
    keys_true : Sequence[int],
    levels    : Sequence[Any] | None = None,
    prefix    : str = "khayyam",
) -> Mapping[str, float]:
    """
    Compute accuracy for the Khayyam dataset.
    The loop safely stops at the shorter of the two inputs, so a stray
    prediction batch can never raise IndexError.
    """
    total = correct = 0
    per_tot, per_ok = defaultdict(int), defaultdict(int)

    for idx, (pred, key) in enumerate(zip(preds_txt, keys_true)):
        chosen = extract_between_asterisks(pred)
        if chosen is None:                       # model didn’t give a choice
            continue
        is_ok = str(chosen) == str(int(key))
        total += 1
        if is_ok:
            correct += 1

        if levels is not None:
            lvl = levels[idx]
            per_tot[lvl] += 1
            if is_ok:
                per_ok[lvl] += 1

    metrics = {f"{prefix}_accuracy": correct / max(total, 1)}
    if levels is not None:
        for lvl in per_tot:
            metrics[f"{prefix}_accuracy_level_{lvl}"] = per_ok[lvl] / max(per_tot[lvl], 1)
    return metrics


def build_prompt(question: str, choices: Sequence[str]) -> str:
    """Full prompt = system template  + user question & options."""
    q_line = f"Question: {question.strip()}"
    choice_lines = [f"{idx + 1}) {c.strip()}" for idx, c in enumerate(choices)]
    return PROMPT_TEMPLATE + "\n\n" + q_line + "\n" + "\n".join(choice_lines)
