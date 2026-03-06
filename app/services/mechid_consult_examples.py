from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ..schemas import MechIDTextAnalyzeResponse


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "mechid_eval_cases.json"


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _query_text(result: MechIDTextAnalyzeResponse) -> str:
    parsed = result.parsed_request
    parts = [result.text]
    if parsed is not None:
        if parsed.organism:
            parts.append(parsed.organism)
        parts.extend(parsed.mentioned_organisms)
        parts.extend(parsed.resistance_phenotypes)
        parts.append(parsed.tx_context.syndrome)
        parts.append(parsed.tx_context.focus_detail)
        parts.append(parsed.tx_context.severity)
    return " ".join(part for part in parts if part and part != "Not specified")


def select_mechid_consult_examples(
    *,
    result: MechIDTextAnalyzeResponse,
    kind: str,
    limit: int = 2,
) -> List[Dict[str, str]]:
    if not DATASET_PATH.exists():
        return []

    key = "assistantFinalTarget" if kind == "final" else "assistantReviewTarget"
    query_tokens = _tokenize(_query_text(result))
    try:
        cases = json.loads(DATASET_PATH.read_text())
    except Exception:
        return []

    scored: List[tuple[int, Dict[str, str]]] = []
    for payload in cases:
        target = str(payload.get(key) or "").strip()
        if not target:
            continue
        text = str(payload.get("text") or "")
        guidance = str(payload.get("assistantGuidance") or "").strip()
        haystack = " ".join([text, guidance, target])
        score = len(query_tokens & _tokenize(haystack))
        if not score:
            continue
        scored.append(
            (
                score,
                {
                    "id": str(payload.get("id") or ""),
                    "text": text,
                    "guidance": guidance,
                    "target": target,
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [item for _, item in scored[:limit]]
