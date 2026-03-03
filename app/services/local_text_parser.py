from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pydantic import TypeAdapter

from ..schemas import AnalyzeRequest, ParserTrainingExample
from .module_store import InMemoryModuleStore
from .text_parser import PRESET_HINT_ALIASES, ParseTextResult, _preset_score, normalize, parse_text_to_request, summarize_parsed_request


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "the",
    "to",
    "with",
    "without",
}

MAX_NEIGHBORS = 5
MIN_EXAMPLE_SCORE = 2.0
MIN_FINDING_SCORE = 2.4


class LocalParserError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexedExample:
    example: ParserTrainingExample
    tokens: frozenset[str]
    normalized_text: str


def _data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "parser_training_examples.json"


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token for token in normalize(text).split() if len(token) >= 2 and token not in STOPWORDS)


def _load_examples() -> Tuple[IndexedExample, ...]:
    path = _data_path()
    if not path.exists():
        return ()

    raw = json.loads(path.read_text())
    adapter = TypeAdapter(List[ParserTrainingExample])
    parsed = adapter.validate_python(raw)
    return tuple(
        IndexedExample(
            example=example,
            tokens=_tokenize(example.text),
            normalized_text=normalize(example.text),
        )
        for example in parsed
    )


def _score_example(
    *,
    query_tokens: frozenset[str],
    query_text: str,
    indexed: IndexedExample,
    module_hint: str | None,
    preset_hint: str | None,
) -> float:
    if not query_tokens:
        return 0.0

    overlap = query_tokens & indexed.tokens
    if not overlap:
        return 0.0

    overlap_count = float(len(overlap))
    precision = overlap_count / max(len(indexed.tokens), 1)
    recall = overlap_count / len(query_tokens)
    phrase_bonus = 0.0
    if indexed.normalized_text and indexed.normalized_text in query_text:
        phrase_bonus = 1.0

    score = overlap_count + precision + recall + phrase_bonus
    if module_hint and indexed.example.module_id == module_hint:
        score += 1.0
    if preset_hint and indexed.example.preset_id == preset_hint:
        score += 0.5
    return score


def _rank_examples(
    *,
    text: str,
    module_hint: str | None,
    preset_hint: str | None,
) -> List[Tuple[IndexedExample, float]]:
    query_text = normalize(text)
    query_tokens = _tokenize(text)
    if not query_tokens:
        raise LocalParserError("Local parser could not find enough signal words in the text.")

    scored: List[Tuple[IndexedExample, float]] = []
    for indexed in _load_examples():
        score = _score_example(
            query_tokens=query_tokens,
            query_text=query_text,
            indexed=indexed,
            module_hint=module_hint,
            preset_hint=preset_hint,
        )
        if score > 0:
            scored.append((indexed, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored or scored[0][1] < MIN_EXAMPLE_SCORE:
        raise LocalParserError("Local parser did not find a close enough match in the training examples.")

    score_cutoff = max(MIN_EXAMPLE_SCORE, scored[0][1] * 0.55)
    return [item for item in scored[:MAX_NEIGHBORS] if item[1] >= score_cutoff]


def _weighted_choice(candidates: Iterable[Tuple[str, float]]) -> Tuple[str | None, List[Tuple[str, float]]]:
    totals: Dict[str, float] = {}
    for key, score in candidates:
        if not key:
            continue
        totals[key] = totals.get(key, 0.0) + score
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, []
    return ranked[0][0], ranked


def _choose_preset(
    *,
    module,
    text: str,
    ranked_examples: List[Tuple[IndexedExample, float]],
    preset_hint: str | None,
) -> Tuple[str | None, List[Tuple[str, float]]]:
    if preset_hint and any(p.id == preset_hint for p in module.pretest_presets):
        return preset_hint, [(preset_hint, 1.0)]

    text_norm = normalize(text)
    if module.id == "vap":
        if any(alias in text_norm for alias in PRESET_HINT_ALIASES["vap_ge5d"]):
            ge5d = next((p.id for p in module.pretest_presets if "ge5d" in p.id), None)
            if ge5d:
                return ge5d, [(ge5d, 999.0)]
        if any(alias in text_norm for alias in PRESET_HINT_ALIASES["vap_gt48h"]):
            gt48h = next((p.id for p in module.pretest_presets if "gt48h" in p.id), None)
            if gt48h:
                return gt48h, [(gt48h, 999.0)]

    scores: Dict[str, float] = {}
    for preset in module.pretest_presets:
        scores[preset.id] = float(_preset_score(text_norm, module, preset))

    for indexed, score in ranked_examples:
        if indexed.example.module_id != module.id:
            continue
        if not indexed.example.preset_id:
            continue
        scores[indexed.example.preset_id] = scores.get(indexed.example.preset_id, 0.0) + score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, []
    return ranked[0][0], ranked


def _ordered_example_ids(example: ParserTrainingExample) -> List[str]:
    if example.ordered_finding_ids:
        return example.ordered_finding_ids
    return list(example.findings.keys())


def _predict_findings(
    *,
    module_id: str,
    preset_id: str | None,
    ranked_examples: List[Tuple[IndexedExample, float]],
    valid_item_ids: set[str],
) -> Tuple[Dict[str, str], List[str]]:
    state_scores: Dict[str, Dict[str, float]] = {}
    order_scores: Dict[str, float] = {}
    filtered_examples = [
        (indexed, score)
        for indexed, score in ranked_examples
        if indexed.example.module_id == module_id and (preset_id is None or indexed.example.preset_id == preset_id)
    ]
    if not filtered_examples:
        filtered_examples = [
            (indexed, score) for indexed, score in ranked_examples if indexed.example.module_id == module_id
        ]

    for indexed, score in filtered_examples:
        example = indexed.example

        ordered_ids = _ordered_example_ids(example)
        for position, finding_id in enumerate(ordered_ids):
            if finding_id in valid_item_ids:
                order_scores[finding_id] = order_scores.get(finding_id, 0.0) + max(score - (position * 0.15), 0.1)

        for finding_id, state in example.findings.items():
            if finding_id not in valid_item_ids:
                continue
            bucket = state_scores.setdefault(finding_id, {})
            bucket[state] = bucket.get(state, 0.0) + score

    predictions: Dict[str, str] = {}
    confidence: Dict[str, float] = {}
    for finding_id, bucket in state_scores.items():
        ranked_states = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        best_state, best_score = ranked_states[0]
        second_score = ranked_states[1][1] if len(ranked_states) > 1 else 0.0
        if best_score < MIN_FINDING_SCORE:
            continue
        if second_score and best_score < (second_score * 1.15):
            continue
        predictions[finding_id] = best_state
        confidence[finding_id] = best_score - second_score

    ordered_ids = sorted(
        predictions.keys(),
        key=lambda finding_id: (-order_scores.get(finding_id, 0.0), -confidence.get(finding_id, 0.0), finding_id),
    )
    return predictions, ordered_ids


def parse_text_with_local_model(
    *,
    store: InMemoryModuleStore,
    text: str,
    module_hint: str | None = None,
    preset_hint: str | None = None,
    include_explanation: bool = True,
) -> ParseTextResult:
    ranked_examples = _rank_examples(text=text, module_hint=module_hint, preset_hint=preset_hint)
    warnings: List[str] = []

    if module_hint:
        module = store.get(module_hint)
        if module is None:
            raise LocalParserError(f"Unknown moduleHint '{module_hint}'.")
    else:
        module_id, ranked_modules = _weighted_choice(
            (indexed.example.module_id or "", score) for indexed, score in ranked_examples
        )
        if module_id is None:
            raise LocalParserError("Training examples did not resolve to a known module.")
        module = store.get(module_id)
        if module is None:
            raise LocalParserError(f"Training data selected unknown module '{module_id}'.")
        if len(ranked_modules) > 1 and ranked_modules[1][1] >= (ranked_modules[0][1] * 0.85):
            warnings.append(f"Training examples were split across modules; selected '{module_id}'.")

    if preset_hint and any(p.id == preset_hint for p in module.pretest_presets):
        preset_id = preset_hint
        ranked_presets = [(preset_hint, 1.0)]
    else:
        preset_id, ranked_presets = _choose_preset(
            module=module,
            text=text,
            ranked_examples=ranked_examples,
            preset_hint=preset_hint,
        )
        if preset_hint and preset_id != preset_hint:
            warnings.append(f"presetHint '{preset_hint}' is not valid for module '{module.id}', using training examples.")
        if len(ranked_presets) > 1 and ranked_presets[1][1] >= (ranked_presets[0][1] * 0.9):
            warnings.append(f"Training examples matched multiple presets; selected '{preset_id}'.")

    rule_result = parse_text_to_request(
        store=store,
        text=text,
        module_hint=module.id,
        preset_hint=preset_id,
        include_explanation=include_explanation,
    )
    rule_request = rule_result.parsed_request

    rule_findings = dict(rule_request.findings) if rule_request is not None else {}
    merged_findings = dict(rule_findings)

    ordered_ids: List[str] = []
    seen: set[str] = set()
    if rule_request is not None:
        for finding_id in rule_request.ordered_finding_ids:
            if finding_id in merged_findings and finding_id not in seen:
                ordered_ids.append(finding_id)
                seen.add(finding_id)
    for finding_id in merged_findings:
        if finding_id not in seen:
            ordered_ids.append(finding_id)
            seen.add(finding_id)

    parsed_request = AnalyzeRequest(
        moduleId=module.id,
        presetId=preset_id or (rule_request.preset_id if rule_request is not None else None),
        findings=merged_findings,
        orderedFindingIds=ordered_ids,
        includeExplanation=include_explanation,
    )

    understood, summary_warnings, requires_confirmation = summarize_parsed_request(store, parsed_request)
    warnings.extend(summary_warnings)

    if rule_request is None:
        warnings.append("Rule-based extraction could not anchor this text, so the local parser relied on training examples.")
        requires_confirmation = True

    return ParseTextResult(
        parsed_request=parsed_request,
        understood=understood,
        warnings=[*warnings, *rule_result.warnings],
        requires_confirmation=requires_confirmation or rule_result.requires_confirmation,
        parser_name="local-trained-v1",
    )
