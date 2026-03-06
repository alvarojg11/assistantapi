from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from fastapi.testclient import TestClient


@dataclass
class EvalStats:
    total: int = 0
    success: int = 0
    parsed_checks: int = 0
    parsed_passes: int = 0
    analysis_checks: int = 0
    analysis_passes: int = 0
    provisional_checks: int = 0
    provisional_passes: int = 0
    assistant_checks: int = 0
    assistant_passes: int = 0
    failures: List[str] = field(default_factory=list)


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def _norm(value: str) -> str:
    return (
        value.lower()
        .replace("β", "beta")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
    )


def _contains_text(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _contains_in_list(values: List[str], expected: str) -> bool:
    return any(_contains_text(value, expected) for value in values)


def _check_equal(actual: Any, expected: Any, label: str, case_id: str, stats: EvalStats) -> None:
    stats.parsed_checks += 1
    if actual == expected:
        stats.parsed_passes += 1
        return
    stats.failures.append(f"{case_id}: {label} expected {expected!r}, got {actual!r}")


def _check_subset_mapping(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    label: str,
    case_id: str,
    stats: EvalStats,
    bucket: str,
) -> None:
    for key, value in expected.items():
        if bucket == "parsed":
            stats.parsed_checks += 1
            if actual.get(key) == value:
                stats.parsed_passes += 1
            else:
                stats.failures.append(f"{case_id}: {label}.{key} expected {value!r}, got {actual.get(key)!r}")
        elif bucket == "analysis":
            stats.analysis_checks += 1
            if actual.get(key) == value:
                stats.analysis_passes += 1
            else:
                stats.failures.append(f"{case_id}: {label}.{key} expected {value!r}, got {actual.get(key)!r}")
        else:
            stats.provisional_checks += 1
            if actual.get(key) == value:
                stats.provisional_passes += 1
            else:
                stats.failures.append(f"{case_id}: {label}.{key} expected {value!r}, got {actual.get(key)!r}")


def _check_contains_list(
    actual: List[str],
    expected: List[str],
    label: str,
    case_id: str,
    stats: EvalStats,
    bucket: str,
) -> None:
    for item in expected:
        if bucket == "parsed":
            stats.parsed_checks += 1
            if _contains_in_list(actual, item):
                stats.parsed_passes += 1
            else:
                stats.failures.append(f"{case_id}: expected {label} to include {item!r}, got {actual!r}")
        elif bucket == "analysis":
            stats.analysis_checks += 1
            if _contains_in_list(actual, item):
                stats.analysis_passes += 1
            else:
                stats.failures.append(f"{case_id}: expected {label} to include {item!r}, got {actual!r}")
        elif bucket == "provisional":
            stats.provisional_checks += 1
            if _contains_in_list(actual, item):
                stats.provisional_passes += 1
            else:
                stats.failures.append(f"{case_id}: expected {label} to include {item!r}, got {actual!r}")
        else:
            stats.assistant_checks += 1
            if _contains_text(actual if isinstance(actual, str) else " ".join(actual), item):
                stats.assistant_passes += 1
            else:
                stats.failures.append(f"{case_id}: expected {label} to include {item!r}")


def evaluate_mechid_case(
    *,
    client: TestClient,
    case: Dict[str, Any],
    stats: EvalStats,
    check_assistant: bool,
) -> None:
    case_id = case.get("id", f"case_{stats.total + 1}")
    stats.total += 1

    parser_strategy = case.get("parserStrategy", "rule")
    response = client.post(
        "/v1/mechid/analyze-text",
        json={
            "text": case["text"],
            "parserStrategy": parser_strategy,
            "allowFallback": False,
        },
    )
    if response.status_code != 200:
        stats.failures.append(f"{case_id}: mechid endpoint returned HTTP {response.status_code}")
        return

    body = response.json()
    stats.success += 1

    if "expectedRequiresConfirmation" in case:
        stats.parsed_checks += 1
        if body.get("requiresConfirmation") == case["expectedRequiresConfirmation"]:
            stats.parsed_passes += 1
        else:
            stats.failures.append(
                f"{case_id}: requiresConfirmation expected {case['expectedRequiresConfirmation']!r}, got {body.get('requiresConfirmation')!r}"
            )

    parsed = body.get("parsedRequest") or {}
    parsed_expect = case.get("expectedParsed", {})
    if parsed_expect:
        if "organism" in parsed_expect:
            _check_equal(parsed.get("organism"), parsed_expect["organism"], "organism", case_id, stats)
        if "syndrome" in parsed_expect:
            _check_equal(parsed.get("txContext", {}).get("syndrome"), parsed_expect["syndrome"], "syndrome", case_id, stats)
        if "severity" in parsed_expect:
            _check_equal(parsed.get("txContext", {}).get("severity"), parsed_expect["severity"], "severity", case_id, stats)
        if "focusDetail" in parsed_expect:
            _check_equal(parsed.get("txContext", {}).get("focusDetail"), parsed_expect["focusDetail"], "focusDetail", case_id, stats)
        if "oralPreference" in parsed_expect:
            _check_equal(
                parsed.get("txContext", {}).get("oralPreference"),
                parsed_expect["oralPreference"],
                "oralPreference",
                case_id,
                stats,
            )
        if "carbapenemaseResult" in parsed_expect:
            _check_equal(
                parsed.get("txContext", {}).get("carbapenemaseResult"),
                parsed_expect["carbapenemaseResult"],
                "carbapenemaseResult",
                case_id,
                stats,
            )
        if "carbapenemaseClass" in parsed_expect:
            _check_equal(
                parsed.get("txContext", {}).get("carbapenemaseClass"),
                parsed_expect["carbapenemaseClass"],
                "carbapenemaseClass",
                case_id,
                stats,
            )
        if "mentionedOrganismsContains" in parsed_expect:
            _check_contains_list(
                parsed.get("mentionedOrganisms", []),
                parsed_expect["mentionedOrganismsContains"],
                "mentionedOrganisms",
                case_id,
                stats,
                "parsed",
            )
        if "resistancePhenotypesContains" in parsed_expect:
            _check_contains_list(
                parsed.get("resistancePhenotypes", []),
                parsed_expect["resistancePhenotypesContains"],
                "resistancePhenotypes",
                case_id,
                stats,
                "parsed",
            )
        if "susceptibilityResultsSubset" in parsed_expect:
            _check_subset_mapping(
                parsed.get("susceptibilityResults", {}),
                parsed_expect["susceptibilityResultsSubset"],
                "susceptibilityResults",
                case_id,
                stats,
                "parsed",
            )

    analysis = body.get("analysis")
    expected_analysis_present = case.get("expectedAnalysisPresent")
    if expected_analysis_present is not None:
        stats.analysis_checks += 1
        if bool(analysis) == bool(expected_analysis_present):
            stats.analysis_passes += 1
        else:
            stats.failures.append(f"{case_id}: expected analysis presence {expected_analysis_present!r}, got {bool(analysis)!r}")

    if analysis:
        if "expectedMechanismsContains" in case:
            _check_contains_list(
                analysis.get("mechanisms", []),
                case["expectedMechanismsContains"],
                "mechanisms",
                case_id,
                stats,
                "analysis",
            )
        if "expectedTherapyNotesContains" in case:
            _check_contains_list(
                analysis.get("therapyNotes", []),
                case["expectedTherapyNotesContains"],
                "therapyNotes",
                case_id,
                stats,
                "analysis",
            )
        if "expectedFinalResultsSubset" in case:
            _check_subset_mapping(
                analysis.get("finalResults", {}),
                case["expectedFinalResultsSubset"],
                "finalResults",
                case_id,
                stats,
                "analysis",
            )

    provisional = body.get("provisionalAdvice")
    expected_provisional_present = case.get("expectedProvisionalPresent")
    if expected_provisional_present is not None:
        stats.provisional_checks += 1
        if bool(provisional) == bool(expected_provisional_present):
            stats.provisional_passes += 1
        else:
            stats.failures.append(
                f"{case_id}: expected provisional advice presence {expected_provisional_present!r}, got {bool(provisional)!r}"
            )

    if provisional:
        if "expectedRecommendedOptionsContains" in case:
            _check_contains_list(
                provisional.get("recommendedOptions", []),
                case["expectedRecommendedOptionsContains"],
                "recommendedOptions",
                case_id,
                stats,
                "provisional",
            )
        if "expectedOralOptionsContains" in case:
            _check_contains_list(
                provisional.get("oralOptions", []),
                case["expectedOralOptionsContains"],
                "oralOptions",
                case_id,
                stats,
                "provisional",
            )
        if "expectedMissingSusceptibilitiesContains" in case:
            _check_contains_list(
                provisional.get("missingSusceptibilities", []),
                case["expectedMissingSusceptibilitiesContains"],
                "missingSusceptibilities",
                case_id,
                stats,
                "provisional",
            )

    if not check_assistant:
        return

    if "assistantReviewContains" not in case and "assistantFinalContains" not in case:
        return

    review_response = client.post(
        "/v1/assistant/turn",
        json={
            "message": case["text"],
            "parserStrategy": parser_strategy,
            "allowFallback": False,
        },
    )
    if review_response.status_code != 200:
        stats.failures.append(f"{case_id}: assistant review returned HTTP {review_response.status_code}")
        return

    review_body = review_response.json()
    if review_body.get("state", {}).get("workflow") != "mechid":
        stats.failures.append(f"{case_id}: assistant did not enter mechid workflow")
        return

    if "assistantReviewContains" in case:
        _check_contains_list(
            [review_body.get("assistantMessage", "")],
            case["assistantReviewContains"],
            "assistant review message",
            case_id,
            stats,
            "assistant",
        )

    if "assistantFinalContains" not in case:
        return

    final_response = client.post(
        "/v1/assistant/turn",
        json={
            "state": review_body.get("state"),
            "selection": "run_assessment",
        },
    )
    if final_response.status_code != 200:
        stats.failures.append(f"{case_id}: assistant final returned HTTP {final_response.status_code}")
        return

    final_body = final_response.json()
    _check_contains_list(
        [final_body.get("assistantMessage", "")],
        case["assistantFinalContains"],
        "assistant final message",
        case_id,
        stats,
        "assistant",
    )
