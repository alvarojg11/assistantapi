from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app
from app.schemas import ParserTrainingExample


DEFAULT_DATASET = BACKEND_ROOT / "app" / "data" / "parser_eval_cases.json"


@dataclass
class ParserStats:
    parser_name: str
    total: int = 0
    success: int = 0
    module_correct: int = 0
    preset_correct: int = 0
    exact_finding_match: int = 0
    order_match: int = 0
    predicted_findings: int = 0
    expected_findings: int = 0
    true_positive_findings: int = 0
    failures: List[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def _load_cases(path: Path) -> List[ParserTrainingExample]:
    raw = json.loads(path.read_text())
    adapter = TypeAdapter(List[ParserTrainingExample])
    return adapter.validate_python(raw)


def _relative_order_matches(expected_order: List[str], predicted_order: List[str], expected_findings: Dict[str, str]) -> bool:
    if not expected_order:
        expected_order = list(expected_findings.keys())

    if any(fid not in expected_findings for fid in expected_order):
        return False

    filtered = [fid for fid in predicted_order if fid in expected_findings]
    return filtered[: len(expected_order)] == expected_order


def _evaluate_parser(
    *,
    client: TestClient,
    cases: List[ParserTrainingExample],
    parser_name: str,
) -> ParserStats:
    stats = ParserStats(parser_name=parser_name)

    for index, case in enumerate(cases, start=1):
        stats.total += 1
        response = client.post(
            "/v1/analyze-text",
            json={
                "text": case.text,
                "parserStrategy": parser_name,
                "allowFallback": False,
                "runAnalyze": False,
                "includeExplanation": False,
            },
        )

        if response.status_code != 200:
            stats.failures.append(f"Case {index}: HTTP {response.status_code} - {response.text}")
            continue

        body = response.json()
        parsed = body.get("parsedRequest")
        if not parsed:
            stats.failures.append(f"Case {index}: parser returned no parsedRequest")
            continue

        stats.success += 1
        predicted_module = parsed.get("moduleId")
        predicted_preset = parsed.get("presetId")
        predicted_findings = parsed.get("findings", {})
        predicted_order = parsed.get("orderedFindingIds", [])
        expected_findings = case.findings

        if predicted_module == case.module_id:
            stats.module_correct += 1
        else:
            stats.failures.append(
                f"Case {index}: module expected {case.module_id}, got {predicted_module}"
            )

        if predicted_preset == case.preset_id:
            stats.preset_correct += 1
        else:
            stats.failures.append(
                f"Case {index}: preset expected {case.preset_id}, got {predicted_preset}"
            )

        if predicted_findings == expected_findings:
            stats.exact_finding_match += 1
        else:
            stats.failures.append(
                f"Case {index}: findings expected {expected_findings}, got {predicted_findings}"
            )

        if _relative_order_matches(case.ordered_finding_ids, predicted_order, expected_findings):
            stats.order_match += 1
        else:
            stats.failures.append(
                f"Case {index}: order expected {case.ordered_finding_ids}, got {predicted_order}"
            )

        expected_pairs = set(expected_findings.items())
        predicted_pairs = set(predicted_findings.items())
        stats.expected_findings += len(expected_pairs)
        stats.predicted_findings += len(predicted_pairs)
        stats.true_positive_findings += len(expected_pairs & predicted_pairs)

    return stats


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _print_summary(stats: ParserStats, show_failures: bool) -> None:
    precision = (
        stats.true_positive_findings / stats.predicted_findings if stats.predicted_findings else 0.0
    )
    recall = stats.true_positive_findings / stats.expected_findings if stats.expected_findings else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"Parser: {stats.parser_name}")
    print(f"  Success: {stats.success}/{stats.total} ({_pct(stats.success, stats.total):.1f}%)")
    print(f"  Module accuracy: {stats.module_correct}/{stats.total} ({_pct(stats.module_correct, stats.total):.1f}%)")
    print(f"  Preset accuracy: {stats.preset_correct}/{stats.total} ({_pct(stats.preset_correct, stats.total):.1f}%)")
    print(
        f"  Exact finding match: {stats.exact_finding_match}/{stats.total} "
        f"({_pct(stats.exact_finding_match, stats.total):.1f}%)"
    )
    print(f"  Expected-order match: {stats.order_match}/{stats.total} ({_pct(stats.order_match, stats.total):.1f}%)")
    print(f"  Finding precision: {precision * 100:.1f}%")
    print(f"  Finding recall: {recall * 100:.1f}%")
    print(f"  Finding F1: {f1 * 100:.1f}%")

    if show_failures and stats.failures:
        print("  Failures:")
        for line in stats.failures:
            print(f"    {line}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ProbID text parser quality against labeled cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to evaluation dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--parsers",
        nargs="+",
        default=["local", "rule"],
        choices=["auto", "local", "rule", "openai"],
        help="Parser strategies to evaluate.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print per-case mismatches.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    cases = _load_cases(args.dataset)
    client = TestClient(app)

    for parser_name in args.parsers:
        stats = _evaluate_parser(client=client, cases=cases, parser_name=parser_name)
        _print_summary(stats, args.show_failures)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
