from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402
from app.services.mechid_eval import EvalStats, evaluate_mechid_case, pct  # noqa: E402


DEFAULT_DATASET = BACKEND_ROOT / "app" / "data" / "mechid_eval_cases.json"


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MechID parsing and treatment guidance against labeled cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to MechID evaluation dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--check-assistant",
        action="store_true",
        help="Also evaluate assistant review/final messages. Best used with deterministic narration (for example without OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print per-case failures.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    client = TestClient(app)
    stats = EvalStats()
    for case in _load_cases(args.dataset):
        evaluate_mechid_case(client=client, case=case, stats=stats, check_assistant=args.check_assistant)

    print(f"Cases: {stats.success}/{stats.total} endpoint successes ({pct(stats.success, stats.total):.1f}%)")
    print(f"Parsed checks: {stats.parsed_passes}/{stats.parsed_checks} ({pct(stats.parsed_passes, stats.parsed_checks):.1f}%)")
    print(f"Analysis checks: {stats.analysis_passes}/{stats.analysis_checks} ({pct(stats.analysis_passes, stats.analysis_checks):.1f}%)")
    print(
        f"Provisional checks: {stats.provisional_passes}/{stats.provisional_checks} "
        f"({pct(stats.provisional_passes, stats.provisional_checks):.1f}%)"
    )
    if args.check_assistant:
        print(
            f"Assistant checks: {stats.assistant_passes}/{stats.assistant_checks} "
            f"({pct(stats.assistant_passes, stats.assistant_checks):.1f}%)"
        )

    if stats.failures and args.show_failures:
        print("Failures:")
        for failure in stats.failures:
            print(f"- {failure}")

    return 1 if stats.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
