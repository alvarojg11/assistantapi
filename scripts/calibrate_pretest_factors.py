from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pretest_factors import (
    PretestFactorSpec,
    get_pretest_factor_tuning,
    resolve_pretest_factor_specs,
)
from app.schemas import SyndromeModule
from app.services.module_store import InMemoryModuleStore


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _applied_multiplier(module_id: str, weights: Iterable[float]) -> tuple[float, float, bool]:
    raw = 1.0
    for weight in weights:
        raw *= weight
    tuning = get_pretest_factor_tuning(module_id)
    applied = clamp(pow(raw, tuning.shrink_exponent), 1.0, tuning.max_multiplier)
    capped = raw > 1.0 and abs(applied - tuning.max_multiplier) < 1e-9
    return raw, applied, capped


def _fmt_multiplier(value: float) -> str:
    return f"{value:.2f}x"


def _load_modules() -> List[SyndromeModule]:
    store = InMemoryModuleStore()
    summaries = store.list_summaries()
    modules: List[SyndromeModule] = []
    for summary in summaries:
        module = store.get(summary.id)
        if module is not None:
            modules.append(module)
    return modules


def _print_factor_rows(module: SyndromeModule, specs: Sequence[PretestFactorSpec], *, top: int, show_all: bool) -> None:
    rows = []
    for spec in specs:
        raw, applied, capped = _applied_multiplier(module.id, [spec.weight])
        rows.append((applied, raw, capped, spec))
    rows.sort(key=lambda row: (-row[0], -row[1], row[3].label.lower()))

    visible = rows if show_all else rows[:top]
    print("  Factors:")
    for applied, raw, capped, spec in visible:
        cap_note = " (capped)" if capped else ""
        group_note = f" [{spec.context_group}]" if spec.context_group else ""
        print(
            f"    {spec.id}: {spec.label}{group_note} | "
            f"raw {_fmt_multiplier(raw)} -> applied {_fmt_multiplier(applied)}{cap_note}"
        )
        print(f"      Source: {spec.source_note}")
    hidden = len(rows) - len(visible)
    if hidden > 0:
        print(f"    ... {hidden} more factor(s). Use --show-all to print everything.")


def _print_module_report(
    module: SyndromeModule,
    *,
    top: int,
    show_all: bool,
    selected_ids: Sequence[str],
) -> int:
    specs = resolve_pretest_factor_specs(module)
    if not specs:
        print(f"{module.id} ({module.name})")
        print("  No configured pretest factors.\n")
        return 0

    tuning = get_pretest_factor_tuning(module.id)
    all_raw, all_applied, all_capped = _applied_multiplier(module.id, [spec.weight for spec in specs])
    strongest = max(specs, key=lambda spec: _applied_multiplier(module.id, [spec.weight])[1])

    print(f"{module.id} ({module.name})")
    print(
        f"  Tuning: shrink_exponent={tuning.shrink_exponent:.2f}, "
        f"max_multiplier={tuning.max_multiplier:.2f}x"
    )
    print(
        f"  Coverage: {len(specs)} factor(s), strongest solo effect {strongest.label} "
        f"({_fmt_multiplier(_applied_multiplier(module.id, [strongest.weight])[1])})"
    )
    print(
        f"  Full-stack theoretical max: raw {_fmt_multiplier(all_raw)} -> "
        f"applied {_fmt_multiplier(all_applied)}{' (capped)' if all_capped else ''}"
    )
    _print_factor_rows(module, specs, top=top, show_all=show_all)

    if selected_ids:
        by_id = {spec.id: spec for spec in specs}
        missing = [factor_id for factor_id in selected_ids if factor_id not in by_id]
        if missing:
            print(f"  Selected: unknown factor id(s): {', '.join(missing)}\n")
            return 1
        selected = [by_id[factor_id] for factor_id in selected_ids]
        raw, applied, capped = _applied_multiplier(module.id, [spec.weight for spec in selected])
        print(
            f"  Selected ({', '.join(spec.id for spec in selected)}): "
            f"raw {_fmt_multiplier(raw)} -> applied {_fmt_multiplier(applied)}"
            f"{' (capped)' if capped else ''}"
        )
    print()
    return 0


def _parse_selected(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and calibrate baseline pretest-factor weights using live backend shrink/cap math."
    )
    parser.add_argument(
        "--module",
        help="Limit output to one module id (for example: cap, vap, endo).",
    )
    parser.add_argument(
        "--selected",
        help="Comma-separated factor ids to evaluate as a combination for the chosen module.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many strongest factors to show per module when --show-all is not set.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every configured factor for each reported module.",
    )
    args = parser.parse_args()

    modules = _load_modules()
    modules_by_id = {module.id: module for module in modules}
    selected_ids = _parse_selected(args.selected)

    if selected_ids and not args.module:
        raise SystemExit("--selected requires --module.")

    if args.module:
        module = modules_by_id.get(args.module)
        if module is None:
            available = ", ".join(sorted(modules_by_id))
            raise SystemExit(f"Unknown module '{args.module}'. Available: {available}")
        return _print_module_report(
            module,
            top=max(args.top, 1),
            show_all=args.show_all,
            selected_ids=selected_ids,
        )

    exit_code = 0
    for module in modules:
        exit_code = max(
            exit_code,
            _print_module_report(
                module,
                top=max(args.top, 1),
                show_all=args.show_all,
                selected_ids=(),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
