"""Calibration outcome storage and statistics.

Stores (predicted_probability, actual_outcome) pairs per ProbID module,
then computes calibration curves, Brier scores, and bias estimates.

Storage: JSON file at backend/app/data/calibration_outcomes.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import (
    CalibrationBucket,
    CalibrationOutcome,
    CalibrationReport,
    ModuleCalibrationStats,
)

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "calibration_outcomes.json"

# Number of bins for calibration curve
_NUM_BINS = 10


def _load_outcomes() -> List[dict]:
    if not _DATA_PATH.exists():
        return []
    return json.loads(_DATA_PATH.read_text())


def _save_outcomes(outcomes: List[dict]) -> None:
    _DATA_PATH.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False) + "\n")


def record_outcome(outcome: CalibrationOutcome) -> int:
    """Append a new outcome record. Returns total count after insertion."""
    outcomes = _load_outcomes()
    record = outcome.model_dump(by_alias=True)
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
    outcomes.append(record)
    _save_outcomes(outcomes)
    return len(outcomes)


def get_all_outcomes(module_id: Optional[str] = None) -> List[dict]:
    """Return all stored outcomes, optionally filtered by module."""
    outcomes = _load_outcomes()
    if module_id:
        outcomes = [o for o in outcomes if o.get("moduleId") == module_id]
    return outcomes


def delete_outcomes(module_id: Optional[str] = None) -> int:
    """Delete outcomes. If module_id given, delete only that module's. Returns count deleted."""
    outcomes = _load_outcomes()
    if module_id:
        before = len(outcomes)
        outcomes = [o for o in outcomes if o.get("moduleId") != module_id]
        deleted = before - len(outcomes)
    else:
        deleted = len(outcomes)
        outcomes = []
    _save_outcomes(outcomes)
    return deleted


def _brier_score(pairs: List[tuple]) -> float:
    """Brier score: mean of (predicted - actual)^2. Lower is better. Perfect = 0."""
    if not pairs:
        return 0.0
    return sum((p - a) ** 2 for p, a in pairs) / len(pairs)


def _build_calibration_curve(pairs: List[tuple], num_bins: int = _NUM_BINS) -> List[CalibrationBucket]:
    """Build calibration curve by binning predictions and computing observed rates."""
    bins: Dict[int, List[tuple]] = defaultdict(list)
    bin_width = 1.0 / num_bins

    for predicted, actual in pairs:
        bin_idx = min(int(predicted / bin_width), num_bins - 1)
        bins[bin_idx].append((predicted, actual))

    buckets: List[CalibrationBucket] = []
    for i in range(num_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width
        midpoint = (lower + upper) / 2
        bin_pairs = bins.get(i, [])
        if not bin_pairs:
            continue
        pred_mean = sum(p for p, _ in bin_pairs) / len(bin_pairs)
        outcome_true = sum(1 for _, a in bin_pairs if a)
        outcome_false = len(bin_pairs) - outcome_true
        obs_rate = outcome_true / len(bin_pairs)
        buckets.append(
            CalibrationBucket(
                binLower=round(lower, 2),
                binUpper=round(upper, 2),
                binMidpoint=round(midpoint, 2),
                predictedMean=round(pred_mean, 4),
                observedRate=round(obs_rate, 4),
                count=len(bin_pairs),
                outcomeTrue=outcome_true,
                outcomeFalse=outcome_false,
            )
        )
    return buckets


def compute_calibration_report() -> CalibrationReport:
    """Compute full calibration report across all modules."""
    outcomes = _load_outcomes()

    # Group by module
    by_module: Dict[str, List[dict]] = defaultdict(list)
    for o in outcomes:
        by_module[o["moduleId"]].append(o)

    all_pairs: List[tuple] = []
    module_stats: List[ModuleCalibrationStats] = []

    for module_id, mod_outcomes in sorted(by_module.items()):
        pairs = [
            (o["predictedProbability"], 1.0 if o["actualOutcome"] else 0.0)
            for o in mod_outcomes
        ]
        all_pairs.extend(pairs)

        brier = _brier_score(pairs)
        curve = _build_calibration_curve(pairs)
        mean_pred = sum(p for p, _ in pairs) / len(pairs) if pairs else 0
        mean_obs = sum(a for _, a in pairs) / len(pairs) if pairs else 0

        # Overall accuracy: predicted ≥0.5 → True, <0.5 → False, compare to actual
        correct = sum(
            1 for p, a in pairs
            if (p >= 0.5 and a >= 0.5) or (p < 0.5 and a < 0.5)
        )
        accuracy = correct / len(pairs) if pairs else 0

        module_name = mod_outcomes[0].get("moduleName", module_id)

        module_stats.append(
            ModuleCalibrationStats(
                moduleId=module_id,
                moduleName=module_name,
                totalOutcomes=len(pairs),
                overallAccuracy=round(accuracy, 4),
                brierScore=round(brier, 4),
                calibrationCurve=curve,
                meanPredicted=round(mean_pred, 4),
                meanObserved=round(mean_obs, 4),
                overconfidenceBias=round(mean_pred - mean_obs, 4),
            )
        )

    overall_brier = _brier_score(all_pairs) if all_pairs else 0

    return CalibrationReport(
        totalOutcomes=len(all_pairs),
        modules=module_stats,
        overallBrierScore=round(overall_brier, 4),
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
