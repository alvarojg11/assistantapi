from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from ..schemas import AppliedFinding, AnalyzeRequest, AnalyzeResponse, DecisionThresholds, PretestSummary, StepwiseUpdate, SyndromeModule


TB_UVEITIS_MODULE_ID = "tb_uveitis"
TB_UVEITIS_OBSERVE_THRESHOLD = 0.20
TB_UVEITIS_TREAT_THRESHOLD = 0.60
TB_UVEITIS_MEDIAN_TO_PROBABILITY = {
    1: 0.10,
    2: 0.30,
    3: 0.50,
    4: 0.70,
    5: 0.90,
}
TB_UVEITIS_IQR_TO_CONFIDENCE = {
    0: 0.95,
    1: 0.85,
    2: 0.72,
    3: 0.55,
}
TB_UVEITIS_DEFAULT_CODES = {
    "q2": "n",
    "q3": "ND",
    "q4": "ND",
    "q5": "ND",
}
TB_UVEITIS_GROUPS = (
    ("tbu_phenotype", "q1"),
    ("tbu_endemicity", "q2"),
    ("tbu_tst", "q3"),
    ("tbu_igra", "q4"),
    ("tbu_chest_imaging", "q5"),
)
TB_UVEITIS_ITEM_TO_CODE = {
    "tbu_phenotype_au_first": "auf",
    "tbu_phenotype_au_recurrent": "aur",
    "tbu_phenotype_intermediate": "iu",
    "tbu_phenotype_panuveitis": "pu",
    "tbu_phenotype_rv_active": "rva",
    "tbu_phenotype_rv_inactive": "rvi",
    "tbu_phenotype_choroiditis_serpiginoid": "chs",
    "tbu_phenotype_choroiditis_multifocal": "chm",
    "tbu_phenotype_choroiditis_tuberculoma": "cht",
    "tbu_endemicity_endemic": "e",
    "tbu_endemicity_non_endemic": "n",
    "tbu_tst_positive": "+",
    "tbu_tst_negative": "-",
    "tbu_tst_na": "ND",
    "tbu_igra_positive": "+",
    "tbu_igra_negative": "-",
    "tbu_igra_na": "ND",
    "tbu_chest_imaging_positive": "+",
    "tbu_chest_imaging_negative": "-",
    "tbu_chest_imaging_na": "ND",
}


def _prob_to_odds(probability: float) -> float:
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return p / (1 - p)


@lru_cache(maxsize=1)
def _load_lookup() -> Dict[str, Dict[str, int]]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "tb_uveitis_cots_lookup.json"
    return json.loads(data_path.read_text())


def _selected_item_id_for_group(module: SyndromeModule, findings: Dict[str, str], group: str) -> str | None:
    for item in module.items:
        if item.group != group:
            continue
        if findings.get(item.id) == "present":
            return item.id
    return None


def _scenario_inputs(module: SyndromeModule, findings: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, str], List[str]]:
    selected_item_ids: Dict[str, str] = {}
    selected_codes = dict(TB_UVEITIS_DEFAULT_CODES)
    notes: List[str] = []

    for group_id, code_key in TB_UVEITIS_GROUPS:
        item_id = _selected_item_id_for_group(module, findings, group_id)
        if item_id is None:
            if group_id == "tbu_endemicity":
                notes.append("Endemicity not specified; defaulted to non-endemic for the COTS lookup.")
            continue
        selected_item_ids[group_id] = item_id
        selected_codes[code_key] = TB_UVEITIS_ITEM_TO_CODE[item_id]

    return selected_item_ids, selected_codes, notes


def _lookup_result(codes: Dict[str, str]) -> tuple[int, int]:
    key = "|".join([codes["q1"], codes["q2"], codes["q3"], codes["q4"], codes["q5"]])
    payload = _load_lookup()[key]
    return int(payload["median"]), int(payload["iqr"])


def _recommendation_for_probability(probability: float) -> str:
    if probability >= TB_UVEITIS_TREAT_THRESHOLD:
        return "treat"
    if probability <= TB_UVEITIS_OBSERVE_THRESHOLD:
        return "observe"
    return "test"


def _consensus_summary(iqr: int) -> str:
    if iqr == 0:
        return "absolute consensus"
    if iqr == 1:
        return "moderate consensus"
    if iqr == 2:
        return "weak consensus"
    return "poor consensus"


def _build_stepwise_updates(
    *,
    baseline_probability: float,
    module: SyndromeModule,
    selected_item_ids: Dict[str, str],
    selected_codes: Dict[str, str],
) -> List[StepwiseUpdate]:
    if "tbu_phenotype" not in selected_item_ids:
        return []

    items_by_id = {item.id: item for item in module.items}
    current_codes = dict(TB_UVEITIS_DEFAULT_CODES)
    current_codes["q1"] = TB_UVEITIS_ITEM_TO_CODE[selected_item_ids["tbu_phenotype"]]
    current_probability = TB_UVEITIS_MEDIAN_TO_PROBABILITY[_lookup_result(current_codes)[0]]

    steps = [
        StepwiseUpdate(
            id=selected_item_ids["tbu_phenotype"],
            label=items_by_id[selected_item_ids["tbu_phenotype"]].label,
            state="present",
            lrUsed=max(_prob_to_odds(current_probability) / _prob_to_odds(baseline_probability), 1e-6),
            pAfter=current_probability,
        )
    ]

    for group_id, code_key in TB_UVEITIS_GROUPS[1:]:
        item_id = selected_item_ids.get(group_id)
        if item_id is None:
            continue
        next_code = selected_codes[code_key]
        if next_code == current_codes[code_key]:
            continue
        previous_probability = current_probability
        current_codes[code_key] = next_code
        current_probability = TB_UVEITIS_MEDIAN_TO_PROBABILITY[_lookup_result(current_codes)[0]]
        lr_used = max(_prob_to_odds(current_probability) / _prob_to_odds(previous_probability), 1e-6)
        steps.append(
            StepwiseUpdate(
                id=item_id,
                label=items_by_id[item_id].label,
                state="present",
                lrUsed=lr_used,
                pAfter=current_probability,
            )
        )

    return steps


def _build_applied_findings(steps: List[StepwiseUpdate]) -> List[AppliedFinding]:
    findings = [
        AppliedFinding(
            id=step.id,
            label=step.label,
            state="present",
            lrUsed=step.lr_used,
            impactScore=abs(math.log(max(step.lr_used, 1e-6))),
        )
        for step in steps
    ]
    findings.sort(key=lambda item: item.impact_score, reverse=True)
    return findings


def _build_recommendation_summary(
    *,
    median: int | None,
    iqr: int | None,
    selected_item_ids: Dict[str, str],
) -> tuple[str, List[str]]:
    next_steps: List[str] = []

    if "tbu_phenotype" not in selected_item_ids:
        summary = (
            "The COTS calculator cannot be applied reliably until the ocular phenotype is specified, so this stays an incomplete ophthalmology-stage assessment."
        )
        next_steps.extend(
            [
                "Classify the ocular phenotype explicitly (anterior, intermediate, panuveitis, retinal vasculitis, or choroiditis subtype).",
                "Clarify whether the patient is from a TB-endemic region.",
            ]
        )
        return summary, next_steps

    if "tbu_tst" not in selected_item_ids:
        next_steps.append("Obtain or verify a tuberculin skin test if it has not already been documented.")
    if "tbu_igra" not in selected_item_ids:
        next_steps.append("Obtain or verify an IGRA / QuantiFERON result if available.")
    if "tbu_chest_imaging" not in selected_item_ids:
        next_steps.append("Review chest X-ray or chest CT for healed or active TB stigmata.")

    if median is None or iqr is None:
        return "COTS output is unavailable because the minimum phenotype input is missing.", next_steps

    if median <= 2:
        summary = (
            f"COTS consensus is low for starting antitubercular therapy (median score {median}, IQR {iqr}; {_consensus_summary(iqr)})."
        )
        next_steps.append("Reassess alternative causes of intraocular inflammation before committing to ATT.")
        return summary, next_steps

    if median == 3:
        summary = (
            f"COTS consensus is mixed for starting antitubercular therapy (median score 3, IQR {iqr}; {_consensus_summary(iqr)})."
        )
        next_steps.append("Use the missing systemic TB workup and ophthalmic phenotype details to refine the decision before starting ATT.")
        return summary, next_steps

    summary = (
        f"COTS consensus favors considering antitubercular therapy (median score {median}, IQR {iqr}; {_consensus_summary(iqr)}) once non-TB mimics have been excluded."
    )
    next_steps.append("Corroborate the ocular impression with systemic TB evaluation and specialist review before committing to a full ATT course.")
    return summary, next_steps


def analyze_tb_uveitis(module: SyndromeModule, req: AnalyzeRequest) -> AnalyzeResponse:
    preset = next((preset for preset in module.pretest_presets if preset.id == req.preset_id), None) or module.pretest_presets[0]
    adjusted_pretest = float(req.pretest_probability if req.pretest_probability is not None else preset.p)

    selected_item_ids, selected_codes, notes = _scenario_inputs(module, req.findings)
    thresholds = DecisionThresholds(
        observeProbability=TB_UVEITIS_OBSERVE_THRESHOLD,
        treatProbability=TB_UVEITIS_TREAT_THRESHOLD,
    )

    if "tbu_phenotype" not in selected_item_ids:
        summary, next_steps = _build_recommendation_summary(
            median=None,
            iqr=None,
            selected_item_ids=selected_item_ids,
        )
        response = AnalyzeResponse(
            moduleId=module.id,
            moduleName=module.name,
            pretest=PretestSummary(
                baseProbability=adjusted_pretest,
                adjustedProbability=adjusted_pretest,
                presetId=preset.id,
            ),
            combinedLR=1.0,
            posttestProbability=adjusted_pretest,
            thresholds=thresholds,
            recommendation="test",
            recommendationSummary=summary,
            recommendedNextSteps=next_steps,
            confidence=0.25,
            appliedFindings=[],
            stepwise=[],
            reasons=[
                "Tuberculous uveitis uses a COTS consensus lookup rather than an independent likelihood-ratio stack.",
                "A phenotype selection is required before the published COTS matrix can be applied.",
                *notes,
            ],
            riskFlags=["no_findings_selected"],
            explanationForUser=None,
        )
        return response

    median, iqr = _lookup_result(selected_codes)
    posttest_probability = TB_UVEITIS_MEDIAN_TO_PROBABILITY[median]
    combined_lr = _prob_to_odds(posttest_probability) / _prob_to_odds(adjusted_pretest)
    recommendation = _recommendation_for_probability(posttest_probability)
    stepwise = _build_stepwise_updates(
        baseline_probability=adjusted_pretest,
        module=module,
        selected_item_ids=selected_item_ids,
        selected_codes=selected_codes,
    )
    applied_findings = _build_applied_findings(stepwise)
    recommendation_summary, next_steps = _build_recommendation_summary(
        median=median,
        iqr=iqr,
        selected_item_ids=selected_item_ids,
    )

    risk_flags: List[str] = []
    if not applied_findings:
        risk_flags.append("no_findings_selected")
    if len(applied_findings) == 1:
        risk_flags.append("low_evidence_count")
    if abs(posttest_probability - thresholds.observe_probability) <= 0.03 or abs(posttest_probability - thresholds.treat_probability) <= 0.03:
        risk_flags.append("near_decision_threshold")
    if posttest_probability >= 0.5:
        risk_flags.append("high_posttest_probability")
    if iqr >= 2:
        risk_flags.append("low_consensus")

    response = AnalyzeResponse(
        moduleId=module.id,
        moduleName=module.name,
        pretest=PretestSummary(
            baseProbability=adjusted_pretest,
            adjustedProbability=adjusted_pretest,
            presetId=preset.id,
        ),
        combinedLR=combined_lr,
        posttestProbability=posttest_probability,
        thresholds=thresholds,
        recommendation=recommendation,
        recommendationSummary=recommendation_summary,
        recommendedNextSteps=next_steps,
        confidence=TB_UVEITIS_IQR_TO_CONFIDENCE.get(iqr, 0.55),
        appliedFindings=applied_findings,
        stepwise=stepwise,
        reasons=[
            "Tuberculous uveitis uses the published COTS consensus matrix instead of independent pooled LRs.",
            f"COTS scenario: phenotype {selected_codes['q1']}, endemicity {selected_codes['q2']}, TST {selected_codes['q3']}, IGRA {selected_codes['q4']}, chest imaging {selected_codes['q5']}.",
            f"COTS output: median score {median} and IQR {iqr} ({_consensus_summary(iqr)}).",
            f"Median score {median} was mapped to an approximate ATT-initiation probability midpoint of {int(posttest_probability * 100)}%.",
            "This output estimates expert willingness to initiate ATT, not a microbiologic gold-standard disease probability.",
            *notes,
        ],
        riskFlags=risk_flags,
        explanationForUser=None,
    )
    return response
