from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .pretest_factors import get_pretest_factor_spec, get_pretest_factor_tuning
from .schemas import (
    AnalyzeRequest,
    AppliedFinding,
    DecisionThresholds,
    EndoRiskModifiersInput,
    FindingState,
    HarmInputs,
    LRItem,
    ProbIDControlsInput,
    StepwiseUpdate,
    SyndromeModule,
    VAPRiskModifiersInput,
)


EPS = 1e-6

ENDO_SCORE_ITEM_IDS = [
    "endo_virsta_high",
    "endo_virsta_na",
    "endo_denova_high",
    "endo_denova_na",
    "endo_handoc_high",
    "endo_handoc_na",
]

HANDOC_SPECIES_POINTS: Dict[str, int] = {
    "unspecified_other": 0,
    "s_anginosus_group": -1,
    "s_gallolyticus_bovis_group": 1,
    "s_mutans_group": 1,
    "s_sanguinis_group": 1,
    "s_mitis_oralis_group": 0,
    "s_salivarius_group": 0,
}


@dataclass
class HarmEvidence:
    short: str
    url: str | None = None


@dataclass
class HarmDriver:
    label: str
    delta: float
    evidence: HarmEvidence | None = None


@dataclass
class HarmEstimate:
    base_missed_dx: float
    base_unnecessary_tx: float
    base_evidence: HarmEvidence | None
    missed_dx: float
    unnecessary_tx: float
    rationale: List[str]
    missed_dx_drivers: List[HarmDriver]


@dataclass
class ProbIDPreparation:
    analysis_findings: Dict[str, FindingState]
    harm_findings: Dict[str, FindingState]
    effective_pretest_odds_multiplier: float
    notes: List[str]


BASE_HARM_BY_MODULE: Dict[str, dict] = {
    "cap": {
        "missedDx": 10,
        "unnecessaryTx": 3,
        "evidence": {"short": "Metlay et al. ATS/IDSA", "url": "https://doi.org/10.1164/rccm.201908-1581ST"},
    },
    "cdi": {
        "missedDx": 11,
        "unnecessaryTx": 4,
        "evidence": {"short": "Johnson et al. IDSA/SHEA", "url": "https://doi.org/10.1093/cid/ciab549"},
    },
    "uti": {
        "missedDx": 7,
        "unnecessaryTx": 4,
        "evidence": {"short": "Bent et al. JAMA", "url": "https://doi.org/10.1001/jama.287.20.2701"},
    },
    "endo": {
        "missedDx": 20,
        "unnecessaryTx": 6,
        "evidence": {"short": "Delgado et al. ESC Endocarditis", "url": "https://doi.org/10.1093/eurheartj/ehad193"},
    },
    "active_tb": {
        "missedDx": 18,
        "unnecessaryTx": 8,
        "evidence": {"short": "WHO Global TB Report 2024", "url": "https://www.who.int/publications/i/item/9789240101531"},
    },
    "pjp": {
        "missedDx": 16,
        "unnecessaryTx": 5,
        "evidence": {"short": "Mappin-Kasirer et al. BMC Infect Dis", "url": "https://doi.org/10.1186/s12879-024-09957-y"},
    },
    "pji": {
        "missedDx": 14,
        "unnecessaryTx": 8,
        "evidence": {"short": "Cortes-Penfield et al. Clin Infect Dis", "url": "https://doi.org/10.1093/cid/ciac992"},
    },
    "inv_mold": {
        "missedDx": 18,
        "unnecessaryTx": 9,
        "evidence": {"short": "Donnelly et al. Clin Infect Dis", "url": "https://doi.org/10.1093/cid/ciz1008"},
    },
    "inv_candida": {
        "missedDx": 16,
        "unnecessaryTx": 7,
        "evidence": {"short": "Pappas et al. IDSA Candidiasis", "url": "https://doi.org/10.1093/cid/civ933"},
    },
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def prob_to_odds(p: float) -> float:
    pp = clamp(p, EPS, 1 - EPS)
    return pp / (1 - pp)


def odds_to_prob(odds: float) -> float:
    oo = max(odds, 0)
    return oo / (1 + oo)


def clamp_lr(lr: float, lower: float = 0.05, upper: float = 20) -> float:
    if (not math.isfinite(lr)) or lr <= 0:
        return 1.0
    return clamp(lr, lower, upper)


def lr_for_state(item: LRItem, state: FindingState) -> float | None:
    if state == "present" and item.lr_pos is not None:
        return item.lr_pos
    if state == "absent" and item.lr_neg is not None:
        return item.lr_neg
    return None


def post_test_prob(pretest_probability: float, lr: float) -> float:
    pre_odds = prob_to_odds(pretest_probability)
    post_odds = pre_odds * clamp_lr(lr, 0.001, 1000)
    return odds_to_prob(post_odds)


def resolve_pretest(
    module: SyndromeModule,
    req: AnalyzeRequest,
    *,
    odds_multiplier: float | None = None,
) -> Tuple[float, float, str | None]:
    preset = None
    if req.preset_id:
        preset = next((p for p in module.pretest_presets if p.id == req.preset_id), None)
    if preset is None and module.pretest_presets:
        preset = module.pretest_presets[0]

    base = req.pretest_probability if req.pretest_probability is not None else (preset.p if preset else 0.05)
    base = clamp(base, EPS, 1 - EPS)
    adjusted_odds = prob_to_odds(base) * (odds_multiplier if odds_multiplier is not None else req.pretest_odds_multiplier)
    adjusted = clamp(odds_to_prob(adjusted_odds), EPS, 1 - EPS)
    return base, adjusted, (preset.id if preset else None)


def iter_applied_findings(items: Iterable[LRItem], findings: Dict[str, FindingState]) -> Iterable[Tuple[LRItem, FindingState, float]]:
    for item in items:
        state = findings.get(item.id, "unknown")
        raw = lr_for_state(item, state)
        if raw is None:
            continue
        yield item, state, clamp_lr(raw)


def combined_lr(items: Iterable[LRItem], findings: Dict[str, FindingState]) -> float:
    lr = 1.0
    for _, _, used in iter_applied_findings(items, findings):
        lr *= used
    return clamp_lr(lr, 0.001, 1000)


def build_stepwise_path(
    *,
    pretest_probability: float,
    module: SyndromeModule,
    findings: Dict[str, FindingState],
    ordered_ids: List[str],
) -> List[StepwiseUpdate]:
    items_by_id = {item.id: item for item in module.items}
    ordered_unique: List[str] = []
    seen: set[str] = set()
    for fid in ordered_ids:
        if fid in items_by_id and fid not in seen:
            ordered_unique.append(fid)
            seen.add(fid)

    for item in module.items:
        if item.id not in seen and findings.get(item.id, "unknown") != "unknown":
            ordered_unique.append(item.id)
            seen.add(item.id)

    odds = prob_to_odds(pretest_probability)
    steps: List[StepwiseUpdate] = []
    for fid in ordered_unique:
        item = items_by_id[fid]
        state = findings.get(fid, "unknown")
        lr_used_raw = lr_for_state(item, state)
        if lr_used_raw is None:
            continue
        lr_used = clamp_lr(lr_used_raw)
        odds *= lr_used
        steps.append(
            StepwiseUpdate(
                id=fid,
                label=item.label,
                lrUsed=lr_used,
                state=state,
                pAfter=odds_to_prob(odds),
            )
        )
    return steps


def _compute_virsta_score(v) -> int:
    device_or_prior_points = 4 if (v.intracardiac_device or v.prior_endocarditis) else 0
    acquisition_points = 2 if v.acquisition == "community_or_nhca" else 0
    return (
        (5 if v.emboli else 0)
        + (5 if v.meningitis else 0)
        + device_or_prior_points
        + (3 if v.native_valve_disease else 0)
        + (4 if v.ivdu else 0)
        + (3 if v.persistent_bacteremia_48h else 0)
        + (2 if v.vertebral_osteomyelitis else 0)
        + acquisition_points
        + (1 if v.severe_sepsis_shock else 0)
        + (1 if v.crp_gt_190 else 0)
    )


def _compute_denova_score(d) -> int:
    return sum(
        1
        for flag in [
            d.duration_7d,
            d.embolization,
            d.num_positive_2,
            d.origin_unknown,
            d.valve_disease,
            d.auscultation_murmur,
        ]
        if flag
    )


def _compute_handoc_score(h) -> int:
    return (
        (1 if h.heart_murmur_valve else 0)
        + HANDOC_SPECIES_POINTS[h.species]
        + (1 if h.num_positive_2 else 0)
        + (1 if h.duration_7d else 0)
        + (1 if h.only_one_species else 0)
        + (1 if h.community_acquired else 0)
    )


def _apply_vap_risk_modifiers(module_id: str, ctrl: VAPRiskModifiersInput | None) -> tuple[float, str | None]:
    if module_id != "vap" or ctrl is None or not ctrl.enabled:
        return 1.0, None
    selected = [sid for sid in ctrl.selected_ids if get_pretest_factor_spec(module_id, sid) is not None]
    raw = 1.0
    for sid in selected:
        spec = get_pretest_factor_spec(module_id, sid)
        if spec is not None:
            raw *= spec.weight
    tuning = get_pretest_factor_tuning(module_id)
    applied = clamp(pow(raw, tuning.shrink_exponent), 1.0, tuning.max_multiplier)
    note = (
        f"VAP pretest risk modifiers applied ({len(selected)} factors): raw OR product {raw:.2f}, "
        f"shrunk/capped multiplier {applied:.2f}."
    )
    return applied, note


def _apply_endo_risk_modifiers(
    module_id: str,
    ctrl: EndoRiskModifiersInput | None,
    *,
    active_score_ids: set[str],
) -> tuple[float, set[str], str | None]:
    if module_id != "endo" or ctrl is None or not ctrl.enabled:
        return 1.0, set(), None

    selected = [sid for sid in ctrl.selected_ids if get_pretest_factor_spec(module_id, sid) is not None]
    selected_set = set(selected)
    applied: List[str] = []
    suppressed_count = 0
    for sid in selected:
        spec = get_pretest_factor_spec(module_id, sid)
        if spec is None:
            continue
        if any(score_id in active_score_ids for score_id in spec.suppressed_by_scores):
            suppressed_count += 1
            continue
        applied.append(sid)

    raw = 1.0
    for sid in applied:
        spec = get_pretest_factor_spec(module_id, sid)
        if spec is not None:
            raw *= spec.weight
    tuning = get_pretest_factor_tuning(module_id)
    applied_multiplier = clamp(pow(raw, tuning.shrink_exponent), 1.0, tuning.max_multiplier)
    active_score_labels = ", ".join(score_id.upper() for score_id in sorted(active_score_ids))
    note = (
        f"Endocarditis pretest host/context modifiers applied ({len(applied)}/{len(selected)} selected"
        + (
            f", {suppressed_count} overlap factor(s) suppressed by {active_score_labels}"
            if suppressed_count and active_score_labels
            else ""
        )
        + f"): raw product {raw:.2f}, shrunk/capped multiplier {applied_multiplier:.2f}."
    )
    return applied_multiplier, selected_set, note


def prepare_probid_inputs(module: SyndromeModule, req: AnalyzeRequest) -> ProbIDPreparation:
    controls: ProbIDControlsInput | None = req.probid_controls
    analysis_findings: Dict[str, FindingState] = dict(req.findings)
    harm_findings: Dict[str, FindingState] = dict(req.findings)
    effective_multiplier = req.pretest_odds_multiplier
    direct_pretest_multiplier = req.pretest_odds_multiplier
    notes: List[str] = []

    active_endo_score_ids: set[str] = set()
    if module.id == "endo" and controls and controls.endo_scores:
        scores = controls.endo_scores
        auto_states: Dict[str, FindingState] = {item_id: "unknown" for item_id in ENDO_SCORE_ITEM_IDS}

        if scores.virsta and scores.virsta.enabled:
            active_endo_score_ids.add("virsta")
            virsta_score = _compute_virsta_score(scores.virsta)
            auto_states["endo_virsta_high"] = "present" if virsta_score >= 3 else "absent"
            notes.append(f"Auto-score VIRSTA = {virsta_score} (threshold >=3).")

        if scores.denova and scores.denova.enabled:
            active_endo_score_ids.add("denova")
            denova_score = _compute_denova_score(scores.denova)
            auto_states["endo_denova_high"] = "present" if denova_score >= 3 else "absent"
            notes.append(f"Auto-score DENOVA = {denova_score} (threshold >=3).")

        if scores.handoc and scores.handoc.enabled:
            active_endo_score_ids.add("handoc")
            handoc_score = _compute_handoc_score(scores.handoc)
            auto_states["endo_handoc_high"] = "present" if handoc_score >= 3 else "absent"
            notes.append(f"Auto-score HANDOC = {handoc_score} (threshold >=3).")

        analysis_findings.update(auto_states)
        harm_findings.update(auto_states)

    vap_multiplier, vap_note = _apply_vap_risk_modifiers(
        module.id,
        controls.vap_risk_modifiers if controls else None,
    )
    effective_multiplier *= vap_multiplier
    if vap_note:
        notes.append(vap_note)

    endo_multiplier, endo_selected_set, endo_note = _apply_endo_risk_modifiers(
        module.id,
        controls.endo_risk_modifiers if controls else None,
        active_score_ids=active_endo_score_ids,
    )
    effective_multiplier *= endo_multiplier
    if endo_note:
        notes.append(endo_note)

    # UI parity: endo risk modifiers also influence harm estimation through these synthetic host-risk states.
    if module.id == "endo" and controls and controls.endo_risk_modifiers and controls.endo_risk_modifiers.enabled:
        harm_findings["endo_prosthetic_valve"] = "present" if "prosthetic_valve" in endo_selected_set else "unknown"
        harm_findings["endo_cied"] = "present" if "cied" in endo_selected_set else "unknown"

    if direct_pretest_multiplier != 1.0:
        notes.append(f"Baseline pretest modifiers adjusted the pretest odds by x{direct_pretest_multiplier:.2f}.")

    return ProbIDPreparation(
        analysis_findings=analysis_findings,
        harm_findings=harm_findings,
        effective_pretest_odds_multiplier=effective_multiplier,
        notes=notes,
    )


def _evidence(short: str, url: str) -> HarmEvidence:
    return HarmEvidence(short=short, url=url)


def estimate_harms(module_id: str, states: Dict[str, FindingState]) -> HarmEstimate:
    base = BASE_HARM_BY_MODULE.get(module_id, {"missedDx": 10, "unnecessaryTx": 4})
    missed_dx = float(base["missedDx"])
    unnecessary_tx = float(base["unnecessaryTx"])
    base_evidence = None
    if base.get("evidence"):
        base_evidence = HarmEvidence(short=base["evidence"]["short"], url=base["evidence"].get("url"))

    rationale: List[str] = []
    drivers: List[HarmDriver] = []

    def has(item_id: str) -> bool:
        return states.get(item_id, "unknown") == "present"

    def add_missed_dx_driver(delta: float, label: str, evidence: HarmEvidence | None = None) -> None:
        nonlocal missed_dx
        missed_dx += delta
        drivers.append(HarmDriver(label=label, delta=delta, evidence=evidence))
        rationale.append(label)

    if module_id == "cap":
        if has("cap_hypox") or has("cap_rr"):
            add_missed_dx_driver(
                3,
                "Higher severity physiology selected (hypoxemia/tachypnea).",
                _evidence("Metlay et al. ATS/IDSA", "https://doi.org/10.1164/rccm.201908-1581ST"),
            )
        if has("cap_cxr_consolidation"):
            add_missed_dx_driver(
                2,
                "Radiographic consolidation selected.",
                _evidence("Metlay et al. ATS/IDSA", "https://doi.org/10.1164/rccm.201908-1581ST"),
            )

    if module_id == "uti":
        if has("uti_cva") or has("uti_fever"):
            add_missed_dx_driver(
                2,
                "Systemic/upper-tract features selected.",
                _evidence("Bent et al. JAMA", "https://doi.org/10.1001/jama.287.20.2701"),
            )
        if has("uti_catheter") or has("uti_obstruction"):
            add_missed_dx_driver(
                1,
                "Complicated host factors selected.",
                _evidence("Gupta et al. IDSA/ESCMID", "https://doi.org/10.1093/cid/ciq257"),
            )

    if module_id == "endo":
        if has("endo_prosthetic_valve") or has("endo_cied"):
            add_missed_dx_driver(
                3,
                "Prosthetic/device host risk selected.",
                _evidence("Delgado et al. ESC Endocarditis", "https://doi.org/10.1093/eurheartj/ehad193"),
            )
        if has("endo_bcx_major_typical") or has("endo_bcx_major_persistent"):
            add_missed_dx_driver(
                4,
                "Major microbiology criterion selected.",
                _evidence("Fowler et al. Duke-ISCVID", "https://doi.org/10.1093/cid/ciad271"),
            )
        if has("endo_tte") or has("endo_tee"):
            add_missed_dx_driver(
                2,
                "Positive endocarditis imaging selected.",
                _evidence("Bai et al. JASE", "https://doi.org/10.1016/j.echo.2017.03.007"),
            )

    if module_id == "active_tb":
        if has("tb_contact") or has("tb_hiv_or_immunosuppression"):
            add_missed_dx_driver(
                3,
                "Major TB epidemiologic/host risk selected.",
                _evidence("Fox et al. PLoS Med", "https://doi.org/10.1371/journal.pmed.1001432"),
            )
        if has("tb_incarceration") or has("tb_homelessness"):
            add_missed_dx_driver(
                2,
                "High-risk transmission setting selected.",
                _evidence("Cords et al. Lancet Public Health", "https://doi.org/10.1016/S2468-2667(21)00025-6"),
            )

    if module_id == "pjp":
        if has("pjp_host_hiv_cd4_sot") or has("pjp_host_no_ppx"):
            add_missed_dx_driver(
                4,
                "High-risk host/prophylaxis context selected.",
                _evidence("Mappin-Kasirer et al. BMC Infect Dis", "https://doi.org/10.1186/s12879-024-09957-y"),
            )
        if has("pjp_vital_hypoxemia"):
            add_missed_dx_driver(
                2,
                "Hypoxemia selected.",
                _evidence("Mappin-Kasirer et al. BMC Infect Dis", "https://doi.org/10.1186/s12879-024-09957-y"),
            )

    if module_id == "pji":
        if has("pji_exam_sinus_tract"):
            add_missed_dx_driver(
                4,
                "Sinus tract (major exam criterion) selected.",
                _evidence("Parvizi et al. J Arthroplasty", "https://doi.org/10.1016/j.arth.2018.09.028"),
            )
        if has("pji_alpha_defensin_elisa") or has("pji_synovial_fluid_culture"):
            add_missed_dx_driver(
                2,
                "Strong synovial/microbiologic evidence selected.",
                _evidence("Cortes-Penfield et al. Clin Infect Dis", "https://doi.org/10.1093/cid/ciac992"),
            )

    if module_id == "inv_mold":
        if has("imi_host_neutropenia_hsct") or has("imi_host_hematologic_malignancy"):
            add_missed_dx_driver(
                4,
                "High-risk mold host profile selected.",
                _evidence("Donnelly et al. Clin Infect Dis", "https://doi.org/10.1093/cid/ciz1008"),
            )
        if has("imi_mucorales_pcr_bal") or has("imi_aspergillus_pcr_bal"):
            add_missed_dx_driver(
                2,
                "Specific mold molecular evidence selected.",
                _evidence("Brown et al. Int J Infect Dis", "https://doi.org/10.1016/j.ijid.2025.107941"),
            )

    if module_id == "inv_candida":
        if has("icand_component_severe_sepsis") or has("icand_component_multifocal_colonization"):
            add_missed_dx_driver(
                3,
                "High-risk candidiasis host context selected.",
                _evidence("Leon et al. Crit Care Med", "https://doi.org/10.1097/01.CCM.0000202208.37364.7D"),
            )
        if has("icand_t2candida") or has("icand_pcr_blood"):
            add_missed_dx_driver(
                2,
                "Candida molecular evidence selected.",
                _evidence("Tang et al. BMC Infect Dis", "https://doi.org/10.1186/s12879-019-4419-z"),
            )

    if module_id == "cdi":
        if has("cdi_naat_pos_tox_pos"):
            add_missed_dx_driver(
                2,
                "Concordant CDI molecular/toxin evidence selected.",
                _evidence("Kraft et al. Clin Microbiol Rev", "https://doi.org/10.1128/CMR.00032-18"),
            )

    if not rationale:
        rationale = ["No additional high-impact risk modifiers selected; using syndrome baseline harms."]

    return HarmEstimate(
        base_missed_dx=float(base["missedDx"]),
        base_unnecessary_tx=float(base["unnecessaryTx"]),
        base_evidence=base_evidence,
        missed_dx=clamp(missed_dx, 1, 30),
        unnecessary_tx=clamp(unnecessary_tx, 1, 30),
        rationale=rationale,
        missed_dx_drivers=drivers,
    )


def derive_decision_thresholds(harms: HarmInputs) -> DecisionThresholds:
    # Exact frontend formula from `ProbID/probidDecision.ts`.
    treat = clamp(harms.unnecessary_treatment / (harms.unnecessary_treatment + harms.missed_diagnosis), 0.001, 0.999)
    observe = clamp(treat * 0.5, 0.001, 0.999)
    return DecisionThresholds(observeProbability=observe, treatProbability=treat)


def recommendation_for_probability(posttest_probability: float, thresholds: DecisionThresholds) -> str:
    if posttest_probability >= thresholds.treat_probability:
        return "treat"
    if posttest_probability <= thresholds.observe_probability:
        return "observe"
    return "test"


def confidence_from_thresholds(posttest_probability: float, thresholds: DecisionThresholds) -> float:
    distances = [
        abs(posttest_probability - thresholds.observe_probability),
        abs(posttest_probability - thresholds.treat_probability),
    ]
    nearest = min(distances)
    return clamp(nearest / 0.25, 0.05, 0.99)


def applied_finding_summaries(module: SyndromeModule, findings: Dict[str, FindingState]) -> List[AppliedFinding]:
    summaries: List[AppliedFinding] = []
    for item, state, lr_used in iter_applied_findings(module.items, findings):
        impact_score = abs(math.log(max(lr_used, EPS)))
        summaries.append(
            AppliedFinding(
                id=item.id,
                label=item.label,
                state=state,
                lrUsed=lr_used,
                impactScore=impact_score,
            )
        )
    summaries.sort(key=lambda x: x.impact_score, reverse=True)
    return summaries


def resolve_harms(module: SyndromeModule, req: AnalyzeRequest, *, states_override: Dict[str, FindingState] | None = None) -> HarmInputs:
    if req.harms is not None:
        return req.harms
    if module.default_harms is not None:
        return module.default_harms

    est = estimate_harms(module.id, states_override if states_override is not None else req.findings)
    return HarmInputs(unnecessary_treatment=est.unnecessary_tx, missed_diagnosis=est.missed_dx)
