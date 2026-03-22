from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .pretest_factors import get_pretest_factor_spec, get_pretest_factor_tuning
from .schemas import (
    AnalyzeRequest,
    AppliedFinding,
    ClinicalScoreResult,
    DecisionThresholds,
    EndoRiskModifiersInput,
    FindingState,
    HarmInputs,
    LRItem,
    NextBestTest,
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

ENDO_TYPICAL_MAJOR_MICRO_IDS = {
    "endo_bcx_major_typical",
    "endo_bcx_saureus_multi",
    "endo_bcx_cons_prosthetic_multi",
    "endo_bcx_efaecalis_multi",
    "endo_bcx_enterococcus_prosthetic_multi",
    "endo_bcx_nbhs_multi",
    "endo_coxiella_major",
}

ENDO_PERSISTENT_BACTEREMIA_INCREMENTAL_LR = 2.5

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
    "tb_uveitis": {
        "missedDx": 14,
        "unnecessaryTx": 7,
        "evidence": {"short": "ATT toxicity calibration: hepatotoxicity + ethambutol optic toxicity literature", "url": "https://pubmed.ncbi.nlm.nih.gov/36249736/"},
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
    "septic_arthritis": {
        "missedDx": 18,
        "unnecessaryTx": 6,
        "evidence": {"short": "Margaretten et al. JAMA", "url": None},
    },
    "bacterial_meningitis": {
        "missedDx": 22,
        "unnecessaryTx": 5,
        "evidence": {"short": "Brouwer et al. Lancet", "url": None},
    },
    "encephalitis": {
        "missedDx": 20,
        "unnecessaryTx": 7,
        "evidence": {"short": "Tunkel et al. IDSA", "url": None},
    },
    "spinal_epidural_abscess": {
        "missedDx": 24,
        "unnecessaryTx": 6,
        "evidence": {"short": "Davis et al. J Emerg Med", "url": None},
    },
    "brain_abscess": {
        "missedDx": 22,
        "unnecessaryTx": 7,
        "evidence": {"short": "Brouwer et al. Curr Opin Infect Dis", "url": None},
    },
    "necrotizing_soft_tissue_infection": {
        "missedDx": 26,
        "unnecessaryTx": 8,
        "evidence": {"short": "IDSA SSTI Guideline", "url": None},
    },
    "diabetic_foot_infection": {
        "missedDx": 14,
        "unnecessaryTx": 6,
        "evidence": {"short": "IWGDF/IDSA DFI Guideline", "url": "https://doi.org/10.1093/cid/ciad527"},
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


def compute_next_best_tests(
    *,
    current_posttest: float,
    module: SyndromeModule,
    findings: Dict[str, FindingState],
    max_results: int = 5,
) -> List[NextBestTest]:
    """For each unevaluated finding with both LR+ and LR-, calculate the
    probability swing if that test were positive vs negative.  Return the
    top *max_results* tests ranked by largest swing (most informative)."""
    evaluated = {fid for fid, state in findings.items() if state != "unknown"}
    candidates: List[NextBestTest] = []
    current_odds = prob_to_odds(clamp(current_posttest, EPS, 1 - EPS))
    for item in module.items:
        if item.id in evaluated:
            continue
        if item.lr_pos is None or item.lr_neg is None:
            continue
        # Skip neutral / "not done" items (LR effectively 1.0)
        if abs((item.lr_pos or 1.0) - 1.0) < 0.01 and abs((item.lr_neg or 1.0) - 1.0) < 0.01:
            continue
        p_if_pos = odds_to_prob(current_odds * clamp_lr(item.lr_pos))
        p_if_neg = odds_to_prob(current_odds * clamp_lr(item.lr_neg))
        swing = abs(p_if_pos - p_if_neg)
        source_short = None
        if hasattr(item, "source") and item.source:
            src = item.source
            if isinstance(src, dict):
                source_short = src.get("short")
            elif hasattr(src, "short"):
                source_short = src.short
        candidates.append(
            NextBestTest(
                id=item.id,
                label=item.label,
                category=item.category,
                probabilityIfPositive=round(p_if_pos, 4),
                probabilityIfNegative=round(p_if_neg, 4),
                probabilitySwing=round(swing, 4),
                sourceShort=source_short,
            )
        )
    candidates.sort(key=lambda t: t.probability_swing, reverse=True)
    return candidates[:max_results]


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

    if module.id == "endo" and analysis_findings.get("endo_bcx_major_persistent") == "present":
        if any(analysis_findings.get(item_id) == "present" for item_id in ENDO_TYPICAL_MAJOR_MICRO_IDS):
            persistent_item = next((item for item in module.items if item.id == "endo_bcx_major_persistent"), None)
            if persistent_item is not None:
                persistent_item.lr_pos = ENDO_PERSISTENT_BACTEREMIA_INCREMENTAL_LR
                notes.append(
                    "Persistent bacteremia was counted as an incremental modifier (LR 2.50) on top of the concurrent major typical-organism blood-culture signal, rather than as a second full independent Duke-major microbiology LR, to limit double counting of correlated endocarditis evidence."
                )

    return ProbIDPreparation(
        analysis_findings=analysis_findings,
        harm_findings=harm_findings,
        effective_pretest_odds_multiplier=effective_multiplier,
        notes=notes,
    )


def _evidence(short: str, url: str | None = None) -> HarmEvidence:
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

    def add_unnecessary_tx_driver(delta: float, label: str, evidence: HarmEvidence | None = None) -> None:
        nonlocal unnecessary_tx
        unnecessary_tx += delta
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
        if any(has(item_id) for item_id in (ENDO_TYPICAL_MAJOR_MICRO_IDS | {"endo_bcx_major_persistent"})):
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

    if module_id == "tb_uveitis":
        has_cld_severity = any(
            has(item_id)
            for item_id in (
                "tbu_harm_cld_mild",
                "tbu_harm_cld_moderate",
                "tbu_harm_cld_severe",
            )
        )
        if has("tbu_phenotype_choroiditis_tuberculoma") or has("tbu_phenotype_choroiditis_serpiginoid") or has("tbu_harm_macular_or_vision_threatening_lesion"):
            add_missed_dx_driver(
                5,
                "Posterior segment or macula-threatening phenotype selected, which raises the harm of delayed treatment.",
                _evidence("Agrawal et al. Ophthalmology", "https://doi.org/10.1016/j.ophtha.2020.01.008"),
            )
        if has("tbu_phenotype_panuveitis") or has("tbu_phenotype_rv_active") or has("tbu_harm_progressive_vision_loss_or_severe_inflammation"):
            add_missed_dx_driver(
                4,
                "Progressive inflammation, active retinal vasculitis, or panuveitis selected, which increases the risk of visual loss if TBU is missed.",
                _evidence("Agrawal et al. Ophthalmology", "https://doi.org/10.1016/j.ophtha.2020.06.052"),
            )
        if has("tbu_harm_bilateral_or_only_seeing_eye"):
            add_missed_dx_driver(
                6,
                "Bilateral disease or an only-seeing eye selected, which makes a missed diagnosis much more consequential.",
                _evidence("COTS Calculator", "https://www.oculartb.net/cots-calc"),
            )
        if has("tbu_harm_immunosuppressed"):
            add_missed_dx_driver(
                2,
                "Host immunosuppression selected, which raises concern about uncontrolled ocular TB if therapy is withheld.",
                _evidence("COTS Calculator", "https://www.oculartb.net/cots-calc"),
            )
        if has("tbu_harm_cld_mild"):
            add_unnecessary_tx_driver(
                1,
                "Mild chronic liver disease severity selected (for example Child-Pugh A or MELD-Na under 10).",
                _evidence("ATS/CDC/ERS/IDSA TB Treatment Guideline", "https://www.idsociety.org/practice-guideline/treatment-of-drug-susceptible-tb/"),
            )
        if has("tbu_harm_cld_moderate"):
            add_unnecessary_tx_driver(
                3,
                "Moderate chronic liver disease severity selected (for example Child-Pugh B or MELD-Na 10-19).",
                _evidence("ATS/CDC/ERS/IDSA TB Treatment Guideline", "https://www.idsociety.org/practice-guideline/treatment-of-drug-susceptible-tb/"),
            )
        if has("tbu_harm_cld_severe"):
            add_unnecessary_tx_driver(
                5,
                "Severe chronic liver disease severity selected (for example Child-Pugh C or MELD-Na 20 or higher).",
                _evidence("ATS/CDC/ERS/IDSA TB Treatment Guideline", "https://www.idsociety.org/practice-guideline/treatment-of-drug-susceptible-tb/"),
            )
        if has("tbu_harm_hepatotoxicity_risk") and not has_cld_severity:
            add_unnecessary_tx_driver(
                3,
                "Major hepatotoxicity risk selected, which raises the expected harm of empiric ATT.",
                _evidence("Wang et al. ATLI meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/36249736/"),
            )
        if has("tbu_harm_ethambutol_ocular_risk"):
            add_unnecessary_tx_driver(
                2,
                "Ethambutol ocular-toxicity risk selected, which raises the expected harm of empiric ATT in an ophthalmology patient.",
                _evidence("Kim et al. Ethambutol optic neuropathy cohort", "https://pubmed.ncbi.nlm.nih.gov/39474613/"),
            )
        if has("tbu_harm_major_drug_interaction_or_intolerance"):
            add_unnecessary_tx_driver(
                2,
                "Major rifamycin interaction, prior ATT intolerance, or treatment-complexity risk selected.",
                _evidence("ATS/CDC/ERS/IDSA TB Treatment Guideline", "https://www.idsociety.org/practice-guideline/treatment-of-drug-susceptible-tb/"),
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

    if module_id == "septic_arthritis":
        if has("sa_host_immunosuppression") or has("sa_host_bacteremia_or_overlying_ssti"):
            add_missed_dx_driver(
                3,
                "High-risk host or hematogenous/contiguous source context selected.",
                _evidence("Mathews et al. Lancet"),
            )
        if has("sa_synovial_wbc_ge50k") or has("sa_gram_stain") or has("sa_synovial_culture"):
            add_missed_dx_driver(
                3,
                "Strong synovial or microbiologic evidence selected.",
                _evidence("Carpenter et al. Acad Emerg Med"),
            )

    if module_id == "bacterial_meningitis":
        if has("bm_host_csf_leak_or_neurosurgery") or has("bm_host_bacteremia_sepsis"):
            add_missed_dx_driver(
                4,
                "High-risk structural or bacteremic context selected.",
                _evidence("Brouwer et al. Lancet"),
            )
        if has("bm_csf_gram_stain") or has("bm_csf_culture") or has("bm_csf_bacterial_pcr") or has("bm_csf_glucose_ratio_low"):
            add_missed_dx_driver(
                4,
                "Strong CSF or microbiologic evidence selected.",
                _evidence("Brouwer et al. Lancet"),
            )

    if module_id == "encephalitis":
        if has("enc_host_immunocompromised") or has("enc_host_transplant_or_biologic"):
            add_missed_dx_driver(
                3,
                "High-risk host context selected.",
                _evidence("Tunkel et al. IDSA"),
            )
        if has("enc_hsv_pcr") or has("enc_mri_temporal") or has("enc_eeg_temporal"):
            add_missed_dx_driver(
                4,
                "Strong HSV-weighted molecular, MRI, or EEG evidence selected.",
                _evidence("Tunkel et al. IDSA"),
            )

    if module_id == "spinal_epidural_abscess":
        if has("sea_host_recent_spinal_procedure") or has("sea_host_bacteremia_or_ssti"):
            add_missed_dx_driver(
                4,
                "High-risk local inoculation or bacteremic context selected.",
                _evidence("Arko et al. Surg Neurol Int"),
            )
        if has("sea_mri_positive") or has("sea_exam_neuro_deficit") or has("sea_exam_bowel_bladder"):
            add_missed_dx_driver(
                4,
                "Strong MRI or neurologic compression evidence selected.",
                _evidence("Davis et al. J Emerg Med"),
            )

    if module_id == "brain_abscess":
        if has("ba_host_otogenic_sinus_dental") or has("ba_host_endocarditis_bacteremia") or has("ba_host_neurosurgery_trauma"):
            add_missed_dx_driver(
                4,
                "High-risk contiguous, hematogenous, or postoperative source context selected.",
                _evidence("Helweg-Larsen et al. Open Forum Infect Dis"),
            )
        if has("ba_mri_dwi_positive") or has("ba_aspirate_culture_positive") or has("ba_exam_focal_deficit"):
            add_missed_dx_driver(
                4,
                "Strong MRI, operative microbiology, or focal neurologic evidence selected.",
                _evidence("Brouwer et al. Curr Opin Infect Dis"),
            )

    if module_id == "necrotizing_soft_tissue_infection":
        if has("nsti_host_recent_surgery_or_trauma") or has("nsti_host_perineal_or_chronic_wound_source"):
            add_missed_dx_driver(
                4,
                "High-risk local source context selected.",
                _evidence("IDSA SSTI Guideline"),
            )
        if has("nsti_vital_hypotension") or has("nsti_exam_bullae_or_necrosis") or has("nsti_operative_findings") or has("nsti_ct_positive"):
            add_missed_dx_driver(
                5,
                "Strong systemic toxicity, operative, or imaging evidence selected.",
                _evidence("Fernando et al. Ann Surg"),
            )

    if module_id == "diabetic_foot_infection":
        if has("dfi_systemic_toxicity") or has("dfi_deep_abscess_or_gangrene"):
            add_missed_dx_driver(
                8,
                "Systemic toxicity, gangrene, or other destructive diabetic foot feature selected.",
                _evidence("IWGDF/IDSA DFI Guideline", "https://doi.org/10.1093/cid/ciad527"),
            )
        if has("dfi_host_pad_or_ischemia"):
            add_missed_dx_driver(
                3,
                "PAD or ischemia selected, which raises limb risk if infection is undertreated.",
                _evidence("IWGDF/IDSA DFI Guideline", "https://doi.org/10.1093/cid/ciad527"),
            )
        if has("dfi_bone_biopsy_culture_pos") or has("dfi_bone_histology_pos"):
            add_missed_dx_driver(
                4,
                "Bone-sampling evidence selected, which raises the consequence of undertreating diabetic foot osteomyelitis.",
                _evidence("IWGDF/IDSA DFI Guideline", "https://doi.org/10.1093/cid/ciad527"),
            )

    if module_id == "inv_mold":
        if has("imi_host_neutropenia_hsct") or has("imi_host_hematologic_malignancy"):
            add_missed_dx_driver(
                4,
                "High-risk mold host profile selected.",
                _evidence("Donnelly et al. Clin Infect Dis", "https://doi.org/10.1093/cid/ciz1008"),
            )
        if has("imi_mucorales_pcr_bal") or has("imi_aspergillus_pcr_bal") or has("imi_aspergillus_pcr_plasma") or has("imi_aspergillus_culture_resp"):
            add_missed_dx_driver(
                2,
                "Specific mold microbiology selected.",
                _evidence("Aspergillus/Mucorales PCR studies"),
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


# ---------------------------------------------------------------------------
# Clinical prediction rule calculators
# ---------------------------------------------------------------------------

def _is_present(findings: Dict[str, FindingState], item_id: str) -> bool:
    return findings.get(item_id) == "present"


def _is_absent(findings: Dict[str, FindingState], item_id: str) -> bool:
    return findings.get(item_id) == "absent"


def _is_evaluated(findings: Dict[str, FindingState], item_id: str) -> bool:
    return findings.get(item_id, "unknown") != "unknown"


def compute_clinical_scores(
    module_id: str,
    findings: Dict[str, FindingState],
) -> List[ClinicalScoreResult]:
    """Compute all applicable clinical prediction rule scores for a module."""
    scores: List[ClinicalScoreResult] = []
    if module_id == "cap":
        s = _compute_psi(findings)
        if s:
            scores.append(s)
        q = _compute_qsofa(findings)
        if q:
            scores.append(q)
    elif module_id == "endo":
        d = _compute_duke(findings)
        if d:
            scores.append(d)
    elif module_id == "necrotizing_soft_tissue_infection":
        l = _compute_lrinec(findings)
        if l:
            scores.append(l)
    elif module_id == "febrile_neutropenia":
        m = _compute_mascc(findings)
        if m:
            scores.append(m)
    elif module_id == "bacterial_meningitis":
        b = _compute_bms(findings)
        if b:
            scores.append(b)
    return scores


# --- PSI/PORT (Fine, NEJM 1997) ---
# Simplified: uses ProbID findings to estimate PSI class.
# Full PSI requires 20 variables; we use available findings to approximate.

def _compute_psi(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    # Check if we have enough CAP findings to score
    evaluated = {fid for fid, s in findings.items() if s != "unknown" and fid.startswith("cap_")}
    if len(evaluated) < 3:
        return None

    met: List[str] = []
    not_met: List[str] = []
    points = 0

    # Demographics
    if _is_present(findings, "cap_age_ge65"):
        met.append("Age ≥65 years (+age points)")
        points += 70  # approximate midpoint
    elif _is_evaluated(findings, "cap_age_ge65"):
        not_met.append("Age ≥65")

    # Nursing home
    if _is_present(findings, "cap_nursing_home"):
        met.append("Nursing home resident (+10)")
        points += 10
    elif _is_evaluated(findings, "cap_nursing_home"):
        not_met.append("Nursing home resident")

    # Comorbidities
    comorbidity_items = {
        "cap_copd": ("COPD/chronic lung disease", 10),
        "cap_hf": ("Heart failure", 10),
        "cap_liver_disease": ("Liver disease", 20),
        "cap_renal_disease": ("Renal disease", 10),
        "cap_neoplastic_disease": ("Neoplastic disease", 30),
        "cap_dm": ("Diabetes mellitus", 10),
        "cap_cerebrovascular": ("Cerebrovascular disease", 10),
    }
    for item_id, (label, pts) in comorbidity_items.items():
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            points += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    # Physical exam findings
    exam_items = {
        "cap_rr": ("Tachypnea ≥30/min", 20),
        "cap_hypotension": ("Hypotension SBP <90", 20),
        "cap_hr": ("Tachycardia ≥125/min", 10),
        "cap_hypothermia": ("Temperature <35°C", 15),
        "cap_ams": ("Altered mental status", 20),
        "cap_hypox": ("Hypoxemia SpO₂ <90%", 10),
    }
    for item_id, (label, pts) in exam_items.items():
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            points += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    # Lab findings
    lab_items = {
        "cap_bun_elevated": ("BUN >30 mg/dL", 20),
        "cap_sodium_low": ("Sodium <130 mEq/L", 20),
        "cap_glucose_high": ("Glucose ≥250 mg/dL", 10),
        "cap_hematocrit_low": ("Hematocrit <30%", 10),
        "cap_pao2_low": ("PaO₂ <60 mmHg", 10),
        "cap_arterial_ph_low": ("Arterial pH <7.35", 30),
        "cap_pleural_effusion": ("Pleural effusion", 10),
    }
    for item_id, (label, pts) in lab_items.items():
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            points += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    # Determine PSI class
    if points <= 50:
        risk_class = "I-II"
        interp = "Low risk (0.1-0.7% mortality)"
        rec = "Outpatient treatment appropriate. Consider oral antibiotics."
    elif points <= 70:
        risk_class = "III"
        interp = "Low-moderate risk (0.9-2.8% mortality)"
        rec = "Brief observation or outpatient with close follow-up."
    elif points <= 90:
        risk_class = "IV"
        interp = "Moderate risk (8.2-9.3% mortality)"
        rec = "Inpatient treatment recommended."
    elif points <= 130:
        risk_class = "V"
        interp = "High risk (27-31% mortality)"
        rec = "Inpatient treatment, consider ICU admission."
    else:
        risk_class = "V+"
        interp = "Very high risk (>31% mortality)"
        rec = "ICU admission strongly recommended."

    return ClinicalScoreResult(
        scoreName="PSI/PORT Score",
        scoreValue=points,
        riskClass=f"Class {risk_class}",
        interpretation=interp,
        recommendation=rec,
        componentsMet=met,
        componentsNotMet=not_met,
        source="Fine et al. NEJM 1997",
    )


# --- Modified Duke Criteria (Fowler, CID 2023 — 2023 Duke-ISCVID) ---

def _compute_duke(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    evaluated = {fid for fid, s in findings.items() if s != "unknown" and fid.startswith("endo_")}
    if len(evaluated) < 3:
        return None

    major_met: List[str] = []
    minor_met: List[str] = []
    major_not: List[str] = []
    minor_not: List[str] = []

    # Major criterion 1: Microbiology
    micro_major_ids = {
        "endo_bcx_major_typical": "Typical organism in ≥2 sets",
        "endo_bcx_major_persistent": "Persistently positive blood cultures",
        "endo_bcx_saureus_multi": "S. aureus in ≥2 sets",
        "endo_bcx_cons_prosthetic_multi": "CoNS in ≥2 sets with prosthetic",
        "endo_bcx_efaecalis_multi": "E. faecalis in ≥2 sets",
        "endo_bcx_nbhs_multi": "NBHS in ≥2 sets",
        "endo_coxiella_major": "Coxiella Phase I IgG ≥1:800",
    }
    micro_major_count = 0
    for item_id, label in micro_major_ids.items():
        if _is_present(findings, item_id):
            major_met.append(f"Micro major: {label}")
            micro_major_count += 1
        elif _is_evaluated(findings, item_id):
            major_not.append(f"Micro major: {label}")

    # Major criterion 2: Imaging (echo/PET)
    imaging_major = False
    for item_id, label in [
        ("endo_tte", "TTE positive (vegetation/regurgitation/perforation)"),
        ("endo_tee", "TEE positive (vegetation/regurgitation/perforation)"),
        ("endo_pet", "FDG PET/CT positive"),
    ]:
        if _is_present(findings, item_id):
            major_met.append(f"Imaging major: {label}")
            imaging_major = True
        elif _is_evaluated(findings, item_id):
            major_not.append(f"Imaging major: {label}")

    total_major = min(micro_major_count, 1) + (1 if imaging_major else 0)

    # Minor criteria
    minor_items = {
        "endo_fever": "Fever ≥38°C",
        "endo_vascular": "Vascular phenomena",
        "endo_immune": "Immunologic phenomena",
        "endo_virsta_prosthetic_valve": "Predisposing heart condition (prosthetic valve/CIED)",
        "endo_virsta_ivdu": "IVDU",
        "endo_bcx_pos_not_major": "Positive BCx not meeting major criterion",
    }
    minor_count = 0
    for item_id, label in minor_items.items():
        if _is_present(findings, item_id):
            minor_met.append(label)
            minor_count += 1
        elif _is_evaluated(findings, item_id):
            minor_not.append(label)

    # Classification
    all_met = major_met + minor_met
    all_not = major_not + minor_not

    if total_major >= 2 or (total_major >= 1 and minor_count >= 3) or minor_count >= 5:
        risk_class = "Definite"
        interp = "Definite infective endocarditis by Modified Duke Criteria"
        rec = "Treat as IE. Prolonged IV antibiotics (4-6 weeks). Surgical evaluation if indication present."
    elif (total_major >= 1 and minor_count >= 1) or minor_count >= 3:
        risk_class = "Possible"
        interp = "Possible infective endocarditis — cannot be ruled out"
        rec = "Further workup: TEE if not done, FDG-PET/CT if prosthetic. Consider repeat blood cultures. Treat empirically if clinical suspicion high."
    else:
        risk_class = "Rejected"
        interp = "Duke criteria not met — IE unlikely but not excluded"
        rec = "Alternative diagnosis likely. If clinical suspicion persists, consider repeat imaging or extended culture protocol."

    return ClinicalScoreResult(
        scoreName="Modified Duke Criteria (2023 Duke-ISCVID)",
        scoreValue=None,
        riskClass=risk_class,
        interpretation=interp,
        recommendation=rec,
        componentsMet=all_met,
        componentsNotMet=all_not,
        source="Fowler et al. CID 2023",
    )


# --- qSOFA (Seymour, JAMA 2016 — Sepsis-3) ---

def _compute_qsofa(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    # qSOFA uses 3 criteria: RR ≥22, AMS, SBP ≤100
    # Map to available CAP findings (these overlap with PSI items)
    criteria = {
        "cap_rr": "Respiratory rate ≥22/min",
        "cap_ams": "Altered mental status",
        "cap_hypotension": "Systolic BP ≤100 mmHg",
    }
    evaluated_count = sum(1 for fid in criteria if _is_evaluated(findings, fid))
    if evaluated_count < 2:
        return None

    met: List[str] = []
    not_met: List[str] = []
    score = 0
    for item_id, label in criteria.items():
        if _is_present(findings, item_id):
            met.append(label)
            score += 1
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    if score >= 2:
        interp = "qSOFA ≥2 — high risk of poor outcome (mortality 3-14x higher)"
        rec = "Assess for organ dysfunction (full SOFA). Consider ICU level of care. Ensure Hour-1 sepsis bundle."
    elif score == 1:
        interp = "qSOFA 1 — intermediate risk"
        rec = "Monitor closely. Reassess if clinical trajectory worsens."
    else:
        interp = "qSOFA 0 — lower risk of poor outcome"
        rec = "Low qSOFA does NOT exclude sepsis. Clinical judgment remains paramount."

    return ClinicalScoreResult(
        scoreName="qSOFA",
        scoreValue=score,
        riskClass=f"{score}/3",
        interpretation=interp,
        recommendation=rec,
        componentsMet=met,
        componentsNotMet=not_met,
        source="Seymour et al. JAMA 2016 (Sepsis-3)",
    )


# --- LRINEC (Wong, Crit Care Med 2004) ---

def _compute_lrinec(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    evaluated = {fid for fid, s in findings.items() if s != "unknown" and fid.startswith("nsti_")}
    if len(evaluated) < 3:
        return None

    met: List[str] = []
    not_met: List[str] = []
    score = 0

    # LRINEC components mapped to NSTI module findings
    components = {
        "nsti_wbc_high": ("WBC elevated (>15,000 or <4,000/µL)", 1),
        "nsti_hgb_low": ("Hemoglobin <13.5 g/dL", 2),
        "nsti_sodium_low": ("Sodium <135 mEq/L", 2),
        "nsti_creatinine_elevated": ("Creatinine >1.6 mg/dL", 2),
        "nsti_glucose_elevated": ("Glucose >180 mg/dL", 1),
        "nsti_crp_gt150": ("CRP >150 mg/L", 4),
    }
    for item_id, (label, pts) in components.items():
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            score += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    if score >= 8:
        risk_class = "High risk"
        interp = f"LRINEC {score} ≥8 — strongly predictive of NF (PPV ~93%)"
        rec = "Emergent surgical exploration. Do NOT wait for imaging. Start broad-spectrum empirics (vancomycin + pip-tazo + clindamycin)."
    elif score >= 6:
        risk_class = "Intermediate risk"
        interp = f"LRINEC {score} (6-7) — suspicious for NF"
        rec = "Urgent surgical consultation. CT/MRI if surgery not immediately available. LRINEC sensitivity is only ~60-80% — clinical suspicion trumps score."
    else:
        risk_class = "Low risk"
        interp = f"LRINEC {score} <6 — lower probability of NF"
        rec = "NF not excluded by low LRINEC (sensitivity limited). If rapid progression, crepitus, or disproportionate pain → surgical exploration regardless of score."

    return ClinicalScoreResult(
        scoreName="LRINEC Score",
        scoreValue=score,
        riskClass=risk_class,
        interpretation=interp,
        recommendation=rec,
        componentsMet=met,
        componentsNotMet=not_met,
        source="Wong et al. Crit Care Med 2004",
    )


# --- MASCC (Klastersky, JCO 2000) ---

def _compute_mascc(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    evaluated = {fid for fid, s in findings.items() if s != "unknown" and fid.startswith("fn_")}
    if len(evaluated) < 3:
        return None

    met: List[str] = []
    not_met: List[str] = []
    score = 0

    # MASCC components — score is ADDITIVE (higher = lower risk)
    components = {
        "fn_mascc_burden_mild": ("Mild/no symptoms", 5),
        "fn_mascc_burden_moderate": ("Moderate symptoms", 3),
        "fn_mascc_no_hypotension": ("No hypotension (SBP ≥90)", 5),
        "fn_mascc_no_copd": ("No active COPD", 4),
        "fn_mascc_solid_tumor_or_no_fungal": ("Solid tumor or no prior fungal", 4),
        "fn_mascc_no_dehydration": ("No dehydration requiring IV fluids", 3),
        "fn_mascc_outpatient": ("Outpatient at onset", 3),
        "fn_mascc_age_lt60": ("Age <60 years", 2),
    }
    # Burden of illness: mild and moderate are mutually exclusive; pick the one present
    burden_mild = _is_present(findings, "fn_mascc_burden_mild")
    burden_moderate = _is_present(findings, "fn_mascc_burden_moderate")

    for item_id, (label, pts) in components.items():
        # Skip mild if moderate is present (mutually exclusive)
        if item_id == "fn_mascc_burden_mild" and burden_moderate:
            continue
        if item_id == "fn_mascc_burden_moderate" and burden_mild:
            continue
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            score += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    if score >= 21:
        risk_class = "Low risk"
        interp = f"MASCC {score} ≥21 — low risk of serious complications (PPV 91%, mortality <3%)"
        rec = "Consider outpatient oral antibiotics (ciprofloxacin + amoxicillin-clavulanate). Requires: tolerating PO, adherence, close follow-up within 24-48h, no prior FQ prophylaxis."
    elif score >= 15:
        risk_class = "Intermediate risk"
        interp = f"MASCC {score} (15-20) — intermediate risk"
        rec = "Inpatient IV antibiotics recommended. Anti-pseudomonal beta-lactam monotherapy (cefepime, pip-tazo, or meropenem)."
    else:
        risk_class = "High risk"
        interp = f"MASCC {score} <15 — high risk of serious complications"
        rec = "Inpatient IV antibiotics mandatory. Anti-pseudomonal beta-lactam. Add vancomycin if suspected CLABSI, MRSA, mucositis, or hemodynamic instability. Consider antifungal if neutropenia >7 days."

    return ClinicalScoreResult(
        scoreName="MASCC Score",
        scoreValue=score,
        riskClass=risk_class,
        interpretation=interp,
        recommendation=rec,
        componentsMet=met,
        componentsNotMet=not_met,
        source="Klastersky et al. JCO 2000",
    )


# --- Bacterial Meningitis Score (Nigrovic, JAMA 2007) ---

def _compute_bms(findings: Dict[str, FindingState]) -> ClinicalScoreResult | None:
    evaluated = {fid for fid, s in findings.items() if s != "unknown" and fid.startswith("bm_")}
    if len(evaluated) < 3:
        return None

    met: List[str] = []
    not_met: List[str] = []
    score = 0

    components = {
        "bm_bms_gram_stain_positive": ("CSF Gram stain positive", 2),
        "bm_bms_csf_protein_ge80": ("CSF protein ≥80 mg/dL", 1),
        "bm_bms_blood_anc_ge10k": ("Blood ANC ≥10,000/µL", 1),
        "bm_bms_csf_anc_ge1000": ("CSF ANC ≥1,000/µL", 1),
        "bm_bms_seizure": ("Seizure at/before presentation", 1),
    }
    for item_id, (label, pts) in components.items():
        if _is_present(findings, item_id):
            met.append(f"{label} (+{pts})")
            score += pts
        elif _is_evaluated(findings, item_id):
            not_met.append(label)

    if score == 0 and len(met) == 0 and len(not_met) >= 3:
        risk_class = "Very low risk"
        interp = "BMS = 0 — very low risk of bacterial meningitis (NPV 99.7%)"
        rec = "Bacterial meningitis essentially excluded. Consider viral etiology. Close follow-up if discharged. Note: validated primarily in children; use with caution in adults."
    elif score >= 3:
        risk_class = "High risk"
        interp = f"BMS = {score} — high probability of bacterial meningitis"
        rec = "Treat empirically immediately. Ceftriaxone + vancomycin ± dexamethasone (give dexamethasone BEFORE or WITH first antibiotic dose). Do not delay treatment for imaging."
    elif score >= 1:
        risk_class = "Moderate risk"
        interp = f"BMS = {score} — bacterial meningitis cannot be excluded"
        rec = "Empiric antibiotics recommended. Observe closely. Repeat LP if clinical course uncertain."
    else:
        risk_class = "Insufficient data"
        interp = "BMS — insufficient criteria evaluated"
        rec = "Complete CSF analysis, Gram stain, and peripheral blood ANC to fully score."

    return ClinicalScoreResult(
        scoreName="Bacterial Meningitis Score",
        scoreValue=score,
        riskClass=risk_class,
        interpretation=interp,
        recommendation=rec,
        componentsMet=met,
        componentsNotMet=not_met,
        source="Nigrovic et al. JAMA 2007",
    )
