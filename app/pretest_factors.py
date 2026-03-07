from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schemas import LRItem, SyndromeModule


@dataclass(frozen=True)
class PretestFactorRef:
    id: str
    module_item_id: Optional[str] = None
    label: Optional[str] = None
    weight: Optional[float] = None
    source_note: Optional[str] = None
    source_url: Optional[str] = None
    context_group: Optional[str] = None
    suppressed_by_scores: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PretestFactorSpec:
    id: str
    label: str
    weight: float
    source_note: str
    source_url: Optional[str] = None
    context_group: Optional[str] = None
    suppressed_by_scores: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PretestFactorTuning:
    shrink_exponent: float
    max_multiplier: float


DEFAULT_PRETEST_FACTOR_TUNING = PretestFactorTuning(shrink_exponent=0.5, max_multiplier=5.0)


def _item_ref(item_id: str, *, source_note: Optional[str] = None) -> PretestFactorRef:
    return PretestFactorRef(id=item_id, module_item_id=item_id, source_note=source_note)


def _explicit_ref(
    factor_id: str,
    *,
    label: str,
    weight: float,
    source_note: str,
    source_url: Optional[str] = None,
    context_group: Optional[str] = None,
    suppressed_by_scores: Tuple[str, ...] = (),
) -> PretestFactorRef:
    return PretestFactorRef(
        id=factor_id,
        label=label,
        weight=weight,
        source_note=source_note,
        source_url=source_url,
        context_group=context_group,
        suppressed_by_scores=suppressed_by_scores,
    )


PRETEST_FACTOR_TUNING: Dict[str, PretestFactorTuning] = {
    "vap": PretestFactorTuning(shrink_exponent=0.5, max_multiplier=6.0),
    "endo": PretestFactorTuning(shrink_exponent=0.5, max_multiplier=5.0),
    "tb_uveitis": PretestFactorTuning(shrink_exponent=0.5, max_multiplier=3.5),
}


PRETEST_FACTOR_CONFIG: Dict[str, Tuple[PretestFactorRef, ...]] = {
    "cap": (
        _item_ref("cap_age_ge65"),
        _item_ref("cap_copd"),
        _item_ref("cap_hf"),
        _item_ref("cap_ckd"),
        _item_ref("cap_dm"),
        _item_ref("cap_liver_disease"),
        _item_ref("cap_alcohol_use_disorder"),
        _item_ref("cap_active_malignancy"),
        _item_ref("cap_asplenia"),
        _item_ref("cap_current_smoker"),
        _item_ref("cap_malnutrition"),
    ),
    "vap": (
        _explicit_ref(
            "male_sex",
            label="Male sex",
            weight=1.3,
            source_note="VAP epidemiologic host-risk modifier.",
        ),
        _explicit_ref(
            "copd",
            label="COPD",
            weight=1.52,
            source_note="VAP epidemiologic host-risk modifier.",
        ),
        _explicit_ref(
            "trauma",
            label="Trauma",
            weight=1.47,
            source_note="VAP epidemiologic host-risk modifier.",
        ),
        _explicit_ref(
            "impaired_consciousness",
            label="Impaired consciousness",
            weight=3.14,
            source_note="VAP aspiration-risk modifier.",
        ),
        _explicit_ref(
            "prior_antibiotics",
            label="Prior antibiotics",
            weight=1.52,
            source_note="VAP prior antibiotic exposure modifier.",
        ),
        _explicit_ref(
            "reintubation",
            label="Reintubation",
            weight=5.11,
            source_note="VAP high-impact airway manipulation modifier.",
        ),
        _explicit_ref(
            "tracheostomy",
            label="Tracheostomy",
            weight=3.44,
            source_note="VAP airway-device risk modifier.",
        ),
        _explicit_ref(
            "enteral_feeding",
            label="Enteral feeding",
            weight=4.73,
            source_note="VAP aspiration-associated risk modifier.",
        ),
        _explicit_ref(
            "nasogastric_tube",
            label="Nasogastric tube",
            weight=2.94,
            source_note="VAP aspiration-associated risk modifier.",
        ),
        _explicit_ref(
            "h2_blocker",
            label="H2 blocker",
            weight=2.24,
            source_note="VAP acid-suppression risk modifier.",
        ),
    ),
    "cdi": (
        _item_ref("cdi_abx"),
        _item_ref("cdi_healthcare"),
        _item_ref("cdi_ppi"),
        _item_ref("cdi_prev"),
        _item_ref("cdi_age_ge65"),
        _item_ref("cdi_immuno"),
        _item_ref("cdi_ibd"),
        _item_ref("cdi_enteral_feeding"),
    ),
    "uti": (
        _item_ref("uti_female"),
        _item_ref("uti_male"),
        _item_ref("uti_age_ge65"),
        _item_ref("uti_diabetes"),
        _item_ref("uti_ckd"),
        _item_ref("uti_immuno"),
        _item_ref("uti_catheter"),
        _item_ref("uti_obstruction"),
        _item_ref("uti_stones"),
        _item_ref("uti_recurrent"),
    ),
    "endo": (
        _explicit_ref(
            "prosthetic_valve",
            label="Prosthetic valve",
            weight=2.5,
            source_note="Endocarditis structural-device risk modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "chd",
            label="Congenital heart disease",
            weight=1.8,
            source_note="Endocarditis structural heart disease modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "hemodialysis",
            label="Hemodialysis",
            weight=2.0,
            source_note="Endocarditis healthcare-exposure modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "central_venous_catheter",
            label="Central venous catheter",
            weight=1.8,
            source_note="Endocarditis intravascular device modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "immunosuppression",
            label="Immunosuppression",
            weight=1.5,
            source_note="Endocarditis host-risk modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "recent_healthcare_or_invasive_exposure",
            label="Recent healthcare or invasive exposure",
            weight=1.4,
            source_note="Endocarditis healthcare-associated exposure modifier.",
            context_group="general_ie",
        ),
        _explicit_ref(
            "ivdu",
            label="Injection drug use",
            weight=2.5,
            source_note="Endocarditis S. aureus score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("virsta",),
        ),
        _explicit_ref(
            "prior_endo",
            label="Prior endocarditis",
            weight=2.5,
            source_note="Endocarditis S. aureus score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("virsta",),
        ),
        _explicit_ref(
            "native_valve_disease",
            label="Native valve disease",
            weight=1.8,
            source_note="Endocarditis S. aureus native-valve score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("virsta",),
        ),
        _explicit_ref(
            "cied",
            label="Cardiac implantable electronic device",
            weight=2.2,
            source_note="Endocarditis S. aureus device-related score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("virsta",),
        ),
        _explicit_ref(
            "enterococcus_unknown_source",
            label="Enterococcal bacteremia with unclear source",
            weight=1.7,
            source_note="Endocarditis enterococcal source-uncertain score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("denova",),
        ),
        _explicit_ref(
            "enterococcus_valve_disease",
            label="Known valve disease",
            weight=1.7,
            source_note="Endocarditis enterococcal valve-disease score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("denova",),
        ),
        _explicit_ref(
            "viridans_community_acquired",
            label="Community-acquired viridans/NBHS bacteremia",
            weight=1.6,
            source_note="Endocarditis viridans-group community-acquisition score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("handoc",),
        ),
        _explicit_ref(
            "viridans_single_species",
            label="Single-species viridans/NBHS bacteremia",
            weight=1.4,
            source_note="Endocarditis viridans-group monomicrobial score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("handoc",),
        ),
        _explicit_ref(
            "viridans_valve_history",
            label="Known valve disease or longstanding murmur history",
            weight=1.7,
            source_note="Endocarditis viridans-group valve-history score-overlap modifier.",
            context_group="score_overlap",
            suppressed_by_scores=("handoc",),
        ),
    ),
    "active_tb": (
        _item_ref("tb_contact"),
        _item_ref("tb_birth_travel_high_incidence"),
        _item_ref("tb_incarceration"),
        _item_ref("tb_homelessness"),
        _item_ref("tb_hiv_or_immunosuppression"),
        _item_ref("tb_diabetes"),
        _item_ref("tb_smoking"),
        _item_ref("tb_undernutrition"),
        _item_ref("tb_alcohol_use"),
    ),
    "tb_uveitis": (
        _explicit_ref(
            "tbu_pretest_prior_tb_or_ltbi",
            label="Prior TB disease or known latent TB infection",
            weight=2.0,
            source_note="General TB epidemiologic pretest modifier calibrated for ocular TB; not a pooled ocular-specific OR.",
            source_url="https://www.who.int/publications/i/item/9789240101531",
        ),
        _explicit_ref(
            "tbu_pretest_close_tb_contact",
            label="Close TB contact or household TB exposure",
            weight=1.8,
            source_note="General TB contact-risk pretest modifier calibrated for ocular TB; not a pooled ocular-specific OR.",
            source_url="https://doi.org/10.1371/journal.pmed.1001432",
        ),
    ),
    "pjp": (
        _item_ref("pjp_host_hiv_cd4_sot"),
        _item_ref("pjp_host_no_ppx"),
        _item_ref("pjp_host_prolonged_steroids"),
        _item_ref("pjp_host_heme_hsct"),
        _item_ref("pjp_host_other_tcell_immunosuppression"),
    ),
    "inv_candida": (
        _item_ref("icand_component_tpn"),
        _item_ref("icand_component_surgery"),
        _item_ref("icand_host_broad_abx"),
        _item_ref("icand_host_cvc"),
        _item_ref("icand_host_dialysis"),
        _item_ref("icand_component_multifocal_colonization"),
        _item_ref("icand_component_severe_sepsis"),
    ),
    "inv_mold": (
        _item_ref("imi_host_neutropenia_hsct"),
        _item_ref("imi_host_hematologic_malignancy"),
        _item_ref("imi_host_solid_organ_transplant"),
        _item_ref("imi_host_steroids_tcell"),
        _item_ref("imi_host_icu_viral_steroid"),
    ),
    "pji": (
        _item_ref("pji_host_revision_arthroplasty"),
        _item_ref("pji_host_obesity"),
        _item_ref("pji_host_diabetes"),
        _item_ref("pji_host_ra_immunosuppression"),
        _item_ref("pji_host_prior_pji"),
        _item_ref("pji_host_tobacco_use"),
        _item_ref("pji_host_malnutrition"),
        _item_ref("pji_host_ckd"),
        _item_ref("pji_host_liver_disease"),
        _item_ref("pji_host_anemia"),
        _item_ref("pji_host_alcohol_use"),
    ),
    "septic_arthritis": (
        _item_ref("sa_host_age_gt80"),
        _item_ref("sa_host_diabetes"),
        _item_ref("sa_host_ra"),
        _item_ref("sa_host_immunosuppression"),
        _item_ref("sa_host_ivdu"),
        _item_ref("sa_host_recent_joint_surgery_or_injection"),
        _item_ref("sa_host_bacteremia_or_overlying_ssti"),
    ),
    "bacterial_meningitis": (
        _item_ref("bm_host_age_ge50"),
        _item_ref("bm_host_immunocompromised"),
        _item_ref("bm_host_csf_leak_or_neurosurgery"),
        _item_ref("bm_host_otitis_sinusitis"),
        _item_ref("bm_host_bacteremia_sepsis"),
    ),
    "encephalitis": (
        _item_ref("enc_host_immunocompromised"),
        _item_ref("enc_host_transplant_or_biologic"),
        _item_ref("enc_host_vector_travel_exposure"),
    ),
    "spinal_epidural_abscess": (
        _item_ref("sea_host_ivdu"),
        _item_ref("sea_host_diabetes"),
        _item_ref("sea_host_immunocompromised"),
        _item_ref("sea_host_recent_spinal_procedure"),
        _item_ref("sea_host_bacteremia_or_ssti"),
    ),
    "brain_abscess": (
        _item_ref("ba_host_otogenic_sinus_dental"),
        _item_ref("ba_host_neurosurgery_trauma"),
        _item_ref("ba_host_endocarditis_bacteremia"),
        _item_ref("ba_host_immunocompromised"),
    ),
    "necrotizing_soft_tissue_infection": (
        _item_ref("nsti_host_diabetes"),
        _item_ref("nsti_host_immunocompromised"),
        _item_ref("nsti_host_ivdu"),
        _item_ref("nsti_host_recent_surgery_or_trauma"),
        _item_ref("nsti_host_perineal_or_chronic_wound_source"),
    ),
}


def get_pretest_factor_tuning(module_id: str) -> PretestFactorTuning:
    return PRETEST_FACTOR_TUNING.get(module_id, DEFAULT_PRETEST_FACTOR_TUNING)


def _resolve_item(module: SyndromeModule, item_id: str) -> Optional[LRItem]:
    return next((item for item in module.items if item.id == item_id), None)


def _resolve_source_note(item: Optional[LRItem], ref: PretestFactorRef) -> str:
    if ref.source_note:
        return ref.source_note
    if item and item.source and item.source.short:
        return item.source.short
    if item and item.notes:
        return item.notes
    return "Module-defined baseline risk factor."


def resolve_pretest_factor_specs(module: SyndromeModule) -> List[PretestFactorSpec]:
    refs = PRETEST_FACTOR_CONFIG.get(module.id, ())
    if not refs:
        return []

    specs: List[PretestFactorSpec] = []
    for ref in refs:
        item = _resolve_item(module, ref.module_item_id) if ref.module_item_id else None
        if ref.module_item_id and item is None:
            continue
        label = ref.label or (item.label if item is not None else ref.id.replace("_", " ").title())
        weight = ref.weight if ref.weight is not None else float(item.lr_pos or 1.0)
        source_url = ref.source_url or (item.source.url if item and item.source else None)
        specs.append(
            PretestFactorSpec(
                id=ref.id,
                label=label,
                weight=weight,
                source_note=_resolve_source_note(item, ref),
                source_url=source_url,
                context_group=ref.context_group,
                suppressed_by_scores=ref.suppressed_by_scores,
            )
        )
    return specs


def get_pretest_factor_spec(
    module_id: str,
    factor_id: str,
    *,
    module: Optional[SyndromeModule] = None,
) -> Optional[PretestFactorSpec]:
    refs = PRETEST_FACTOR_CONFIG.get(module_id, ())
    ref = next((entry for entry in refs if entry.id == factor_id), None)
    if ref is None:
        return None
    if ref.module_item_id and module is None:
        return None
    if module is None:
        return PretestFactorSpec(
            id=ref.id,
            label=ref.label or ref.id.replace("_", " ").title(),
            weight=ref.weight or 1.0,
            source_note=ref.source_note or "Configured baseline risk factor.",
            source_url=ref.source_url,
            context_group=ref.context_group,
            suppressed_by_scores=ref.suppressed_by_scores,
        )
    specs = resolve_pretest_factor_specs(module)
    return next((spec for spec in specs if spec.id == factor_id), None)
