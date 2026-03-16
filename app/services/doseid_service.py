from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional


BioSex = Literal["male", "female"]
RenalMode = Literal["standard", "ihd", "crrt"]
WeightBasis = Literal["tbw", "ibw", "adjbw", "lbw"]

ANTIBACTERIAL_SOURCE = (
    "Cross-check: UCSF IDMP + Nebraska Medicine Antimicrobial Renal Dosing Guidance"
)
TB_SOURCE = "Cross-check: UCSF IDMP + CDC/ATS/IDSA TB guidance"
ANTIFUNGAL_SOURCE = (
    "Cross-check: UCSF IDMP + Nebraska Medicine Antimicrobial Renal Dosing Guidance"
)
ANTIVIRAL_SOURCE = (
    "Cross-check: UCSF IDMP + Nebraska Medicine Antimicrobial Renal Dosing Guidance"
)


class DoseIDError(RuntimeError):
    pass


@dataclass(frozen=True)
class DoseWeight:
    basis: WeightBasis
    kg: float


@dataclass(frozen=True)
class NormalizedPatient:
    age_years: int
    sex: BioSex
    total_body_weight_kg: float
    height_cm: float
    serum_creatinine_mg_dl: float
    bmi: float
    ibw_kg: float
    adjbw_kg: float
    lbw_kg: float
    crcl_weight_kg: float
    crcl_ml_min: float


@dataclass(frozen=True)
class DoseResult:
    regimen: str
    renal_bucket: str
    notes: List[str]
    dose_weight: Optional[DoseWeight] = None


@dataclass(frozen=True)
class MedicationIndication:
    id: str
    label: str


@dataclass(frozen=True)
class MedicationRule:
    id: str
    name: str
    category: str
    indications: List[MedicationIndication]
    source_pages: str
    calculate: Callable[[NormalizedPatient, str, RenalMode], DoseResult]


def round_dose(value: float, step: int = 50) -> int:
    return int(round(value / step) * step)


def bmi_from_kg_cm(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    if height_m <= 0:
        return 0.0
    return weight_kg / (height_m * height_m)


def ibw_kg(sex: BioSex, height_cm: float) -> float:
    height_in = height_cm / 2.54
    base = 50.0 if sex == "male" else 45.0
    over_5_ft = max(0.0, height_in - 60.0)
    return base + (2.3 * over_5_ft)


def adjbw_kg(tbw_kg: float, ibw_value: float) -> float:
    return ibw_value + (0.4 * (tbw_kg - ibw_value))


def lbw_kg(sex: BioSex, tbw_kg: float, bmi: float) -> float:
    if sex == "male":
        return (9270.0 * tbw_kg) / (6680.0 + (216.0 * bmi))
    return (9270.0 * tbw_kg) / (8780.0 + (244.0 * bmi))


def crcl_weight_kg(tbw_kg: float, ibw_value: float, bmi: float) -> float:
    if bmi >= 30:
        return adjbw_kg(tbw_kg, ibw_value)
    return tbw_kg


def cockcroft_gault_ml_min(age_years: int, sex: BioSex, scr_mg_dl: float, weight_kg: float) -> float:
    if age_years <= 0 or scr_mg_dl <= 0 or weight_kg <= 0:
        return 0.0
    base = ((140 - age_years) * weight_kg) / (72 * scr_mg_dl)
    return base * 0.85 if sex == "female" else base


def normalize_patient(
    *,
    age_years: int,
    sex: BioSex,
    total_body_weight_kg: float,
    height_cm: float,
    serum_creatinine_mg_dl: float,
) -> NormalizedPatient:
    bmi = bmi_from_kg_cm(total_body_weight_kg, height_cm)
    ibw_value = ibw_kg(sex, height_cm)
    adjbw_value = adjbw_kg(total_body_weight_kg, ibw_value)
    lbw_value = lbw_kg(sex, total_body_weight_kg, bmi)
    crcl_weight_value = crcl_weight_kg(total_body_weight_kg, ibw_value, bmi)
    crcl_value = cockcroft_gault_ml_min(age_years, sex, serum_creatinine_mg_dl, crcl_weight_value)
    return NormalizedPatient(
        age_years=age_years,
        sex=sex,
        total_body_weight_kg=total_body_weight_kg,
        height_cm=height_cm,
        serum_creatinine_mg_dl=serum_creatinine_mg_dl,
        bmi=bmi,
        ibw_kg=ibw_value,
        adjbw_kg=adjbw_value,
        lbw_kg=lbw_value,
        crcl_weight_kg=crcl_weight_value,
        crcl_ml_min=crcl_value,
    )


def normalize_patient_from_available_inputs(
    *,
    total_body_weight_kg: float | None = None,
    age_years: int | None = None,
    sex: BioSex | None = None,
    height_cm: float | None = None,
    serum_creatinine_mg_dl: float | None = None,
    crcl_ml_min: float | None = None,
    renal_mode: RenalMode = "standard",
) -> tuple[NormalizedPatient, List[str]]:
    assumptions: List[str] = []

    if total_body_weight_kg is None or total_body_weight_kg <= 0:
        total_body_weight_kg = 70.0
        assumptions.append("Weight was not provided, so a 70 kg placeholder was used for scaffolding and any weight-based pathway should be confirmed.")
    if age_years is None or age_years <= 0:
        age_years = 50
        assumptions.append("Age was not provided, so 50 years was used for renal and body-size scaffolding.")
    if sex is None:
        sex = "male"
        assumptions.append("Sex was not provided, so male was used for body-size estimates.")
    if height_cm is None or height_cm <= 0:
        height_cm = 170.0
        assumptions.append("Height was not provided, so 170 cm was used for obesity-adjusted body-size estimates.")

    bmi = bmi_from_kg_cm(total_body_weight_kg, height_cm)
    ibw_value = ibw_kg(sex, height_cm)
    adjbw_value = adjbw_kg(total_body_weight_kg, ibw_value)
    lbw_value = lbw_kg(sex, total_body_weight_kg, bmi)
    crcl_weight_value = crcl_weight_kg(total_body_weight_kg, ibw_value, bmi)

    if crcl_ml_min is None:
        if renal_mode == "standard":
            if serum_creatinine_mg_dl is None or serum_creatinine_mg_dl <= 0:
                raise DoseIDError("Need either serum creatinine or creatinine clearance for non-dialysis dosing.")
            crcl_value = cockcroft_gault_ml_min(age_years, sex, serum_creatinine_mg_dl, crcl_weight_value)
        else:
            crcl_value = 0.0
            if serum_creatinine_mg_dl is None or serum_creatinine_mg_dl <= 0:
                serum_creatinine_mg_dl = 1.0
                assumptions.append("Serum creatinine was not provided, but the selected dialysis pathway does not require it for many fixed-dose regimens.")
    else:
        crcl_value = crcl_ml_min
        if serum_creatinine_mg_dl is None or serum_creatinine_mg_dl <= 0:
            serum_creatinine_mg_dl = 1.0
            assumptions.append("Direct creatinine clearance was used; serum creatinine was not provided and is not driving the renal bucket.")

    if serum_creatinine_mg_dl is None or serum_creatinine_mg_dl <= 0:
        serum_creatinine_mg_dl = 1.0

    return (
        NormalizedPatient(
            age_years=age_years,
            sex=sex,
            total_body_weight_kg=total_body_weight_kg,
            height_cm=height_cm,
            serum_creatinine_mg_dl=serum_creatinine_mg_dl,
            bmi=bmi,
            ibw_kg=ibw_value,
            adjbw_kg=adjbw_value,
            lbw_kg=lbw_value,
            crcl_weight_kg=crcl_weight_value,
            crcl_ml_min=crcl_value,
        ),
        assumptions,
    )


def obesity_adjusted_weight(patient: NormalizedPatient) -> DoseWeight:
    if patient.bmi >= 30:
        return DoseWeight(basis="adjbw", kg=patient.adjbw_kg)
    return DoseWeight(basis="tbw", kg=patient.total_body_weight_kg)


def adjusted_weight_over_120_ibw(patient: NormalizedPatient) -> DoseWeight:
    if patient.total_body_weight_kg > patient.ibw_kg * 1.2:
        return DoseWeight(basis="adjbw", kg=patient.adjbw_kg)
    return DoseWeight(basis="tbw", kg=patient.total_body_weight_kg)


def foscarnet_adjusted_crcl_ml_min_per_kg(patient: NormalizedPatient) -> float:
    if patient.serum_creatinine_mg_dl <= 0:
        return 0.0
    sex_factor = 0.85 if patient.sex == "female" else 1.0
    return ((140 - patient.age_years) * sex_factor) / (72 * patient.serum_creatinine_mg_dl)


def no_renal_adjust_bucket(mode: RenalMode) -> str:
    if mode == "standard":
        return "No routine renal adjustment in major references"
    if mode == "ihd":
        return "Intermittent hemodialysis"
    return "Continuous renal replacement therapy (CRRT)"


def renal_mode_label(mode: RenalMode) -> str:
    if mode == "standard":
        return "Standard renal pathway"
    if mode == "ihd":
        return "Intermittent hemodialysis (iHD)"
    return "Continuous renal replacement therapy (CRRT)"


def crcl_band(patient: NormalizedPatient, bands: List[int]) -> str:
    crcl = patient.crcl_ml_min
    upper, mid, low = bands
    if crcl > upper:
        return f"CrCl > {upper} mL/min"
    if crcl > mid:
        return f"CrCl {mid + 1}-{upper} mL/min"
    if crcl > low:
        return f"CrCl {low + 1}-{mid} mL/min"
    return f"CrCl <= {low} mL/min"


def mg_from_weight(mg_per_kg: float, weight_kg: float, step: int = 50, max_mg: int | None = None) -> int:
    rounded = round_dose(mg_per_kg * weight_kg, step)
    if max_mg is not None:
        return min(max_mg, rounded)
    return rounded


def _tmp_range(
    patient: NormalizedPatient,
    min_mg_per_kg_per_day: float,
    max_mg_per_kg_per_day: float,
    factor: float = 1.0,
) -> tuple[str, DoseWeight]:
    dose_weight = obesity_adjusted_weight(patient)
    low = mg_from_weight(min_mg_per_kg_per_day * factor, dose_weight.kg, 40)
    high = mg_from_weight(max_mg_per_kg_per_day * factor, dose_weight.kg, 40)
    return (str(low) if low == high else f"{low}-{high}"), dose_weight


def _cefepime(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    if renal_mode == "ihd":
        return DoseResult(
            regimen="2 g IV post-HD after each session",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Alternative institutional approach: 1 g IV qPM with post-HD timing alignment.",
                "CNS infection may need intensive PK/PD optimization.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q8h",
            renal_bucket="CRRT",
            notes=[
                "CRRT dosing is modality and effluent-dependent; confirm with ICU pharmacist.",
                "Template favors high beta-lactam exposure in critical illness.",
            ],
        )
    regimen = "1 g IV q24h"
    if patient.crcl_ml_min > 60:
        regimen = "2 g IV q8h"
    elif patient.crcl_ml_min > 30:
        regimen = "2 g IV q12h"
    elif patient.crcl_ml_min > 10:
        regimen = "2 g IV q24h"
    return DoseResult(
        regimen=regimen,
        renal_bucket=crcl_band(patient, [60, 30, 10]),
        notes=[
            "Cross-institution default; final dose depends on source control, MIC, and neurotoxicity risk.",
            "Dialysis and CRRT pathways can be selected in the renal function section.",
        ],
    )


def _piperacillin_tazobactam(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    high_inoculum = indication_id == "high_inoculum_pseudomonal"
    if renal_mode == "ihd":
        return DoseResult(
            regimen=(
                "2.25 g IV q6h (short infusion), dose after HD on dialysis days"
                if high_inoculum
                else "2.25 g IV q8h (short infusion), dose after HD on dialysis days"
            ),
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Dialysis pathway is a template; local protocol may use alternative schedules.",
                "Use extended infusion when feasible for time-above-MIC optimization.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(
                "4.5 g IV q6h (4-hour infusion)"
                if high_inoculum
                else "4.5 g IV q8h (4-hour infusion)"
            ),
            renal_bucket="CRRT",
            notes=[
                "CRRT regimens vary with effluent flow and residual renal function.",
                "Confirm strategy with ICU antimicrobial stewardship/pharmacy.",
            ],
        )
    regimen = "3.375 g IV q8h (short infusion)"
    if high_inoculum:
        regimen = "4.5 g IV q6h (4-hour infusion)" if patient.crcl_ml_min > 20 else "3.375 g IV q6h (short infusion)"
    elif patient.crcl_ml_min > 20:
        regimen = "4.5 g IV q8h (4-hour infusion)"
    return DoseResult(
        regimen=regimen,
        renal_bucket="CrCl > 20 mL/min" if patient.crcl_ml_min > 20 else "CrCl <= 20 mL/min",
        notes=[
            "Low-CrCl pathways vary across institutions; short-infusion fallback is common.",
            "Ensure indication-specific source control assumptions are met.",
        ],
    )


def _meropenem(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    is_cns = indication_id == "cns_meningitis"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="1 g IV qPM (after HD on dialysis days)" if is_cns else "500 mg IV qPM (after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "This mirrors common qPM post-HD institutional schedules.",
                "Use local susceptibility data to guide escalation/de-escalation.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q8h" if is_cns else "1 g IV q8h",
            renal_bucket="CRRT",
            notes=[
                "CRRT template assumes high-intensity critical care infection management.",
                "May need prolonged infusion per ICU protocol.",
            ],
        )
    regimen = "500 mg IV q24h"
    if is_cns:
        if patient.crcl_ml_min > 50:
            regimen = "2 g IV q8h"
        elif patient.crcl_ml_min > 25:
            regimen = "2 g IV q12h"
        elif patient.crcl_ml_min > 10:
            regimen = "1 g IV q12h"
        else:
            regimen = "1 g IV q24h"
    else:
        if patient.crcl_ml_min > 50:
            regimen = "1 g IV q8h"
        elif patient.crcl_ml_min > 25:
            regimen = "1 g IV q12h"
        elif patient.crcl_ml_min > 10:
            regimen = "500 mg IV q12h"
    return DoseResult(
        regimen=regimen,
        renal_bucket=(
            "CrCl > 50 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 26-50 mL/min"
            if patient.crcl_ml_min > 25
            else "CrCl 11-25 mL/min"
            if patient.crcl_ml_min > 10
            else "CrCl <= 10 mL/min"
        ),
        notes=["Higher regimens are indication-specific; use local guidance for resistant pathogens."],
    )


def _daptomycin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    mg_per_kg = 10 if indication_id == "vre_high_burden" else 8
    dose_weight = obesity_adjusted_weight(patient)
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
    if renal_mode == "ihd":
        return DoseResult(
            regimen=f"{dose_mg} mg IV post-HD (3x weekly)",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "For long interdialytic intervals, many protocols increase the final weekly dose.",
                "Check CK baseline and serially; evaluate for myopathy symptoms.",
            ],
        )
    if renal_mode == "crrt":
        crrt_mg_per_kg = 8 if indication_id == "vre_high_burden" else 6
        crrt_dose_mg = mg_from_weight(crrt_mg_per_kg, dose_weight.kg, 50)
        return DoseResult(
            regimen=f"{crrt_dose_mg} mg IV q24h ({crrt_mg_per_kg} mg/kg)",
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                "CRRT interval may vary by filter and effluent intensity.",
                "Higher CRRT doses can be considered for deep-seated VRE infection.",
            ],
        )
    return DoseResult(
        regimen=f"{dose_mg} mg IV {'q24h' if patient.crcl_ml_min > 30 else 'q48h'} ({mg_per_kg} mg/kg)",
        renal_bucket="CrCl > 30 mL/min" if patient.crcl_ml_min > 30 else "CrCl <= 30 mL/min",
        dose_weight=dose_weight,
        notes=[
            "This tool uses AdjBW when BMI >=30; otherwise TBW.",
            "Dose optimization should incorporate infection source and organism MIC when available.",
        ],
    )


def _ampicillin_sulbactam(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    intra_abdominal = indication_id == "surgical_or_intraabdominal"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="3 g IV q12h (administer after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Post-dialysis timing is preferred on HD days.",
                "For high-inoculum infection, confirm interval/intensity with local protocol.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="3 g IV q6h",
            renal_bucket="CRRT",
            notes=[
                "CRRT pathway is an educational template and may vary by effluent rate.",
                "Extended-infusion strategies can be considered per ICU protocol.",
            ],
        )
    regimen = "3 g IV q6h" if intra_abdominal else "1.5-3 g IV q6h"
    if patient.crcl_ml_min <= 15:
        regimen = "3 g IV q24h"
    elif patient.crcl_ml_min <= 30:
        regimen = "3 g IV q12h"
    elif not intra_abdominal:
        regimen = "3 g IV q6h"
    return DoseResult(
        regimen=regimen,
        renal_bucket=(
            "CrCl > 30 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 16-30 mL/min"
            if patient.crcl_ml_min > 15
            else "CrCl <= 15 mL/min"
        ),
        notes=[
            "Renal interval extension aligns with common antimicrobial dosing references.",
            "Use organism and source-specific targets for final regimen selection.",
        ],
    )


def _aztreonam(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    uncomplicated_uti = indication_id == "uncomplicated_uti"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="1 g IV x1, then 1 g IV qPM" if uncomplicated_uti else "2 g IV x1, then 2 g IV qPM",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "qPM maintenance is a common post-HD pathway.",
                "Use higher-intensity pathways for deep or high-inoculum infections.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q12h" if uncomplicated_uti else "2 g IV q8h",
            renal_bucket="CRRT",
            notes=[
                "CRRT dosing varies by modality and residual renal function.",
                "Confirm final interval with ICU antimicrobial stewardship when available.",
            ],
        )
    if uncomplicated_uti:
        return DoseResult(
            regimen=(
                "1 g IV q8h"
                if patient.crcl_ml_min > 30
                else "500 mg IV q8h"
                if patient.crcl_ml_min >= 10
                else "500 mg IV q12h"
            ),
            renal_bucket=(
                "CrCl > 30 mL/min"
                if patient.crcl_ml_min > 30
                else "CrCl 10-30 mL/min"
                if patient.crcl_ml_min >= 10
                else "CrCl < 10 mL/min"
            ),
            notes=[
                "Uncomplicated UTI pathway uses lower exposure targets.",
                "Escalate for bacteremia or complicated urinary source.",
            ],
        )
    return DoseResult(
        regimen=(
            "2 g IV q8h"
            if patient.crcl_ml_min > 30
            else "2 g IV q12h"
            if patient.crcl_ml_min >= 10
            else "1 g IV q12h"
        ),
        renal_bucket=(
            "CrCl > 30 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 10-30 mL/min"
            if patient.crcl_ml_min >= 10
            else "CrCl < 10 mL/min"
        ),
        notes=[
            "Systemic pathway reflects common renal-adjustment intervals.",
            "Consider infusion optimization for difficult-to-treat pathogens.",
        ],
    )


def _cefazolin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    complicated = indication_id == "complicated_or_deep"
    if renal_mode == "ihd":
        return DoseResult(
            regimen=(
                "2 g IV post-HD (consider 2 g / 2 g / 3 g across weekly HD sessions)"
                if complicated
                else "2 g IV post-HD"
            ),
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Many protocols use higher third-weekly doses for long interdialytic intervals.",
                "Align dose timing with HD schedule.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q12h",
            renal_bucket="CRRT",
            notes=[
                "CRRT pathways vary with effluent intensity and infection severity.",
                "Consult ICU pharmacy for protocol-specific adjustments.",
            ],
        )
    if complicated:
        regimen = (
            "2 g IV q8h"
            if patient.crcl_ml_min > 30
            else "2 g IV q12h"
            if patient.crcl_ml_min >= 10
            else "1 g IV q24h"
        )
        bucket = (
            "CrCl > 30 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 10-29 mL/min"
            if patient.crcl_ml_min >= 10
            else "CrCl < 10 mL/min"
        )
        notes = [
            "Complicated-pathway exposure aligns with high-burden MSSA and deep-source dosing.",
            "Source control remains essential for definitive outcomes.",
        ]
    else:
        regimen = (
            "1 g IV q8h"
            if patient.crcl_ml_min > 30
            else "1 g IV q12h"
            if patient.crcl_ml_min >= 10
            else "1 g IV q24h"
        )
        bucket = (
            "CrCl > 30 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 10-29 mL/min"
            if patient.crcl_ml_min >= 10
            else "CrCl < 10 mL/min"
        )
        notes = [
            "Uncomplicated pathway is a simplified reference regimen.",
            "Escalate to complicated pathway when infection burden is high.",
        ]
    return DoseResult(regimen=regimen, renal_bucket=bucket, notes=notes)


def _ceftriaxone(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    if indication_id == "meningitis":
        return DoseResult(
            regimen="2 g IV q12h",
            renal_bucket=no_renal_adjust_bucket(renal_mode),
            notes=[
                "No routine renal adjustment in major adult references.",
                "For CNS infection, optimize adjunctive management per syndrome guidelines.",
            ],
        )
    if indication_id == "serious_infection":
        return DoseResult(
            regimen="2 g IV q24h",
            renal_bucket=no_renal_adjust_bucket(renal_mode),
            notes=[
                "No routine renal adjustment in major adult references.",
                "Typical severe-infection pathway uses 2 g daily.",
            ],
        )
    return DoseResult(
        regimen="2 g IV q24h" if patient.bmi >= 30 else "1 g IV q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal adjustment in major adult references.",
            "In obesity, many protocols prefer at least 2 g daily for systemic infection.",
        ],
    )


def _ceftazidime(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    severe = indication_id == "pseudomonal_or_severe"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="1 g IV x1, then 1 g IV post-HD",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Post-HD administration is preferred for dialysis days.",
                "Higher-intensity strategies may be required for resistant pathogens.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q8h" if severe else "2 g IV q12h",
            renal_bucket="CRRT",
            notes=[
                "CRRT pathway depends on effluent flow and target attainment strategy.",
                "Prolonged infusion can be considered for difficult-to-treat organisms.",
            ],
        )
    regimen = "1 g IV q24h"
    if patient.crcl_ml_min > 50:
        regimen = "2 g IV q8h" if severe else "2 g IV q12h"
    elif patient.crcl_ml_min > 30:
        regimen = "2 g IV q12h"
    elif patient.crcl_ml_min > 15:
        regimen = "2 g IV q24h"
    return DoseResult(
        regimen=regimen,
        renal_bucket=(
            "CrCl > 50 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 31-50 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 16-30 mL/min"
            if patient.crcl_ml_min > 15
            else "CrCl <= 15 mL/min"
        ),
        notes=[
            "Renal intervals follow common non-dialysis reference pathways.",
            "Use susceptibility and syndrome context for final regimen selection.",
        ],
    )


def _ertapenem(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500 mg IV q24h (administer after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Some programs use post-HD three-times-weekly alternatives for stable schedules.",
                "Template favors daily pathway for simplicity.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="1 g IV q24h",
            renal_bucket="CRRT",
            notes=[
                "CRRT clearance is variable; confirm maintenance with ICU protocol.",
                "Daily regimen is commonly used as a starting point in CRRT.",
            ],
        )
    return DoseResult(
        regimen="1 g IV q24h" if patient.crcl_ml_min > 30 else "500 mg IV q24h",
        renal_bucket="CrCl > 30 mL/min" if patient.crcl_ml_min > 30 else "CrCl <= 30 mL/min",
        notes=[
            "Reference pathway uses reduced maintenance at lower CrCl.",
            "Dose finalization should include infection severity and source control.",
        ],
    )


def _linezolid(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    long_course = indication_id == "mycobacterial_or_long_course"
    return DoseResult(
        regimen="600 mg IV/PO q24h" if long_course else "600 mg IV/PO q12h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal adjustment in major references.",
            "For prolonged therapy, monitor CBC closely and consider expert-guided dose individualization.",
        ],
    )


def _levofloxacin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_dose = indication_id == "pneumonia_or_pseudomonas"
    if renal_mode == "ihd":
        return DoseResult(
            regimen=(
                "750 mg IV/PO x1, then 500 mg IV/PO q48h"
                if high_dose
                else "500 mg IV/PO x1, then 250 mg IV/PO q48h"
            ),
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Load-maintenance strategy aligns with common HD pathways.",
                "Use ECG/QT and drug-interaction review where appropriate.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(
                "750 mg IV/PO q24h"
                if high_dose
                else "750 mg IV/PO x1, then 250 mg IV/PO q24h"
            ),
            renal_bucket="CRRT",
            notes=[
                "CRRT pathway is an educational reference and may vary by modality.",
                "For severe infection, ensure PK/PD target attainment with local guidance.",
            ],
        )
    if high_dose:
        return DoseResult(
            regimen=(
                "750 mg IV/PO q24h"
                if patient.crcl_ml_min > 50
                else "750 mg IV/PO q48h"
                if patient.crcl_ml_min >= 20
                else "750 mg IV/PO x1, then 500 mg IV/PO q48h"
            ),
            renal_bucket=(
                "CrCl > 50 mL/min"
                if patient.crcl_ml_min > 50
                else "CrCl 20-49 mL/min"
                if patient.crcl_ml_min >= 20
                else "CrCl < 20 mL/min"
            ),
            notes=[
                "Higher-dose pathway is used for pneumonia and pseudomonal targets.",
                "Avoid duplicate QT-prolonging combinations when possible.",
            ],
        )
    return DoseResult(
        regimen=(
            "500 mg IV/PO q24h"
            if patient.crcl_ml_min > 50
            else "250 mg IV/PO q24h"
            if patient.crcl_ml_min >= 20
            else "500 mg IV/PO x1, then 250 mg IV/PO q48h"
        ),
        renal_bucket=(
            "CrCl > 50 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 20-49 mL/min"
            if patient.crcl_ml_min >= 20
            else "CrCl < 20 mL/min"
        ),
        notes=[
            "Standard pathway aligns with common renal-adjusted systemic use.",
            "Reassess QT, tendinopathy, aortic-risk, and drug interaction profile.",
        ],
    )


def _metronidazole(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500 mg IV/PO q12h after HD",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Dialysis-day timing should be post-HD when feasible.",
                "Fulminant anaerobic infection may still warrant q8h strategy with local review.",
            ],
        )
    if indication_id == "intraabdominal_coverage":
        return DoseResult(
            regimen="500 mg IV/PO q8h",
            renal_bucket="CRRT" if renal_mode == "crrt" else "CrCl >= 10 mL/min",
            notes=[
                "Adjunctive intra-abdominal pathway is a simplified educational regimen.",
                "Final duration should follow source-control status.",
            ],
        )
    return DoseResult(
        regimen="500 mg IV/PO q8h" if patient.crcl_ml_min >= 10 else "500 mg IV/PO q12h",
        renal_bucket="CRRT" if renal_mode == "crrt" else "CrCl >= 10 mL/min" if patient.crcl_ml_min >= 10 else "CrCl < 10 mL/min",
        notes=[
            "Renal adjustment is usually modest except at very low CrCl.",
            "In severe disease, maintain adequate exposure and reassess daily.",
        ],
    )


def _tmp_smx(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    pjp_prophylaxis = indication_id == "pjp_prophylaxis"
    pjp_treatment = indication_id == "pjp_treatment"
    steno = indication_id == "stenotrophomonas"
    cystitis = indication_id == "uncomplicated_cystitis"
    ssti = indication_id == "ssti"

    if pjp_prophylaxis:
        if renal_mode == "ihd":
            return DoseResult(
                regimen="1 SS tablet PO daily after HD (or 1 DS tablet PO three times weekly after HD)",
                renal_bucket="Intermittent hemodialysis",
                notes=[
                    "PJP prophylaxis pathway is oral and usually dose-timed after HD sessions.",
                    "Approximate IV equivalent (TMP component): 80-160 mg TMP/day.",
                    "Monitor potassium, creatinine, and blood counts during chronic prophylaxis.",
                ],
            )
        if renal_mode == "crrt":
            return DoseResult(
                regimen="1 SS tablet PO daily",
                renal_bucket="CRRT",
                notes=[
                    "CRRT prophylaxis data are limited; this is a pragmatic reference starting regimen.",
                    "Approximate IV equivalent (TMP component): 80 mg TMP/day.",
                    "Adjust with local protocol and tolerance monitoring.",
                ],
            )
        if patient.crcl_ml_min > 30:
            return DoseResult(
                regimen="1 DS tablet PO daily (or 1 DS tablet PO three times weekly)",
                renal_bucket="CrCl > 30 mL/min",
                notes=[
                    "Prophylaxis strategy can be daily or three-times-weekly based on tolerance and local protocol.",
                    "Approximate IV equivalent (TMP component): 160 mg TMP/day (or 160 mg TMP on prophylaxis days for TIW strategy).",
                    "Monitor potassium, creatinine, and blood counts during chronic prophylaxis.",
                ],
            )
        if patient.crcl_ml_min >= 15:
            return DoseResult(
                regimen="1 SS tablet PO daily (or 1 DS tablet PO three times weekly)",
                renal_bucket="CrCl 15-30 mL/min",
                notes=[
                    "Renal pathway uses reduced prophylaxis intensity.",
                    "Approximate IV equivalent (TMP component): 80-160 mg TMP/day depending on chosen prophylaxis schedule.",
                    "Monitor potassium, creatinine, and blood counts during chronic prophylaxis.",
                ],
            )
        return DoseResult(
            regimen="1 SS tablet PO three times weekly (specialist-guided at very low CrCl)",
            renal_bucket="CrCl < 15 mL/min",
            notes=[
                "At very low CrCl, prophylaxis should be individualized by ID/pharmacy.",
                "Approximate IV equivalent (TMP component): 80 mg TMP on prophylaxis days.",
                "Monitor for hyperkalemia, kidney function changes, and cytopenias.",
            ],
        )

    if cystitis:
        if renal_mode == "ihd":
            return DoseResult(
                regimen="1 DS tablet PO q24h after HD",
                renal_bucket="Intermittent hemodialysis",
                notes=[
                    "Cystitis pathway is derived from indication-based clinical-use guidance plus dialysis timing.",
                    "Approximate IV equivalent (TMP component): 160 mg TMP/day.",
                    "Dose after HD on dialysis days.",
                ],
            )
        if renal_mode == "crrt":
            return DoseResult(
                regimen="1 DS tablet PO q12-24h",
                renal_bucket="CRRT",
                notes=[
                    "CRRT oral interval is a practical reference range.",
                    "Approximate IV equivalent (TMP component): 160-320 mg TMP/day.",
                    "Adjust to clinical response and local practice.",
                ],
            )
        return DoseResult(
            regimen=(
                "1 DS tablet PO q12h"
                if patient.crcl_ml_min > 30
                else "1 DS tablet PO q24h"
                if patient.crcl_ml_min >= 15
                else "Not routinely recommended at CrCl <15 mL/min; if required, 1 SS tablet PO q24h with close monitoring"
            ),
            renal_bucket=(
                "CrCl > 30 mL/min"
                if patient.crcl_ml_min > 30
                else "CrCl 15-30 mL/min"
                if patient.crcl_ml_min >= 15
                else "CrCl < 15 mL/min"
            ),
            notes=[
                "Clinical-use pathway: uncomplicated cystitis oral dosing.",
                (
                    "Approximate IV equivalent (TMP component): 320 mg TMP/day."
                    if patient.crcl_ml_min > 30
                    else "Approximate IV equivalent (TMP component): 160 mg TMP/day."
                    if patient.crcl_ml_min >= 15
                    else "Approximate IV equivalent (TMP component): ~80 mg TMP/day if therapy is used."
                ),
                "At very low CrCl, use only with specialist oversight.",
            ],
        )

    if ssti:
        if renal_mode == "ihd":
            return DoseResult(
                regimen="1 DS tablet PO q24h after HD (up to 2 DS/day in selected severe cases)",
                renal_bucket="Intermittent hemodialysis",
                notes=[
                    "SSTI pathway reflects a common clinical-use range (1-2 DS q12h baseline).",
                    "Approximate IV equivalent (TMP component): 160-320 mg TMP/day (up to 320 mg/day for higher oral exposure).",
                    "Use higher exposure only when clinically indicated.",
                ],
            )
        if renal_mode == "crrt":
            return DoseResult(
                regimen="1-2 DS tablets PO q12h",
                renal_bucket="CRRT",
                notes=[
                    "CRRT pathway uses standard exposure range as reference.",
                    "Approximate IV equivalent (TMP component): 320-640 mg TMP/day.",
                    "Monitor potassium, renal function, and blood counts.",
                ],
            )
        return DoseResult(
            regimen=(
                "1-2 DS tablets PO q12h"
                if patient.crcl_ml_min > 30
                else "1 DS tablet PO q12h"
                if patient.crcl_ml_min >= 15
                else "Not routinely recommended at CrCl <15 mL/min; if required, 1 DS tablet PO q24h with close monitoring"
            ),
            renal_bucket=(
                "CrCl > 30 mL/min"
                if patient.crcl_ml_min > 30
                else "CrCl 15-30 mL/min"
                if patient.crcl_ml_min >= 15
                else "CrCl < 15 mL/min"
            ),
            notes=[
                "Clinical-use pathway: skin/soft tissue infection oral range.",
                (
                    "Approximate IV equivalent (TMP component): 320-640 mg TMP/day."
                    if patient.crcl_ml_min > 30
                    else "Approximate IV equivalent (TMP component): 320 mg TMP/day."
                    if patient.crcl_ml_min >= 15
                    else "Approximate IV equivalent (TMP component): ~160 mg TMP/day if therapy is used."
                ),
                "At very low CrCl, use only with specialist oversight.",
            ],
        )

    base_min = 8
    base_max = 10
    interval = "q8-12h"
    if indication_id == "staph_bone_joint":
        base_min = 8
        base_max = 8
    elif steno:
        base_min = 15
        base_max = 15
        interval = "q8h"
    elif pjp_treatment:
        base_min = 15
        base_max = 20
        interval = "q6-8h"

    if renal_mode == "ihd":
        hd_min = 5 if pjp_treatment else 7.5 if steno else 2.5
        hd_max = 7.5 if (pjp_treatment or steno) else 5
        range_text, dose_weight = _tmp_range(patient, hd_min, hd_max)
        practical_regimen = (
            "1-2 DS tablets PO q24h after HD"
            if pjp_treatment
            else "2 DS tablets PO q24h after HD"
            if steno
            else "1 DS tablet PO q24h after HD"
        )
        return DoseResult(
            regimen=practical_regimen,
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Dose displayed as trimethoprim (TMP) component.",
                f"Approximate IV equivalent (TMP component): {range_text} mg TMP/day.",
                (
                    "PJP pathway uses high-target dosing; oral suggestion is 1-2 DS tablets q24h after HD."
                    if pjp_treatment
                    else "Stenotrophomonas pathway uses the maximum-target approach; oral suggestion is 2 DS tablets q24h after HD."
                    if steno
                    else "Oral suggestion uses a practical tablet-based pathway."
                ),
                (
                    "HD pathway for PJP uses 5-7.5 mg TMP/kg/day q24h."
                    if pjp_treatment
                    else "HD pathway for Stenotrophomonas uses 5-7.5 mg TMP/kg/day q24h."
                    if steno
                    else "HD pathway for non-PJP severe indications uses 2.5-5 mg TMP/kg/day q24h."
                ),
                "Administer after HD and monitor potassium, renal function, and blood counts closely.",
            ],
        )

    if renal_mode == "crrt":
        crrt_min = 10 if pjp_treatment else 15 if steno else 5
        crrt_max = 15 if (pjp_treatment or steno) else 10
        crrt_interval = "q8h" if (pjp_treatment or steno) else "q12h"
        range_text, dose_weight = _tmp_range(patient, crrt_min, crrt_max)
        practical_regimen = (
            "2 DS tablets PO q8-12h"
            if pjp_treatment
            else "2 DS tablets PO q8h"
            if steno
            else "1-2 DS tablets PO q12h"
        )
        return DoseResult(
            regimen=practical_regimen,
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                f"Approximate IV equivalent (TMP component): {range_text} mg TMP/day divided {crrt_interval}.",
                (
                    "PJP pathway uses high-target dosing in CRRT with practical oral suggestion of 2 DS tablets q8-12h."
                    if pjp_treatment
                    else "Stenotrophomonas pathway uses maximum-target dosing in CRRT with practical oral suggestion of 2 DS tablets q8h."
                    if steno
                    else "Oral suggestion uses a practical tablet-based pathway."
                ),
                (
                    "CRRT pathway for PJP: 10-15 mg TMP/kg/day."
                    if pjp_treatment
                    else "CRRT pathway for Stenotrophomonas: 10-15 mg TMP/kg/day."
                    if steno
                    else "CRRT pathway for most severe non-PJP indications: 5-10 mg TMP/kg/day."
                ),
                "CRRT clearance varies by modality and intensity; confirm final regimen with ICU pharmacy when possible.",
            ],
        )

    factor = 1.0 if patient.crcl_ml_min > 30 else 0.5 if patient.crcl_ml_min >= 15 else 0.25
    range_text, dose_weight = _tmp_range(patient, base_min, base_max, factor)
    low25 = mg_from_weight(base_min * 0.25, dose_weight.kg, 40)
    high50 = mg_from_weight(base_max * 0.5, dose_weight.kg, 40)
    practical_regimen = (
        "2 DS tablets PO q8h"
        if patient.crcl_ml_min > 30 and (pjp_treatment or steno)
        else "2 DS tablets PO q12h"
        if patient.crcl_ml_min > 30
        else "2 DS tablets PO q12h"
        if patient.crcl_ml_min >= 15 and (pjp_treatment or steno)
        else "1 DS tablet PO q12h"
        if patient.crcl_ml_min >= 15
        else "1 DS tablet PO q12h (specialist-guided)"
        if pjp_treatment or steno
        else "1 DS tablet PO q24h (specialist-guided)"
    )
    return DoseResult(
        regimen=practical_regimen,
        renal_bucket=(
            "CrCl > 30 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 15-30 mL/min"
            if patient.crcl_ml_min >= 15
            else "CrCl < 15 mL/min"
        ),
        dose_weight=dose_weight,
        notes=[
            "Dose displayed as trimethoprim (TMP) component; uses adjusted body weight in obesity.",
            (
                "PJP pathway uses high target 15-20 mg TMP/kg/day when feasible."
                if pjp_treatment
                else "Stenotrophomonas pathway uses maximum target 15 mg TMP/kg/day when feasible."
                if steno
                else "For non-Stenotrophomonas severe indications, target selection follows indication-specific ranges."
            ),
            (
                "Approximate oral option: 2 DS tablets PO q8h (~960 mg TMP/day). Approximate IV option: TMP-based target above divided q6-8h."
                if pjp_treatment and patient.crcl_ml_min > 30
                else "Approximate oral option: 2 DS tablets PO q12h (~640 mg TMP/day). Approximate IV option: TMP-based renal-reduced target above divided q12h."
                if pjp_treatment and patient.crcl_ml_min >= 15
                else "Approximate oral option: 1 DS tablet PO q12h (~320 mg TMP/day) if treatment is pursued. Approximate IV option: low-end specialist-guided renal-reduced TMP target."
                if pjp_treatment
                else "Approximate oral option: 2 DS tablets PO q8h (~960 mg TMP/day). Approximate IV option: TMP-based target above divided q8h."
                if steno and patient.crcl_ml_min > 30
                else "Approximate oral option: 2 DS tablets PO q12h (~640 mg TMP/day). Approximate IV option: TMP-based renal-reduced target above divided q12h."
                if steno and patient.crcl_ml_min >= 15
                else "Approximate oral option: 1 DS tablet PO q12h (~320 mg TMP/day) if treatment is pursued. Approximate IV option: low-end specialist-guided renal-reduced TMP target."
                if steno
                else "Approximate oral and IV options should follow the selected indication range and renal bucket."
            ),
            (
                f"Approximate IV equivalent (TMP component): {range_text} mg TMP/day divided {interval}."
                if patient.crcl_ml_min > 30
                else f"Approximate IV equivalent with renal reduction (TMP component): {range_text} mg TMP/day."
                if patient.crcl_ml_min >= 15
                else f"At very low CrCl, if therapy is used, approximate IV equivalent target (TMP component): {low25}-{high50} mg TMP/day with specialist-guided monitoring."
            ),
            "At CrCl <15 mL/min, use is generally avoided unless benefit outweighs risk and close monitoring is available.",
        ],
    )


def _amoxicillin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_dose = indication_id == "high_dose_oral"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500 mg PO q12h (after HD on dialysis days)" if high_dose else "500 mg PO q24h (after HD)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day doses should be given after HD when feasible."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="875 mg PO q12h" if high_dose else "500 mg PO q8h",
            renal_bucket="CRRT",
            notes=["CRRT pathway is an educational template and should be locally confirmed."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="1 g PO q8h" if high_dose else "500 mg PO q8h",
            renal_bucket="CrCl > 30 mL/min",
            notes=["Oral pathway follows common renal-adjustment intervals in adult references."],
        )
    if patient.crcl_ml_min > 10:
        return DoseResult(
            regimen="500 mg PO q12h",
            renal_bucket="CrCl 11-30 mL/min",
            notes=["Lower-CrCl pathway uses interval extension."],
        )
    return DoseResult(
        regimen="500 mg PO q24h",
        renal_bucket="CrCl <= 10 mL/min",
        notes=["Advanced renal dysfunction generally requires daily maintenance."],
    )


def _amoxicillin_clavulanate(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    high_exposure = indication_id == "high_exposure_oral"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500/125 mg PO q24h (after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Avoid ER formulations in advanced renal dysfunction."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="875/125 mg PO q12h" if high_exposure else "500/125 mg PO q8h",
            renal_bucket="CRRT",
            notes=["CRRT oral pathway should be confirmed with local protocol when available."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="875/125 mg PO q12h" if high_exposure else "500/125 mg PO q8h",
            renal_bucket="CrCl > 30 mL/min",
            notes=["Avoid ER formulations in low CrCl pathways."],
        )
    if patient.crcl_ml_min > 10:
        return DoseResult(
            regimen="500/125 mg PO q12h",
            renal_bucket="CrCl 11-30 mL/min",
            notes=["Lower-CrCl pathway uses interval extension."],
        )
    return DoseResult(
        regimen="500/125 mg PO q24h",
        renal_bucket="CrCl <= 10 mL/min",
        notes=["Avoid ER formulations in advanced renal dysfunction."],
    )


def _ampicillin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_exposure = indication_id == "high_exposure"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="2 g IV q8h (after HD on dialysis days)" if high_exposure else "2 g IV q12h (after HD)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day dosing should be synchronized post-HD."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2 g IV q4h" if high_exposure else "2 g IV q6h",
            renal_bucket="CRRT",
            notes=["CRRT pathway depends on modality and effluent flow."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(
            regimen="2 g IV q4h" if high_exposure else "2 g IV q6h",
            renal_bucket="CrCl > 50 mL/min",
            notes=["Higher-intensity pathway should be guided by source and organism profile."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="2 g IV q6h",
            renal_bucket="CrCl 31-50 mL/min",
            notes=["Renal interval extension is commonly used in reference protocols."],
        )
    if patient.crcl_ml_min > 15:
        return DoseResult(
            regimen="2 g IV q8h",
            renal_bucket="CrCl 16-30 mL/min",
            notes=["Renal interval extension is commonly used in reference protocols."],
        )
    return DoseResult(
        regimen="2 g IV q12h",
        renal_bucket="CrCl <= 15 mL/min",
        notes=["Severe infection may require specialist review at very low CrCl."],
    )


def _cefiderocol(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_clearance = indication_id == "high_clearance_or_critical"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="750 mg IV q12h (3-hour infusion; dose after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis schedule and infection severity should guide final interval."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="1.5 g IV q8h (3-hour infusion)",
            renal_bucket="CRRT",
            notes=["CRRT regimens vary by effluent intensity and residual renal function."],
        )
    if patient.crcl_ml_min > 120:
        return DoseResult(
            regimen="2 g IV q6h (3-hour infusion)",
            renal_bucket="CrCl > 120 mL/min",
            notes=["Augmented renal clearance pathway."],
        )
    if patient.crcl_ml_min > 60:
        return DoseResult(
            regimen="2 g IV q6h (3-hour infusion)" if high_clearance else "2 g IV q8h (3-hour infusion)",
            renal_bucket="CrCl 61-120 mL/min",
            notes=["Use higher frequency when exposure targets are difficult to achieve."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="1.5 g IV q8h (3-hour infusion)",
            renal_bucket="CrCl 31-60 mL/min",
            notes=["Interval and dose reduction align with common reference pathways."],
        )
    if patient.crcl_ml_min > 15:
        return DoseResult(
            regimen="1 g IV q8h (3-hour infusion)",
            renal_bucket="CrCl 16-30 mL/min",
            notes=["Use susceptibility and infection site to finalize dose."],
        )
    return DoseResult(
        regimen="750 mg IV q12h (3-hour infusion)",
        renal_bucket="CrCl <= 15 mL/min",
        notes=["Very low CrCl pathways should be reviewed with stewardship/pharmacy."],
    )


def _ceftaroline(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_exposure = indication_id == "high_exposure_mrsa"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="200 mg IV q12h (after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day doses should be given after HD when feasible."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="400 mg IV q8h" if high_exposure else "400 mg IV q12h",
            renal_bucket="CRRT",
            notes=["CRRT pathway should be individualized to modality and severity."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(
            regimen="600 mg IV q8h" if high_exposure else "600 mg IV q12h",
            renal_bucket="CrCl > 50 mL/min",
            notes=["High-exposure pathway is for selected high-burden scenarios."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="400 mg IV q12h",
            renal_bucket="CrCl 31-50 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    if patient.crcl_ml_min > 15:
        return DoseResult(
            regimen="300 mg IV q12h",
            renal_bucket="CrCl 16-30 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    return DoseResult(
        regimen="200 mg IV q12h",
        renal_bucket="CrCl <= 15 mL/min",
        notes=["Very low CrCl pathway should be confirmed with local policy."],
    )


def _ceftazidime_avibactam(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    high_exposure = indication_id == "high_exposure_critical"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="0.94 g IV q24h (2-hour infusion), dose after HD on dialysis days",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis pathway may require supplemental dosing in prolonged sessions."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="2.5 g IV q8h (2-hour infusion)" if high_exposure else "1.25 g IV q8h (2-hour infusion)",
            renal_bucket="CRRT",
            notes=["CRRT strategy should be matched to effluent intensity and organism MIC."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(
            regimen="2.5 g IV q8h (2-hour infusion)",
            renal_bucket="CrCl > 50 mL/min",
            notes=["Standard preserved-renal-function pathway."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="1.25 g IV q8h (2-hour infusion)",
            renal_bucket="CrCl 31-50 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    if patient.crcl_ml_min > 15:
        return DoseResult(
            regimen="0.94 g IV q12h (2-hour infusion)",
            renal_bucket="CrCl 16-30 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    if patient.crcl_ml_min > 5:
        return DoseResult(
            regimen="0.94 g IV q24h (2-hour infusion)",
            renal_bucket="CrCl 6-15 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    return DoseResult(
        regimen="0.94 g IV q48h (2-hour infusion)",
        renal_bucket="CrCl <= 5 mL/min",
        notes=["Very low CrCl pathway should be confirmed with local protocol."],
    )


def _ciprofloxacin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_exposure = indication_id == "high_exposure_pseudomonal"
    if renal_mode == "ihd":
        return DoseResult(
            regimen=(
                "400 mg IV qPM (or 500 mg PO qPM) after HD"
                if high_exposure
                else "400 mg IV qPM (or 500 mg PO qPM)"
            ),
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day timing should be post-HD when feasible."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(
                "400 mg IV q8h (or 750 mg PO q12h)"
                if high_exposure
                else "400 mg IV q12h (or 500 mg PO q12h)"
            ),
            renal_bucket="CRRT",
            notes=["CRRT pathway should be matched to target exposure and local protocol."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(
            regimen=(
                "400 mg IV q8h (or 750 mg PO q12h)"
                if high_exposure
                else "400 mg IV q12h (or 500 mg PO q12h)"
            ),
            renal_bucket="CrCl > 50 mL/min",
            notes=["High-exposure pathway is intended for selected severe Gram-negative scenarios."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="400 mg IV q12h (or 500 mg PO q12h)",
            renal_bucket="CrCl 31-50 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    return DoseResult(
        regimen="400 mg IV q24h (or 500 mg PO q24h)",
        renal_bucket="CrCl <= 30 mL/min",
        notes=["Renal-adjusted pathway."],
    )


def _clindamycin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    bone_joint = indication_id == "bone_joint_infection"
    return DoseResult(
        regimen="900 mg IV/PO q8h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal adjustment in major adult references.",
            "Clindamycin has high oral bioavailability; PO step-down is favored when clinically appropriate and tolerated.",
            "Obesity-focused guidance supports upper-end routine dosing with matched maximum IV and PO schedules.",
            "Equivalent high-dose obesity options include 900 mg q8h or 600 mg q6h.",
            (
                "Bone and joint pathway uses maximized routine dosing to support tissue exposure."
                if bone_joint
                else "High-dose pathway selected as default to avoid underexposure in higher body-weight patients."
            ),
        ],
    )


def _imipenem_cilastatin(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    high_exposure = indication_id == "high_exposure_resistant"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500 mg IV q8h (after HD on dialysis days)" if high_exposure else "200 mg IV q12h (after HD)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day dosing should be synchronized post-HD."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="500 mg IV q6h" if high_exposure else "500 mg IV q8h",
            renal_bucket="CRRT",
            notes=["CRRT pathway is a teaching template and may vary by modality."],
        )
    if patient.crcl_ml_min > 90:
        return DoseResult(
            regimen="1 g IV q6h" if high_exposure else "500 mg IV q6h",
            renal_bucket="CrCl > 90 mL/min",
            notes=["Higher-intensity pathway should be used selectively for severe resistant infection."],
        )
    if patient.crcl_ml_min > 60:
        return DoseResult(
            regimen="400 mg IV q6h",
            renal_bucket="CrCl 61-90 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    if patient.crcl_ml_min > 30:
        return DoseResult(
            regimen="300 mg IV q6h",
            renal_bucket="CrCl 31-60 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    if patient.crcl_ml_min > 15:
        return DoseResult(
            regimen="200 mg IV q6h",
            renal_bucket="CrCl 16-30 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    return DoseResult(
        regimen="200 mg IV q12h",
        renal_bucket="CrCl <= 15 mL/min",
        notes=["Very low CrCl pathways should be confirmed with local protocol."],
    )


def _nafcillin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_burden = indication_id == "mssa_high_burden"
    return DoseResult(
        regimen="2 g IV q4h" if high_burden else "2 g IV q4-6h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=["No routine renal adjustment in major references; monitor hepatic and sodium load context."],
    )


def _penicillin_g(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_exposure = indication_id == "cns_or_high_exposure"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="3 million units IV q6h (after HD on dialysis days)" if high_exposure else "2 million units IV q6h (after HD)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day doses should be timed after HD."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="4 million units IV q4h" if high_exposure else "4 million units IV q6h",
            renal_bucket="CRRT",
            notes=["CRRT pathway should be individualized to target exposure and local protocol."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(
            regimen="4 million units IV q4h",
            renal_bucket="CrCl > 50 mL/min",
            notes=["Preserved-renal-function pathway."],
        )
    if patient.crcl_ml_min > 10:
        return DoseResult(
            regimen="3 million units IV q4h",
            renal_bucket="CrCl 11-50 mL/min",
            notes=["Renal-adjusted pathway."],
        )
    return DoseResult(
        regimen="3 million units IV q6h",
        renal_bucket="CrCl <= 10 mL/min",
        notes=["Renal interval extension for very low CrCl."],
    )


def _vancomycin_iv(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    serious = indication_id == "serious_mrsa_or_invasive"
    dose_weight = DoseWeight(basis="tbw", kg=patient.total_body_weight_kg)
    if renal_mode == "ihd":
        load_mg = mg_from_weight(25 if serious else 20, dose_weight.kg, 250, 3000)
        post_hd_mg = mg_from_weight(15 if serious else 10, dose_weight.kg, 250, 2000)
        return DoseResult(
            regimen=f"{load_mg} mg IV once, then {post_hd_mg} mg IV post-HD (AUC/level-guided)",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Maintenance must be adjusted with levels and dialysis schedule.",
                "AUC-guided monitoring is preferred.",
            ],
        )
    if renal_mode == "crrt":
        load_mg = mg_from_weight(25 if serious else 20, dose_weight.kg, 250, 3000)
        maint_mg = mg_from_weight(15 if serious else 10, dose_weight.kg, 250, 2000)
        return DoseResult(
            regimen=f"{load_mg} mg IV once, then {maint_mg} mg IV q12-24h (AUC/level-guided)",
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                "CRRT vancomycin pathways require protocol-specific level timing.",
                "AUC-guided monitoring is preferred.",
            ],
        )
    mg_per_kg = 20 if serious else 15
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 250, 2500)
    interval = "q48h or by levels"
    if patient.crcl_ml_min > 90:
        interval = "q8h"
    elif patient.crcl_ml_min > 50:
        interval = "q12h"
    elif patient.crcl_ml_min > 30:
        interval = "q24h"
    return DoseResult(
        regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg, AUC/level-guided)",
        renal_bucket=(
            "CrCl > 90 mL/min"
            if patient.crcl_ml_min > 90
            else "CrCl 51-90 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 31-50 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl <= 30 mL/min"
        ),
        dose_weight=dose_weight,
        notes=[
            "Final maintenance must be adjusted to measured levels and AUC targets.",
            "Use local vancomycin monitoring protocol for definitive dosing.",
        ],
    )


def _isoniazid(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    intermittent = indication_id == "tb_intermittent"
    dose_mg = (
        min(900, mg_from_weight(15, patient.total_body_weight_kg, 50, 900))
        if intermittent
        else min(300, mg_from_weight(5, patient.total_body_weight_kg, 25, 300))
    )
    return DoseResult(
        regimen=(
            f"{dose_mg} mg PO three times weekly (max 900 mg)"
            if intermittent
            else f"{dose_mg} mg PO daily (max 300 mg/day)"
        ),
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        dose_weight=DoseWeight(basis="tbw", kg=patient.total_body_weight_kg),
        notes=[
            "No routine renal dose reduction in major TB references; in iHD dose after dialysis on HD days.",
            "Add pyridoxine when clinically indicated.",
        ],
    )


def _rifampin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    hardware = indication_id == "hardware_adjuvant"
    return DoseResult(
        regimen="300 mg PO q12h" if hardware else "600 mg PO daily",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "Rifampin is interaction-heavy; always perform full medication reconciliation.",
            "In iHD, dose after dialysis sessions when practical.",
        ],
    )


def _ethambutol(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_dose_intermittent = indication_id == "tb_high_dose_intermittent"
    mg_per_kg = 25 if high_dose_intermittent else 20
    dose_weight = DoseWeight(basis="lbw", kg=patient.lbw_kg)
    max_dose = 2400 if high_dose_intermittent else 1600
    dose_mg = min(max_dose, mg_from_weight(mg_per_kg, dose_weight.kg, 100, max_dose))
    if renal_mode == "ihd":
        return DoseResult(
            regimen=f"{dose_mg} mg PO three times weekly post-HD",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Template uses lean body weight for obesity dosing.",
                "Use ophthalmologic toxicity monitoring per TB protocol.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(f"{dose_mg} mg PO three times weekly" if high_dose_intermittent else f"{dose_mg} mg PO q24h"),
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=["CRRT pathway is a teaching template; confirm with TB pharmacy/ID team."],
        )
    return DoseResult(
        regimen=(
            f"{dose_mg} mg PO three times weekly"
            if high_dose_intermittent
            else f"{dose_mg} mg PO {'daily' if patient.crcl_ml_min >= 30 else 'three times weekly'}"
        ),
        renal_bucket="CrCl >= 30 mL/min" if patient.crcl_ml_min >= 30 else "CrCl < 30 mL/min",
        dose_weight=dose_weight,
        notes=[
            "Daily dose aligned to common institutional range (~20 mg/kg; upper ranges up to 24 mg/kg exist).",
            "Renal dysfunction usually requires interval extension for standard regimens.",
        ],
    )


def _pyrazinamide(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_dose_intermittent = indication_id == "tb_high_dose_intermittent"
    mg_per_kg = 35 if high_dose_intermittent else 20
    dose_weight = DoseWeight(basis="lbw", kg=patient.lbw_kg)
    max_dose = 3000 if high_dose_intermittent else 2000
    dose_mg = min(max_dose, mg_from_weight(mg_per_kg, dose_weight.kg, 100, max_dose))
    if renal_mode == "ihd":
        return DoseResult(
            regimen=f"{dose_mg} mg PO three times weekly post-HD",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Dialysis pathway generally uses intermittent post-HD administration.",
                "Monitor liver tests and uric acid when clinically indicated.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(f"{dose_mg} mg PO three times weekly" if high_dose_intermittent else f"{dose_mg} mg PO q24h"),
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=["CRRT pathway is a template and requires specialist confirmation."],
        )
    return DoseResult(
        regimen=(
            f"{dose_mg} mg PO three times weekly"
            if high_dose_intermittent
            else f"{dose_mg} mg PO {'daily' if patient.crcl_ml_min >= 30 else 'three times weekly'}"
        ),
        renal_bucket="CrCl >= 30 mL/min" if patient.crcl_ml_min >= 30 else "CrCl < 30 mL/min",
        dose_weight=dose_weight,
        notes=[
            "Daily pathway aligned to common institutional range (~20-25 mg/kg).",
            "In renal dysfunction, interval extension is usually preferred.",
        ],
    )


def _moxifloxacin_tb(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    return DoseResult(
        regimen="400 mg PO/IV q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal adjustment in major references.",
            "QT interval and drug-interaction review are recommended.",
        ],
    )


def _rifabutin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    low_clearance = indication_id == "renal_impairment_low_clearance" or patient.crcl_ml_min < 30
    return DoseResult(
        regimen="150 mg PO q24h" if low_clearance else "300 mg PO q24h",
        renal_bucket=(
            "CrCl >= 30 mL/min"
            if renal_mode == "standard" and patient.crcl_ml_min >= 30
            else "CrCl < 30 mL/min"
            if renal_mode == "standard"
            else renal_mode_label(renal_mode)
        ),
        notes=[
            "Major pathway concern is drug-drug interaction burden (CYP induction).",
            "Dialysis-specific data are limited; use specialist confirmation for iHD/CRRT.",
        ],
    )


def _caspofungin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    return DoseResult(
        regimen="70 mg IV once, then 50 mg IV q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=["No routine renal adjustment in major references."],
    )


def _isavuconazole(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    stepdown = indication_id == "stepdown_oral"
    return DoseResult(
        regimen="372 mg PO q24h (after loading)" if stepdown else "372 mg IV/PO q8h x6 doses, then 372 mg IV/PO q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=["No routine renal adjustment in major references."],
    )


def _posaconazole(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    return DoseResult(
        regimen="300 mg PO/IV q12h x2 doses, then 300 mg PO/IV q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal dose adjustment in major references.",
            "For IV route in low CrCl, vehicle accumulation concerns may favor oral delayed-release tablets when feasible.",
        ],
    )


def _fluconazole(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    invasive = indication_id == "candidemia_invasive"
    if renal_mode == "ihd":
        return DoseResult(
            regimen=(
                "800 mg PO/IV once, then 400 mg after each HD session"
                if invasive
                else "200 mg PO/IV once, then 100-200 mg after each HD session"
            ),
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Give maintenance dose after dialysis sessions.",
                "For invasive Candida, verify organism susceptibility and source control.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen=(
                "800 mg PO/IV once, then 800-1200 mg/day divided q12-24h"
                if invasive
                else "200 mg PO/IV once, then 200 mg q24h"
            ),
            renal_bucket="CRRT",
            notes=[
                "CRRT may clear fluconazole substantially; higher maintenance can be required.",
                "Use local ICU antifungal protocol where available.",
            ],
        )
    if invasive:
        return DoseResult(
            regimen=(
                "800 mg PO/IV once, then 400 mg q24h"
                if patient.crcl_ml_min > 50
                else "800 mg PO/IV once, then 200 mg q24h"
            ),
            renal_bucket="CrCl > 50 mL/min" if patient.crcl_ml_min > 50 else "CrCl <= 50 mL/min",
            notes=[
                "Cross-institution references typically reduce maintenance by ~50% when CrCl <=50 mL/min.",
                "Use susceptibility, source control, and species context for final maintenance.",
            ],
        )
    return DoseResult(
        regimen=(
            "200 mg PO/IV once, then 100 mg q24h"
            if patient.crcl_ml_min > 50
            else "200 mg PO/IV once, then 50 mg q24h"
        ),
        renal_bucket="CrCl > 50 mL/min" if patient.crcl_ml_min > 50 else "CrCl <= 50 mL/min",
        notes=[
            "Mucosal pathway is a simplified educational regimen.",
            "Adjust by syndrome severity and treatment duration guidance.",
        ],
    )


def _micafungin(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    esophageal = indication_id == "esophageal_candidiasis"
    return DoseResult(
        regimen="150 mg IV q24h" if esophageal else "100 mg IV q24h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        notes=[
            "No routine renal adjustment in major references.",
            "Track hepatic profile and treatment response during therapy.",
        ],
    )


def _voriconazole(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    treatment = indication_id == "invasive_mold_treatment"
    dose_weight = adjusted_weight_over_120_ibw(patient)
    if not treatment:
        if patient.bmi >= 30:
            prophylaxis_mg = mg_from_weight(4, dose_weight.kg, 50)
            return DoseResult(
                regimen=f"{prophylaxis_mg} mg IV/PO q12h (4 mg/kg AdjBW obesity pathway)",
                renal_bucket=no_renal_adjust_bucket(renal_mode),
                dose_weight=dose_weight,
                notes=[
                    "Obesity prophylaxis pathway (BMI >=30) uses weight-based dosing: 4 mg/kg q12h with adjusted body weight.",
                    "No routine renal dose adjustment in major references.",
                    "For prolonged therapy with CrCl <50 mL/min, oral route is commonly preferred over IV due to SBECD vehicle exposure.",
                ],
            )
        return DoseResult(
            regimen="200 mg PO/IV q12h",
            renal_bucket=no_renal_adjust_bucket(renal_mode),
            notes=[
                "Non-obesity prophylaxis pathway: fixed 200 mg q12h.",
                "No routine renal dose adjustment in major references.",
                "For prolonged therapy with CrCl <50 mL/min, oral route is commonly preferred over IV due to SBECD vehicle exposure.",
            ],
        )
    load_mg = mg_from_weight(6, dose_weight.kg, 50)
    maint_mg = mg_from_weight(4, dose_weight.kg, 50)
    return DoseResult(
        regimen=f"{load_mg} mg IV q12h x2 doses, then {maint_mg} mg IV/PO q12h",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        dose_weight=dose_weight,
        notes=[
            "Weight-based pathway uses AdjBW when TBW >120% of IBW.",
            "Therapeutic drug monitoring is recommended when available.",
        ],
    )


def _liposomal_amphotericin_b(
    patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode
) -> DoseResult:
    cns = indication_id == "cryptococcal_cns_induction"
    dose_weight = adjusted_weight_over_120_ibw(patient)
    mg_per_kg = 4 if cns else 5
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
    return DoseResult(
        regimen=f"{dose_mg} mg IV q24h ({mg_per_kg} mg/kg)",
        renal_bucket=no_renal_adjust_bucket(renal_mode),
        dose_weight=dose_weight,
        notes=[
            "No routine renal dose adjustment; nephrotoxicity risk remains significant.",
            "Monitor creatinine, potassium, and magnesium frequently during therapy.",
        ],
    )


def _foscarnet(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    cmv_induction = indication_id == "cmv_disease_induction"
    dose_weight = adjusted_weight_over_120_ibw(patient)
    adjusted_crcl = foscarnet_adjusted_crcl_ml_min_per_kg(patient)
    renal_band = (
        "> 1.4 mL/min/kg" if adjusted_crcl > 1.4 else
        "1.0-1.4 mL/min/kg" if adjusted_crcl > 1.0 else
        "0.8-1.0 mL/min/kg" if adjusted_crcl > 0.8 else
        "0.6-0.8 mL/min/kg" if adjusted_crcl > 0.6 else
        "0.5-0.6 mL/min/kg" if adjusted_crcl > 0.5 else
        "0.4-0.5 mL/min/kg" if adjusted_crcl >= 0.4 else
        "< 0.4 mL/min/kg"
    )
    adjusted_label = f"Adjusted CrCl {adjusted_crcl:.2f} mL/min/kg ({renal_band})"
    if renal_mode == "ihd":
        mg_per_kg = 60 if cmv_induction else 45
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 500)
        return DoseResult(
            regimen=f"{dose_mg} mg IV post-HD ({mg_per_kg} mg/kg; level/clinical-guided redosing)",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Dialysis pathway should be individualized with specialist support.",
                "Adjusted CrCl formula is used for non-dialysis pathways; HD uses post-dialysis dosing templates.",
                "Aggressive hydration and electrolyte monitoring are required.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="No standardized CRRT foscarnet dose in major references. Use an individualized protocol with ID/pharmacy support.",
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                "CRRT foscarnet pathways vary substantially by modality and effluent rate.",
                "Monitor creatinine, calcium, magnesium, phosphate, and potassium closely.",
            ],
        )
    if adjusted_crcl < 0.4:
        return DoseResult(
            regimen="Not recommended at adjusted CrCl < 0.4 mL/min/kg without specialist-guided individualized dosing",
            renal_bucket=adjusted_label,
            dose_weight=dose_weight,
            notes=[
                "Adjusted CrCl (mL/min/kg) = ((140 - age) x sex factor) / (72 x SCr), with sex factor 0.85 for females.",
                "For obesity, dosing weight is TBW unless TBW >120% of IBW, then AdjBW.",
                "Aggressive hydration and electrolyte repletion are mandatory when foscarnet is used.",
            ],
        )
    if cmv_induction:
        mg_per_kg = 50
        interval = "q24h"
        if adjusted_crcl > 1.4:
            mg_per_kg, interval = 90, "q12h"
        elif adjusted_crcl > 1.0:
            mg_per_kg, interval = 70, "q12h"
        elif adjusted_crcl > 0.8:
            mg_per_kg, interval = 50, "q12h"
        elif adjusted_crcl > 0.6:
            mg_per_kg, interval = 80, "q24h"
        elif adjusted_crcl > 0.5:
            mg_per_kg, interval = 60, "q24h"
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 500)
        return DoseResult(
            regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
            renal_bucket=adjusted_label,
            dose_weight=dose_weight,
            notes=[
                "Adjusted CrCl (mL/min/kg) = ((140 - age) x sex factor) / (72 x SCr), with sex factor 0.85 for females.",
                "For obesity, dosing weight is TBW unless TBW >120% of IBW, then AdjBW.",
                "Hydration and electrolyte repletion are mandatory during therapy.",
            ],
        )
    mg_per_kg = 35
    interval = "q24h"
    if adjusted_crcl > 1.4:
        mg_per_kg, interval = 40, "q8h"
    elif adjusted_crcl > 1.0:
        mg_per_kg, interval = 30, "q8h"
    elif adjusted_crcl > 0.8:
        mg_per_kg, interval = 35, "q12h"
    elif adjusted_crcl > 0.6:
        mg_per_kg, interval = 25, "q12h"
    elif adjusted_crcl > 0.5:
        mg_per_kg, interval = 40, "q24h"
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 500)
    return DoseResult(
        regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
        renal_bucket=adjusted_label,
        dose_weight=dose_weight,
        notes=[
            "Adjusted CrCl (mL/min/kg) = ((140 - age) x sex factor) / (72 x SCr), with sex factor 0.85 for females.",
            "For obesity, dosing weight is TBW unless TBW >120% of IBW, then AdjBW.",
            "Monitor renal function and electrolytes frequently throughout treatment.",
        ],
    )


def _famciclovir(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    if renal_mode == "ihd":
        if indication_id == "herpes_zoster":
            return DoseResult(
                regimen="250 mg PO after each hemodialysis session",
                renal_bucket="Intermittent hemodialysis",
                notes=[
                    "Hemodialysis pathway is directly from FDA/DailyMed renal dosing table.",
                    "Initiate at first sign of symptoms for maximal benefit.",
                ],
            )
        if indication_id == "recurrent_genital_hsv":
            return DoseResult(
                regimen="250 mg PO single dose following dialysis",
                renal_bucket="Intermittent hemodialysis",
                notes=[
                    "Single-day episodic pathway is from FDA/DailyMed renal dosing table.",
                    "Start at prodrome or earliest lesion onset.",
                ],
            )
        return DoseResult(
            regimen="125 mg PO following each dialysis",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Suppressive pathway is from FDA/DailyMed renal dosing table.",
                "Dose reductions are required to reduce renal toxicity risk.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="No standardized CRRT famciclovir regimen in major references; use specialist-guided individualized dosing",
            renal_bucket="CRRT",
            notes=[
                "Published dose tables specify CrCl- and HD-based adjustments, with limited CRRT-specific data.",
                "Review antiviral strategy with ID/pharmacy and monitor renal function closely.",
            ],
        )
    if indication_id == "herpes_zoster":
        if patient.crcl_ml_min >= 60:
            return DoseResult(regimen="500 mg PO q8h for 7 days", renal_bucket="CrCl >= 60 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        if patient.crcl_ml_min >= 40:
            return DoseResult(regimen="500 mg PO q12h for 7 days", renal_bucket="CrCl 40-59 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        if patient.crcl_ml_min >= 20:
            return DoseResult(regimen="500 mg PO q24h for 7 days", renal_bucket="CrCl 20-39 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        return DoseResult(regimen="250 mg PO q24h for 7 days", renal_bucket="CrCl < 20 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
    if indication_id == "recurrent_genital_hsv":
        if patient.crcl_ml_min >= 60:
            return DoseResult(regimen="1000 mg PO q12h for 1 day (2 doses total)", renal_bucket="CrCl >= 60 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        if patient.crcl_ml_min >= 40:
            return DoseResult(regimen="500 mg PO q12h for 1 day (2 doses total)", renal_bucket="CrCl 40-59 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        if patient.crcl_ml_min >= 20:
            return DoseResult(regimen="500 mg PO single dose", renal_bucket="CrCl 20-39 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
        return DoseResult(regimen="250 mg PO single dose", renal_bucket="CrCl < 20 mL/min", notes=["FDA/DailyMed adult renal table pathway."])
    if patient.crcl_ml_min >= 40:
        return DoseResult(regimen="250 mg PO q12h", renal_bucket="CrCl >= 40 mL/min", notes=["FDA/DailyMed suppressive-therapy renal table pathway."])
    if patient.crcl_ml_min >= 20:
        return DoseResult(regimen="125 mg PO q12h", renal_bucket="CrCl 20-39 mL/min", notes=["FDA/DailyMed suppressive-therapy renal table pathway."])
    return DoseResult(regimen="125 mg PO q24h", renal_bucket="CrCl < 20 mL/min", notes=["FDA/DailyMed suppressive-therapy renal table pathway."])


def _acyclovir_po(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    zoster = indication_id == "zoster_or_severe_hsv"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="800 mg PO q12h (after HD on dialysis days)" if zoster else "200 mg PO q12h (after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day administration should be timed after HD."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="800 mg PO q8h" if zoster else "400 mg PO q8h",
            renal_bucket="CRRT",
            notes=["CRRT pathway is an educational template and may vary by modality."],
        )
    if zoster:
        return DoseResult(
            regimen=(
                "800 mg PO five times daily (q4h while awake)"
                if patient.crcl_ml_min > 25
                else "800 mg PO q8h"
                if patient.crcl_ml_min >= 10
                else "800 mg PO q12h"
            ),
            renal_bucket=(
                "CrCl > 25 mL/min"
                if patient.crcl_ml_min > 25
                else "CrCl 10-25 mL/min"
                if patient.crcl_ml_min >= 10
                else "CrCl < 10 mL/min"
            ),
            notes=["High-frequency oral pathway should be matched to indication and tolerability."],
        )
    return DoseResult(
        regimen=(
            "400 mg PO q8h"
            if patient.crcl_ml_min > 25
            else "200 mg PO q8h"
            if patient.crcl_ml_min >= 10
            else "200 mg PO q12h"
        ),
        renal_bucket=(
            "CrCl > 25 mL/min"
            if patient.crcl_ml_min > 25
            else "CrCl 10-25 mL/min"
            if patient.crcl_ml_min >= 10
            else "CrCl < 10 mL/min"
        ),
        notes=["Renal interval extension follows common oral acyclovir reference pathways."],
    )


def _acyclovir_iv(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    high_dose = indication_id == "hsv_encephalitis_or_disseminated"
    dose_weight = obesity_adjusted_weight(patient)
    if renal_mode == "ihd":
        mg_per_kg = 10 if high_dose else 5
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
        return DoseResult(
            regimen=f"{dose_mg} mg IV qPM (give after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Ensure aggressive hydration and renal monitoring.",
                "Dialysis pathway is a template; verify local protocol.",
            ],
        )
    if renal_mode == "crrt":
        mg_per_kg = 10 if high_dose else 5
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
        return DoseResult(
            regimen=f"{dose_mg} mg IV q12h ({mg_per_kg} mg/kg)",
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                "CRRT acyclovir regimens vary by modality and clearance intensity.",
                "Track renal function trends daily while on therapy.",
            ],
        )
    if high_dose:
        mg_per_kg = 10 if patient.crcl_ml_min >= 10 else 5
        interval = "q8h" if patient.crcl_ml_min > 50 else "q12h" if patient.crcl_ml_min > 25 else "q24h"
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
        return DoseResult(
            regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
            renal_bucket=(
                "CrCl > 50 mL/min"
                if patient.crcl_ml_min > 50
                else "CrCl 26-50 mL/min"
                if patient.crcl_ml_min > 25
                else "CrCl 10-25 mL/min"
                if patient.crcl_ml_min >= 10
                else "CrCl < 10 mL/min"
            ),
            dose_weight=dose_weight,
            notes=[
                "For CrCl <10, this simplified pathway drops to 5 mg/kg q24h.",
                "Consider therapeutic drug monitoring where available for prolonged therapy.",
            ],
        )
    mg_per_kg = 5 if patient.crcl_ml_min >= 10 else 2.5
    interval = "q8h" if patient.crcl_ml_min > 50 else "q12h" if patient.crcl_ml_min > 25 else "q24h"
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 50)
    return DoseResult(
        regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
        renal_bucket=(
            "CrCl > 50 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 26-50 mL/min"
            if patient.crcl_ml_min > 25
            else "CrCl 10-25 mL/min"
            if patient.crcl_ml_min >= 10
            else "CrCl < 10 mL/min"
        ),
        dose_weight=dose_weight,
        notes=[
            "This tool uses AdjBW when BMI >=30; otherwise TBW for acyclovir dosing weight.",
            "Hydration and nephrotoxicity monitoring remain essential.",
        ],
    )


def _valacyclovir(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    suppression = indication_id == "hsv_suppression"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="500 mg PO after HD (three times weekly)" if suppression else "500 mg PO q24h (after HD on dialysis days)",
            renal_bucket="Intermittent hemodialysis",
            notes=["Dialysis-day administration should be timed after HD."],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="500 mg PO q24h" if suppression else "500 mg PO q8h",
            renal_bucket="CRRT",
            notes=["CRRT pathway is an educational template and may vary by modality."],
        )
    if suppression:
        return DoseResult(
            regimen=(
                "500 mg PO q24h"
                if patient.crcl_ml_min > 30
                else "500 mg PO q48h"
                if patient.crcl_ml_min > 10
                else "500 mg PO every 72h"
            ),
            renal_bucket=(
                "CrCl > 30 mL/min"
                if patient.crcl_ml_min > 30
                else "CrCl 11-30 mL/min"
                if patient.crcl_ml_min > 10
                else "CrCl <= 10 mL/min"
            ),
            notes=["Suppressive dosing should be individualized to recurrence burden and clinical context."],
        )
    if patient.crcl_ml_min > 50:
        return DoseResult(regimen="1 g PO q8h", renal_bucket="CrCl > 50 mL/min", notes=["Standard treatment pathway for preserved renal function."])
    if patient.crcl_ml_min > 30:
        return DoseResult(regimen="1 g PO q12h", renal_bucket="CrCl 31-50 mL/min", notes=["Renal-adjusted treatment pathway."])
    if patient.crcl_ml_min > 10:
        return DoseResult(regimen="1 g PO q24h", renal_bucket="CrCl 11-30 mL/min", notes=["Renal-adjusted treatment pathway."])
    return DoseResult(regimen="500 mg PO q24h", renal_bucket="CrCl <= 10 mL/min", notes=["Very low CrCl pathway should be confirmed with local protocol."])


def _ganciclovir_iv(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    treatment = indication_id == "cmv_treatment"
    dose_weight = adjusted_weight_over_120_ibw(patient)
    if renal_mode == "ihd":
        mg_per_kg = 1.25 if treatment else 0.625
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 25)
        return DoseResult(
            regimen=f"{dose_mg} mg IV x1 now and after HD sessions ({mg_per_kg} mg/kg)",
            renal_bucket="Intermittent hemodialysis",
            dose_weight=dose_weight,
            notes=[
                "Dose displayed as IV ganciclovir total mg.",
                "CBC and renal monitoring are required during therapy.",
            ],
        )
    if renal_mode == "crrt":
        mg_per_kg = 2.5
        interval = "q12h" if treatment else "q24h"
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 25)
        return DoseResult(
            regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
            renal_bucket="CRRT",
            dose_weight=dose_weight,
            notes=[
                "CRRT pathway is an educational reference and may vary by modality.",
                "Track marrow suppression risk with serial CBC.",
            ],
        )
    if treatment:
        mg_per_kg = 1.25
        interval = "q24h"
        if patient.crcl_ml_min > 70:
            mg_per_kg, interval = 5, "q12h"
        elif patient.crcl_ml_min > 50:
            mg_per_kg, interval = 2.5, "q12h"
        elif patient.crcl_ml_min > 25:
            mg_per_kg, interval = 2.5, "q24h"
        dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 25)
        return DoseResult(
            regimen=f"{dose_mg} mg IV {interval} ({mg_per_kg} mg/kg)",
            renal_bucket=(
                "CrCl > 70 mL/min"
                if patient.crcl_ml_min > 70
                else "CrCl 51-70 mL/min"
                if patient.crcl_ml_min > 50
                else "CrCl 26-50 mL/min"
                if patient.crcl_ml_min > 25
                else "CrCl <= 25 mL/min"
            ),
            dose_weight=dose_weight,
            notes=[
                "Weight-based pathway uses AdjBW when TBW >120% of IBW.",
                "Therapy should be guided by virologic context and toxicity monitoring.",
            ],
        )
    mg_per_kg = 5 if patient.crcl_ml_min > 70 else 2.5 if patient.crcl_ml_min > 50 else 1.25 if patient.crcl_ml_min > 25 else 0.625
    dose_mg = mg_from_weight(mg_per_kg, dose_weight.kg, 25)
    return DoseResult(
        regimen=f"{dose_mg} mg IV q24h ({mg_per_kg} mg/kg)",
        renal_bucket=(
            "CrCl > 70 mL/min"
            if patient.crcl_ml_min > 70
            else "CrCl 51-70 mL/min"
            if patient.crcl_ml_min > 50
            else "CrCl 26-50 mL/min"
            if patient.crcl_ml_min > 25
            else "CrCl 11-25 mL/min"
            if patient.crcl_ml_min > 10
            else "CrCl <= 10 mL/min"
        ),
        dose_weight=dose_weight,
        notes=[
            "Prophylaxis pathway uses simplified once-daily maintenance.",
            "CBC and renal function monitoring remain required.",
        ],
    )


def _valganciclovir(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    treatment = indication_id == "cmv_treatment"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="450 mg PO post-HD" if treatment else "450 mg PO twice weekly post-HD",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Administer with food when feasible.",
                "For unstable renal function, IV ganciclovir may be preferred.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="450 mg PO q24h" if treatment else "450 mg PO q48h",
            renal_bucket="CRRT",
            notes=[
                "CRRT pathway is a simplified teaching regimen.",
                "Use CBC and renal trend monitoring throughout therapy.",
            ],
        )
    if treatment:
        if patient.crcl_ml_min > 60:
            return DoseResult(regimen="900 mg PO q12h", renal_bucket="CrCl > 60 mL/min", notes=["Standard treatment pathway in adults with preserved renal function."])
        if patient.crcl_ml_min > 40:
            return DoseResult(regimen="450 mg PO q12h", renal_bucket="CrCl 41-60 mL/min", notes=["Renal-adjusted treatment pathway."])
        if patient.crcl_ml_min > 25:
            return DoseResult(regimen="450 mg PO q24h", renal_bucket="CrCl 26-40 mL/min", notes=["Renal-adjusted treatment pathway."])
        if patient.crcl_ml_min > 10:
            return DoseResult(regimen="450 mg PO q48h", renal_bucket="CrCl 11-25 mL/min", notes=["Consider IV ganciclovir if rapid control is needed."])
        return DoseResult(
            regimen="Insufficient oral data for CrCl <= 10 mL/min; use IV ganciclovir specialist pathway",
            renal_bucket="CrCl <= 10 mL/min",
            notes=["Very low renal function generally requires individualized IV dosing."],
        )
    if patient.crcl_ml_min > 60:
        return DoseResult(regimen="900 mg PO q24h", renal_bucket="CrCl > 60 mL/min", notes=["Standard prophylaxis pathway in adults with preserved renal function."])
    if patient.crcl_ml_min > 40:
        return DoseResult(regimen="450 mg PO q24h", renal_bucket="CrCl 41-60 mL/min", notes=["Renal-adjusted prophylaxis pathway."])
    if patient.crcl_ml_min > 25:
        return DoseResult(regimen="450 mg PO q48h", renal_bucket="CrCl 26-40 mL/min", notes=["Renal-adjusted prophylaxis pathway."])
    if patient.crcl_ml_min > 10:
        return DoseResult(regimen="450 mg PO twice weekly", renal_bucket="CrCl 11-25 mL/min", notes=["Low-CrCl prophylaxis pathway should include close CBC follow-up."])
    return DoseResult(
        regimen="Insufficient oral data for CrCl <= 10 mL/min; use IV ganciclovir specialist pathway",
        renal_bucket="CrCl <= 10 mL/min",
        notes=["Very low renal function generally requires individualized IV dosing."],
    )


def _oseltamivir(patient: NormalizedPatient, indication_id: str, renal_mode: RenalMode) -> DoseResult:
    prophylaxis = indication_id == "influenza_prophylaxis"
    if renal_mode == "ihd":
        return DoseResult(
            regimen="30 mg PO once weekly post-HD" if prophylaxis else "30 mg PO x1 now, then 30 mg PO after each HD",
            renal_bucket="Intermittent hemodialysis",
            notes=[
                "Start treatment as early as possible after symptom onset when clinically indicated.",
                "Post-HD dosing is standard for iHD pathways.",
            ],
        )
    if renal_mode == "crrt":
        return DoseResult(
            regimen="30 mg PO q24h" if prophylaxis else "75 mg PO q12h",
            renal_bucket="CRRT",
            notes=[
                "CRRT pathway is a simplified educational template.",
                "Adjust to local virology and critical-care protocol when applicable.",
            ],
        )
    if prophylaxis:
        return DoseResult(
            regimen=(
                "75 mg PO q24h"
                if patient.crcl_ml_min > 60
                else "30 mg PO q24h"
                if patient.crcl_ml_min > 30
                else "30 mg PO every other day"
                if patient.crcl_ml_min > 10
                else "Not routinely recommended for CrCl <= 10 mL/min without specialist guidance"
            ),
            renal_bucket=(
                "CrCl > 60 mL/min"
                if patient.crcl_ml_min > 60
                else "CrCl 31-60 mL/min"
                if patient.crcl_ml_min > 30
                else "CrCl 11-30 mL/min"
                if patient.crcl_ml_min > 10
                else "CrCl <= 10 mL/min"
            ),
            notes=[
                "Prophylaxis should be coordinated with current outbreak and exposure context.",
                "Use local public-health recommendations for duration.",
            ],
        )
    return DoseResult(
        regimen=(
            "75 mg PO q12h"
            if patient.crcl_ml_min > 60
            else "30 mg PO q12h"
            if patient.crcl_ml_min > 30
            else "30 mg PO q24h"
            if patient.crcl_ml_min > 10
            else "Not routinely recommended for CrCl <= 10 mL/min without specialist guidance"
        ),
        renal_bucket=(
            "CrCl > 60 mL/min"
            if patient.crcl_ml_min > 60
            else "CrCl 31-60 mL/min"
            if patient.crcl_ml_min > 30
            else "CrCl 11-30 mL/min"
            if patient.crcl_ml_min > 10
            else "CrCl <= 10 mL/min"
        ),
        notes=[
            "Treatment efficacy is greatest when started early in illness.",
            "Clinical severity and local resistance trends should guide final antiviral plan.",
        ],
    )


MEDICATIONS: Dict[str, MedicationRule] = {
    "cefepime": MedicationRule(
        id="cefepime",
        name="Cefepime",
        category="antibacterial",
        indications=[
            MedicationIndication("severe_non_cns", "Severe non-CNS infection"),
            MedicationIndication("cns_meningitis", "Meningitis / CNS infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_cefepime,
    ),
    "piperacillin_tazobactam": MedicationRule(
        id="piperacillin_tazobactam",
        name="Piperacillin/Tazobactam",
        category="antibacterial",
        indications=[
            MedicationIndication("severe_ei", "Severe infection (extended infusion)"),
            MedicationIndication("high_inoculum_pseudomonal", "High inoculum / high-risk pseudomonal infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_piperacillin_tazobactam,
    ),
    "meropenem": MedicationRule(
        id="meropenem",
        name="Meropenem",
        category="antibacterial",
        indications=[
            MedicationIndication("severe_non_cns", "Severe non-CNS infection"),
            MedicationIndication("cns_meningitis", "Meningitis / CNS infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_meropenem,
    ),
    "daptomycin": MedicationRule(
        id="daptomycin",
        name="Daptomycin",
        category="antibacterial",
        indications=[
            MedicationIndication("bacteremia_endovascular", "Bacteremia / endovascular infection"),
            MedicationIndication("vre_high_burden", "High-burden VRE infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_daptomycin,
    ),
    "ampicillin_sulbactam": MedicationRule(
        id="ampicillin_sulbactam",
        name="Ampicillin/Sulbactam",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_systemic", "Standard systemic infection"),
            MedicationIndication("surgical_or_intraabdominal", "Intra-abdominal or polymicrobial source"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ampicillin_sulbactam,
    ),
    "aztreonam": MedicationRule(
        id="aztreonam",
        name="Aztreonam",
        category="antibacterial",
        indications=[
            MedicationIndication("systemic_gram_negative", "Systemic Gram-negative infection"),
            MedicationIndication("uncomplicated_uti", "Uncomplicated UTI"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_aztreonam,
    ),
    "cefazolin": MedicationRule(
        id="cefazolin",
        name="Cefazolin",
        category="antibacterial",
        indications=[
            MedicationIndication("uncomplicated_infection", "Uncomplicated infection"),
            MedicationIndication("complicated_or_deep", "Complicated / deep-seated infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_cefazolin,
    ),
    "ceftriaxone": MedicationRule(
        id="ceftriaxone",
        name="Ceftriaxone",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_dose", "Standard infection"),
            MedicationIndication("serious_infection", "Serious infection"),
            MedicationIndication("meningitis", "Meningitis / CNS infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ceftriaxone,
    ),
    "ceftazidime": MedicationRule(
        id="ceftazidime",
        name="Ceftazidime",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_systemic", "Standard systemic infection"),
            MedicationIndication("pseudomonal_or_severe", "Severe / pseudomonal infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ceftazidime,
    ),
    "ertapenem": MedicationRule(
        id="ertapenem",
        name="Ertapenem",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_systemic", "Standard systemic infection"),
            MedicationIndication("esbl_targeted", "Targeted ESBL infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ertapenem,
    ),
    "linezolid": MedicationRule(
        id="linezolid",
        name="Linezolid",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_bacterial", "Standard bacterial infection"),
            MedicationIndication("mycobacterial_or_long_course", "Mycobacterial or prolonged-course use"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_linezolid,
    ),
    "levofloxacin": MedicationRule(
        id="levofloxacin",
        name="Levofloxacin",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_infection", "Standard infection"),
            MedicationIndication("pneumonia_or_pseudomonas", "Pneumonia / pseudomonal infection"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_levofloxacin,
    ),
    "metronidazole": MedicationRule(
        id="metronidazole",
        name="Metronidazole",
        category="antibacterial",
        indications=[
            MedicationIndication("anaerobic_systemic", "Anaerobic systemic infection"),
            MedicationIndication("intraabdominal_coverage", "Intra-abdominal adjunctive coverage"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_metronidazole,
    ),
    "tmp_smx": MedicationRule(
        id="tmp_smx",
        name="Trimethoprim-Sulfamethoxazole",
        category="antibacterial",
        indications=[
            MedicationIndication("uncomplicated_cystitis", "Uncomplicated cystitis"),
            MedicationIndication("ssti", "Skin/soft tissue infection"),
            MedicationIndication("staph_bone_joint", "S. aureus bone/joint infection"),
            MedicationIndication("gnr_bacteremia", "Gram-negative rod bacteremia"),
            MedicationIndication("stenotrophomonas", "Stenotrophomonas infection"),
            MedicationIndication("pjp_treatment", "Pneumocystis jirovecii pneumonia (PJP) treatment"),
            MedicationIndication("pjp_prophylaxis", "PJP prophylaxis"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_tmp_smx,
    ),
    "amoxicillin": MedicationRule(
        id="amoxicillin",
        name="Amoxicillin",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_oral", "Standard oral infection"),
            MedicationIndication("high_dose_oral", "High-dose oral pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_amoxicillin,
    ),
    "amoxicillin_clavulanate": MedicationRule(
        id="amoxicillin_clavulanate",
        name="Amoxicillin/Clavulanate",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_oral", "Standard oral infection"),
            MedicationIndication("high_exposure_oral", "Higher oral exposure pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_amoxicillin_clavulanate,
    ),
    "ampicillin": MedicationRule(
        id="ampicillin",
        name="Ampicillin",
        category="antibacterial",
        indications=[
            MedicationIndication("systemic_standard", "Systemic infection"),
            MedicationIndication("high_exposure", "High-exposure pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ampicillin,
    ),
    "cefiderocol": MedicationRule(
        id="cefiderocol",
        name="Cefiderocol",
        category="antibacterial",
        indications=[
            MedicationIndication("resistant_gram_negative", "Resistant Gram-negative infection"),
            MedicationIndication("high_clearance_or_critical", "High-clearance / critical illness pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_cefiderocol,
    ),
    "ceftaroline": MedicationRule(
        id="ceftaroline",
        name="Ceftaroline",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_serious", "Standard serious infection"),
            MedicationIndication("high_exposure_mrsa", "High-exposure MRSA pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ceftaroline,
    ),
    "ceftazidime_avibactam": MedicationRule(
        id="ceftazidime_avibactam",
        name="Ceftazidime/Avibactam",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_resistant_gn", "Resistant Gram-negative pathway"),
            MedicationIndication("high_exposure_critical", "High-exposure critical illness pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ceftazidime_avibactam,
    ),
    "ciprofloxacin": MedicationRule(
        id="ciprofloxacin",
        name="Ciprofloxacin",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_systemic", "Standard systemic infection"),
            MedicationIndication("high_exposure_pseudomonal", "High-exposure pseudomonal pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_ciprofloxacin,
    ),
    "clindamycin": MedicationRule(
        id="clindamycin",
        name="Clindamycin",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_systemic_max", "Systemic infection (high-dose pathway)"),
            MedicationIndication("bone_joint_infection", "Bone and joint infection pathway"),
            MedicationIndication("adjunctive_toxin_suppression", "Adjunctive toxin-suppression pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_clindamycin,
    ),
    "imipenem_cilastatin": MedicationRule(
        id="imipenem_cilastatin",
        name="Imipenem/Cilastatin",
        category="antibacterial",
        indications=[
            MedicationIndication("standard_severe", "Standard severe infection"),
            MedicationIndication("high_exposure_resistant", "High-exposure resistant-pathogen pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_imipenem_cilastatin,
    ),
    "nafcillin": MedicationRule(
        id="nafcillin",
        name="Nafcillin",
        category="antibacterial",
        indications=[
            MedicationIndication("mssa_standard", "MSSA systemic infection pathway"),
            MedicationIndication("mssa_high_burden", "MSSA high-burden pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_nafcillin,
    ),
    "penicillin_g": MedicationRule(
        id="penicillin_g",
        name="Penicillin G",
        category="antibacterial",
        indications=[
            MedicationIndication("serious_streptococcal", "Serious streptococcal/systemic pathway"),
            MedicationIndication("cns_or_high_exposure", "CNS or high-exposure pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_penicillin_g,
    ),
    "vancomycin_iv": MedicationRule(
        id="vancomycin_iv",
        name="Vancomycin IV",
        category="antibacterial",
        indications=[
            MedicationIndication("serious_mrsa_or_invasive", "Serious MRSA/invasive pathway"),
            MedicationIndication("standard_systemic", "Standard systemic pathway"),
        ],
        source_pages=ANTIBACTERIAL_SOURCE,
        calculate=_vancomycin_iv,
    ),
    "isoniazid": MedicationRule(
        id="isoniazid",
        name="Isoniazid",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_daily", "Active TB daily regimen"),
            MedicationIndication("tb_intermittent", "Active TB intermittent regimen (3x weekly)"),
        ],
        source_pages=TB_SOURCE,
        calculate=_isoniazid,
    ),
    "rifampin": MedicationRule(
        id="rifampin",
        name="Rifampin",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_daily", "Active TB daily regimen"),
            MedicationIndication("hardware_adjuvant", "Staphylococcal hardware adjuvant use"),
        ],
        source_pages=TB_SOURCE,
        calculate=_rifampin,
    ),
    "ethambutol": MedicationRule(
        id="ethambutol",
        name="Ethambutol",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_standard", "Active TB standard regimen"),
            MedicationIndication("tb_high_dose_intermittent", "High-dose intermittent TB regimen"),
        ],
        source_pages=TB_SOURCE,
        calculate=_ethambutol,
    ),
    "pyrazinamide": MedicationRule(
        id="pyrazinamide",
        name="Pyrazinamide",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_standard", "Active TB standard regimen"),
            MedicationIndication("tb_high_dose_intermittent", "High-dose intermittent TB regimen"),
        ],
        source_pages=TB_SOURCE,
        calculate=_pyrazinamide,
    ),
    "moxifloxacin_tb": MedicationRule(
        id="moxifloxacin_tb",
        name="Moxifloxacin (TB/NTM)",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_alt_backbone", "TB alternative backbone regimen"),
            MedicationIndication("ntm_or_salvage", "NTM or salvage regimen component"),
        ],
        source_pages=TB_SOURCE,
        calculate=_moxifloxacin_tb,
    ),
    "rifabutin": MedicationRule(
        id="rifabutin",
        name="Rifabutin",
        category="mycobacterial_tb",
        indications=[
            MedicationIndication("tb_or_ntm_standard", "TB/NTM regimen component"),
            MedicationIndication("renal_impairment_low_clearance", "Low CrCl pathway"),
        ],
        source_pages=TB_SOURCE,
        calculate=_rifabutin,
    ),
    "caspofungin": MedicationRule(
        id="caspofungin",
        name="Caspofungin",
        category="antifungal",
        indications=[
            MedicationIndication("invasive_candidiasis", "Invasive candidiasis pathway"),
            MedicationIndication("invasive_aspergillosis_salvage", "Invasive aspergillosis salvage pathway"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_caspofungin,
    ),
    "isavuconazole": MedicationRule(
        id="isavuconazole",
        name="Isavuconazole",
        category="antifungal",
        indications=[
            MedicationIndication("invasive_mold_treatment", "Invasive mold treatment"),
            MedicationIndication("stepdown_oral", "Step-down oral pathway"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_isavuconazole,
    ),
    "posaconazole": MedicationRule(
        id="posaconazole",
        name="Posaconazole",
        category="antifungal",
        indications=[
            MedicationIndication("mold_prophylaxis", "Mold prophylaxis pathway"),
            MedicationIndication("invasive_fungal_treatment", "Invasive fungal treatment pathway"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_posaconazole,
    ),
    "fluconazole": MedicationRule(
        id="fluconazole",
        name="Fluconazole",
        category="antifungal",
        indications=[
            MedicationIndication("candidemia_invasive", "Candidemia / invasive candidiasis"),
            MedicationIndication("mucosal_candidiasis", "Mucosal candidiasis"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_fluconazole,
    ),
    "micafungin": MedicationRule(
        id="micafungin",
        name="Micafungin",
        category="antifungal",
        indications=[
            MedicationIndication("candidemia_invasive", "Candidemia / invasive candidiasis"),
            MedicationIndication("esophageal_candidiasis", "Esophageal candidiasis"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_micafungin,
    ),
    "voriconazole": MedicationRule(
        id="voriconazole",
        name="Voriconazole",
        category="antifungal",
        indications=[
            MedicationIndication("invasive_mold_treatment", "Invasive mold treatment"),
            MedicationIndication("mold_prophylaxis", "Mold prophylaxis"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_voriconazole,
    ),
    "liposomal_amphotericin_b": MedicationRule(
        id="liposomal_amphotericin_b",
        name="Liposomal Amphotericin B",
        category="antifungal",
        indications=[
            MedicationIndication("invasive_mold_or_severe_yeast", "Invasive mold / severe yeast infection"),
            MedicationIndication("cryptococcal_cns_induction", "Cryptococcal CNS induction pathway"),
        ],
        source_pages=ANTIFUNGAL_SOURCE,
        calculate=_liposomal_amphotericin_b,
    ),
    "foscarnet": MedicationRule(
        id="foscarnet",
        name="Foscarnet",
        category="antiviral",
        indications=[
            MedicationIndication("cmv_induction", "CMV induction / resistant CMV treatment"),
            MedicationIndication("cmv_maintenance_or_hsv", "CMV maintenance / HSV salvage pathway"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_foscarnet,
    ),
    "famciclovir": MedicationRule(
        id="famciclovir",
        name="Famciclovir",
        category="antiviral",
        indications=[
            MedicationIndication("herpes_zoster", "Herpes zoster treatment"),
            MedicationIndication("recurrent_genital_hsv", "Recurrent genital HSV"),
            MedicationIndication("suppression", "Suppressive HSV therapy"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_famciclovir,
    ),
    "acyclovir_po": MedicationRule(
        id="acyclovir_po",
        name="Acyclovir PO",
        category="antiviral",
        indications=[
            MedicationIndication("standard_hsv", "Standard HSV treatment"),
            MedicationIndication("zoster_or_severe_hsv", "Herpes zoster / severe HSV"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_acyclovir_po,
    ),
    "acyclovir_iv": MedicationRule(
        id="acyclovir_iv",
        name="Acyclovir IV",
        category="antiviral",
        indications=[
            MedicationIndication("standard_hsv_systemic", "Systemic HSV treatment"),
            MedicationIndication("hsv_encephalitis_or_disseminated", "HSV encephalitis / disseminated HSV"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_acyclovir_iv,
    ),
    "valacyclovir": MedicationRule(
        id="valacyclovir",
        name="Valacyclovir",
        category="antiviral",
        indications=[
            MedicationIndication("zoster_or_treatment", "Herpes zoster / treatment pathway"),
            MedicationIndication("hsv_suppression", "HSV suppression"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_valacyclovir,
    ),
    "ganciclovir_iv": MedicationRule(
        id="ganciclovir_iv",
        name="Ganciclovir IV",
        category="antiviral",
        indications=[
            MedicationIndication("cmv_treatment", "CMV treatment"),
            MedicationIndication("cmv_prophylaxis", "CMV prophylaxis"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_ganciclovir_iv,
    ),
    "valganciclovir": MedicationRule(
        id="valganciclovir",
        name="Valganciclovir",
        category="antiviral",
        indications=[
            MedicationIndication("cmv_treatment", "CMV treatment"),
            MedicationIndication("cmv_prophylaxis", "CMV prophylaxis"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_valganciclovir,
    ),
    "oseltamivir": MedicationRule(
        id="oseltamivir",
        name="Oseltamivir",
        category="antiviral",
        indications=[
            MedicationIndication("influenza_treatment", "Influenza treatment"),
            MedicationIndication("influenza_prophylaxis", "Influenza prophylaxis"),
        ],
        source_pages=ANTIVIRAL_SOURCE,
        calculate=_oseltamivir,
    ),
}


TEXT_MEDICATION_MAP: List[tuple[str, str]] = [
    ("Ceftazidime/Avibactam", "ceftazidime_avibactam"),
    ("Piperacillin/Tazobactam", "piperacillin_tazobactam"),
    ("Ampicillin/Sulbactam", "ampicillin_sulbactam"),
    ("Trimethoprim/Sulfamethoxazole", "tmp_smx"),
    ("Cefiderocol", "cefiderocol"),
    ("Ceftaroline", "ceftaroline"),
    ("Imipenem", "imipenem_cilastatin"),
    ("Meropenem", "meropenem"),
    ("Ertapenem", "ertapenem"),
    ("Cefepime", "cefepime"),
    ("Ceftriaxone", "ceftriaxone"),
    ("Ceftazidime", "ceftazidime"),
    ("Aztreonam", "aztreonam"),
    ("Vancomycin", "vancomycin_iv"),
    ("Linezolid", "linezolid"),
    ("Daptomycin", "daptomycin"),
    ("Ampicillin", "ampicillin"),
    ("Rifabutin", "rifabutin"),
    ("Rifampin", "rifampin"),
    ("Isoniazid", "isoniazid"),
    ("Pyrazinamide", "pyrazinamide"),
    ("Ethambutol", "ethambutol"),
    ("Nafcillin/Oxacillin", "nafcillin"),
    ("Nafcillin", "nafcillin"),
    ("Oxacillin", "nafcillin"),
    ("Penicillin", "penicillin_g"),
    ("Levofloxacin", "levofloxacin"),
    ("Ciprofloxacin", "ciprofloxacin"),
    ("Clindamycin", "clindamycin"),
    ("Metronidazole", "metronidazole"),
    ("Amoxicillin/Clavulanate", "amoxicillin_clavulanate"),
    ("Amoxicillin", "amoxicillin"),
]


def list_medications(category: str | None = None) -> List[MedicationRule]:
    meds = sorted(MEDICATIONS.values(), key=lambda item: item.name.lower())
    if category is None:
        return meds
    return [med for med in meds if med.category == category]


def get_medication(medication_id: str) -> MedicationRule:
    try:
        return MEDICATIONS[medication_id]
    except KeyError as exc:
        raise DoseIDError(f"Unsupported medication id: {medication_id}") from exc


def default_indication_id(medication_id: str) -> str:
    med = get_medication(medication_id)
    if not med.indications:
        raise DoseIDError(f"No dosing indications configured for {medication_id}")
    return med.indications[0].id


def calculate_medication(
    *,
    medication_id: str,
    patient: NormalizedPatient,
    renal_mode: RenalMode,
    indication_id: str | None = None,
) -> Dict[str, object]:
    med = get_medication(medication_id)
    selected_indication = indication_id or default_indication_id(medication_id)
    indication = next((item for item in med.indications if item.id == selected_indication), None)
    if indication is None:
        raise DoseIDError(f"Unsupported indication '{selected_indication}' for {med.name}")
    result = med.calculate(patient, indication.id, renal_mode)
    payload: Dict[str, object] = {
        "medication_id": med.id,
        "medication_name": med.name,
        "category": med.category,
        "indication_id": indication.id,
        "indication_label": indication.label,
        "regimen": result.regimen,
        "renal_bucket": result.renal_bucket,
        "notes": list(result.notes),
        "source_pages": med.source_pages,
    }
    if result.dose_weight is not None:
        payload["dose_weight"] = {
            "basis": result.dose_weight.basis,
            "kg": round(result.dose_weight.kg, 1),
        }
    return payload


def _extract_medications_from_texts(texts: Iterable[str]) -> List[str]:
    combined = " ".join(texts).lower()
    matches: List[str] = []
    for needle, med_id in TEXT_MEDICATION_MAP:
        if needle.lower() in combined and med_id not in matches:
            matches.append(med_id)
    return matches


def _default_indication_for_mechid(
    *,
    medication_id: str,
    organism: str,
    final_results: Dict[str, str],
    tx_context: Dict[str, str] | None,
    therapy_notes: Iterable[str],
) -> str:
    syndrome = (tx_context or {}).get("syndrome", "Not specified")
    severity = (tx_context or {}).get("severity", "Not specified")
    carbapenemase_result = (tx_context or {}).get("carbapenemaseResult", "Not specified")
    notes_text = " ".join(therapy_notes).lower()
    severe = severity == "Severe / septic shock"
    deep = syndrome in {
        "Bloodstream infection",
        "Pneumonia (HAP/VAP or severe CAP)",
        "CNS infection",
        "Bone/joint infection",
        "Other deep-seated / high-inoculum focus",
    }

    if medication_id in {"cefepime", "meropenem"}:
        return "cns_meningitis" if syndrome == "CNS infection" else "severe_non_cns"
    if medication_id == "piperacillin_tazobactam":
        if organism == "Pseudomonas aeruginosa" or syndrome == "Pneumonia (HAP/VAP or severe CAP)" or severe:
            return "high_inoculum_pseudomonal"
        return "severe_ei"
    if medication_id == "daptomycin":
        if organism.startswith("Enterococcus") and final_results.get("Vancomycin") == "Resistant":
            return "vre_high_burden"
        if "vre" in notes_text:
            return "vre_high_burden"
        return "bacteremia_endovascular"
    if medication_id == "ampicillin_sulbactam":
        return "surgical_or_intraabdominal" if syndrome == "Intra-abdominal infection" else "standard_systemic"
    if medication_id == "aztreonam":
        return "uncomplicated_uti" if syndrome == "Uncomplicated cystitis" else "systemic_gram_negative"
    if medication_id == "cefazolin":
        return "complicated_or_deep" if deep or severe else "uncomplicated_infection"
    if medication_id == "ceftriaxone":
        if syndrome == "CNS infection":
            return "meningitis"
        return "serious_infection" if deep or severe else "standard_dose"
    if medication_id == "ceftazidime":
        return "pseudomonal_or_severe" if organism == "Pseudomonas aeruginosa" or severe else "standard_systemic"
    if medication_id == "ertapenem":
        return "esbl_targeted" if "esbl" in notes_text or final_results.get("Ceftriaxone") == "Resistant" else "standard_systemic"
    if medication_id == "linezolid":
        return "standard_bacterial"
    if medication_id == "isoniazid":
        return "tb_daily"
    if medication_id == "rifampin":
        if "hardware" in notes_text and not organism.startswith("Mycobacterium"):
            return "hardware_adjuvant"
        return "tb_daily"
    if medication_id == "ethambutol":
        return "tb_standard"
    if medication_id == "pyrazinamide":
        return "tb_standard"
    if medication_id == "moxifloxacin_tb":
        return "ntm_or_salvage" if "ntm" in notes_text or organism.startswith("Mycobacterium avium") else "tb_alt_backbone"
    if medication_id == "rifabutin":
        return "tb_or_ntm_standard"
    if medication_id == "levofloxacin":
        return "pneumonia_or_pseudomonas" if syndrome == "Pneumonia (HAP/VAP or severe CAP)" or organism == "Pseudomonas aeruginosa" else "standard_infection"
    if medication_id == "metronidazole":
        return "intraabdominal_coverage" if syndrome == "Intra-abdominal infection" else "anaerobic_systemic"
    if medication_id == "tmp_smx":
        if organism == "Stenotrophomonas maltophilia":
            return "stenotrophomonas"
        if syndrome == "Uncomplicated cystitis":
            return "uncomplicated_cystitis"
        if syndrome == "Bone/joint infection":
            return "staph_bone_joint"
        if syndrome == "Bloodstream infection":
            return "gnr_bacteremia"
        return "ssti"
    if medication_id == "amoxicillin":
        return "high_dose_oral" if deep or severe else "standard_oral"
    if medication_id == "amoxicillin_clavulanate":
        return "high_exposure_oral" if deep or severe or syndrome == "Intra-abdominal infection" else "standard_oral"
    if medication_id == "ampicillin":
        return "high_exposure" if deep or severe else "systemic_standard"
    if medication_id == "cefiderocol":
        return "high_clearance_or_critical" if severe or syndrome == "Pneumonia (HAP/VAP or severe CAP)" else "resistant_gram_negative"
    if medication_id == "ceftaroline":
        return "high_exposure_mrsa" if organism == "Staphylococcus aureus" and final_results.get("Nafcillin/Oxacillin") == "Resistant" else "standard_serious"
    if medication_id == "ceftazidime_avibactam":
        return "high_exposure_critical" if severe or carbapenemase_result == "Positive" else "standard_resistant_gn"
    if medication_id == "ciprofloxacin":
        return "high_exposure_pseudomonal" if organism == "Pseudomonas aeruginosa" else "standard_systemic"
    if medication_id == "clindamycin":
        return "bone_joint_infection" if syndrome == "Bone/joint infection" else "standard_systemic_max"
    if medication_id == "imipenem_cilastatin":
        return "high_exposure_resistant" if severe or carbapenemase_result == "Positive" else "standard_severe"
    if medication_id == "nafcillin":
        return "mssa_high_burden" if deep or severe else "mssa_standard"
    if medication_id == "penicillin_g":
        return "cns_or_high_exposure" if syndrome == "CNS infection" else "serious_streptococcal"
    if medication_id == "vancomycin_iv":
        if severe or deep or organism == "Staphylococcus aureus":
            return "serious_mrsa_or_invasive"
        return "standard_systemic"
    return default_indication_id(medication_id)


def suggest_mechid_doses(
    *,
    organism: str,
    final_results: Dict[str, str],
    therapy_notes: Iterable[str],
    tx_context: Dict[str, str] | None,
    patient: NormalizedPatient,
    renal_mode: RenalMode,
    max_suggestions: int = 3,
) -> List[Dict[str, object]]:
    medication_ids = _extract_medications_from_texts(therapy_notes)
    suggestions: List[Dict[str, object]] = []
    for medication_id in medication_ids[:max(0, max_suggestions)]:
        indication_id = _default_indication_for_mechid(
            medication_id=medication_id,
            organism=organism,
            final_results=final_results,
            tx_context=tx_context,
            therapy_notes=therapy_notes,
        )
        suggestions.append(
            calculate_medication(
                medication_id=medication_id,
                patient=patient,
                renal_mode=renal_mode,
                indication_id=indication_id,
            )
        )
    return suggestions
