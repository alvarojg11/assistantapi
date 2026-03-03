from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import AnalyzeRequest, ParsedUnderstanding, SyndromeModule
from .module_store import InMemoryModuleStore


NEGATION_RE = re.compile(r"\b(no|not|without|denies|deny|negative|normal|absent|none)\b")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


MODULE_ALIASES: Dict[str, List[str]] = {
    "cap": ["cap", "community acquired pneumonia", "pneumonia", "pna"],
    "vap": ["vap", "ventilator associated pneumonia", "ventilated pneumonia"],
    "cdi": ["cdi", "c diff", "cdiff", "clostridioides difficile", "clostridium difficile"],
    "uti": ["uti", "urinary tract infection", "pyelonephritis", "cystitis"],
    "endo": [
        "endo",
        "endocarditis",
        "infective endocarditis",
        "ie",
        "sab bacteremia",
        "staphylococcus aureus bacteremia",
        "staph aureus bacteremia",
        "s aureus bacteremia",
        "viridans bacteremia",
        "viridans streptococcal bacteremia",
        "nbhs bacteremia",
        "enterococcus bacteremia",
        "enterococcus faecalis bacteremia",
        "e faecalis bacteremia",
    ],
    "active_tb": ["tb", "active tb", "tuberculosis", "pulmonary tb"],
    "pjp": ["pjp", "pcp", "pneumocystis"],
    "inv_candida": ["invasive candidiasis", "candida", "candidemia"],
    "inv_mold": ["invasive mold", "aspergillus", "mucor", "mold infection"],
    "pji": ["pji", "prosthetic joint infection", "joint prosthesis infection"],
}


PRESET_HINT_ALIASES: Dict[str, List[str]] = {
    "ed": ["ed", "emergency department", "er"],
    "pc": ["primary care", "clinic"],
    "icu": ["icu", "intensive care"],
    "sab": ["sab", "staph aureus bacteremia", "s aureus bacteremia"],
    "vap_ge5d": ["5 days", "five days", "day 5", "5th day"],
    "vap_gt48h": ["48 hours", "2 days", "two days", "greater than 48 hours"],
}


COMMON_FINDING_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "cap_fever": {"present": ["fever", "febrile"], "absent": ["afebrile", "no fever"]},
    "cap_rr": {"present": ["tachypnea", "tachypnoea", "rr high", "respiratory rate high"]},
    "cap_hypox": {"present": ["hypoxemia", "hypoxia", "o2 sat low", "oxygen low", "low oxygen saturation", "oxygen saturation 92"]},
    "cap_cxr_consolidation": {
        "present": [
            "cxr infiltrate",
            "chest xray infiltrate",
            "xray infiltrate",
            "consolidation on cxr",
            "cxr consolidation",
            "lobar infiltrate",
            "chest radiograph infiltrate",
            "cxr with lobar or multilobar consolidation",
        ],
        "absent": [
            "no infiltrate on cxr",
            "cxr no infiltrate",
            "clear cxr",
            "no consolidation on cxr",
            "cxr no consolidation",
            "cxr without consolidation",
        ],
    },
    "cap_procal_high": {"present": ["procalcitonin high", "procalcitonin elevated"], "absent": ["procalcitonin low", "procalcitonin normal"]},
    "cap_crackles": {"present": ["crackles", "rales"]},
    "cap_cxr_not_done": {
        "present": ["cxr not done"],
        "absent": ["cxr completed"],
    },
    "cap_rvp_pos": {
        "present": ["respiratory viral panel positive"],
        "absent": ["respiratory viral panel negative"],
    },
    "cap_rvp_na": {
        "present": ["respiratory viral panel not done"],
        "absent": ["respiratory viral panel completed"],
    },
    "cap_active_malignancy": {
        "present": ["active malignancy present"],
        "absent": ["no active malignancy"],
    },
    "vap_fever": {"present": ["fever"]},
    "vap_cxr_infiltrate": {
        "present": ["new infiltrate", "progressive infiltrate", "cxr infiltrate", "chest radiograph with new or progressive infiltrate"],
        "absent": ["chest radiograph without new or progressive infiltrate"],
    },
    "vap_purulent_secretions": {"present": ["purulent secretions", "purulent tracheal secretions"]},
    "vap_cxr_na": {
        "present": ["vap cxr not done"],
        "absent": ["vap cxr completed"],
    },
    "vap_leukocytosis": {
        "present": ["leukocytosis", "wbc high", "white count elevated", "wbc at least 12 for vap"],
        "absent": ["wbc below 12 for vap"],
    },
    "vap_hypoxemia_pf240": {
        "present": ["pao2 fio2 at or below 240", "pf ratio 240 or less"],
        "absent": ["pao2 fio2 above 240", "pf ratio above 240"],
    },
    "vap_cpis_gt6": {
        "present": ["cpis greater than 6", "cpis above 6"],
        "absent": ["cpis 6 or lower", "cpis at most 6"],
    },
    "vap_cpis_na": {
        "present": ["cpis not used"],
        "absent": ["cpis completed"],
    },
    "vap_bal_qcx": {"present": ["bal culture positive", "bal culture growing", "bal culture growing a pathogen"]},
    "vap_resp_micro_na": {
        "present": ["respiratory sampling not done"],
        "absent": ["respiratory sampling completed"],
    },
    "vap_pct_elevated": {
        "present": ["procalcitonin elevated for vap"],
        "absent": ["procalcitonin not elevated for vap"],
    },
    "vap_pct_na": {
        "present": ["procalcitonin not done for vap"],
        "absent": ["procalcitonin completed for vap"],
    },
    "endo_fever": {"present": ["fever", "febrile"]},
    "endo_new_murmur": {
        "present": [
            "new murmur",
            "new heart murmur",
            "new cardiac murmur",
            "new regurgitant murmur",
        ]
    },
    "endo_tte": {
        "present": ["tte positive", "vegetation on tte", "echo vegetation", "echo positive"],
        "absent": ["tte negative", "tte without vegetation", "negative tte"],
    },
    "endo_tee": {
        "present": ["tee positive", "tee showing vegetation", "vegetation on tee"],
        "absent": ["tee negative", "tee without vegetation", "negative tee"],
    },
    "endo_bcx_major_typical": {"present": ["typical blood cultures", "major blood culture criterion"]},
    "endo_bcx_major_persistent": {"present": ["persistent bacteremia", "persistent positive blood cultures"]},
    "endo_prosthetic_valve": {"present": ["prosthetic valve"]},
    "endo_cied": {"present": ["cied", "pacemaker", "icd", "cardiac device"]},
    "cdi_freq": {
        "present": [
            "3 unformed stools",
            "more than 3 bowel movements",
            "more than 3 bm",
            "3 bowel movements in 24 hours",
            "3 loose stools in 24 hours",
            "frequent diarrhea",
            "rectal tube output 1.5 l",
            "more than 1.5 l stool output",
            "1.5 l in rectal tube",
            "at least 3 unformed stools in 24 hours",
        ],
        "absent": ["fewer than 3 unformed stools in 24 hours"],
    },
    "cdi_watery": {
        "present": ["watery diarrhea", "profuse diarrhea", "loose watery stools", "watery diarrhea present"],
        "absent": ["no watery diarrhea"],
    },
    "cdi_abx": {"present": ["recent antibiotics", "antibiotics recently", "after recent antibiotics"]},
    "cdi_abd_pain": {"present": ["abdominal pain", "abdominal cramping", "cramping"]},
    "cdi_test_na": {
        "present": ["stool testing not done"],
        "absent": ["stool testing completed"],
    },
    "cdi_naat_neg": {
        "present": ["c diff naat negative", "cdiff naat negative"],
        "absent": ["c diff naat not negative"],
    },
    "cdi_naat_pos_tox_pos": {
        "present": ["toxin detected", "toxin positive", "pcr positive with toxin detected", "c diff naat positive and toxin positive"],
        "absent": ["c diff naat toxin not both positive"],
    },
    "cdi_naat_pos_tox_neg": {
        "present": ["c diff naat positive and toxin negative"],
        "absent": ["c diff naat toxin pattern not naat positive toxin negative"],
    },
    "cdi_naat_pos_tox_na": {
        "present": ["c diff naat positive and toxin not sent"],
        "absent": ["c diff naat pattern not naat positive with toxin not sent"],
    },
    "uti_freq": {"present": ["frequency", "urgency", "urinary urgency", "urinary frequency"]},
    "uti_vaginitis": {
        "present": ["vaginal discharge or irritation present", "vaginal discharge present", "vaginal irritation present"],
        "absent": ["no vaginal discharge or irritation"],
    },
    "uti_obstruction": {
        "present": ["urinary obstruction or anatomic abnormality present"],
        "absent": ["no urinary obstruction or anatomic abnormality"],
    },
    "ua_le_pos": {
        "present": ["urine leukocyte esterase positive"],
        "absent": ["urine leukocyte esterase negative"],
    },
    "ua_nit_pos": {
        "present": ["urine nitrite positive"],
        "absent": ["urine nitrite negative"],
    },
    "ua_pyuria_pos": {
        "present": ["pyuria", "pyuria on urinalysis", "pyuria on ua", "pyuria present on microscopy"],
        "absent": ["no pyuria on microscopy"],
    },
    "ua_bact_pos": {
        "present": ["bacteriuria present on microscopy"],
        "absent": ["no bacteriuria on microscopy"],
    },
    "uti_cx_pos": {
        "present": ["urine culture above 100000 cfu", "urine culture over 100000 cfu"],
        "absent": ["urine culture below 100000 cfu"],
    },
    "tb_contact": {
        "present": ["tb exposure", "household tb exposure", "tb contact", "close household tb exposure"],
        "absent": ["no tb exposure", "no household tb exposure", "no close household tb exposure"],
    },
    "tb_sym_any": {
        "present": [
            "tb symptoms",
            "who tb symptom screen positive",
            "cough fever night sweats weight loss",
            "who tb symptom cough",
            "who tb symptom fever",
            "who tb symptom night sweats",
            "who tb symptom weight loss",
        ],
        "absent": ["no tb symptoms", "no who tb symptoms"],
    },
    "tb_sym_cough_2w": {
        "present": ["cough for more than two weeks", "cough for more than 2 weeks", "cough for over two weeks", "cough for over 2 weeks"],
        "absent": ["cough for less than two weeks", "cough for less than 2 weeks"],
    },
    "tb_sym_na": {
        "present": ["symptom screen not done", "tb symptom screen not done"],
        "absent": ["symptom screen completed", "tb symptom screen completed"],
    },
    "tb_qft": {
        "present": ["quantiferon positive", "igra positive", "positive quantiferon"],
        "absent": ["quantiferon negative", "igra negative", "negative quantiferon"],
    },
    "tb_tst": {
        "present": ["tuberculin skin test positive", "tst positive", "ppd positive"],
        "absent": ["tuberculin skin test negative", "tst negative", "ppd negative"],
    },
    "tb_immune_na": {
        "present": ["qft tst not done", "qft/tst not done", "igra not done"],
        "absent": ["qft tst completed", "qft/tst completed", "igra completed"],
    },
    "tb_mtbpcr_sputum": {
        "present": ["mtb pcr positive on sputum", "xpert positive on sputum", "mtb pcr sputum positive"],
        "absent": ["mtb pcr negative on sputum", "xpert negative on sputum", "mtb pcr sputum negative"],
    },
    "tb_mtbpcr_bal": {
        "present": ["mtb pcr positive on bal", "xpert positive on bal", "mtb pcr bal positive"],
        "absent": ["mtb pcr negative on bal", "xpert negative on bal", "mtb pcr bal negative"],
    },
    "tb_afb_smear_sputum": {
        "present": ["positive sputum smear", "sputum smear positive", "afb smear positive", "afb smear positive on sputum"],
        "absent": ["afb smear negative", "afb smear negative on sputum", "sputum smear negative"],
    },
    "tb_culture_sputum": {
        "present": ["mycobacterial culture positive from sputum", "tb culture positive from sputum", "sputum culture positive for tb"],
        "absent": ["mycobacterial culture negative from sputum", "tb culture negative from sputum", "sputum culture negative for tb"],
    },
    "tb_culture_bal": {
        "present": ["mycobacterial culture positive from bal", "tb culture positive from bal", "bal culture positive for tb"],
        "absent": ["mycobacterial culture negative from bal", "tb culture negative from bal", "bal culture negative for tb"],
    },
    "tb_cxr_suggestive": {
        "present": [
            "xray suspicious for active tuberculosis",
            "chest xray suspicious for active tuberculosis",
            "chest x ray suspicious for active tuberculosis",
            "cxr concerning for active tb",
            "chest xray concerning for active tb",
            "chest x ray concerning for active tb",
        ],
        "absent": ["cxr not suggestive of active pulmonary tb", "chest xray not suggestive of active pulmonary tb"],
    },
    "tb_cxr_na": {
        "present": ["cxr not done", "chest xray not done"],
        "absent": ["cxr completed", "chest xray completed"],
    },
    "tb_ct_suggestive": {
        "present": [
            "chest ct suggestive of active pulmonary tb",
            "ct chest suggestive of active pulmonary tb",
            "ct scan chest suggestive of active pulmonary tb",
            "ct scan of the chest suggestive of active pulmonary tb",
            "chest ct suspicious for active tuberculosis",
            "ct chest suspicious for active tuberculosis",
            "chest ct concerning for active tb",
            "ct chest concerning for active tb",
            "ct scan chest concerning for active tb",
            "ct scan of the chest concerning for active tb",
        ],
        "absent": [
            "chest ct not suggestive of active pulmonary tb",
            "ct chest not suggestive of active pulmonary tb",
            "ct scan chest not suggestive of active pulmonary tb",
            "ct scan of the chest not suggestive of active pulmonary tb",
        ],
    },
    "tb_ct_na": {
        "present": ["chest ct not done", "ct chest not done"],
        "absent": ["chest ct completed", "ct chest completed"],
    },
    "pjp_host_no_ppx": {
        "present": ["lack of tmp smx prophylaxis despite indication", "tmp smx prophylaxis missing despite indication"],
        "absent": ["receiving tmp smx prophylaxis when indicated", "on tmp smx prophylaxis when indicated"],
    },
    "pjp_host_heme_hsct": {
        "present": ["hematologic malignancy or stem cell transplant", "hematologic malignancy", "stem cell transplant"],
        "absent": ["no hematologic malignancy or stem cell transplant", "no hematologic malignancy"],
    },
    "pjp_bdg_serum": {
        "present": ["beta d glucan positive", "beta-d-glucan positive", "serum beta d glucan positive", "positive serum beta d glucan"],
        "absent": ["serum beta d glucan negative", "beta d glucan negative", "negative serum beta d glucan"],
    },
    "pjp_bdg_na": {
        "present": ["serum bdg not done", "serum beta d glucan not done"],
        "absent": ["serum bdg completed", "serum beta d glucan completed"],
    },
    "pjp_ldh_high": {
        "present": ["elevated ldh", "ldh elevated", "serum ldh elevated"],
        "absent": ["serum ldh not elevated", "ldh not elevated", "ldh normal"],
    },
    "pjp_pcr_bal": {
        "present": ["pjp pcr positive on bal", "pneumocystis pcr positive on bal"],
        "absent": ["pjp pcr negative on bal", "pneumocystis pcr negative on bal"],
    },
    "pjp_pcr_induced_sputum": {
        "present": ["pjp pcr positive on induced sputum", "pneumocystis pcr positive on induced sputum"],
        "absent": ["pjp pcr negative on induced sputum", "pneumocystis pcr negative on induced sputum"],
    },
    "pjp_pcr_upper_airway": {
        "present": ["pjp pcr positive on upper airway sample", "pneumocystis pcr positive on upper airway sample"],
        "absent": ["pjp pcr negative on upper airway sample", "pneumocystis pcr negative on upper airway sample"],
    },
    "pjp_pcr_na": {
        "present": ["respiratory pjp pcr not done", "respiratory pjp pcr not performed"],
        "absent": ["respiratory pjp pcr completed", "respiratory pjp pcr performed"],
    },
    "pjp_dfa": {
        "present": ["pjp dfa ifa positive", "pneumocystis dfa ifa positive"],
        "absent": ["pjp dfa ifa negative", "pneumocystis dfa ifa negative"],
    },
    "pjp_dfa_na": {
        "present": ["pjp dfa ifa not done", "pneumocystis dfa ifa not done"],
        "absent": ["pjp dfa ifa completed", "pneumocystis dfa ifa completed"],
    },
    "pjp_cxr_typical": {
        "present": ["cxr typical of pjp", "chest xray typical of pjp"],
        "absent": ["cxr not typical of pjp", "chest xray not typical of pjp"],
    },
    "pjp_ct_typical": {
        "present": ["ct typical for pneumocystis", "ct typical for pjp", "ground glass opacities typical for pjp"],
        "absent": ["ct not typical of pjp", "ct not typical for pneumocystis"],
    },
    "pjp_imaging_na": {
        "present": ["chest imaging not done", "chest imaging not performed"],
        "absent": ["chest imaging completed", "chest imaging performed"],
    },
    "icand_component_tpn": {"present": ["on tpn", "total parenteral nutrition", "receiving tpn"]},
    "icand_component_surgery": {"present": ["after surgery", "postoperative", "post operative"]},
    "icand_component_multifocal_colonization": {
        "present": ["multifocal candida colonization present", "multifocal candida colonization"],
        "absent": ["no multifocal candida colonization"],
    },
    "icand_component_severe_sepsis": {
        "present": ["severe sepsis", "septic shock", "severe sepsis or septic shock"],
        "absent": ["no severe sepsis", "no septic shock", "no severe sepsis or septic shock"],
    },
    "icand_host_dialysis": {
        "present": ["on dialysis", "on renal replacement therapy", "dialysis", "renal replacement therapy"],
        "absent": ["not on dialysis", "not on renal replacement therapy"],
    },
    "icand_bdg_serum": {
        "present": ["serum beta d glucan positive", "beta-d-glucan positive"],
        "absent": ["serum beta d glucan negative", "beta d glucan negative"],
    },
    "icand_bdg_na": {
        "present": ["serum bdg not done", "candida bdg not done"],
        "absent": ["serum bdg completed", "candida bdg completed"],
    },
    "icand_mannan_antimannan": {
        "present": ["mannan anti mannan assay positive", "mannan anti mannan positive"],
        "absent": ["mannan anti mannan assay negative", "mannan anti mannan negative"],
    },
    "icand_mannan_na": {
        "present": ["mannan anti mannan testing not done", "mannan anti mannan not done"],
        "absent": ["mannan anti mannan testing completed", "mannan anti mannan completed"],
    },
    "icand_t2candida": {
        "present": ["t2candida positive", "positive t2candida"],
        "absent": ["t2candida negative", "negative t2candida"],
    },
    "icand_t2_na": {
        "present": ["t2candida not done", "t2candida not performed"],
        "absent": ["t2candida completed", "t2candida performed"],
    },
    "icand_pcr_blood": {
        "present": ["candida pcr positive from blood", "blood candida pcr positive"],
        "absent": ["candida pcr negative from blood", "blood candida pcr negative"],
    },
    "icand_pcr_na": {
        "present": ["candida pcr not done", "candida pcr not performed"],
        "absent": ["candida pcr completed", "candida pcr performed"],
    },
    "icand_culture_positive": {
        "present": ["blood culture growing candida", "culture growing candida", "candida blood culture positive", "blood or sterile site culture positive for candida"],
        "absent": ["blood or sterile site culture negative for candida", "candida culture negative"],
    },
    "icand_culture_na": {
        "present": ["candida culture strategy not done", "candida culture strategy not performed"],
        "absent": ["candida culture strategy completed", "candida culture strategy performed"],
    },
    "imi_host_neutropenia_hsct": {
        "present": ["neutropenic", "profound neutropenia", "recent allogeneic hsct"],
        "absent": ["no profound neutropenia", "no recent allogeneic hsct"],
    },
    "imi_host_hematologic_malignancy": {
        "present": ["active hematologic malignancy", "aml", "mds", "relapsed leukemia"],
        "absent": ["no active hematologic malignancy"],
    },
    "imi_fever_refractory": {
        "present": ["refractory fever", "persistent fever despite antibiotics", "refractory fever despite broad spectrum antibacterials"],
        "absent": ["no refractory fever", "no persistent fever despite antibiotics"],
    },
    "imi_ct_halo_sign": {
        "present": ["halo sign on ct", "ct halo sign", "chest ct halo sign present"],
        "absent": ["no chest ct halo sign", "ct without halo sign"],
    },
    "imi_ct_na": {
        "present": ["chest ct not done", "chest ct not performed"],
        "absent": ["chest ct completed", "chest ct performed"],
    },
    "imi_serum_gm_odi10": {
        "present": ["serum galactomannan positive", "galactomannan positive"],
        "absent": ["serum galactomannan negative", "galactomannan negative"],
    },
    "imi_bal_gm_odi10": {
        "present": ["bal galactomannan positive"],
        "absent": ["bal galactomannan negative"],
    },
    "imi_gm_na": {
        "present": ["galactomannan testing not done", "galactomannan not done"],
        "absent": ["galactomannan testing completed", "galactomannan completed"],
    },
    "imi_serum_bdg": {
        "present": ["serum beta d glucan positive", "beta d glucan positive"],
        "absent": ["serum beta d glucan negative", "beta d glucan negative"],
    },
    "imi_bdg_na": {
        "present": ["serum bdg not done", "mold bdg not done"],
        "absent": ["serum bdg completed", "mold bdg completed"],
    },
    "imi_aspergillus_lfd": {
        "present": ["aspergillus lfd lfa positive", "aspergillus lateral flow positive"],
        "absent": ["aspergillus lfd lfa negative", "aspergillus lateral flow negative"],
    },
    "imi_lfd_na": {
        "present": ["aspergillus lfd lfa not done", "aspergillus lateral flow not done"],
        "absent": ["aspergillus lfd lfa completed", "aspergillus lateral flow completed"],
    },
    "imi_aspergillus_pcr_bal": {
        "present": ["bal aspergillus pcr positive", "aspergillus pcr positive from bal"],
        "absent": ["aspergillus pcr negative from bal", "bal aspergillus pcr negative"],
    },
    "imi_aspergillus_pcr_na": {
        "present": ["aspergillus pcr not done", "aspergillus pcr not performed"],
        "absent": ["aspergillus pcr completed", "aspergillus pcr performed"],
    },
    "imi_mucorales_pcr_bal": {
        "present": ["mucorales pcr positive from bal", "bal mucorales pcr positive"],
        "absent": ["mucorales pcr negative from bal", "bal mucorales pcr negative"],
    },
    "imi_mucorales_pcr_blood": {
        "present": ["mucorales pcr positive from blood", "blood mucorales pcr positive"],
        "absent": ["mucorales pcr negative from blood", "blood mucorales pcr negative"],
    },
    "imi_mucorales_pcr_na": {
        "present": ["mucorales pcr not done", "mucorales pcr not performed"],
        "absent": ["mucorales pcr completed", "mucorales pcr performed"],
    },
    "pji_sym_joint_pain": {"present": ["prosthetic joint pain", "worsening prosthetic joint pain"]},
    "pji_sym_local_inflammation": {"present": ["erythema and drainage", "swollen prosthetic knee", "warmth and drainage", "drainage"]},
    "pji_crp": {
        "present": ["crp elevated", "elevated crp", "c reactive protein elevated"],
        "absent": ["crp not elevated", "normal crp"],
    },
    "pji_esr": {
        "present": ["esr elevated", "elevated esr"],
        "absent": ["esr not elevated", "normal esr"],
    },
    "pji_alpha_defensin_elisa": {
        "present": ["alpha defensin positive", "positive synovial alpha defensin test", "synovial alpha defensin positive", "synovial alpha defensin elisa positive"],
        "absent": ["synovial alpha defensin elisa negative", "alpha defensin negative"],
    },
    "pji_alpha_defensin_lateral_flow": {
        "present": ["synovial alpha defensin lateral flow positive"],
        "absent": ["synovial alpha defensin lateral flow negative"],
    },
    "pji_leukocyte_esterase": {
        "present": ["synovial leukocyte esterase positive", "leukocyte esterase positive"],
        "absent": ["synovial leukocyte esterase negative", "leukocyte esterase negative"],
    },
    "pji_synovial_marker_na": {
        "present": ["synovial biomarker testing not done", "synovial biomarker not done"],
        "absent": ["synovial biomarker testing completed", "synovial biomarker completed"],
    },
    "pji_synovial_fluid_culture": {
        "present": ["synovial fluid culture positive"],
        "absent": ["synovial fluid culture negative"],
    },
    "pji_intraop_tissue_culture": {
        "present": ["intraoperative tissue culture positive"],
        "absent": ["intraoperative tissue culture negative"],
    },
    "pji_sonication_culture": {
        "present": ["sonication fluid culture positive"],
        "absent": ["sonication fluid culture negative"],
    },
    "pji_culture_na": {
        "present": ["pji culture strategy not done", "culture strategy not done for pji"],
        "absent": ["pji culture strategy completed", "culture strategy completed for pji"],
    },
    "pji_synovial_pcr": {
        "present": ["synovial pcr positive"],
        "absent": ["synovial pcr negative"],
    },
    "pji_pcr_na": {
        "present": ["synovial pcr not done", "synovial pcr not performed"],
        "absent": ["synovial pcr completed", "synovial pcr performed"],
    },
    "pji_xray_supportive": {
        "present": ["plain radiograph supportive of infection", "xray supportive of infection"],
        "absent": ["plain radiograph not supportive of infection", "xray not supportive of infection"],
    },
    "pji_imaging_na": {
        "present": ["pji imaging not done", "imaging not done for pji"],
        "absent": ["pji imaging completed", "imaging completed for pji"],
    },
}


@dataclass
class ParseTextResult:
    parsed_request: Optional[AnalyzeRequest]
    understood: ParsedUnderstanding
    warnings: List[str]
    requires_confirmation: bool
    parser_name: str = "rule-based-v1"


def empty_understanding() -> ParsedUnderstanding:
    return ParsedUnderstanding(
        moduleId=None,
        moduleName=None,
        presetId=None,
        presetLabel=None,
        findingsPresent=[],
        findingsAbsent=[],
        unknownMentions=[],
    )


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", NON_ALNUM_RE.sub(" ", text.lower())).strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _find_all_positions(text: str, phrase: str) -> List[int]:
    phrase_norm = normalize(phrase)
    if not phrase_norm:
        return []
    return [m.start() for m in _phrase_pattern(phrase_norm).finditer(text)]


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = normalize(phrase)
    if not phrase_norm:
        return False
    return _phrase_pattern(phrase_norm).search(text) is not None


def _module_score(text_norm: str, module: SyndromeModule) -> int:
    score = 0
    for alias in MODULE_ALIASES.get(module.id, []):
        if _contains_phrase(text_norm, alias):
            score += 3
    if _contains_phrase(text_norm, module.name):
        score += 2
    return score


def _choose_module(store: InMemoryModuleStore, text: str, module_hint: str | None) -> tuple[Optional[SyndromeModule], List[str]]:
    warnings: List[str] = []
    text_norm = normalize(text)

    if module_hint:
        hinted = store.get(module_hint)
        if hinted:
            return hinted, warnings
        warnings.append(f"Unknown moduleHint '{module_hint}', using text inference.")

    modules = [store.get(m.id) for m in store.list_summaries()]
    modules = [m for m in modules if m is not None]

    scored = [(m, _module_score(text_norm, m)) for m in modules]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0] if scored else (None, 0)

    if best[0] is None:
        return None, warnings
    if best[1] == 0:
        warnings.append("Could not infer syndrome from text. Add a disease name or use `moduleHint`.")
        return None, warnings

    if len(scored) > 1 and scored[1][1] == best[1]:
        warnings.append(f"Multiple syndromes matched text; selected '{best[0].id}'. Please confirm.")
    return best[0], warnings


def resolve_module_from_request(
    *,
    store: InMemoryModuleStore,
    parsed_request: AnalyzeRequest,
    module_hint: str | None = None,
    text: str | None = None,
) -> tuple[Optional[SyndromeModule], List[str]]:
    warnings: List[str] = []

    if parsed_request.module is not None:
        return parsed_request.module, warnings

    if parsed_request.module_id:
        module = store.get(parsed_request.module_id)
        if module:
            return module, warnings
        warnings.append(f"Parsed moduleId '{parsed_request.module_id}' is not known.")

    if module_hint:
        hinted = store.get(module_hint)
        if hinted:
            warnings.append(f"Using `moduleHint` '{module_hint}' because parsed module was missing/invalid.")
            return hinted, warnings

    if text:
        inferred, infer_warnings = _choose_module(store, text, None)
        warnings.extend(infer_warnings)
        if inferred:
            warnings.append(f"Using inferred module '{inferred.id}' because parsed module was missing/invalid.")
            return inferred, warnings

    return None, warnings


def _preset_score(text_norm: str, module: SyndromeModule, preset) -> int:
    score = 0
    label_norm = normalize(preset.label)
    if label_norm and _contains_phrase(text_norm, label_norm):
        score += 4
    for token in label_norm.split():
        if len(token) >= 3 and _contains_phrase(text_norm, token):
            score += 1
    for group, aliases in PRESET_HINT_ALIASES.items():
        if any(_contains_phrase(text_norm, a) for a in aliases):
            if group == "ed" and (
                "ed" in preset.id
                or "emergency" in label_norm
                or _contains_phrase(label_norm, "ed")
                or _contains_phrase(label_norm, "inpatient")
            ):
                score += 2
            if group == "pc" and ("pc" in preset.id or "primary" in label_norm):
                score += 2
            if group == "icu" and "icu" in label_norm:
                score += 2
            if group == "sab" and ("sab" in preset.id or "aureus" in label_norm):
                score += 2
            if group == "vap_ge5d" and "ge5d" in preset.id:
                score += 6
            if group == "vap_gt48h" and "gt48h" in preset.id:
                score += 6
    return score


def _choose_preset(module: SyndromeModule, text: str, preset_hint: str | None) -> tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    presets = module.pretest_presets
    if not presets:
        return None, warnings

    if preset_hint:
        for p in presets:
            if p.id == preset_hint:
                return p.id, warnings
        warnings.append(f"Unknown presetHint '{preset_hint}' for module '{module.id}', using text inference/default.")

    text_norm = normalize(text)
    scored = [(p, _preset_score(text_norm, module, p)) for p in presets]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    if best[1] == 0:
        return presets[0].id, warnings
    if len(scored) > 1 and scored[1][1] == best[1]:
        warnings.append(f"Multiple presets matched text; selected '{best[0].id}'.")
    return best[0].id, warnings


def _significant_label_aliases(item_id: str, label: str) -> List[str]:
    aliases: List[str] = []
    label_norm = normalize(label)
    if label_norm:
        aliases.append(label_norm)

    id_parts = item_id.split("_")[1:] if "_" in item_id else item_id.split("_")
    id_phrase = normalize(" ".join(id_parts))
    if id_phrase and len(id_phrase) >= 4:
        aliases.append(id_phrase)

    is_status_item = any(token in item_id for token in ["not_done", "unknown", "not_used", "na"])
    is_status_label = any(token in label.lower() for token in ["not done", "unknown", "not used", "n/a"])

    # small curated simplifications for common abbreviations
    if "cxr" in label.lower() and not is_status_item and not is_status_label:
        aliases.append("cxr")
    if "tte" in label.lower() and not is_status_item and not is_status_label:
        aliases.append("tte")
    if "tee" in label.lower() and not is_status_item and not is_status_label:
        aliases.append("tee")
    if "wbc" in label.lower() and not is_status_item and not is_status_label:
        aliases.append("wbc")
    return list(dict.fromkeys([a for a in aliases if len(a) >= 3]))


def _has_local_negation(text_norm: str, pos: int) -> bool:
    window_start = max(0, pos - 24)
    prefix = text_norm[window_start:pos].strip()
    if not prefix:
        return False
    last_token = prefix.split()[-1]
    return NEGATION_RE.fullmatch(last_token) is not None


def _has_nonmention_suffix(text_norm: str, end_pos: int) -> bool:
    suffix = text_norm[end_pos : min(len(text_norm), end_pos + 28)].strip()
    return suffix.startswith("not mentioned") or suffix.startswith("not documented") or suffix.startswith("not reported")


def _match_item_state(text_norm: str, item) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    alias_groups = COMMON_FINDING_ALIASES.get(item.id, {})
    has_explicit_absent_alias = bool(alias_groups.get("absent"))

    # Explicit aliases first (state-specific)
    explicit_candidates: List[Tuple[int, int, str, str]] = []
    for state in ("absent", "present"):
        for phrase in alias_groups.get(state, []):
            phrase_norm = normalize(phrase)
            for pos in _find_all_positions(text_norm, phrase_norm):
                if _has_nonmention_suffix(text_norm, pos + len(phrase_norm)):
                    continue
                resolved_state = state
                if state == "present":
                    if any(token in phrase_norm for token in ("not done", "unknown", "not used", "negative", "not sent")):
                        resolved_state = "present"
                    elif not has_explicit_absent_alias and _has_local_negation(text_norm, pos):
                        resolved_state = "absent"
                explicit_candidates.append((pos, pos + len(phrase_norm), resolved_state, phrase_norm))
    if explicit_candidates:
        filtered_candidates: List[Tuple[int, int, str, str]] = []
        for candidate in explicit_candidates:
            start, end, _, _ = candidate
            contained = False
            for other in explicit_candidates:
                other_start, other_end, _, _ = other
                if other is candidate:
                    continue
                if other_start <= start and other_end >= end and (other_end - other_start) > (end - start):
                    contained = True
                    break
            if not contained:
                filtered_candidates.append(candidate)
        pos, _, state, phrase_norm = max(filtered_candidates, key=lambda entry: entry[0])
        return state, pos, phrase_norm

    # Generic label matching
    generic_candidates: List[Tuple[int, int, str, str]] = []
    for phrase in _significant_label_aliases(item.id, item.label):
        for pos in _find_all_positions(text_norm, phrase):
            if _has_nonmention_suffix(text_norm, pos + len(phrase)):
                continue
            # "not done"/"unknown" style neutral items should be marked present if explicitly mentioned.
            if any(x in phrase for x in ["not done", "unknown", "not used", "na"]):
                generic_candidates.append(
                    (pos, pos + len(phrase), "absent" if _has_local_negation(text_norm, pos) else "present", phrase)
                )
                continue

            generic_candidates.append(
                (pos, pos + len(phrase), "absent" if _has_local_negation(text_norm, pos) else "present", phrase)
            )
    if generic_candidates:
        filtered_candidates: List[Tuple[int, int, str, str]] = []
        for candidate in generic_candidates:
            start, end, _, _ = candidate
            contained = False
            for other in generic_candidates:
                other_start, other_end, _, _ = other
                if other is candidate:
                    continue
                if other_start <= start and other_end >= end and (other_end - other_start) > (end - start):
                    contained = True
                    break
            if not contained:
                filtered_candidates.append(candidate)
        pos, _, state, phrase = max(filtered_candidates, key=lambda entry: entry[0])
        return state, pos, phrase

    return None, None, None


def _extract_findings(module: SyndromeModule, text: str) -> tuple[Dict[str, str], List[str], Dict[str, str]]:
    text_norm = normalize(text)
    findings: Dict[str, str] = {}
    match_pos: Dict[str, int] = {}
    match_alias: Dict[str, str] = {}
    warnings: List[str] = []

    for item in module.items:
        state, pos, alias = _match_item_state(text_norm, item)
        if state is None or pos is None or alias is None:
            continue
        findings[item.id] = state
        match_pos[item.id] = pos
        match_alias[item.id] = alias

    if not findings:
        warnings.append("No findings/tests were confidently extracted from text.")
    return findings, warnings, match_alias


def summarize_parsed_request(store: InMemoryModuleStore, parsed: AnalyzeRequest) -> tuple[ParsedUnderstanding, List[str], bool]:
    warnings: List[str] = []
    requires_confirmation = False
    module, mod_warnings = resolve_module_from_request(store=store, parsed_request=parsed)
    warnings.extend(mod_warnings)
    if module is None:
        warnings.append("Parsed request does not include a valid module.")
        return empty_understanding(), warnings, True

    preset_id = parsed.preset_id
    preset_label = None
    if preset_id:
        preset = next((p for p in module.pretest_presets if p.id == preset_id), None)
        if preset:
            preset_label = preset.label
        else:
            warnings.append(f"Parsed presetId '{preset_id}' is not valid for module '{module.id}'.")
            requires_confirmation = True
    elif module.pretest_presets:
        warnings.append("No presetId parsed; backend will use the module default preset.")
        requires_confirmation = True

    items_by_id = {item.id: item for item in module.items}
    present_labels: List[str] = []
    absent_labels: List[str] = []
    unknown_mentions: List[str] = []
    valid_findings_count = 0

    for fid, state in parsed.findings.items():
        item = items_by_id.get(fid)
        if item is None:
            warnings.append(f"Unknown finding id '{fid}' for module '{module.id}'.")
            unknown_mentions.append(fid)
            requires_confirmation = True
            continue
        valid_findings_count += 1
        if state == "present":
            present_labels.append(item.label)
        elif state == "absent":
            absent_labels.append(item.label)
        else:
            unknown_mentions.append(item.label)

    if valid_findings_count == 0:
        warnings.append("No valid findings parsed.")
        requires_confirmation = True
    elif valid_findings_count < 2:
        warnings.append("Low extraction confidence: fewer than 2 findings parsed.")
        requires_confirmation = True

    if parsed.ordered_finding_ids:
        invalid_ordered = [fid for fid in parsed.ordered_finding_ids if fid not in items_by_id]
        if invalid_ordered:
            warnings.append(f"Some orderedFindingIds are invalid for module '{module.id}': {invalid_ordered}")
            requires_confirmation = True

    understood = ParsedUnderstanding(
        moduleId=module.id,
        moduleName=module.name,
        presetId=preset_id,
        presetLabel=preset_label,
        findingsPresent=present_labels,
        findingsAbsent=absent_labels,
        unknownMentions=unknown_mentions,
    )
    return understood, warnings, requires_confirmation


def parse_text_to_request(
    *,
    store: InMemoryModuleStore,
    text: str,
    module_hint: str | None = None,
    preset_hint: str | None = None,
    include_explanation: bool = True,
) -> ParseTextResult:
    warnings: List[str] = []
    module, module_warnings = _choose_module(store, text, module_hint)
    warnings.extend(module_warnings)

    if module is None:
        return ParseTextResult(
            parsed_request=None,
            understood=empty_understanding(),
            warnings=warnings,
            requires_confirmation=True,
        )

    preset_id, preset_warnings = _choose_preset(module, text, preset_hint)
    warnings.extend(preset_warnings)
    preset_label = next((p.label for p in module.pretest_presets if p.id == preset_id), None)

    findings, finding_warnings, match_aliases = _extract_findings(module, text)
    warnings.extend(finding_warnings)

    ordered_ids = sorted(findings.keys(), key=lambda fid: normalize(text).find(match_aliases.get(fid, "")))
    parsed = AnalyzeRequest(
        moduleId=module.id,
        presetId=preset_id,
        findings=findings,
        orderedFindingIds=ordered_ids,
        includeExplanation=include_explanation,
    )

    understood, summary_warnings, summary_requires_confirmation = summarize_parsed_request(store, parsed)
    warnings.extend(summary_warnings)
    requires_confirmation = (
        summary_requires_confirmation
        or any("Multiple" in w for w in warnings)
        or any("Could not infer" in w for w in warnings)
    )
    if len(findings) < 2 and all("Low extraction confidence" not in w for w in warnings):
        warnings.append("Low extraction confidence: consider reviewing parsed findings before relying on the result.")
        requires_confirmation = True

    return ParseTextResult(
        parsed_request=parsed,
        understood=understood,
        warnings=warnings,
        requires_confirmation=requires_confirmation,
    )
