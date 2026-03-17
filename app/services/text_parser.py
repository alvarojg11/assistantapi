from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import AnalyzeRequest, ParsedUnderstanding, SyndromeModule
from .module_store import InMemoryModuleStore


NEGATION_RE = re.compile(r"\b(no|not|without|denies|deny|negative|normal|absent|none)\b")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

TB_UVEITIS_ENDEMIC_COUNTRY_ALIASES = (
    "mexico",
    "india",
    "pakistan",
    "bangladesh",
    "afghanistan",
    "nepal",
    "myanmar",
    "cambodia",
    "laos",
    "thailand",
    "mongolia",
    "philippines",
    "vietnam",
    "china",
    "indonesia",
    "papua new guinea",
    "haiti",
    "peru",
    "brazil",
    "bolivia",
    "ecuador",
    "colombia",
    "venezuela",
    "guatemala",
    "honduras",
    "el salvador",
    "nicaragua",
    "dominican republic",
    "somalia",
    "ethiopia",
    "eritrea",
    "djibouti",
    "kenya",
    "uganda",
    "tanzania",
    "nigeria",
    "cameroon",
    "ghana",
    "sierra leone",
    "liberia",
    "democratic republic of the congo",
    "congo",
    "angola",
    "zambia",
    "mozambique",
    "zimbabwe",
    "malawi",
    "madagascar",
    "rwanda",
    "burundi",
    "south africa",
)


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
        "staphylococcus epidermidis bacteremia",
        "s epidermidis bacteremia",
        "coagulase negative staph bacteremia",
        "coagulase negative staphylococcal bacteremia",
        "coagulase-negative staph bacteremia",
        "coagulase-negative staphylococcal bacteremia",
        "viridans bacteremia",
        "viridans streptococcal bacteremia",
        "nbhs bacteremia",
        "enterococcus bacteremia",
        "enterococcal bacteremia",
        "enterococcus faecalis bacteremia",
        "e faecalis bacteremia",
        "enterococcus faecium bacteremia",
        "e faecium bacteremia",
        "prosthetic valve endocarditis",
        "tavi endocarditis",
        "tavr endocarditis",
    ],
    "active_tb": ["tb", "active tb", "tuberculosis", "pulmonary tb"],
    "tb_uveitis": [
        "tuberculous uveitis",
        "tubercular uveitis",
        "tb uveitis",
        "ocular tb",
        "ocular tuberculosis",
        "tuberculous choroiditis",
    ],
    "pjp": ["pjp", "pcp", "pneumocystis"],
    "inv_candida": ["invasive candidiasis", "candida", "candidemia"],
    "inv_mold": ["invasive mold", "aspergillus", "mucor", "mold infection"],
    "septic_arthritis": ["septic arthritis", "infectious arthritis", "septic joint", "native joint infection"],
    "bacterial_meningitis": ["bacterial meningitis", "meningitis", "pyogenic meningitis", "bacterial meningitis concern"],
    "encephalitis": ["encephalitis", "viral encephalitis", "infectious encephalitis", "hsv encephalitis", "herpes encephalitis"],
    "spinal_epidural_abscess": ["spinal epidural abscess", "sea", "epidural abscess", "spinal epidural infection"],
    "brain_abscess": ["brain abscess", "cerebral abscess", "intracranial abscess", "pyogenic brain abscess"],
    "necrotizing_soft_tissue_infection": ["necrotizing soft tissue infection", "nsti", "necrotizing fasciitis", "flesh eating infection", "fournier gangrene"],
    "diabetic_foot_infection": [
        "diabetic foot infection",
        "infected diabetic foot ulcer",
        "diabetic foot ulcer infection",
        "diabetic foot osteomyelitis",
        "diabetic foot wound infection",
        "diabetic wound infection",
        "diabetic foot wound",
        "dfi",
    ],
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

PRESET_GENERIC_LABEL_TOKENS = {
    "evaluation",
    "workup",
    "consult",
    "specialty",
    "suspected",
    "possible",
    "acute",
    "severe",
    "low",
    "high",
    "concern",
    "pathway",
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
    "endo_bcx_major_typical": {
        "present": [
            "typical blood cultures",
            "major blood culture criterion",
            "typical endocarditis blood cultures in at least 2 sets",
        ]
    },
    "endo_bcx_major_persistent": {
        "present": [
            "persistent bacteremia",
            "persistent positive blood cultures",
            "blood cultures persistently positive",
        ],
        "absent": ["blood cultures cleared without persistent positivity"],
    },
    "endo_bcx_saureus_multi": {
        "present": [
            "staphylococcus aureus in at least 2 blood culture sets",
            "staphylococcus aureus in 2 blood culture sets",
            "staphylococcus aureus in >=2 blood culture sets",
            "staphylococcus aureus in >=2 sets",
            "s aureus in at least 2 blood culture sets",
            "s aureus in >=2 blood culture sets",
            "staphylococcus aureus bacteremia",
            "staph aureus bacteremia",
            "mssa bacteremia",
            "mrsa bacteremia",
        ],
        "absent": ["no staphylococcus aureus in at least 2 blood culture sets"],
    },
    "endo_bcx_cons_prosthetic_multi": {
        "present": [
            "coagulase negative staph in at least 2 blood culture sets with prosthetic valve",
            "coagulase-negative staph in at least 2 blood culture sets with prosthetic valve",
            "coagulase negative staph in >=2 blood culture sets with prosthetic valve",
            "coagulase-negative staph in >=2 blood culture sets with prosthetic valve",
            "coagulase negative staphylococci in at least 2 blood culture sets with prosthetic valve",
            "coagulase-negative staphylococci in at least 2 blood culture sets with prosthetic valve",
            "coagulase negative staphylococci in >=2 blood culture sets with prosthetic valve",
            "coagulase-negative staphylococci in >=2 blood culture sets with prosthetic valve",
            "coagulase negative staph bacteremia with prosthetic valve",
            "prosthetic valve with coagulase negative staph bacteremia",
            "prosthetic valve with coagulase-negative staph bacteremia",
            "prosthetic valve endocarditis concern from coagulase negative staph bacteremia",
            "prosthetic valve endocarditis concern from coagulase-negative staph bacteremia",
            "coagulase-negative staph bacteremia with prosthetic valve",
            "coagulase negative staph bacteremia after tavr",
            "coagulase-negative staph bacteremia after tavr",
            "coagulase negative staph bacteremia after tavi",
            "coagulase-negative staph bacteremia after tavi",
            "tavr patient with coagulase negative staph bacteremia",
            "tavr patient with coagulase-negative staph bacteremia",
            "tavi patient with coagulase negative staph bacteremia",
            "tavi patient with coagulase-negative staph bacteremia",
            "staphylococcus epidermidis bacteremia with prosthetic valve",
            "prosthetic valve with staphylococcus epidermidis bacteremia",
            "prosthetic valve endocarditis concern from staphylococcus epidermidis bacteremia",
            "s epidermidis bacteremia with prosthetic valve",
            "prosthetic valve with s epidermidis bacteremia",
            "staphylococcus epidermidis bacteremia after tavr",
            "staphylococcus epidermidis bacteremia after tavi",
            "tavr patient with staphylococcus epidermidis bacteremia",
            "tavi patient with staphylococcus epidermidis bacteremia",
            "coagulase negative staph bacteremia with cardiac device",
            "coagulase-negative staph bacteremia with cardiac device",
            "coagulase negative staph bacteremia with cied",
            "coagulase-negative staph bacteremia with cied",
            "staphylococcus epidermidis bacteremia with cardiac device",
            "staphylococcus epidermidis bacteremia with cied",
        ],
        "absent": ["no coagulase negative staphylococcal prosthetic/device bacteremia"],
    },
    "endo_bcx_enterococcus_prosthetic_multi": {
        "present": [
            "enterococcus bacteremia with prosthetic valve",
            "prosthetic valve with enterococcus bacteremia",
            "enterococcus bacteremia after tavr",
            "enterococcus bacteremia after tavi",
            "tavr patient with enterococcus bacteremia",
            "tavi patient with enterococcus bacteremia",
            "enterococcal bacteremia with prosthetic valve",
            "prosthetic valve with enterococcal bacteremia",
            "enterococcal bacteremia after tavr",
            "enterococcal bacteremia after tavi",
            "tavr patient with enterococcal bacteremia",
            "tavi patient with enterococcal bacteremia",
            "enterococcus faecium bacteremia with prosthetic valve",
            "prosthetic valve with enterococcus faecium bacteremia",
            "e faecium bacteremia with prosthetic valve",
            "prosthetic valve with e faecium bacteremia",
            "enterococcus faecium bacteremia after tavr",
            "enterococcus faecium bacteremia after tavi",
            "tavr patient with enterococcus faecium bacteremia",
            "tavi patient with enterococcus faecium bacteremia",
            "enterococcus bacteremia with cardiac device",
            "enterococcus bacteremia with cied",
            "enterococcal bacteremia with cardiac device",
            "enterococcal bacteremia with cied",
        ],
        "absent": ["no enterococcal prosthetic/device bacteremia"],
    },
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
    "tbu_phenotype_au_first": {
        "present": [
            "anterior uveitis first episode",
            "first episode anterior uveitis",
            "first episode of anterior uveitis",
            "first anterior uveitis episode",
        ],
    },
    "tbu_phenotype_au_recurrent": {
        "present": [
            "anterior uveitis recurrent",
            "recurrent anterior uveitis",
            "recurrent episode anterior uveitis",
            "recurrent episode of anterior uveitis",
        ],
    },
    "tbu_phenotype_intermediate": {
        "present": ["intermediate uveitis"],
    },
    "tbu_phenotype_panuveitis": {
        "present": ["panuveitis"],
    },
    "tbu_phenotype_rv_active": {
        "present": ["active retinal vasculitis", "retinal vasculitis active"],
    },
    "tbu_phenotype_rv_inactive": {
        "present": ["inactive retinal vasculitis", "retinal vasculitis inactive"],
    },
    "tbu_phenotype_choroiditis_serpiginoid": {
        "present": ["serpiginoid choroiditis", "serpiginous like choroiditis", "tb slc"],
    },
    "tbu_phenotype_choroiditis_multifocal": {
        "present": ["multifocal choroiditis", "non serpiginoid choroiditis", "nonserpiginoid choroiditis"],
    },
    "tbu_phenotype_choroiditis_tuberculoma": {
        "present": ["choroidal tuberculoma", "choroidal nodule", "tuberculoma"],
    },
    "tbu_endemicity_endemic": {
        "present": ["tb endemic region", "from endemic region", "from tb endemic area", "high burden tb country"],
    },
    "tbu_endemicity_non_endemic": {
        "present": ["tb non endemic region", "from non endemic region", "from tb non endemic area"],
    },
    "tbu_tst_positive": {
        "present": ["tst positive", "mantoux positive", "tuberculin skin test positive", "ppd positive"],
    },
    "tbu_tst_negative": {
        "present": ["tst negative", "mantoux negative", "tuberculin skin test negative", "ppd negative"],
    },
    "tbu_tst_na": {
        "present": ["tst not done", "mantoux not done", "tuberculin skin test not done", "ppd not done"],
    },
    "tbu_igra_positive": {
        "present": [
            "igra positive",
            "igra is positive",
            "quantiferon positive",
            "quantiferon is positive",
            "qft positive",
            "qft is positive",
            "t spot positive",
            "t spot is positive",
        ],
    },
    "tbu_igra_negative": {
        "present": [
            "igra negative",
            "igra is negative",
            "quantiferon negative",
            "quantiferon is negative",
            "qft negative",
            "qft is negative",
            "t spot negative",
            "t spot is negative",
        ],
    },
    "tbu_igra_na": {
        "present": ["igra not done", "quantiferon not done", "qft not done", "t spot not done"],
    },
    "tbu_chest_imaging_positive": {
        "present": [
            "chest x ray positive for tb",
            "chest xray positive for tb",
            "chest ct positive for tb",
            "chest imaging positive for tb",
            "chest imaging with healed tb signs",
            "chest imaging with active tb signs",
        ],
    },
    "tbu_chest_imaging_negative": {
        "present": [
            "chest x ray negative for tb",
            "chest xray negative for tb",
            "chest ct negative for tb",
            "chest imaging negative for tb",
            "normal chest radiograph",
            "chest radiograph is normal",
            "normal chest x ray",
            "chest x ray is normal",
            "normal chest xray",
            "chest xray is normal",
            "chest radiograph normal",
        ],
    },
    "tbu_chest_imaging_na": {
        "present": ["chest imaging not done", "chest x ray not done", "chest xray not done", "chest ct not done"],
    },
    "tbu_pretest_prior_tb_or_ltbi": {
        "present": [
            "prior tb",
            "previous tb",
            "history of tb",
            "latent tb infection",
            "ltbi",
            "treated tuberculosis",
            "prior pulmonary tb",
        ],
    },
    "tbu_pretest_close_tb_contact": {
        "present": [
            "tb contact",
            "close tb contact",
            "household tb contact",
            "family member with tb",
            "tb exposure",
            "known exposure to tuberculosis",
        ],
    },
    "tbu_harm_macular_or_vision_threatening_lesion": {
        "present": [
            "macular involvement",
            "macula involved",
            "posterior pole involvement",
            "foveal involvement",
            "vision threatening lesion",
            "sight threatening lesion",
            "central vision threatening lesion",
        ],
    },
    "tbu_harm_bilateral_or_only_seeing_eye": {
        "present": [
            "bilateral uveitis",
            "bilateral disease",
            "only seeing eye",
            "only functional eye",
            "monocular patient",
            "one seeing eye",
        ],
    },
    "tbu_harm_progressive_vision_loss_or_severe_inflammation": {
        "present": [
            "progressive vision loss",
            "worsening vision",
            "declining vision",
            "severe inflammation",
            "vision rapidly worsening",
            "steroid dependent uveitis",
            "steroid dependent inflammation",
        ],
    },
    "tbu_harm_immunosuppressed": {
        "present": [
            "immunosuppressed",
            "immunocompromised",
            "on immunosuppression",
            "receiving biologic therapy",
            "transplant recipient",
            "hiv positive",
        ],
    },
    "tbu_harm_hepatotoxicity_risk": {
        "present": [
            "baseline liver disease",
            "cirrhosis",
            "chronic hepatitis",
            "high hepatotoxicity risk",
            "baseline transaminitis",
            "significant liver injury risk",
        ],
    },
    "tbu_harm_cld_mild": {
        "present": [
            "child pugh a",
            "child-pugh a",
            "child class a",
            "compensated cirrhosis",
            "mild chronic liver disease",
        ],
    },
    "tbu_harm_cld_moderate": {
        "present": [
            "child pugh b",
            "child-pugh b",
            "child class b",
            "moderate chronic liver disease",
        ],
    },
    "tbu_harm_cld_severe": {
        "present": [
            "child pugh c",
            "child-pugh c",
            "child class c",
            "decompensated cirrhosis",
            "hepatic decompensation",
            "severe chronic liver disease",
        ],
    },
    "tbu_harm_ethambutol_ocular_risk": {
        "present": [
            "high ethambutol ocular toxicity risk",
            "high ethambutol optic neuropathy risk",
            "baseline optic neuropathy",
            "history of optic neuropathy",
            "renal dysfunction",
            "chronic kidney disease",
            "older age with ethambutol risk",
        ],
    },
    "tbu_harm_major_drug_interaction_or_intolerance": {
        "present": [
            "major rifamycin interaction",
            "major drug interaction with rifampin",
            "rifampin interaction",
            "prior att intolerance",
            "prior tb drug intolerance",
            "history of severe att toxicity",
            "concern for drug resistant tb",
            "previous tuberculosis treatment",
        ],
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
        "present": [
            "serum galactomannan positive",
            "serum galactomannan odi above 0.5",
            "serum galactomannan odi greater than 0.5",
            "serum gm above 0.5",
        ],
        "absent": [
            "serum galactomannan negative",
            "serum galactomannan odi 0.5 or below",
            "serum gm 0.5 or below",
        ],
    },
    "imi_bal_gm_odi10": {
        "present": [
            "bal galactomannan positive",
            "bal galactomannan odi above 1.0",
            "bal galactomannan odi greater than 1.0",
            "bal gm above 1.0",
        ],
        "absent": [
            "bal galactomannan negative",
            "bal galactomannan odi 1.0 or below",
            "bal gm 1.0 or below",
        ],
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
    "imi_aspergillus_culture_resp": {
        "present": [
            "bal culture grows aspergillus",
            "bal culture growing aspergillus",
            "bal fungal culture grows aspergillus",
            "bal fungal culture growing aspergillus",
            "respiratory culture grows aspergillus",
            "respiratory culture growing aspergillus",
            "sputum culture grows aspergillus",
            "sputum culture growing aspergillus",
            "sputum culture positive for aspergillus",
            "endotracheal aspirate culture grows aspergillus",
            "endotracheal aspirate culture growing aspergillus",
            "lower respiratory culture grows aspergillus",
            "lower respiratory culture growing aspergillus",
        ],
    },
    "imi_aspergillus_culture_na": {
        "present": [
            "respiratory fungal culture not done",
            "aspergillus respiratory culture not done",
            "bal fungal culture not done",
        ],
        "absent": [
            "respiratory fungal culture completed",
            "aspergillus respiratory culture completed",
            "bal fungal culture completed",
        ],
    },
    "imi_aspergillus_pcr_plasma": {
        "present": [
            "plasma aspergillus pcr positive",
            "aspergillus pcr positive from plasma",
            "blood aspergillus pcr positive",
        ],
        "absent": [
            "aspergillus pcr negative from plasma",
            "plasma aspergillus pcr negative",
            "blood aspergillus pcr negative",
        ],
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
    "sa_host_ra": {
        "present": ["rheumatoid arthritis", "inflammatory arthritis"],
        "absent": ["no rheumatoid arthritis", "no inflammatory arthritis"],
    },
    "sa_host_diabetes": {
        "present": ["diabetes", "diabetes mellitus", "diabetic"],
        "absent": ["no diabetes", "not diabetic"],
    },
    "sa_host_immunosuppression": {
        "present": ["immunosuppressed", "on chemotherapy", "transplant recipient", "on biologic therapy", "high dose steroids"],
        "absent": ["not immunosuppressed", "no immunosuppression"],
    },
    "sa_host_ivdu": {
        "present": ["injection drug use", "ivdu", "injects drugs"],
        "absent": ["no injection drug use", "no ivdu"],
    },
    "sa_host_recent_joint_surgery_or_injection": {
        "present": ["recent joint injection", "recent arthroscopy", "recent joint surgery", "penetrating trauma to the joint"],
        "absent": ["no recent joint injection or surgery", "no penetrating trauma to the joint"],
    },
    "sa_host_bacteremia_or_overlying_ssti": {
        "present": ["bacteremia", "bloodstream infection", "overlying cellulitis", "skin infection over the joint"],
        "absent": ["no bacteremia", "no overlying cellulitis", "no bloodstream infection"],
    },
    "sa_sym_monoarthritis": {
        "present": ["acute monoarthritis", "hot swollen joint", "painful swollen joint", "single swollen joint", "swollen painful knee"],
        "absent": ["no acute monoarthritis", "no swollen joint"],
    },
    "sa_vital_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "sa_exam_painful_rom": {
        "present": ["pain with passive range of motion", "painful range of motion", "severe pain with movement"],
        "absent": ["painless range of motion", "full range of motion without pain"],
    },
    "sa_exam_warmth_effusion": {
        "present": ["warm swollen joint", "joint warmth", "joint effusion", "erythematous joint", "swollen knee"],
        "absent": ["no joint effusion", "no joint warmth", "joint not swollen"],
    },
    "sa_crp": {
        "present": ["crp elevated", "elevated crp", "c reactive protein elevated"],
        "absent": ["crp not elevated", "normal crp"],
    },
    "sa_esr": {
        "present": ["esr elevated", "elevated esr"],
        "absent": ["esr not elevated", "normal esr"],
    },
    "sa_synovial_wbc_ge50k": {
        "present": ["synovial wbc above 50000", "synovial fluid wbc above 50000", "synovial white count above 50000", "synovial wbc at least 50000"],
        "absent": ["synovial wbc below 50000", "synovial fluid wbc below 50000", "synovial white count below 50000"],
    },
    "sa_synovial_na": {
        "present": ["arthrocentesis not done", "synovial cell count not done", "joint aspiration not done"],
        "absent": ["arthrocentesis completed", "joint aspiration completed"],
    },
    "sa_synovial_pmn_ge90": {
        "present": ["synovial pmn above 90", "synovial pmn 90 percent", "neutrophils above 90 percent in synovial fluid"],
        "absent": ["synovial pmn below 90", "neutrophils below 90 percent in synovial fluid"],
    },
    "sa_gram_stain": {
        "present": ["synovial gram stain positive", "gram stain positive from synovial fluid", "organisms seen on gram stain"],
        "absent": ["synovial gram stain negative", "gram stain negative from synovial fluid"],
    },
    "sa_gram_stain_na": {
        "present": ["synovial gram stain not done", "gram stain not done on synovial fluid"],
        "absent": ["synovial gram stain completed", "gram stain completed on synovial fluid"],
    },
    "sa_synovial_culture": {
        "present": ["synovial culture positive", "joint aspirate culture positive", "synovial fluid culture positive"],
        "absent": ["synovial culture negative", "joint aspirate culture negative", "synovial fluid culture negative"],
    },
    "sa_synovial_culture_na": {
        "present": ["synovial culture not done", "joint aspirate culture not done"],
        "absent": ["synovial culture completed", "joint aspirate culture completed"],
    },
    "sa_blood_culture_positive": {
        "present": ["blood culture positive with matching pathogen", "bacteremia with matching organism", "blood cultures growing the same organism"],
        "absent": ["blood culture negative", "blood cultures negative"],
    },
    "sa_blood_culture_na": {
        "present": ["blood cultures not done", "blood culture not done"],
        "absent": ["blood cultures completed", "blood culture completed"],
    },
    "sa_ultrasound_effusion": {
        "present": ["ultrasound shows joint effusion", "joint ultrasound with effusion", "imaging shows joint effusion"],
        "absent": ["ultrasound without effusion", "joint ultrasound without effusion"],
    },
    "sa_imaging_na": {
        "present": ["joint imaging not done", "ultrasound not done for the joint"],
        "absent": ["joint imaging completed", "ultrasound completed for the joint"],
    },
    "bm_host_immunocompromised": {
        "present": ["immunocompromised", "immunosuppressed", "on chemotherapy", "transplant recipient", "high dose steroids"],
        "absent": ["not immunocompromised", "no immunosuppression"],
    },
    "bm_host_csf_leak_or_neurosurgery": {
        "present": ["csf leak", "recent neurosurgery", "vp shunt", "cns device", "recent skull base fracture"],
        "absent": ["no csf leak", "no recent neurosurgery"],
    },
    "bm_host_otitis_sinusitis": {
        "present": ["otitis", "mastoiditis", "sinusitis", "ear infection", "severe sinus infection"],
        "absent": ["no otitis", "no mastoiditis", "no sinusitis"],
    },
    "bm_host_bacteremia_sepsis": {
        "present": ["bacteremia", "bloodstream infection", "sepsis", "septic shock"],
        "absent": ["no bacteremia", "no sepsis"],
    },
    "bm_sym_headache": {
        "present": ["headache", "severe headache"],
        "absent": ["no headache"],
    },
    "bm_vital_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "bm_exam_neck_stiffness": {
        "present": ["neck stiffness", "meningismus", "nuchal rigidity", "stiff neck"],
        "absent": ["no neck stiffness", "no meningismus", "supple neck"],
    },
    "bm_exam_ams": {
        "present": ["altered mental status", "confused", "encephalopathic", "obtunded"],
        "absent": ["normal mental status", "alert and oriented"],
    },
    "bm_exam_petechiae": {
        "present": ["petechial rash", "purpuric rash", "petechiae", "purpura"],
        "absent": ["no petechiae", "no purpura", "no rash"],
    },
    "bm_exam_seizure": {
        "present": ["seizure", "new seizure", "convulsion"],
        "absent": ["no seizure"],
    },
    "bm_serum_procalcitonin": {
        "present": ["serum procalcitonin elevated", "procalcitonin elevated", "high procalcitonin"],
        "absent": ["serum procalcitonin not elevated", "procalcitonin normal", "low procalcitonin"],
    },
    "bm_csf_wbc_ge1000": {
        "present": ["csf wbc above 1000", "csf white count above 1000", "csf wbc at least 1000", "csf pleocytosis above 1000"],
        "absent": ["csf wbc below 1000", "csf white count below 1000"],
    },
    "bm_csf_cell_count_na": {
        "present": ["lp not done", "lumbar puncture not done", "csf cell count not done"],
        "absent": ["lp completed", "lumbar puncture completed", "csf cell count completed"],
    },
    "bm_csf_pmn_ge80": {
        "present": ["csf neutrophils above 80 percent", "csf pmn above 80", "csf neutrophils at least 80 percent"],
        "absent": ["csf neutrophils below 80 percent", "csf pmn below 80"],
    },
    "bm_csf_glucose_ratio_low": {
        "present": ["csf glucose low", "csf serum glucose ratio below 0.4", "csf glucose ratio below 0.4", "low csf glucose"],
        "absent": ["csf glucose not low", "csf serum glucose ratio not low"],
    },
    "bm_csf_protein_high": {
        "present": ["csf protein elevated", "high csf protein"],
        "absent": ["csf protein not elevated", "normal csf protein"],
    },
    "bm_csf_lactate_high": {
        "present": ["csf lactate elevated", "high csf lactate"],
        "absent": ["csf lactate not elevated", "normal csf lactate"],
    },
    "bm_csf_gram_stain": {
        "present": ["csf gram stain positive", "gram stain positive from csf", "organisms seen on csf gram stain"],
        "absent": ["csf gram stain negative", "gram stain negative from csf"],
    },
    "bm_csf_gram_na": {
        "present": ["csf gram stain not done", "gram stain not done on csf"],
        "absent": ["csf gram stain completed", "gram stain completed on csf"],
    },
    "bm_csf_culture": {
        "present": ["csf culture positive", "cerebrospinal fluid culture positive"],
        "absent": ["csf culture negative", "cerebrospinal fluid culture negative"],
    },
    "bm_csf_culture_na": {
        "present": ["csf culture not done", "cerebrospinal fluid culture not done"],
        "absent": ["csf culture completed", "cerebrospinal fluid culture completed"],
    },
    "bm_blood_culture_positive": {
        "present": ["blood culture positive with meningitis pathogen", "blood cultures growing pneumococcus", "blood cultures growing meningococcus", "blood cultures positive with matching organism"],
        "absent": ["blood culture negative", "blood cultures negative"],
    },
    "bm_blood_culture_na": {
        "present": ["blood cultures not done", "blood culture not done"],
        "absent": ["blood cultures completed", "blood culture completed"],
    },
    "bm_csf_bacterial_pcr": {
        "present": ["csf bacterial pcr positive", "meningitis panel positive for bacteria", "multiplex csf panel positive for bacteria"],
        "absent": ["csf bacterial pcr negative", "meningitis panel negative for bacteria"],
    },
    "bm_csf_pcr_na": {
        "present": ["csf bacterial pcr not done", "meningitis panel not done"],
        "absent": ["csf bacterial pcr completed", "meningitis panel completed"],
    },
    "bm_imaging_supportive": {
        "present": ["imaging supportive of meningitis", "mri supportive of meningitis", "ct supportive of meningitis"],
        "absent": ["imaging not supportive of meningitis", "mri not supportive of meningitis", "ct not supportive of meningitis"],
    },
    "bm_imaging_na": {
        "present": ["neuroimaging not done", "brain imaging not done", "head ct not done", "brain mri not done"],
        "absent": ["neuroimaging completed", "brain imaging completed"],
    },
    "enc_host_immunocompromised": {
        "present": ["immunocompromised", "immunosuppressed", "on chemotherapy", "advanced hiv", "high dose steroids"],
        "absent": ["not immunocompromised", "no immunosuppression"],
    },
    "enc_host_transplant_or_biologic": {
        "present": ["transplant recipient", "on biologic therapy", "on tacrolimus", "on major immunomodulator"],
        "absent": ["no transplant", "not on biologic therapy"],
    },
    "enc_host_vector_travel_exposure": {
        "present": ["mosquito exposure", "tick exposure", "recent travel", "animal bite", "arboviral exposure"],
        "absent": ["no travel exposure", "no vector exposure"],
    },
    "enc_sym_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "enc_exam_ams": {
        "present": ["altered mental status", "encephalopathy", "confused", "obtunded", "encephalopathic"],
        "absent": ["normal mental status", "alert and oriented"],
    },
    "enc_exam_behavioral_change": {
        "present": ["behavioral change", "personality change", "memory change", "short term memory loss"],
        "absent": ["no behavioral change", "no memory change"],
    },
    "enc_exam_focal_deficit": {
        "present": ["focal neurologic deficit", "aphasia", "hemiparesis", "new neurologic deficit"],
        "absent": ["no focal neurologic deficit"],
    },
    "enc_exam_seizure": {
        "present": ["seizure", "new seizure", "convulsion"],
        "absent": ["no seizure"],
    },
    "enc_csf_pleocytosis": {
        "present": ["csf pleocytosis", "csf white count elevated", "lymphocytic pleocytosis", "csf wbc elevated"],
        "absent": ["no csf pleocytosis", "normal csf cell count"],
    },
    "enc_csf_cell_count_na": {
        "present": ["lp not done", "lumbar puncture not done", "csf cell count not done"],
        "absent": ["lp completed", "lumbar puncture completed", "csf cell count completed"],
    },
    "enc_csf_protein_high": {
        "present": ["csf protein elevated", "high csf protein"],
        "absent": ["csf protein normal", "csf protein not elevated"],
    },
    "enc_csf_rbc_high": {
        "present": ["csf rbc elevated", "hemorrhagic csf", "rbc in csf elevated", "bloody csf"],
        "absent": ["csf rbc not elevated", "no rbc elevation in csf"],
    },
    "enc_hsv_pcr": {
        "present": ["csf hsv pcr positive", "hsv pcr positive in csf", "positive hsv from csf"],
        "absent": ["csf hsv pcr negative", "hsv pcr negative in csf"],
    },
    "enc_hsv_pcr_na": {
        "present": ["csf hsv pcr not done", "hsv pcr not done on csf"],
        "absent": ["csf hsv pcr completed", "hsv pcr completed on csf"],
    },
    "enc_csf_viral_pcr": {
        "present": ["csf viral pcr positive", "viral meningitis encephalitis panel positive", "positive viral csf panel"],
        "absent": ["csf viral pcr negative", "viral csf panel negative"],
    },
    "enc_csf_viral_pcr_na": {
        "present": ["csf viral pcr not done", "viral csf panel not done"],
        "absent": ["csf viral pcr completed", "viral csf panel completed"],
    },
    "enc_mri_temporal": {
        "present": ["temporal lobe mri abnormality", "mri temporal hyperintensity", "frontotemporal mri abnormality", "insular mri abnormality", "mri compatible with hsv encephalitis"],
        "absent": ["mri without temporal abnormality", "mri not supportive of encephalitis"],
    },
    "enc_mri_na": {
        "present": ["brain mri not done", "mri not done"],
        "absent": ["brain mri completed", "mri completed"],
    },
    "enc_eeg_temporal": {
        "present": ["eeg temporal slowing", "eeg periodic discharges", "temporal periodic discharges on eeg", "eeg compatible with encephalitis"],
        "absent": ["eeg without temporal slowing", "eeg not supportive of encephalitis"],
    },
    "enc_eeg_na": {
        "present": ["eeg not done"],
        "absent": ["eeg completed"],
    },
    "sea_host_ivdu": {
        "present": ["injection drug use", "ivdu", "injects drugs"],
        "absent": ["no injection drug use", "no ivdu"],
    },
    "sea_host_diabetes": {
        "present": ["diabetes", "diabetes mellitus", "diabetic"],
        "absent": ["no diabetes", "not diabetic"],
    },
    "sea_host_immunocompromised": {
        "present": ["immunocompromised", "immunosuppressed", "on chemotherapy", "high dose steroids"],
        "absent": ["not immunocompromised", "no immunosuppression"],
    },
    "sea_host_recent_spinal_procedure": {
        "present": ["recent spinal procedure", "recent epidural injection", "recent spinal surgery", "spinal hardware", "recent lumbar puncture"],
        "absent": ["no recent spinal procedure", "no recent spinal surgery"],
    },
    "sea_host_bacteremia_or_ssti": {
        "present": ["bacteremia", "bloodstream infection", "cellulitis", "skin infection", "soft tissue infection"],
        "absent": ["no bacteremia", "no skin infection", "no soft tissue infection"],
    },
    "sea_sym_back_pain": {
        "present": ["severe back pain", "focal back pain", "neck pain", "spine pain"],
        "absent": ["no back pain", "no neck pain"],
    },
    "sea_vital_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "sea_exam_spinal_tenderness": {
        "present": ["midline spinal tenderness", "spinal tenderness", "vertebral tenderness"],
        "absent": ["no spinal tenderness", "no midline tenderness"],
    },
    "sea_exam_neuro_deficit": {
        "present": ["focal neurologic deficit", "new weakness", "leg weakness", "paraparesis", "neurologic deficit"],
        "absent": ["no focal neurologic deficit", "no weakness"],
    },
    "sea_exam_bowel_bladder": {
        "present": ["urinary retention", "bowel bladder dysfunction", "saddle anesthesia with retention", "new incontinence"],
        "absent": ["no urinary retention", "no bowel bladder dysfunction"],
    },
    "sea_esr_high": {
        "present": ["esr elevated", "markedly elevated esr", "esr high"],
        "absent": ["esr not elevated", "normal esr"],
    },
    "sea_crp_high": {
        "present": ["crp elevated", "elevated crp", "crp high"],
        "absent": ["crp not elevated", "normal crp"],
    },
    "sea_wbc_high": {
        "present": ["wbc elevated", "white count elevated", "leukocytosis"],
        "absent": ["wbc not elevated", "normal white count", "no leukocytosis"],
    },
    "sea_blood_culture_positive": {
        "present": ["blood culture positive", "blood cultures growing staph", "positive blood cultures with matching organism"],
        "absent": ["blood culture negative", "blood cultures negative"],
    },
    "sea_blood_culture_na": {
        "present": ["blood cultures not done", "blood culture not done"],
        "absent": ["blood cultures completed", "blood culture completed"],
    },
    "sea_mri_positive": {
        "present": ["mri shows epidural abscess", "spine mri positive for epidural abscess", "epidural phlegmon on mri", "mri with spinal epidural abscess"],
        "absent": ["mri without epidural abscess", "spine mri negative for epidural abscess"],
    },
    "sea_mri_na": {
        "present": ["mri not done", "spine mri not done"],
        "absent": ["mri completed", "spine mri completed"],
    },
    "sea_discitis_osteo": {
        "present": ["discitis on imaging", "vertebral osteomyelitis on imaging", "discitis osteomyelitis"],
        "absent": ["no discitis", "no vertebral osteomyelitis"],
    },
    "ba_host_otogenic_sinus_dental": {
        "present": ["sinusitis", "mastoiditis", "otitis", "dental infection", "dental abscess", "ear infection"],
        "absent": ["no sinusitis", "no dental infection", "no ear infection"],
    },
    "ba_host_neurosurgery_trauma": {
        "present": ["recent neurosurgery", "craniotomy", "penetrating head trauma", "cranial hardware"],
        "absent": ["no recent neurosurgery", "no head trauma"],
    },
    "ba_host_endocarditis_bacteremia": {
        "present": ["endocarditis", "bacteremia", "bloodstream infection", "hematogenous source"],
        "absent": ["no bacteremia", "no endocarditis"],
    },
    "ba_host_immunocompromised": {
        "present": ["immunocompromised", "immunosuppressed", "on chemotherapy", "high dose steroids"],
        "absent": ["not immunocompromised", "no immunosuppression"],
    },
    "ba_sym_headache": {
        "present": ["headache", "severe headache"],
        "absent": ["no headache"],
    },
    "ba_vital_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "ba_exam_focal_deficit": {
        "present": ["focal neurologic deficit", "aphasia", "hemiparesis", "new weakness", "focal deficit"],
        "absent": ["no focal neurologic deficit", "no focal deficit"],
    },
    "ba_exam_ams": {
        "present": ["altered mental status", "confused", "encephalopathic", "obtunded"],
        "absent": ["normal mental status", "alert and oriented"],
    },
    "ba_exam_seizure": {
        "present": ["seizure", "new seizure", "convulsion"],
        "absent": ["no seizure"],
    },
    "ba_crp_high": {
        "present": ["crp elevated", "elevated crp", "crp high"],
        "absent": ["crp not elevated", "normal crp"],
    },
    "ba_wbc_high": {
        "present": ["wbc elevated", "white count elevated", "leukocytosis"],
        "absent": ["wbc not elevated", "normal white count", "no leukocytosis"],
    },
    "ba_blood_culture_positive": {
        "present": ["blood culture positive", "blood cultures growing streptococcus", "positive blood cultures with matching organism"],
        "absent": ["blood culture negative", "blood cultures negative"],
    },
    "ba_blood_culture_na": {
        "present": ["blood cultures not done", "blood culture not done"],
        "absent": ["blood cultures completed", "blood culture completed"],
    },
    "ba_aspirate_culture_positive": {
        "present": ["abscess aspirate culture positive", "operative culture positive", "pus culture positive from brain abscess"],
        "absent": ["abscess aspirate culture negative", "operative culture negative"],
    },
    "ba_aspirate_culture_na": {
        "present": ["abscess aspirate culture not done", "operative culture not done"],
        "absent": ["abscess aspirate culture completed", "operative culture completed"],
    },
    "ba_mri_dwi_positive": {
        "present": ["mri with restricted diffusion compatible with abscess", "ring enhancing lesion with restricted diffusion", "mri compatible with brain abscess"],
        "absent": ["mri without restricted diffusion compatible with abscess", "mri not supportive of brain abscess"],
    },
    "ba_mri_na": {
        "present": ["brain mri not done", "mri not done"],
        "absent": ["brain mri completed", "mri completed"],
    },
    "ba_ct_ring_enhancing": {
        "present": ["ct with ring enhancing lesion", "ring-enhancing lesion on ct", "contrast ct compatible with abscess"],
        "absent": ["ct without ring enhancing lesion", "ct not supportive of abscess"],
    },
    "ba_ct_na": {
        "present": ["brain ct not done", "ct not done"],
        "absent": ["brain ct completed", "ct completed"],
    },
    "ba_imaging_multifocal": {
        "present": ["multiple lesions on imaging", "multifocal lesions", "surrounding cerebritis", "vasogenic edema around lesion"],
        "absent": ["no multifocal lesions", "no surrounding cerebritis"],
    },
    "nsti_host_diabetes": {
        "present": ["diabetes", "diabetes mellitus", "diabetic"],
        "absent": ["no diabetes", "not diabetic"],
    },
    "nsti_host_immunocompromised": {
        "present": ["immunocompromised", "immunosuppressed", "on chemotherapy", "high dose steroids"],
        "absent": ["not immunocompromised", "no immunosuppression"],
    },
    "nsti_host_ivdu": {
        "present": ["injection drug use", "ivdu", "injects drugs"],
        "absent": ["no injection drug use", "no ivdu"],
    },
    "nsti_host_recent_surgery_or_trauma": {
        "present": ["recent surgery", "recent trauma", "recent wound", "recent injection at the site", "postoperative wound"],
        "absent": ["no recent surgery or trauma", "no wound"],
    },
    "nsti_host_perineal_or_chronic_wound_source": {
        "present": ["perineal source", "fournier", "pressure ulcer", "chronic wound", "skin ulcer"],
        "absent": ["no perineal source", "no chronic wound"],
    },
    "nsti_sym_pain_out_of_proportion": {
        "present": ["pain out of proportion", "severe pain out of proportion", "exquisite pain out of proportion"],
        "absent": ["pain not out of proportion"],
    },
    "nsti_sym_rapid_progression": {
        "present": ["rapid progression", "worsening over hours", "rapidly progressive soft tissue infection"],
        "absent": ["not rapidly progressive"],
    },
    "nsti_vital_fever": {
        "present": ["fever", "febrile"],
        "absent": ["afebrile", "no fever"],
    },
    "nsti_vital_hypotension": {
        "present": ["hypotension", "shock", "septic shock"],
        "absent": ["normotensive", "no hypotension", "no shock"],
    },
    "nsti_exam_bullae_or_necrosis": {
        "present": ["hemorrhagic bullae", "skin necrosis", "ecchymosis", "necrotic skin"],
        "absent": ["no bullae", "no skin necrosis"],
    },
    "nsti_exam_crepitus": {
        "present": ["crepitus", "subcutaneous gas on exam"],
        "absent": ["no crepitus"],
    },
    "nsti_exam_cutaneous_anesthesia": {
        "present": ["cutaneous anesthesia", "skin numbness over lesion", "sensory loss over lesion"],
        "absent": ["no sensory loss over lesion"],
    },
    "nsti_crp_high": {
        "present": ["crp markedly elevated", "crp elevated", "high crp"],
        "absent": ["crp not elevated", "normal crp"],
    },
    "nsti_wbc_high": {
        "present": ["wbc elevated", "white count elevated", "leukocytosis"],
        "absent": ["wbc not elevated", "normal white count", "no leukocytosis"],
    },
    "nsti_sodium_low": {
        "present": ["hyponatremia", "sodium low", "low sodium"],
        "absent": ["no hyponatremia", "sodium normal"],
    },
    "nsti_lactate_high": {
        "present": ["lactate elevated", "high lactate"],
        "absent": ["lactate not elevated", "normal lactate"],
    },
    "nsti_blood_culture_positive": {
        "present": ["blood culture positive", "blood cultures positive", "positive blood cultures with matching organism"],
        "absent": ["blood culture negative", "blood cultures negative"],
    },
    "nsti_blood_culture_na": {
        "present": ["blood cultures not done", "blood culture not done"],
        "absent": ["blood cultures completed", "blood culture completed"],
    },
    "nsti_operative_findings": {
        "present": ["dishwater fluid", "necrotic fascia", "easy fascial dissection", "operative findings classic for necrotizing fasciitis"],
        "absent": ["operative findings not classic for necrotizing fasciitis"],
    },
    "nsti_operative_na": {
        "present": ["operative exploration not done", "surgery not done"],
        "absent": ["operative exploration completed", "surgery completed"],
    },
    "nsti_ct_positive": {
        "present": ["ct compatible with necrotizing fasciitis", "ct with fascial gas", "ct with deep fascial fluid", "ct showing infection crossing fascial planes"],
        "absent": ["ct not compatible with necrotizing fasciitis", "ct without deep fascial gas or fluid"],
    },
    "nsti_ct_na": {
        "present": ["ct not done"],
        "absent": ["ct completed"],
    },
    "nsti_mri_positive": {
        "present": ["mri compatible with necrotizing fasciitis", "deep fascial t2 hyperintensity on mri", "mri showing deep fascial enhancement"],
        "absent": ["mri not compatible with necrotizing fasciitis"],
    },
    "nsti_mri_na": {
        "present": ["mri not done"],
        "absent": ["mri completed"],
    },
    "dfi_host_longstanding_or_recurrent_ulcer": {
        "present": ["chronic ulcer", "nonhealing ulcer", "recurrent ulcer", "longstanding ulcer"],
        "absent": ["acute new ulcer", "not a chronic ulcer"],
    },
    "dfi_host_pad_or_ischemia": {
        "present": ["pad", "peripheral arterial disease", "ischemia", "ischemic foot", "poor perfusion"],
        "absent": ["no pad", "no peripheral arterial disease", "no ischemia"],
    },
    "dfi_host_prior_dfi_or_osteomyelitis": {
        "present": ["prior diabetic foot infection", "prior osteomyelitis", "history of osteomyelitis", "prior same foot amputation"],
        "absent": ["no prior diabetic foot infection", "no prior osteomyelitis"],
    },
    "dfi_local_inflammation_2plus": {
        "present": [
            "clinically infected ulcer",
            "local signs of infection",
            "2 local signs of infection",
            "cellulitis around the ulcer",
            "foot ulcer with erythema warmth tenderness",
            "warmth and erythema",
        ],
        "absent": ["clinically uninfected ulcer", "no local signs of infection", "no erythema warmth tenderness or swelling"],
    },
    "dfi_purulence": {
        "present": ["purulence", "purulent drainage", "pus from ulcer", "pus"],
        "absent": ["no purulence", "no purulent drainage", "dry ulcer"],
    },
    "dfi_erythema_ge2cm_or_deep": {
        "present": ["erythema more than 2 cm", "erythema greater than 2 cm", "infection extends deeper than skin", "deep tissue infection"],
        "absent": ["erythema less than 2 cm", "superficial only"],
    },
    "dfi_systemic_toxicity": {
        "present": ["hemodynamic instability", "shock", "septic shock", "systemic toxicity", "sirs", "unstable from infection"],
        "absent": ["hemodynamically stable", "no systemic toxicity", "stable vital signs"],
    },
    "dfi_deep_abscess_or_gangrene": {
        "present": ["deep abscess", "gangrene", "necrosis", "necrotic tissue", "limb threatening infection", "deep space infection"],
        "absent": ["no gangrene", "no deep abscess", "no necrosis"],
    },
    "dfi_probe_to_bone_positive": {
        "present": ["probe to bone positive", "ptb positive", "probe-to-bone positive"],
        "absent": ["probe to bone negative", "ptb negative", "probe-to-bone negative"],
    },
    "dfi_probe_to_bone_na": {
        "present": ["probe to bone not done", "ptb not done"],
        "absent": ["probe to bone completed", "ptb completed"],
    },
    "dfi_exposed_bone": {
        "present": ["exposed bone", "visible bone", "bone exposed", "bone visible in ulcer"],
        "absent": ["no exposed bone", "no visible bone"],
    },
    "dfi_forefoot_only": {
        "present": ["forefoot only", "forefoot osteomyelitis", "toe osteomyelitis", "metatarsal head osteomyelitis"],
        "absent": ["midfoot involvement", "hindfoot involvement"],
    },
    "dfi_esr_high": {
        "present": ["esr elevated", "esr high", "esr 70", "esr above 70", "esr markedly elevated"],
        "absent": ["esr normal", "esr not elevated"],
    },
    "dfi_crp_high": {
        "present": ["crp elevated", "crp high", "crp markedly elevated"],
        "absent": ["crp normal", "crp not elevated"],
    },
    "dfi_wbc_high": {
        "present": ["leukocytosis", "white count elevated", "wbc elevated", "wbc high"],
        "absent": ["no leukocytosis", "wbc normal"],
    },
    "dfi_xray_osteomyelitis": {
        "present": [
            "xray suggests osteomyelitis",
            "xray concerning for osteomyelitis",
            "plain film osteomyelitis",
            "xray with cortical destruction",
            "xray with periosteal reaction",
        ],
        "absent": ["xray not suggestive of osteomyelitis", "plain films negative for osteomyelitis"],
    },
    "dfi_xray_na": {
        "present": ["plain radiographs not done", "xray not done"],
        "absent": ["plain radiographs completed", "xray completed"],
    },
    "dfi_mri_osteomyelitis_or_abscess": {
        "present": ["mri osteomyelitis", "mri compatible with osteomyelitis", "mri with deep abscess", "mri with osteomyelitis or abscess"],
        "absent": ["mri not compatible with osteomyelitis", "mri negative for osteomyelitis"],
    },
    "dfi_mri_na": {
        "present": ["mri not done"],
        "absent": ["mri completed"],
    },
    "dfi_deep_tissue_culture_pos": {
        "present": ["deep tissue culture positive", "operative tissue culture positive", "deep operative culture positive"],
        "absent": ["deep tissue culture negative", "operative tissue culture negative"],
    },
    "dfi_deep_tissue_culture_na": {
        "present": ["deep tissue culture not done", "operative tissue culture not done"],
        "absent": ["deep tissue culture completed", "operative tissue culture completed"],
    },
    "dfi_bone_biopsy_culture_pos": {
        "present": ["bone biopsy culture positive", "bone culture positive", "bone specimen culture positive"],
        "absent": ["bone biopsy culture negative", "bone culture negative"],
    },
    "dfi_bone_histology_pos": {
        "present": ["bone histology positive", "bone pathology positive", "histology consistent with osteomyelitis"],
        "absent": ["bone histology negative", "bone pathology negative"],
    },
    "dfi_bone_biopsy_na": {
        "present": ["bone biopsy not done", "bone histology not done"],
        "absent": ["bone biopsy completed", "bone histology completed"],
    },
    "dfi_surgery_debridement_done": {
        "present": ["debridement performed", "surgical debridement performed", "source control done", "surgery performed"],
        "absent": ["no debridement", "no surgery", "managed nonsurgically"],
    },
    "dfi_minor_amputation_done": {
        "present": ["minor amputation performed", "toe amputation", "ray amputation", "bone resection performed"],
        "absent": ["no amputation", "no bone resection"],
    },
    "dfi_positive_bone_margin": {
        "present": ["positive bone margin", "positive residual bone margin", "residual osteomyelitis at margin"],
        "absent": ["negative bone margin", "clean bone margin"],
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
    if module.id == "tb_uveitis":
        phenotype_terms = (
            "anterior uveitis",
            "intermediate uveitis",
            "panuveitis",
            "retinal vasculitis",
            "serpiginoid choroiditis",
            "serpiginous like choroiditis",
            "multifocal choroiditis",
            "choroidal tuberculoma",
            "choroidal nodule",
        )
        tb_terms = (
            "igra",
            "quantiferon",
            "quanti feron",
            "t spot",
            "tuberculin skin test",
            "mantoux",
            "ppd",
            "tb endemic",
            "high burden tb",
            "chest radiograph",
            "chest xray",
            "chest x ray",
            "ocular tb",
            "tuberculous",
            "tubercular",
        )
        has_phenotype = any(_contains_phrase(text_norm, term) for term in phenotype_terms)
        has_tb_context = any(_contains_phrase(text_norm, term) for term in tb_terms) or any(
            _contains_phrase(text_norm, country) for country in TB_UVEITIS_ENDEMIC_COUNTRY_ALIASES
        )
        if has_phenotype and has_tb_context:
            score += 6
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
    module_tokens = set(normalize(module.name).split())
    if label_norm and _contains_phrase(text_norm, label_norm):
        score += 4
    for token in label_norm.split():
        if (
            len(token) >= 3
            and token not in module_tokens
            and token not in PRESET_GENERIC_LABEL_TOKENS
            and _contains_phrase(text_norm, token)
        ):
            score += 1
    for group, aliases in PRESET_HINT_ALIASES.items():
        if any(_contains_phrase(text_norm, a) for a in aliases):
            if group == "ed" and (
                "ed" in preset.id
                or "emergency" in label_norm
                or _contains_phrase(label_norm, "ed")
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
    if len(presets) == 1:
        return presets[0].id, warnings

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
        warnings.append("Could not infer setting/pretest context from text.")
        return None, warnings
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


def _augment_tb_uveitis_findings(
    text_norm: str,
    findings: Dict[str, str],
    match_alias: Dict[str, str],
) -> None:
    if "tbu_endemicity_endemic" not in findings and "tbu_endemicity_non_endemic" not in findings:
        for country in TB_UVEITIS_ENDEMIC_COUNTRY_ALIASES:
            if _contains_phrase(text_norm, country):
                findings["tbu_endemicity_endemic"] = "present"
                match_alias["tbu_endemicity_endemic"] = country
                break

    if "tbu_chest_imaging_negative" not in findings and "tbu_chest_imaging_positive" not in findings:
        negative_imaging_phrases = (
            "normal chest radiograph",
            "chest radiograph is normal",
            "chest radiograph normal",
            "normal chest x ray",
            "chest x ray is normal",
            "normal chest xray",
            "chest xray is normal",
        )
        for phrase in negative_imaging_phrases:
            if _contains_phrase(text_norm, phrase):
                findings["tbu_chest_imaging_negative"] = "present"
                match_alias["tbu_chest_imaging_negative"] = phrase
                break

    cld_ids = (
        "tbu_harm_cld_mild",
        "tbu_harm_cld_moderate",
        "tbu_harm_cld_severe",
    )
    if not any(item_id in findings for item_id in cld_ids):
        child_pugh_patterns = (
            (r"\bchild(?:\s*-\s*|\s+)pugh\s*a\b|\bchild class a\b", "tbu_harm_cld_mild", "child-pugh a"),
            (r"\bchild(?:\s*-\s*|\s+)pugh\s*b\b|\bchild class b\b", "tbu_harm_cld_moderate", "child-pugh b"),
            (r"\bchild(?:\s*-\s*|\s+)pugh\s*c\b|\bchild class c\b", "tbu_harm_cld_severe", "child-pugh c"),
        )
        for pattern, item_id, alias in child_pugh_patterns:
            if re.search(pattern, text_norm):
                findings[item_id] = "present"
                match_alias[item_id] = alias
                break

    if not any(item_id in findings for item_id in cld_ids):
        meld_match = re.search(r"\bmeld(?:\s*-\s*na|\s+na)?\s*(?:score\s*)?[:=]?\s*(\d{1,2})\b", text_norm)
        if meld_match:
            meld_value = int(meld_match.group(1))
            if meld_value >= 20:
                findings["tbu_harm_cld_severe"] = "present"
                match_alias["tbu_harm_cld_severe"] = f"meld-na {meld_value}"
            elif meld_value >= 10:
                findings["tbu_harm_cld_moderate"] = "present"
                match_alias["tbu_harm_cld_moderate"] = f"meld-na {meld_value}"
            else:
                findings["tbu_harm_cld_mild"] = "present"
                match_alias["tbu_harm_cld_mild"] = f"meld-na {meld_value}"


def _augment_dfi_findings(
    text_norm: str,
    findings: Dict[str, str],
    match_alias: Dict[str, str],
) -> None:
    if "dfi_esr_high" not in findings:
        match = re.search(r"\besr(?:\s*(?:of|is|=))?\s*(\d{2,3})\b", text_norm)
        if match:
            esr_value = int(match.group(1))
            if esr_value >= 70:
                findings["dfi_esr_high"] = "present"
                match_alias["dfi_esr_high"] = f"esr {esr_value}"
            elif esr_value < 40:
                findings["dfi_esr_high"] = "absent"
                match_alias["dfi_esr_high"] = f"esr {esr_value}"

    if "dfi_local_inflammation_2plus" not in findings and all(token in text_norm for token in ("erythema", "warmth")):
        findings["dfi_local_inflammation_2plus"] = "present"
        match_alias["dfi_local_inflammation_2plus"] = "erythema and warmth"


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

    if module.id == "tb_uveitis":
        _augment_tb_uveitis_findings(text_norm, findings, match_alias)
    if module.id == "diabetic_foot_infection":
        _augment_dfi_findings(text_norm, findings, match_alias)

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
