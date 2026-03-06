from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

from .engine import (
    applied_finding_summaries,
    build_stepwise_path,
    combined_lr,
    confidence_from_thresholds,
    derive_decision_thresholds,
    estimate_harms,
    prepare_probid_inputs,
    recommendation_for_probability,
    resolve_harms,
    resolve_pretest,
    post_test_prob,
)
from .pretest_factors import get_pretest_factor_tuning, resolve_pretest_factor_specs
from .schemas import (
    AssistantOption,
    AssistantState,
    AssistantTurnRequest,
    AssistantTurnResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    DecisionThresholds,
    MechIDAnalyzeRequest,
    MechIDAnalyzeResponse,
    MechIDProvisionalAdvice,
    MechIDTextAnalyzeRequest,
    MechIDTextAnalyzeResponse,
    MechIDTextParsedRequest,
    MechIDTrainerEvalCase,
    MechIDTrainerCaseListResponse,
    MechIDTrainerCaseSummary,
    MechIDTrainerDeleteResponse,
    MechIDTrainerEvaluateRequest,
    MechIDTrainerEvaluateResponse,
    MechIDTrainerEvalPatch,
    MechIDTrainerParsedExpectation,
    MechIDTrainerPreviewRequest,
    MechIDTrainerPreviewResponse,
    MechIDTrainerSaveRequest,
    MechIDTrainerSaveResponse,
    PretestSummary,
    ProbIDControlsInput,
    ReferenceEntry,
    RegisterModulesRequest,
    RegisterModulesResponse,
    SyndromeModule,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)
from .services.module_store import InMemoryModuleStore
from .services.consult_narrator import (
    narrate_mechid_assistant_message,
    narrate_mechid_review_message,
    narrate_probid_assistant_message,
    narrate_probid_review_message,
)
from .services.local_text_parser import LocalParserError, parse_text_with_local_model
from .services.mechid_engine import MechIDEngineError, analyze_mechid, list_mechid_organisms
from .services.mechid_eval import EvalStats, evaluate_mechid_case
from .services.mechid_llm_parser import parse_mechid_text_with_openai
from .services.mechid_text_parser import parse_mechid_text
from .services.mechid_trainer_guidance import MechIDTrainerGuidanceError, generate_mechid_trainer_targets
from .services.mechid_trainer_parser import MechIDTrainerParseError, parse_mechid_trainer_correction
from .services.llm_text_parser import LLMParserError, parse_text_with_openai
from .services.text_parser import COMMON_FINDING_ALIASES, parse_text_to_request


app = FastAPI(
    title="ProbID Decision Assistant API",
    version="0.1.0",
    description="FastAPI scaffold for a ProbID-style likelihood-ratio + decision-threshold assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = InMemoryModuleStore()
APP_DIR = Path(__file__).resolve().parent
MECHID_EVAL_DATASET_PATH = APP_DIR / "data" / "mechid_eval_cases.json"

ASSISTANT_MODULE_LABELS = {
    "cap": "Community-acquired pneumonia (CAP)",
    "vap": "Ventilator-associated pneumonia (VAP)",
    "cdi": "Clostridioides difficile infection (CDI)",
    "uti": "Urinary tract infection (UTI)",
    "endo": "Infective endocarditis",
    "active_tb": "Active tuberculosis (TB)",
    "pjp": "Pneumocystis jirovecii pneumonia (PJP)",
    "inv_candida": "Invasive candidiasis",
    "inv_mold": "Invasive mold infection",
    "septic_arthritis": "Septic arthritis",
    "bacterial_meningitis": "Bacterial meningitis",
    "encephalitis": "Encephalitis",
    "spinal_epidural_abscess": "Spinal epidural abscess",
    "brain_abscess": "Brain abscess",
    "necrotizing_soft_tissue_infection": "Necrotizing soft tissue infection",
    "pji": "Prosthetic joint infection (PJI)",
}

MECHID_ASSISTANT_ID = "mechid"
MECHID_ASSISTANT_LABEL = "Resistance mechanism + therapy"
MECHID_ASSISTANT_DESCRIPTION = (
    "Interpret an organism plus susceptibility pattern to estimate likely resistance mechanisms and therapy options."
)
MECHID_INTENT_TOKENS = (
    "mechanism",
    "resistance",
    "resistant",
    "susceptible",
    "sensitive",
    "intermediate",
    "antibiogram",
    "susceptibility",
    "ast",
    "best antibiotic",
    "best therapy",
    "what should i treat with",
)
MECHID_THERAPY_INTENT_TOKENS = (
    "which antibiotics",
    "what antibiotics",
    "what would you treat with",
    "how would you treat",
    "what would you use",
    "would you recommend",
    "recommend antibiotics",
    "recommend therapy",
    "cultures positive",
    "culture positive",
)

MODULE_EVIDENCE_REFERENCES: Dict[str, List[Dict[str, str]]] = {
    "cap": [
        {
            "context": "Evidence base: Community-acquired pneumonia",
            "citation": "Metlay et al. ATS/IDSA CAP guideline (2019)",
            "url": "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/",
        },
        {
            "context": "Evidence base: Community-acquired pneumonia",
            "citation": "Marchello et al. Accuracy of signs and symptoms for CAP: meta-analysis (2020)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/32329557/",
        },
    ],
    "vap": [
        {
            "context": "Evidence base: Ventilator-associated pneumonia",
            "citation": "Kalil et al. ATS/IDSA HAP/VAP guideline (2016)",
            "url": "https://www.idsociety.org/practice-guideline/hap_vap/",
        },
    ],
    "cdi": [
        {
            "context": "Evidence base: Clostridioides difficile infection",
            "citation": "Johnson et al. SHEA/IDSA CDI focused update (2021)",
            "url": "https://www.idsociety.org/practice-guideline/clostridioides-difficile-2021-focused-update/",
        },
        {
            "context": "Evidence base: Clostridioides difficile infection",
            "citation": "McDonald et al. IDSA/SHEA CDI guideline update (2018)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/29562266/",
        },
        {
            "context": "Evidence base: Clostridioides difficile infection",
            "citation": "Binnicker et al. NAAT diagnostic meta-analysis for C. difficile (2019)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/31142497/",
        },
    ],
    "uti": [
        {
            "context": "Evidence base: Urinary tract infection",
            "citation": "Bent et al. Acute uncomplicated UTI diagnostic review (2002)",
            "url": "https://doi.org/10.1001/jama.287.20.2701",
        },
        {
            "context": "Evidence base: Urinary tract infection",
            "citation": "Giesen et al. UTI symptom/sign diagnostic accuracy systematic review (2010)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/20969801/",
        },
        {
            "context": "Evidence base: Urinary tract infection",
            "citation": "Nicolle et al. IDSA asymptomatic bacteriuria guideline update (2019)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/30895288/",
        },
    ],
    "endo": [
        {
            "context": "Evidence base: Infective endocarditis",
            "citation": "ESC endocarditis guideline (2023)",
            "url": "https://academic.oup.com/eurheartj/article/44/39/3948/7243107",
        },
        {
            "context": "Evidence base: Infective endocarditis",
            "citation": "Baddour et al. AHA infective endocarditis scientific statement (2015)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/26373316/",
        },
        {
            "context": "Evidence base: Infective endocarditis",
            "citation": "Duke-ISCVID criteria external validation (2024)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38330166/",
        },
    ],
    "septic_arthritis": [
        {
            "context": "Evidence base: Septic arthritis",
            "citation": "Margaretten et al. Does this adult patient have septic arthritis? (2007)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/17405973/",
        },
        {
            "context": "Evidence base: Septic arthritis",
            "citation": "Carpenter et al. Evidence-based diagnostics: adult septic arthritis (2011)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/21843213/",
        },
        {
            "context": "Evidence base: Septic arthritis",
            "citation": "Mathews et al. Bacterial septic arthritis in adults (2010)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/20206778/",
        },
    ],
    "bacterial_meningitis": [
        {
            "context": "Evidence base: Bacterial meningitis",
            "citation": "Attia et al. Does this adult patient have acute meningitis? (1999)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/10411200/",
        },
        {
            "context": "Evidence base: Bacterial meningitis",
            "citation": "van de Beek et al. Clinical features and prognostic factors in adults with bacterial meningitis (2004)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/15509818/",
        },
        {
            "context": "Evidence base: Bacterial meningitis",
            "citation": "van de Beek et al. Community-acquired bacterial meningitis review (2021)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/34303412/",
        },
        {
            "context": "Evidence base: Bacterial meningitis",
            "citation": "Sakushima et al. CSF lactate meta-analysis (2011)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/21194480/",
        },
    ],
    "encephalitis": [
        {
            "context": "Evidence base: Encephalitis",
            "citation": "Tunkel et al. IDSA encephalitis guideline (2008)",
            "url": "https://www.idsociety.org/practice-guideline/encephalitis/",
        },
        {
            "context": "Evidence base: Encephalitis",
            "citation": "Gaensbauer et al. HSV PCR assay comparison in CNS infection (2023)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/37390620/",
        },
    ],
    "spinal_epidural_abscess": [
        {
            "context": "Evidence base: Spinal epidural abscess",
            "citation": "Arko et al. Medical and surgical management of spinal epidural abscess: systematic review (2014)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/25081964/",
        },
        {
            "context": "Evidence base: Spinal epidural abscess",
            "citation": "Davis et al. Prospective SEA decision guideline evaluation (2011)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/21417700/",
        },
        {
            "context": "Evidence base: Spinal epidural abscess",
            "citation": "Tetsuka et al. Spinal epidural abscess review (2020)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/33324773/",
        },
    ],
    "brain_abscess": [
        {
            "context": "Evidence base: Brain abscess",
            "citation": "Bodilsen et al. Anti-infective treatment of brain abscess (2018)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/29909695/",
        },
        {
            "context": "Evidence base: Brain abscess",
            "citation": "Chow. Brain and spinal epidural abscess review (2018)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/30273242/",
        },
        {
            "context": "Evidence base: Brain abscess",
            "citation": "Leuthardt et al. Diffusion-weighted MRI in the preoperative assessment of brain abscesses (2002)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12517619/",
        },
    ],
    "necrotizing_soft_tissue_infection": [
        {
            "context": "Evidence base: Necrotizing soft tissue infection",
            "citation": "Stevens et al. IDSA skin and soft tissue infection guideline (2014)",
            "url": "https://www.idsociety.org/practice-guideline/skin-and-soft-tissue-infections/",
        },
        {
            "context": "Evidence base: Necrotizing soft tissue infection",
            "citation": "Fernando et al. NSTI diagnostic accuracy meta-analysis (2019)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/29672405/",
        },
    ],
}

ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES = {
    "staph": {
        "label": "Staphylococcus aureus",
        "description": "Use the S. aureus pathway and show VIRSTA-overlap baseline modifiers.",
        "score_id": "virsta",
    },
    "strep": {
        "label": "Viridans group streptococci",
        "description": "Use the viridans/NBHS pathway and show HANDOC-overlap baseline modifiers.",
        "score_id": "handoc",
    },
    "enterococcus": {
        "label": "Enterococcus",
        "description": "Use the enterococcal pathway and show DENOVA-overlap baseline modifiers.",
        "score_id": "denova",
    },
    "other_unknown_pending": {
        "label": "Other / unknown / pending",
        "description": "Skip organism-specific score-overlap factors and use general endocarditis context only.",
        "score_id": None,
    },
}

ENDO_ASSISTANT_SCORE_COMPONENTS = {
    "virsta": (
        ("virsta_emboli", "Cerebral or peripheral emboli"),
        ("virsta_meningitis", "Meningitis"),
        ("virsta_intracardiac_device", "Permanent intracardiac device"),
        ("virsta_prior_endocarditis", "Prior endocarditis"),
        ("virsta_native_valve_disease", "Native valve disease"),
        ("virsta_ivdu", "Injection drug use"),
        ("virsta_persistent_bacteremia_48h", "Persistent bacteremia >48 hours"),
        ("virsta_vertebral_osteomyelitis", "Vertebral osteomyelitis"),
        ("virsta_community_or_nhca", "Community or non-nosocomial healthcare-associated acquisition"),
        ("virsta_severe_sepsis_shock", "Severe sepsis or septic shock"),
        ("virsta_crp_gt_190", "CRP >190 mg/L"),
    ),
    "denova": (
        ("denova_duration_7d", "Symptoms >=7 days"),
        ("denova_embolization", "Embolization"),
        ("denova_num_positive_2", ">=2 positive blood culture sets"),
        ("denova_origin_unknown", "Unknown source"),
        ("denova_valve_disease", "Known valve disease"),
        ("denova_auscultation_murmur", "Auscultation murmur"),
    ),
    "handoc": (
        ("handoc_heart_murmur_valve", "Heart murmur or valve disease"),
        ("handoc_species_high_risk", "Higher-risk species: S. gallolyticus / mutans / sanguinis"),
        ("handoc_species_anginosus", "S. anginosus group"),
        ("handoc_num_positive_2", ">=2 positive blood culture sets"),
        ("handoc_duration_7d", "Symptoms >=7 days"),
        ("handoc_only_one_species", "Only one species in blood cultures"),
        ("handoc_community_acquired", "Community-acquired bacteremia"),
    ),
}

ENDO_ASSISTANT_EXCLUSIVE_SCORE_GROUPS = (
    {"handoc_species_high_risk", "handoc_species_anginosus"},
)

ENDO_ASSISTANT_SCORE_ITEM_IDS = {
    "endo_virsta_high",
    "endo_virsta_na",
    "endo_denova_high",
    "endo_denova_na",
    "endo_handoc_high",
    "endo_handoc_na",
}

ENDO_ASSISTANT_SCORE_TEXT_ALIASES: Dict[str, Dict[str, tuple[str, ...]]] = {
    "virsta": {
        "virsta_emboli": ("emboli", "embolus", "embolic", "stroke", "janeway", "splinter hemorrhage"),
        "virsta_meningitis": ("meningitis",),
        "virsta_intracardiac_device": ("pacemaker", "icd", "cied", "intracardiac device"),
        "virsta_prior_endocarditis": ("prior endocarditis", "previous endocarditis", "history of endocarditis"),
        "virsta_native_valve_disease": ("native valve disease", "known valve disease", "valvular disease"),
        "virsta_ivdu": ("ivdu", "injection drug use", "injects drugs", "intravenous drug use"),
        "virsta_persistent_bacteremia_48h": ("persistent bacteremia", "persistent positive blood cultures", "positive blood cultures for more than 48 hours"),
        "virsta_vertebral_osteomyelitis": ("vertebral osteomyelitis",),
        "virsta_community_or_nhca": ("community acquired", "community-acquired", "healthcare associated", "non nosocomial"),
        "virsta_severe_sepsis_shock": ("severe sepsis", "septic shock"),
        "virsta_crp_gt_190": ("crp >190", "crp greater than 190", "crp 200", "crp 190"),
    },
    "denova": {
        "denova_duration_7d": ("7 days", "seven days", "one week", "1 week"),
        "denova_embolization": ("embolization", "emboli", "embolic"),
        "denova_num_positive_2": ("2 positive blood culture sets", "two positive blood culture sets", ">=2 positive blood culture sets"),
        "denova_origin_unknown": ("unknown source", "unclear source", "source unknown"),
        "denova_valve_disease": ("valve disease", "known valve disease", "valvular disease"),
        "denova_auscultation_murmur": ("murmur", "heart murmur", "new heart murmur"),
    },
    "handoc": {
        "handoc_heart_murmur_valve": ("murmur", "heart murmur", "valve disease", "valvular disease"),
        "handoc_species_high_risk": ("gallolyticus", "bovis", "mutans", "sanguinis"),
        "handoc_species_anginosus": ("anginosus",),
        "handoc_num_positive_2": ("2 positive blood culture sets", "two positive blood culture sets", ">=2 positive blood culture sets"),
        "handoc_duration_7d": ("7 days", "seven days", "one week", "1 week"),
        "handoc_only_one_species": ("one species", "single species", "only one species", "monomicrobial"),
        "handoc_community_acquired": ("community acquired", "community-acquired"),
    },
}

ENDO_CASE_SECTION_ORDER = ("exam_vitals", "lab", "micro", "imaging")
ENDO_CASE_SECTION_LABELS = {
    "exam_vitals": "vital signs and physical exam",
    "lab": "laboratory findings",
    "micro": "microbiology",
    "imaging": "radiology and advanced imaging",
}

ACTIVE_TB_WHO_SYMPTOM_HELPERS = (
    ("tb_sym_any_cough", "Cough (WHO symptom)", "WHO TB symptom: cough", "No WHO TB symptoms"),
    ("tb_sym_any_fever", "Fever (WHO symptom)", "WHO TB symptom: fever", "No WHO TB symptoms"),
    ("tb_sym_any_night_sweats", "Night sweats (WHO symptom)", "WHO TB symptom: night sweats", "No WHO TB symptoms"),
    ("tb_sym_any_weight_loss", "Weight loss (WHO symptom)", "WHO TB symptom: weight loss", "No WHO TB symptoms"),
)

ASSISTANT_CASE_TEXT_OVERRIDES: Dict[str, Dict[str, tuple[str, str]]] = {
    "cap": {
        "cap_cxr_consolidation": (
            "CXR with lobar or multilobar consolidation",
            "CXR without consolidation",
        ),
        "cap_cxr_not_done": (
            "CXR not done",
            "CXR completed",
        ),
        "cap_rvp_pos": (
            "Respiratory viral panel positive",
            "Respiratory viral panel negative",
        ),
        "cap_rvp_na": (
            "Respiratory viral panel not done",
            "Respiratory viral panel completed",
        ),
        "cap_active_malignancy": (
            "Active malignancy present",
            "No active malignancy",
        ),
    },
    "vap": {
        "vap_cxr_infiltrate": (
            "Chest radiograph with new or progressive infiltrate",
            "Chest radiograph without new or progressive infiltrate",
        ),
        "vap_cxr_na": (
            "VAP CXR not done",
            "VAP CXR completed",
        ),
        "vap_leukocytosis": (
            "WBC at least 12 for VAP",
            "WBC below 12 for VAP",
        ),
        "vap_hypoxemia_pf240": (
            "PaO2/FiO2 at or below 240",
            "PaO2/FiO2 above 240",
        ),
        "vap_cpis_gt6": (
            "CPIS greater than 6",
            "CPIS 6 or lower",
        ),
        "vap_cpis_na": (
            "CPIS not used",
            "CPIS completed",
        ),
        "vap_resp_micro_na": (
            "Respiratory sampling not done",
            "Respiratory sampling completed",
        ),
        "vap_pct_elevated": (
            "Procalcitonin elevated for VAP",
            "Procalcitonin not elevated for VAP",
        ),
        "vap_pct_na": (
            "Procalcitonin not done for VAP",
            "Procalcitonin completed for VAP",
        ),
    },
    "cdi": {
        "cdi_freq": (
            "At least 3 unformed stools in 24 hours",
            "Fewer than 3 unformed stools in 24 hours",
        ),
        "cdi_watery": (
            "Watery diarrhea present",
            "No watery diarrhea",
        ),
        "cdi_test_na": (
            "Stool testing not done",
            "Stool testing completed",
        ),
        "cdi_naat_neg": (
            "C diff NAAT negative",
            "C diff NAAT not negative",
        ),
        "cdi_naat_pos_tox_pos": (
            "C diff NAAT positive and toxin positive",
            "C diff NAAT/toxin not both positive",
        ),
        "cdi_naat_pos_tox_neg": (
            "C diff NAAT positive and toxin negative",
            "C diff NAAT/toxin pattern not NAAT positive toxin negative",
        ),
        "cdi_naat_pos_tox_na": (
            "C diff NAAT positive and toxin not sent",
            "C diff NAAT pattern not NAAT positive with toxin not sent",
        ),
    },
    "uti": {
        "uti_vaginitis": (
            "Vaginal discharge or irritation present",
            "No vaginal discharge or irritation",
        ),
        "uti_obstruction": (
            "Urinary obstruction or anatomic abnormality present",
            "No urinary obstruction or anatomic abnormality",
        ),
        "ua_pyuria_pos": (
            "Pyuria present on microscopy",
            "No pyuria on microscopy",
        ),
        "ua_le_pos": (
            "Urine leukocyte esterase positive",
            "Urine leukocyte esterase negative",
        ),
        "ua_nit_pos": (
            "Urine nitrite positive",
            "Urine nitrite negative",
        ),
        "ua_bact_pos": (
            "Bacteriuria present on microscopy",
            "No bacteriuria on microscopy",
        ),
        "uti_cx_pos": (
            "Urine culture above 100000 CFU",
            "Urine culture below 100000 CFU",
        ),
    },
    "active_tb": {
        "tb_contact": (
            "Close/household TB exposure",
            "No close/household TB exposure",
        ),
        "tb_sym_any": (
            "WHO TB symptom screen positive (cough, fever, night sweats, or weight loss)",
            "No WHO TB symptoms",
        ),
        "tb_sym_cough_2w": (
            "Cough for more than 2 weeks",
            "Cough for less than 2 weeks",
        ),
        "tb_sym_na": (
            "TB symptom screen not done",
            "TB symptom screen completed",
        ),
        "tb_qft": (
            "QuantiFERON positive",
            "QuantiFERON negative",
        ),
        "tb_tst": (
            "Tuberculin skin test positive",
            "Tuberculin skin test negative",
        ),
        "tb_immune_na": (
            "QFT/TST not done",
            "QFT/TST completed",
        ),
        "tb_mtbpcr_sputum": (
            "MTB PCR positive on sputum",
            "MTB PCR negative on sputum",
        ),
        "tb_mtbpcr_bal": (
            "MTB PCR positive on BAL",
            "MTB PCR negative on BAL",
        ),
        "tb_afb_smear_sputum": (
            "AFB smear positive on sputum",
            "AFB smear negative on sputum",
        ),
        "tb_culture_sputum": (
            "Mycobacterial culture positive from sputum",
            "Mycobacterial culture negative from sputum",
        ),
        "tb_culture_bal": (
            "Mycobacterial culture positive from BAL",
            "Mycobacterial culture negative from BAL",
        ),
        "tb_cxr_suggestive": (
            "CXR suggestive of active pulmonary TB",
            "CXR not suggestive of active pulmonary TB",
        ),
        "tb_cxr_na": (
            "CXR not done",
            "CXR completed",
        ),
        "tb_ct_suggestive": (
            "Chest CT suggestive of active pulmonary TB",
            "Chest CT not suggestive of active pulmonary TB",
        ),
        "tb_ct_na": (
            "Chest CT not done",
            "Chest CT completed",
        ),
    },
    "pjp": {
        "pjp_host_no_ppx": (
            "Lack of TMP-SMX prophylaxis despite indication",
            "Receiving TMP-SMX prophylaxis when indicated",
        ),
        "pjp_host_heme_hsct": (
            "Hematologic malignancy or stem-cell transplant",
            "No hematologic malignancy or stem-cell transplant",
        ),
        "pjp_bdg_serum": (
            "Serum beta-D-glucan positive",
            "Serum beta-D-glucan negative",
        ),
        "pjp_bdg_na": (
            "Serum BDG not done",
            "Serum BDG completed",
        ),
        "pjp_ldh_high": (
            "Serum LDH elevated",
            "Serum LDH not elevated",
        ),
        "pjp_pcr_bal": (
            "PJP PCR positive on BAL",
            "PJP PCR negative on BAL",
        ),
        "pjp_pcr_induced_sputum": (
            "PJP PCR positive on induced sputum",
            "PJP PCR negative on induced sputum",
        ),
        "pjp_pcr_upper_airway": (
            "PJP PCR positive on upper airway sample",
            "PJP PCR negative on upper airway sample",
        ),
        "pjp_pcr_na": (
            "Respiratory PJP PCR not done",
            "Respiratory PJP PCR completed",
        ),
        "pjp_dfa": (
            "PJP DFA/IFA positive",
            "PJP DFA/IFA negative",
        ),
        "pjp_dfa_na": (
            "PJP DFA/IFA not done",
            "PJP DFA/IFA completed",
        ),
        "pjp_cxr_typical": (
            "CXR typical of PJP",
            "CXR not typical of PJP",
        ),
        "pjp_ct_typical": (
            "CT typical of PJP",
            "CT not typical of PJP",
        ),
        "pjp_imaging_na": (
            "Chest imaging not done",
            "Chest imaging completed",
        ),
    },
    "inv_candida": {
        "icand_host_dialysis": (
            "On dialysis or renal replacement therapy",
            "Not on dialysis or renal replacement therapy",
        ),
        "icand_component_multifocal_colonization": (
            "Multifocal Candida colonization present",
            "No multifocal Candida colonization",
        ),
        "icand_component_severe_sepsis": (
            "Severe sepsis or septic shock",
            "No severe sepsis or septic shock",
        ),
        "icand_bdg_serum": (
            "Serum beta-D-glucan positive",
            "Serum beta-D-glucan negative",
        ),
        "icand_bdg_na": (
            "Serum BDG not done",
            "Serum BDG completed",
        ),
        "icand_mannan_antimannan": (
            "Mannan/anti-mannan assay positive",
            "Mannan/anti-mannan assay negative",
        ),
        "icand_mannan_na": (
            "Mannan/anti-mannan testing not done",
            "Mannan/anti-mannan testing completed",
        ),
        "icand_t2candida": (
            "T2Candida positive",
            "T2Candida negative",
        ),
        "icand_t2_na": (
            "T2Candida not done",
            "T2Candida completed",
        ),
        "icand_pcr_blood": (
            "Candida PCR positive from blood",
            "Candida PCR negative from blood",
        ),
        "icand_pcr_na": (
            "Candida PCR not done",
            "Candida PCR completed",
        ),
        "icand_culture_positive": (
            "Blood or sterile-site culture positive for Candida",
            "Blood or sterile-site culture negative for Candida",
        ),
        "icand_culture_na": (
            "Candida culture strategy not done",
            "Candida culture strategy completed",
        ),
    },
    "inv_mold": {
        "imi_host_neutropenia_hsct": (
            "Profound neutropenia or recent allogeneic HSCT",
            "No profound neutropenia or recent allogeneic HSCT",
        ),
        "imi_host_hematologic_malignancy": (
            "Active hematologic malignancy",
            "No active hematologic malignancy",
        ),
        "imi_fever_refractory": (
            "Refractory fever despite broad-spectrum antibacterials",
            "No refractory fever despite broad-spectrum antibacterials",
        ),
        "imi_ct_halo_sign": (
            "Chest CT halo sign present",
            "No chest CT halo sign",
        ),
        "imi_ct_na": (
            "Chest CT not done",
            "Chest CT completed",
        ),
        "imi_serum_gm_odi10": (
            "Serum galactomannan positive",
            "Serum galactomannan negative",
        ),
        "imi_bal_gm_odi10": (
            "BAL galactomannan positive",
            "BAL galactomannan negative",
        ),
        "imi_gm_na": (
            "Galactomannan testing not done",
            "Galactomannan testing completed",
        ),
        "imi_serum_bdg": (
            "Serum beta-D-glucan positive",
            "Serum beta-D-glucan negative",
        ),
        "imi_bdg_na": (
            "Serum BDG not done",
            "Serum BDG completed",
        ),
        "imi_aspergillus_lfd": (
            "Aspergillus LFD/LFA positive",
            "Aspergillus LFD/LFA negative",
        ),
        "imi_lfd_na": (
            "Aspergillus LFD/LFA not done",
            "Aspergillus LFD/LFA completed",
        ),
        "imi_aspergillus_pcr_bal": (
            "Aspergillus PCR positive from BAL",
            "Aspergillus PCR negative from BAL",
        ),
        "imi_aspergillus_pcr_na": (
            "Aspergillus PCR not done",
            "Aspergillus PCR completed",
        ),
        "imi_mucorales_pcr_bal": (
            "Mucorales PCR positive from BAL",
            "Mucorales PCR negative from BAL",
        ),
        "imi_mucorales_pcr_blood": (
            "Mucorales PCR positive from blood",
            "Mucorales PCR negative from blood",
        ),
        "imi_mucorales_pcr_na": (
            "Mucorales PCR not done",
            "Mucorales PCR completed",
        ),
    },
    "pji": {
        "pji_crp": (
            "CRP elevated",
            "CRP not elevated",
        ),
        "pji_esr": (
            "ESR elevated",
            "ESR not elevated",
        ),
        "pji_alpha_defensin_elisa": (
            "Synovial alpha-defensin ELISA positive",
            "Synovial alpha-defensin ELISA negative",
        ),
        "pji_alpha_defensin_lateral_flow": (
            "Synovial alpha-defensin lateral flow positive",
            "Synovial alpha-defensin lateral flow negative",
        ),
        "pji_leukocyte_esterase": (
            "Synovial leukocyte esterase positive",
            "Synovial leukocyte esterase negative",
        ),
        "pji_synovial_marker_na": (
            "Synovial biomarker testing not done",
            "Synovial biomarker testing completed",
        ),
        "pji_synovial_fluid_culture": (
            "Synovial fluid culture positive",
            "Synovial fluid culture negative",
        ),
        "pji_intraop_tissue_culture": (
            "Intraoperative tissue culture positive",
            "Intraoperative tissue culture negative",
        ),
        "pji_sonication_culture": (
            "Sonication fluid culture positive",
            "Sonication fluid culture negative",
        ),
        "pji_culture_na": (
            "PJI culture strategy not done",
            "PJI culture strategy completed",
        ),
        "pji_synovial_pcr": (
            "Synovial PCR positive",
            "Synovial PCR negative",
        ),
        "pji_pcr_na": (
            "Synovial PCR not done",
            "Synovial PCR completed",
        ),
        "pji_xray_supportive": (
            "Plain radiograph supportive of infection",
            "Plain radiograph not supportive of infection",
        ),
        "pji_imaging_na": (
            "PJI imaging not done",
            "PJI imaging completed",
        ),
    },
    "septic_arthritis": {
        "sa_crp": (
            "CRP elevated",
            "CRP not elevated",
        ),
        "sa_esr": (
            "ESR elevated",
            "ESR not elevated",
        ),
        "sa_synovial_wbc_ge50k": (
            "Synovial WBC at least 50,000",
            "Synovial WBC below 50,000",
        ),
        "sa_synovial_na": (
            "Arthrocentesis cell count not done",
            "Arthrocentesis completed",
        ),
        "sa_synovial_pmn_ge90": (
            "Synovial PMN at least 90%",
            "Synovial PMN below 90%",
        ),
        "sa_gram_stain": (
            "Synovial Gram stain positive",
            "Synovial Gram stain negative",
        ),
        "sa_gram_stain_na": (
            "Synovial Gram stain not done",
            "Synovial Gram stain completed",
        ),
        "sa_synovial_culture": (
            "Synovial fluid culture positive",
            "Synovial fluid culture negative",
        ),
        "sa_synovial_culture_na": (
            "Synovial fluid culture not done",
            "Synovial fluid culture completed",
        ),
        "sa_blood_culture_positive": (
            "Blood culture positive with matching pathogen",
            "Blood culture negative",
        ),
        "sa_blood_culture_na": (
            "Blood cultures not done",
            "Blood cultures completed",
        ),
        "sa_ultrasound_effusion": (
            "Joint ultrasound shows effusion",
            "Joint ultrasound without effusion",
        ),
        "sa_imaging_na": (
            "Joint imaging not done",
            "Joint imaging completed",
        ),
    },
    "bacterial_meningitis": {
        "bm_serum_procalcitonin": (
            "Serum procalcitonin elevated",
            "Serum procalcitonin not elevated",
        ),
        "bm_csf_wbc_ge1000": (
            "CSF WBC at least 1,000",
            "CSF WBC below 1,000",
        ),
        "bm_csf_cell_count_na": (
            "CSF cell count not done",
            "CSF cell count completed",
        ),
        "bm_csf_pmn_ge80": (
            "CSF neutrophils at least 80%",
            "CSF neutrophils below 80%",
        ),
        "bm_csf_glucose_ratio_low": (
            "CSF glucose low or CSF:serum glucose ratio below 0.4",
            "CSF glucose not low",
        ),
        "bm_csf_protein_high": (
            "CSF protein elevated",
            "CSF protein not elevated",
        ),
        "bm_csf_lactate_high": (
            "CSF lactate elevated",
            "CSF lactate not elevated",
        ),
        "bm_csf_gram_stain": (
            "CSF Gram stain positive",
            "CSF Gram stain negative",
        ),
        "bm_csf_gram_na": (
            "CSF Gram stain not done",
            "CSF Gram stain completed",
        ),
        "bm_csf_culture": (
            "CSF culture positive",
            "CSF culture negative",
        ),
        "bm_csf_culture_na": (
            "CSF culture not done",
            "CSF culture completed",
        ),
        "bm_blood_culture_positive": (
            "Blood culture positive with plausible meningitis pathogen",
            "Blood culture negative",
        ),
        "bm_blood_culture_na": (
            "Blood cultures not done",
            "Blood cultures completed",
        ),
        "bm_csf_bacterial_pcr": (
            "CSF bacterial PCR positive",
            "CSF bacterial PCR negative",
        ),
        "bm_csf_pcr_na": (
            "CSF bacterial PCR not done",
            "CSF bacterial PCR completed",
        ),
        "bm_imaging_supportive": (
            "Neuroimaging supportive of meningitis",
            "Neuroimaging not supportive of meningitis",
        ),
        "bm_imaging_na": (
            "Neuroimaging not done",
            "Neuroimaging completed",
        ),
    },
    "encephalitis": {
        "enc_csf_pleocytosis": (
            "CSF pleocytosis present",
            "No CSF pleocytosis",
        ),
        "enc_csf_cell_count_na": (
            "CSF cell count not done",
            "CSF cell count completed",
        ),
        "enc_csf_protein_high": (
            "CSF protein elevated",
            "CSF protein not elevated",
        ),
        "enc_csf_rbc_high": (
            "CSF RBC elevated",
            "CSF RBC not elevated",
        ),
        "enc_hsv_pcr": (
            "CSF HSV PCR positive",
            "CSF HSV PCR negative",
        ),
        "enc_hsv_pcr_na": (
            "CSF HSV PCR not done",
            "CSF HSV PCR completed",
        ),
        "enc_csf_viral_pcr": (
            "Other CSF viral PCR positive",
            "Other CSF viral PCR negative",
        ),
        "enc_csf_viral_pcr_na": (
            "Other CSF viral PCR not done",
            "Other CSF viral PCR completed",
        ),
        "enc_mri_temporal": (
            "MRI with temporal or insular encephalitis pattern",
            "MRI without temporal encephalitis pattern",
        ),
        "enc_mri_na": (
            "Brain MRI not done",
            "Brain MRI completed",
        ),
        "enc_eeg_temporal": (
            "EEG with temporal slowing or periodic discharges",
            "EEG without temporal encephalitis pattern",
        ),
        "enc_eeg_na": (
            "EEG not done",
            "EEG completed",
        ),
    },
    "spinal_epidural_abscess": {
        "sea_esr_high": (
            "ESR markedly elevated",
            "ESR not markedly elevated",
        ),
        "sea_crp_high": (
            "CRP elevated",
            "CRP not elevated",
        ),
        "sea_wbc_high": (
            "Peripheral WBC elevated",
            "Peripheral WBC not elevated",
        ),
        "sea_blood_culture_positive": (
            "Blood culture positive with plausible SEA pathogen",
            "Blood culture negative",
        ),
        "sea_blood_culture_na": (
            "Blood cultures not done",
            "Blood cultures completed",
        ),
        "sea_mri_positive": (
            "MRI spine shows epidural abscess or phlegmon",
            "MRI spine without epidural abscess",
        ),
        "sea_mri_na": (
            "MRI spine not done",
            "MRI spine completed",
        ),
        "sea_discitis_osteo": (
            "Imaging suggests discitis or vertebral osteomyelitis",
            "Imaging does not suggest discitis or vertebral osteomyelitis",
        ),
    },
    "brain_abscess": {
        "ba_crp_high": (
            "CRP elevated",
            "CRP not elevated",
        ),
        "ba_wbc_high": (
            "Peripheral WBC elevated",
            "Peripheral WBC not elevated",
        ),
        "ba_blood_culture_positive": (
            "Blood culture positive with plausible brain abscess pathogen",
            "Blood culture negative",
        ),
        "ba_blood_culture_na": (
            "Blood cultures not done",
            "Blood cultures completed",
        ),
        "ba_aspirate_culture_positive": (
            "Abscess aspirate or operative culture positive",
            "Abscess aspirate or operative culture negative",
        ),
        "ba_aspirate_culture_na": (
            "Abscess aspirate culture not done",
            "Abscess aspirate culture completed",
        ),
        "ba_mri_dwi_positive": (
            "MRI with ring-enhancing lesion and restricted diffusion",
            "MRI without abscess-compatible restricted diffusion",
        ),
        "ba_mri_na": (
            "Brain MRI not done",
            "Brain MRI completed",
        ),
        "ba_ct_ring_enhancing": (
            "Contrast CT with ring-enhancing lesion",
            "Contrast CT without ring-enhancing lesion",
        ),
        "ba_ct_na": (
            "Brain CT not done",
            "Brain CT completed",
        ),
        "ba_imaging_multifocal": (
            "Imaging shows multifocal lesions or surrounding cerebritis",
            "Imaging does not show multifocal lesions or surrounding cerebritis",
        ),
    },
    "necrotizing_soft_tissue_infection": {
        "nsti_crp_high": (
            "CRP markedly elevated",
            "CRP not markedly elevated",
        ),
        "nsti_wbc_high": (
            "Peripheral WBC elevated",
            "Peripheral WBC not elevated",
        ),
        "nsti_sodium_low": (
            "Hyponatremia present",
            "No hyponatremia",
        ),
        "nsti_lactate_high": (
            "Lactate elevated",
            "Lactate not elevated",
        ),
        "nsti_blood_culture_positive": (
            "Blood culture positive with plausible NSTI pathogen",
            "Blood culture negative",
        ),
        "nsti_blood_culture_na": (
            "Blood cultures not done",
            "Blood cultures completed",
        ),
        "nsti_operative_findings": (
            "Operative findings classic for NSTI",
            "Operative findings not classic for NSTI",
        ),
        "nsti_operative_na": (
            "Operative exploration not done",
            "Operative exploration completed",
        ),
        "nsti_ct_positive": (
            "CT compatible with NSTI",
            "CT not compatible with NSTI",
        ),
        "nsti_ct_na": (
            "CT not done",
            "CT completed",
        ),
        "nsti_mri_positive": (
            "MRI compatible with NSTI",
            "MRI not compatible with NSTI",
        ),
        "nsti_mri_na": (
            "MRI not done",
            "MRI completed",
        ),
    },
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/assistant")
def assistant_web() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "assistant.html")


@app.get("/trainer")
def trainer_web() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "trainer.html")


def _slugify_case_id(text: str, fallback: str = "mechid_case") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        return fallback
    parts = [part for part in slug.split("_") if part][:8]
    return "_".join(parts) or fallback


def _base_trainer_case_id(text: str, result: MechIDTextAnalyzeResponse) -> str:
    parsed = result.parsed_request
    if parsed is not None:
        parts: List[str] = []
        if parsed.organism:
            parts.append(parsed.organism)
        elif parsed.mentioned_organisms:
            parts.extend(parsed.mentioned_organisms[:2])
        focus = parsed.tx_context.focus_detail
        if focus and focus != "Not specified":
            parts.append(focus)
        if parts:
            return _slugify_case_id("_".join(parts))
    return _slugify_case_id(text)


def _build_mechid_trainer_base_draft(
    *,
    text: str,
    parser_strategy: str,
    result: MechIDTextAnalyzeResponse,
) -> MechIDTrainerEvalCase:
    expected_parsed = None
    if result.parsed_request is not None:
        parsed = result.parsed_request
        expected_parsed = MechIDTrainerParsedExpectation(
            organism=parsed.organism,
            syndrome=parsed.tx_context.syndrome if parsed.tx_context.syndrome != "Not specified" else None,
            severity=parsed.tx_context.severity if parsed.tx_context.severity != "Not specified" else None,
            focusDetail=parsed.tx_context.focus_detail if parsed.tx_context.focus_detail != "Not specified" else None,
            oralPreference=parsed.tx_context.oral_preference if parsed.tx_context.oral_preference else None,
            mentionedOrganismsContains=list(parsed.mentioned_organisms),
            resistancePhenotypesContains=list(parsed.resistance_phenotypes),
            susceptibilityResultsSubset=dict(parsed.susceptibility_results),
        )

    analysis = result.analysis
    provisional = result.provisional_advice
    return MechIDTrainerEvalCase(
        id=_base_trainer_case_id(text, result),
        text=text,
        parserStrategy=parser_strategy,
        assistantGuidance=None,
        assistantReviewTarget=None,
        assistantFinalTarget=None,
        expectedRequiresConfirmation=result.requires_confirmation,
        expectedParsed=expected_parsed,
        expectedAnalysisPresent=analysis is not None,
        expectedProvisionalPresent=provisional is not None,
        expectedMechanismsContains=list((analysis.mechanisms[:2] if analysis is not None else [])),
        expectedTherapyNotesContains=list((analysis.therapy_notes[:2] if analysis is not None else [])),
        expectedFinalResultsSubset=dict((analysis.final_results if analysis is not None else {})),
        expectedRecommendedOptionsContains=list((provisional.recommended_options if provisional is not None else [])),
        expectedOralOptionsContains=list((provisional.oral_options if provisional is not None else [])),
        expectedMissingSusceptibilitiesContains=list((provisional.missing_susceptibilities if provisional is not None else [])),
        notes=_join_readable(result.warnings) if result.warnings else None,
    )


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_trainer_patch(base: MechIDTrainerEvalCase, patch: MechIDTrainerEvalPatch) -> MechIDTrainerEvalCase:
    merged = _deep_merge_dict(
        base.model_dump(by_alias=True),
        patch.model_dump(by_alias=True, exclude_none=True),
    )
    return MechIDTrainerEvalCase.model_validate(merged)


def _trainer_transient_examples(draft_case: MechIDTrainerEvalCase, *, kind: str) -> List[Dict[str, str]]:
    target = draft_case.assistant_final_target if kind == "final" else draft_case.assistant_review_target
    if not target:
        return []
    return [
        {
            "id": draft_case.id,
            "text": draft_case.text,
            "guidance": draft_case.assistant_guidance or "",
            "target": target,
        }
    ]


def _load_mechid_eval_cases() -> List[Dict[str, Any]]:
    if not MECHID_EVAL_DATASET_PATH.exists():
        return []
    return json.loads(MECHID_EVAL_DATASET_PATH.read_text())


def _write_mechid_eval_cases(cases: List[Dict[str, Any]]) -> None:
    MECHID_EVAL_DATASET_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")


def _mechid_case_summary(payload: Dict[str, Any]) -> MechIDTrainerCaseSummary:
    text = str(payload.get("text") or "")
    preview = text if len(text) <= 110 else text[:107].rstrip() + "..."
    return MechIDTrainerCaseSummary(
        id=str(payload.get("id") or ""),
        textPreview=preview,
        parserStrategy=str(payload.get("parserStrategy") or "rule"),
    )


@app.get("/v1/trainer/mechid/cases", response_model=MechIDTrainerCaseListResponse)
def mechid_trainer_list_cases() -> MechIDTrainerCaseListResponse:
    cases = _load_mechid_eval_cases()
    summaries = [_mechid_case_summary(case) for case in cases if case.get("id")]
    summaries.sort(key=lambda item: item.id)
    return MechIDTrainerCaseListResponse(cases=summaries)


@app.get("/v1/trainer/mechid/cases/{case_id}", response_model=MechIDTrainerEvalCase)
def mechid_trainer_get_case(case_id: str) -> MechIDTrainerEvalCase:
    for payload in _load_mechid_eval_cases():
        if payload.get("id") == case_id:
            return MechIDTrainerEvalCase.model_validate(payload)
    raise HTTPException(status_code=404, detail=f"Trainer case '{case_id}' not found")


@app.post("/v1/trainer/mechid/cases/{case_id}/duplicate", response_model=MechIDTrainerSaveResponse)
def mechid_trainer_duplicate_case(case_id: str) -> MechIDTrainerSaveResponse:
    cases = _load_mechid_eval_cases()
    for payload in cases:
        if payload.get("id") != case_id:
            continue
        draft = MechIDTrainerEvalCase.model_validate(payload)
        base_id = f"{draft.id}_copy"
        candidate_id = base_id
        used_ids = {str(case.get("id") or "") for case in cases}
        suffix = 2
        while candidate_id in used_ids:
            candidate_id = f"{base_id}_{suffix}"
            suffix += 1
        duplicated = draft.model_copy(update={"id": candidate_id})
        cases.append(duplicated.model_dump(by_alias=True, exclude_none=True))
        _write_mechid_eval_cases(cases)
        return MechIDTrainerSaveResponse(
            saved=True,
            path=str(MECHID_EVAL_DATASET_PATH),
            caseId=candidate_id,
            totalCases=len(cases),
        )
    raise HTTPException(status_code=404, detail=f"Trainer case '{case_id}' not found")


@app.delete("/v1/trainer/mechid/cases/{case_id}", response_model=MechIDTrainerDeleteResponse)
def mechid_trainer_delete_case(case_id: str) -> MechIDTrainerDeleteResponse:
    cases = _load_mechid_eval_cases()
    remaining = [case for case in cases if case.get("id") != case_id]
    if len(remaining) == len(cases):
        raise HTTPException(status_code=404, detail=f"Trainer case '{case_id}' not found")
    _write_mechid_eval_cases(remaining)
    return MechIDTrainerDeleteResponse(
        deleted=True,
        caseId=case_id,
        totalCases=len(remaining),
    )


@app.post("/v1/trainer/mechid/evaluate-case", response_model=MechIDTrainerEvaluateResponse)
def mechid_trainer_evaluate_case(req: MechIDTrainerEvaluateRequest) -> MechIDTrainerEvaluateResponse:
    stats = EvalStats()
    client = TestClient(app)
    evaluate_mechid_case(
        client=client,
        case=req.draft_case.model_dump(by_alias=True, exclude_none=True),
        stats=stats,
        check_assistant=req.check_assistant,
    )
    return MechIDTrainerEvaluateResponse(
        passed=not stats.failures,
        caseId=req.draft_case.id,
        failures=stats.failures,
        success=stats.success,
        total=stats.total,
        parsedChecks=stats.parsed_checks,
        parsedPasses=stats.parsed_passes,
        analysisChecks=stats.analysis_checks,
        analysisPasses=stats.analysis_passes,
        provisionalChecks=stats.provisional_checks,
        provisionalPasses=stats.provisional_passes,
        assistantChecks=stats.assistant_checks,
        assistantPasses=stats.assistant_passes,
    )


@app.post("/v1/trainer/mechid/preview", response_model=MechIDTrainerPreviewResponse)
def mechid_trainer_preview(req: MechIDTrainerPreviewRequest) -> MechIDTrainerPreviewResponse:
    result = _build_mechid_text_response(
        req.text,
        parser_strategy=req.parser_strategy,
        parser_model=req.parser_model,
        allow_fallback=req.allow_fallback,
    )
    draft_case = _build_mechid_trainer_base_draft(
        text=req.text,
        parser_strategy=req.parser_strategy,
        result=result,
    )

    correction_applied = False
    correction_warning = None
    recommendation_applied = False
    recommendation_warning = None
    correction_text = (req.correction_text or "").strip()
    if correction_text:
        try:
            patch, correction_warning = parse_mechid_trainer_correction(
                raw_text=req.text,
                correction_text=correction_text,
                mechid_result=result,
                base_draft=draft_case,
                parser_model=req.parser_model,
            )
            if patch is not None:
                draft_case = _apply_trainer_patch(draft_case, patch)
                correction_applied = True
        except MechIDTrainerParseError as exc:
            correction_warning = str(exc)

    recommendation_text = (req.recommendation_text or "").strip()
    if recommendation_text:
        try:
            patch, recommendation_warning = generate_mechid_trainer_targets(
                raw_text=req.text,
                recommendation_text=recommendation_text,
                mechid_result=result,
                parser_model=req.parser_model,
            )
            if patch is not None:
                draft_case = _apply_trainer_patch(draft_case, patch)
                if not draft_case.assistant_guidance:
                    draft_case = draft_case.model_copy(update={"assistantGuidance": recommendation_text})
                recommendation_applied = True
        except MechIDTrainerGuidanceError as exc:
            recommendation_warning = str(exc)

    review_message, review_refined = _assistant_mechid_review_message(
        result,
        final=False,
        transient_examples=_trainer_transient_examples(draft_case, kind="review"),
    )
    final_message, final_refined = _assistant_mechid_review_message(
        result,
        final=True,
        transient_examples=_trainer_transient_examples(draft_case, kind="final"),
    )

    return MechIDTrainerPreviewResponse(
        mechidResult=result,
        assistantReviewMessage=review_message,
        assistantReviewRefined=review_refined,
        assistantFinalMessage=final_message,
        assistantFinalRefined=final_refined,
        draftCase=draft_case,
        correctionApplied=correction_applied,
        correctionWarning=correction_warning,
        recommendationApplied=recommendation_applied,
        recommendationWarning=recommendation_warning,
    )


@app.post("/v1/trainer/mechid/save", response_model=MechIDTrainerSaveResponse)
def mechid_trainer_save(req: MechIDTrainerSaveRequest) -> MechIDTrainerSaveResponse:
    draft = req.draft_case
    cases = _load_mechid_eval_cases()
    payload = draft.model_dump(by_alias=True, exclude_none=True)
    for index, existing in enumerate(cases):
        if existing.get("id") == draft.id:
            cases[index] = payload
            break
    else:
        cases.append(payload)
    _write_mechid_eval_cases(cases)
    return MechIDTrainerSaveResponse(
        saved=True,
        path=str(MECHID_EVAL_DATASET_PATH),
        caseId=draft.id,
        totalCases=len(cases),
    )


@app.get("/v1/modules")
def list_modules() -> dict:
    return {"modules": store.list_summaries()}


@app.get("/v1/modules/{module_id}", response_model=SyndromeModule)
def get_module(module_id: str) -> SyndromeModule:
    module = store.get(module_id)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    return module


def _build_mechid_text_response(
    text: str,
    *,
    parser_strategy: str = "auto",
    parser_model: str | None = None,
    allow_fallback: bool = True,
) -> MechIDTextAnalyzeResponse:
    warnings: List[str] = []
    parser_fallback_used = False
    parsed = None

    if parser_strategy == "rule":
        try:
            parsed = parse_mechid_text(text)
        except MechIDEngineError as exc:
            return MechIDTextAnalyzeResponse(
                text=text,
                parsedRequest=None,
                warnings=[str(exc)],
                requiresConfirmation=True,
                parserFallbackUsed=False,
                analysis=None,
            )
        parser_name = "rule-based-v1"
    elif parser_strategy == "openai":
        try:
            parsed = parse_mechid_text_with_openai(text=text, parser_model=parser_model)
            parser_name = str(parsed.get("parser") or "openai-mechid")
        except LLMParserError as exc:
            if not allow_fallback:
                return MechIDTextAnalyzeResponse(
                    text=text,
                    parsedRequest=None,
                    warnings=[f"OpenAI MechID parser failed: {exc}"],
                    requiresConfirmation=True,
                    parser="openai-mechid",
                    parserFallbackUsed=False,
                    analysis=None,
                )
            try:
                parsed = parse_mechid_text(text)
            except MechIDEngineError as rule_exc:
                return MechIDTextAnalyzeResponse(
                    text=text,
                    parsedRequest=None,
                    warnings=[f"OpenAI MechID parser failed: {exc}", str(rule_exc)],
                    requiresConfirmation=True,
                    parser="rule-based-v1",
                    parserFallbackUsed=True,
                    analysis=None,
                )
            parser_name = "rule-based-v1"
            parser_fallback_used = True
            warnings.append(f"OpenAI MechID parser unavailable/failed, used rule parser fallback: {exc}")
    else:
        openai_err: str | None = None
        if (os.getenv("OPENAI_API_KEY") or "").strip():
            try:
                parsed = parse_mechid_text_with_openai(text=text, parser_model=parser_model)
                parser_name = str(parsed.get("parser") or "openai-mechid")
            except LLMParserError as exc:
                openai_err = str(exc)
        if parsed is None:
            try:
                parsed = parse_mechid_text(text)
            except MechIDEngineError as exc:
                warning_list = [str(exc)]
                if openai_err:
                    warning_list.insert(0, f"OpenAI MechID parser unavailable/failed: {openai_err}")
                return MechIDTextAnalyzeResponse(
                    text=text,
                    parsedRequest=None,
                    warnings=warning_list,
                    requiresConfirmation=True,
                    parser="rule-based-v1",
                    parserFallbackUsed=bool(openai_err),
                    analysis=None,
                )
            parser_name = "rule-based-v1"
            if openai_err:
                parser_fallback_used = True
                warnings.append(f"OpenAI MechID parser unavailable/failed: {openai_err}")
                warnings.append("Used rule parser fallback.")

    parsed_request = None
    analysis = None
    if (
        parsed.get("organism") is not None
        or parsed.get("mentionedOrganisms")
        or parsed.get("susceptibilityResults")
        or parsed.get("resistancePhenotypes")
    ):
        parsed_request = MechIDTextParsedRequest(
            organism=parsed["organism"],
            mentionedOrganisms=parsed.get("mentionedOrganisms", []),
            resistancePhenotypes=parsed.get("resistancePhenotypes", []),
            susceptibilityResults=parsed["susceptibilityResults"],
            txContext=parsed["txContext"],
        )
        if parsed["organism"] is not None and parsed["susceptibilityResults"]:
            try:
                analyzed = analyze_mechid(
                    organism=parsed["organism"],
                    susceptibility_results=parsed["susceptibilityResults"],
                    tx_context=parsed["txContext"],
                )
                analysis = MechIDAnalyzeResponse(
                    organism=analyzed["organism"],
                    panel=analyzed["panel"],
                    submittedResults=analyzed["submitted_results"],
                    inferredResults=analyzed["inferred_results"],
                    finalResults=analyzed["final_results"],
                    rows=analyzed["rows"],
                    mechanisms=analyzed["mechanisms"],
                    cautions=analyzed["cautions"],
                    favorableSignals=analyzed["favorable_signals"],
                    therapyNotes=analyzed["therapy_notes"],
                    references=analyzed["references"],
                    warnings=list(parsed["warnings"]),
                )
            except MechIDEngineError as exc:
                warnings.append(str(exc))

    warnings.extend(parsed["warnings"])
    provisional_advice = _build_mechid_provisional_advice(parsed_request)
    return MechIDTextAnalyzeResponse(
        parser=parser_name,
        text=text,
        parsedRequest=parsed_request,
        warnings=warnings,
        requiresConfirmation=bool(parsed["requiresConfirmation"] or analysis is None),
        parserFallbackUsed=parser_fallback_used,
        analysis=analysis,
        provisionalAdvice=provisional_advice,
    )


@app.post("/v1/modules/register", response_model=RegisterModulesResponse)
def register_modules(payload: RegisterModulesRequest) -> RegisterModulesResponse:
    ids = store.upsert_many(payload.modules)
    return RegisterModulesResponse(registered=len(ids), ids=ids)


@app.get("/v1/mechid/organisms")
def list_mechid_supported_organisms() -> dict:
    try:
        return {"organisms": list_mechid_organisms()}
    except MechIDEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v1/mechid/analyze", response_model=MechIDAnalyzeResponse)
def analyze_mechid_endpoint(req: MechIDAnalyzeRequest) -> MechIDAnalyzeResponse:
    try:
        payload = analyze_mechid(
            organism=req.organism,
            susceptibility_results=req.susceptibility_results,
            tx_context=req.tx_context.model_dump() if req.tx_context is not None else None,
        )
    except MechIDEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return MechIDAnalyzeResponse(
        organism=payload["organism"],
        panel=payload["panel"],
        submittedResults=payload["submitted_results"],
        inferredResults=payload["inferred_results"],
        finalResults=payload["final_results"],
        rows=payload["rows"],
        mechanisms=payload["mechanisms"],
        cautions=payload["cautions"],
        favorableSignals=payload["favorable_signals"],
        therapyNotes=payload["therapy_notes"],
        references=payload["references"],
        warnings=[],
    )


@app.post("/v1/mechid/analyze-text", response_model=MechIDTextAnalyzeResponse)
def analyze_mechid_text_endpoint(req: MechIDTextAnalyzeRequest) -> MechIDTextAnalyzeResponse:
    return _build_mechid_text_response(
        req.text,
        parser_strategy=req.parser_strategy,
        parser_model=req.parser_model,
        allow_fallback=req.allow_fallback,
    )


def _resolve_module(req: AnalyzeRequest) -> SyndromeModule:
    if req.module is not None:
        return req.module
    if not req.module_id:
        raise HTTPException(status_code=400, detail="Provide either `moduleId` or inline `module`")
    module = store.get(req.module_id)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Module '{req.module_id}' not found")
    return module


def _reference_citation(short: str | None, year: int | None, fallback: str | None = None) -> str | None:
    if short and year:
        return f"{short} ({year})"
    if short:
        return short
    return fallback


def _collect_reference_entries(
    *,
    module: SyndromeModule,
    parsed_request: AnalyzeRequest | None,
    selected_pretest_factor_ids: List[str] | None = None,
) -> List[ReferenceEntry]:
    if parsed_request is None:
        return []

    entries: List[ReferenceEntry] = []
    seen: set[tuple[str, str, str]] = set()

    def add_entry(context: str, citation: str | None, url: str | None = None) -> None:
        if not citation:
            return
        key = (context, citation, url or "")
        if key in seen:
            return
        seen.add(key)
        entries.append(ReferenceEntry(context=context, citation=citation, url=url))

    preset = next((p for p in module.pretest_presets if p.id == parsed_request.preset_id), None)
    if preset is not None:
        add_entry(
            f"Preset: {preset.label}",
            _reference_citation(
                preset.source.short if preset.source else None,
                preset.source.year if preset.source else None,
                preset.notes,
            ),
            preset.source.url if preset.source else None,
        )

    items_by_id = {item.id: item for item in module.items}
    for finding_id in parsed_request.findings:
        item = items_by_id.get(finding_id)
        if item is None:
            continue
        add_entry(
            f"Finding: {item.label}",
            _reference_citation(
                item.source.short if item.source else None,
                item.source.year if item.source else None,
                item.notes,
            ),
            item.source.url if item.source else None,
        )

    if selected_pretest_factor_ids:
        specs_by_id = {spec.id: spec for spec in resolve_pretest_factor_specs(module)}
        for factor_id in selected_pretest_factor_ids:
            spec = specs_by_id.get(factor_id)
            if spec is None:
                continue
            add_entry(f"Baseline factor: {spec.label}", spec.source_note, spec.source_url)

    for evidence_ref in MODULE_EVIDENCE_REFERENCES.get(module.id, []):
        add_entry(evidence_ref["context"], evidence_ref["citation"], evidence_ref["url"])

    return entries


def _sync_text_result_references(
    *,
    text_result: TextAnalyzeResponse,
    module: SyndromeModule | None = None,
    selected_pretest_factor_ids: List[str] | None = None,
) -> None:
    parsed_request = text_result.parsed_request
    if parsed_request is None:
        text_result.references = []
        return

    resolved_module = module or parsed_request.module or store.get(parsed_request.module_id or "")
    if resolved_module is None:
        text_result.references = []
        return

    text_result.references = _collect_reference_entries(
        module=resolved_module,
        parsed_request=parsed_request,
        selected_pretest_factor_ids=selected_pretest_factor_ids,
    )


def _build_reasons(
    *,
    module: SyndromeModule,
    base_pretest: float,
    adjusted_pretest: float,
    preset_id: str | None,
    combined_lr_value: float,
    thresholds: DecisionThresholds,
    recommendation: str,
    applied_findings,
    prep_notes: List[str] | None = None,
) -> List[str]:
    reasons: List[str] = []
    preset_label = next((preset.label for preset in module.pretest_presets if preset.id == preset_id), None)

    if abs(adjusted_pretest - base_pretest) > 0.001:
        if preset_label:
            reasons.append(
                f"Pretest probability started from preset '{preset_label}' at {base_pretest:.1%} and was adjusted to {adjusted_pretest:.1%} using odds multiplier."
            )
        else:
            reasons.append(
                f"Pretest probability was adjusted from {base_pretest:.1%} to {adjusted_pretest:.1%} using odds multiplier."
            )
    else:
        if preset_label:
            reasons.append(f"Pretest probability starts at {adjusted_pretest:.1%} from preset '{preset_label}'.")
        else:
            reasons.append(f"Pretest probability starts at {adjusted_pretest:.1%} from the selected preset or override.")

    if applied_findings:
        top = applied_findings[:3]
        parts = [f"{f.label} ({f.state}, LR {f.lr_used:.2f})" for f in top]
        reasons.append("Strongest applied findings: " + ", ".join(parts) + ".")
    else:
        reasons.append("No diagnostic findings were applied, so post-test probability equals pretest probability.")

    reasons.append(
        f"Combined LR = {combined_lr_value:.2f}; thresholds: observe <= {thresholds.observe_probability:.1%}, treat >= {thresholds.treat_probability:.1%}."
    )
    if prep_notes:
        reasons.extend(prep_notes)
    reasons.append(f"Recommended action: {recommendation} for {module.name}.")
    return reasons


def _build_risk_flags(*, posttest_probability: float, thresholds: DecisionThresholds, applied_count: int) -> List[str]:
    flags: List[str] = []
    if applied_count == 0:
        flags.append("no_findings_selected")
    if applied_count == 1:
        flags.append("low_evidence_count")

    near_observe = abs(posttest_probability - thresholds.observe_probability) <= 0.03
    near_treat = abs(posttest_probability - thresholds.treat_probability) <= 0.03
    if near_observe or near_treat:
        flags.append("near_decision_threshold")

    if posttest_probability >= 0.5:
        flags.append("high_posttest_probability")
    if posttest_probability <= 0.05:
        flags.append("low_posttest_probability")
    return flags


def _build_recommendation_summary(
    *,
    module: SyndromeModule,
    recommendation: str,
    prep_findings: dict[str, str],
    preset_id: str | None = None,
) -> tuple[str | None, List[str]]:
    if module.id == "cdi":
        cdi_result_ids = {
            "cdi_naat_neg",
            "cdi_naat_pos_tox_pos",
            "cdi_naat_pos_tox_neg",
            "cdi_naat_pos_tox_na",
        }
        testing_done = any(item_id in prep_findings for item_id in cdi_result_ids)
        explicit_not_done = prep_findings.get("cdi_test_na") == "present"
        no_micro_selected = not testing_done and not explicit_not_done
        testing_pending = explicit_not_done or no_micro_selected
        positive_result = any(prep_findings.get(item_id) == "present" for item_id in cdi_result_ids if item_id != "cdi_naat_neg")
        negative_result = prep_findings.get("cdi_naat_neg") == "present"
        diarrhea_burden = prep_findings.get("cdi_freq") == "present"
        watery_diarrhea = prep_findings.get("cdi_watery") == "present"
        next_steps: List[str] = []

        if recommendation == "observe":
            if negative_result:
                summary = (
                    "C. difficile probability is below the testing threshold, and the negative stool result makes active C. difficile infection less likely. Repeat testing is usually not necessary unless the clinical picture changes substantially."
                )
                return summary, next_steps

            early_community_onset_context = preset_id == "ed_inpt_early"
            if early_community_onset_context and testing_pending and (diarrhea_burden or watery_diarrhea):
                summary = (
                    "C. difficile probability is still below the formal testing threshold, but in this ED or early-admission context stool C. difficile testing can be considered if the patient has unexplained clinically significant diarrhea."
                )
                summary += (
                    " Because presentations in the first 72 hours are often treated as community-onset rather than hospital-onset, reassess competing explanations and test if concern remains."
                )
            else:
                summary = (
                    "C. difficile probability is below the current testing threshold, so immediate stool C. difficile testing is usually not necessary based on the current data."
                )
                if testing_pending and (diarrhea_burden or watery_diarrhea):
                    summary += (
                        " If unexplained diarrhea persists, especially with at least 3 unformed stools in 24 hours, reassess and send stool testing if concern increases."
                    )
            return summary, next_steps

        if recommendation == "test":
            if testing_pending:
                summary = (
                    "C. difficile probability is high enough that stool C. difficile testing should be sent now if the patient has unexplained unformed diarrhea."
                )
                next_steps.append("Send stool C. difficile testing using the local NAAT/PCR plus toxin algorithm.")
                next_steps.append("Avoid testing formed stool or patients without clinically significant unexplained diarrhea.")
                return summary, next_steps

            if positive_result:
                summary = (
                    "The current stool result increases concern for C. difficile, but the overall probability remains in an intermediate range. Interpret the test in the clinical context before committing to treatment or dismissing alternative explanations."
                )
                next_steps.append("Correlate the stool result with diarrhea burden and any competing explanations for symptoms.")
                return summary, next_steps

            summary = (
                "C. difficile probability remains in an intermediate range despite the available stool testing. Reassess the clinical context and alternative causes before repeating testing or escalating treatment."
            )
            next_steps.append("Avoid repeat stool testing unless the clinical picture changes meaningfully.")
            return summary, next_steps

        if recommendation == "treat":
            summary = (
                "C. difficile probability is high enough that C. difficile-directed treatment is justified based on the current risk-benefit balance."
            )
            if testing_pending:
                next_steps.append("If feasible, send stool C. difficile testing while treatment decisions are being made.")
            return summary, next_steps

        return None, []

    if module.id == "active_tb":
        tb_micro_ids = {
            "tb_mtbpcr_sputum",
            "tb_mtbpcr_bal",
            "tb_afb_smear_sputum",
            "tb_culture_sputum",
            "tb_culture_bal",
        }
        tb_pcr_ids = {"tb_mtbpcr_sputum", "tb_mtbpcr_bal"}
        tb_culture_ids = {"tb_culture_sputum", "tb_culture_bal"}
        any_tb_micro_done = any(item_id in prep_findings for item_id in tb_micro_ids)
        tb_pcr_done = any(item_id in prep_findings for item_id in tb_pcr_ids)
        tb_afb_done = "tb_afb_smear_sputum" in prep_findings
        tb_culture_done = any(item_id in prep_findings for item_id in tb_culture_ids)
        immune_positive = any(prep_findings.get(item_id) == "present" for item_id in {"tb_qft", "tb_tst"})
        next_steps: List[str] = []

        if recommendation == "observe":
            summary = (
                "Active TB probability is below the observation threshold, so routine airborne isolation and immediate pulmonary TB rule-out are usually not necessary based on the current data."
            )
            if immune_positive:
                summary += (
                    " A positive QuantiFERON or tuberculin skin test makes latent TB infection more likely than active pulmonary TB; refer for latent TB evaluation and treatment if clinically appropriate once active TB is not otherwise suspected."
                )
                next_steps.append(
                    "If active TB is not otherwise suspected, refer for latent TB infection evaluation/treatment."
                )
            return summary, next_steps

        if recommendation == "test":
            if not any_tb_micro_done:
                summary = (
                    "Active TB probability is above the observation threshold, so this patient should be ruled out for pulmonary TB. Use airborne isolation and obtain respiratory TB testing."
                )
                next_steps.append("Use airborne isolation while the pulmonary TB rule-out is in progress.")
            else:
                summary = (
                    "Active TB probability remains above the observation threshold, so continue pulmonary TB rule-out and airborne isolation until adequate respiratory TB testing is complete or active TB is excluded."
                )
            if not tb_pcr_done:
                next_steps.append("Obtain MTB PCR/Xpert on a respiratory sample if it has not been sent yet.")
            if not tb_afb_done:
                next_steps.append("Obtain AFB sputum smear testing (with serial sputum samples per local protocol).")
            if not tb_culture_done:
                next_steps.append("Send mycobacterial culture from a respiratory sample if available.")
            return summary, next_steps

        if recommendation == "treat":
            if not any_tb_micro_done:
                summary = (
                    "Active TB is sufficiently likely that this patient should be ruled out for pulmonary TB immediately. Airborne isolation and TB-directed evaluation or empiric management are justified while confirmatory respiratory testing is completed."
                )
            else:
                summary = (
                    "Active TB is sufficiently likely that airborne isolation and TB-directed evaluation or empiric management are justified while confirmatory respiratory testing is completed."
                )
            next_steps.append("Maintain airborne isolation while confirmatory testing is pending.")
            if not tb_pcr_done:
                next_steps.append("Obtain MTB PCR/Xpert on a respiratory sample if it has not been sent yet.")
            if not tb_afb_done:
                next_steps.append("Obtain AFB sputum smear testing (with serial sputum samples per local protocol).")
            if not tb_culture_done:
                next_steps.append("Send mycobacterial culture from a respiratory sample if available.")
            return summary, next_steps

        return None, []

    if module.id != "endo":
        return None, []

    tee_done = prep_findings.get("endo_tee", "unknown") != "unknown"
    pet_done = prep_findings.get("endo_pet", "unknown") != "unknown"
    advanced_imaging_done = tee_done or pet_done
    next_steps: List[str] = []

    if recommendation == "observe":
        summary = "Endocarditis probability is below the observation threshold, so treating this as complicated bacteremia is reasonable."
        if not advanced_imaging_done:
            summary += " TEE is probably not necessary unless new features increase concern for endocarditis."
        return summary, next_steps

    if recommendation == "test":
        summary = "Endocarditis probability remains in an intermediate range, so further diagnostic testing is appropriate before escalating to full endocarditis-directed treatment."
        if not tee_done:
            next_steps.append("Consider TEE if it has not been performed yet.")
        if not pet_done:
            next_steps.append("Consider FDG PET/CT if prosthetic valve/device infection remains a concern and it has not been performed yet.")
        return summary, next_steps

    if recommendation == "treat":
        summary = (
            "Endocarditis probability is high enough that endocarditis-directed treatment is justified based on the current risk-benefit balance."
        )
        return summary, next_steps

    label = _assistant_module_label(module)
    if recommendation == "observe":
        return (
            f"{label} probability is currently low enough that I would not escalate syndrome-directed testing or treatment unless new data meaningfully raises concern.",
            [],
        )
    if recommendation == "test":
        return (
            f"{label} remains in an intermediate range, so I would keep it on the differential and get the next highest-yield discriminating data before committing to full syndrome-directed treatment.",
            [],
        )
    if recommendation == "treat":
        return (
            f"{label} probability is high enough that empiric syndrome-directed treatment is justified while the diagnostic picture is being confirmed.",
            [],
        )

    return None, []


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return _analyze_internal(req)


def _analyze_internal(req: AnalyzeRequest) -> AnalyzeResponse:
    module = _resolve_module(req)
    prep = prepare_probid_inputs(module, req)

    base_pretest, adjusted_pretest, preset_id = resolve_pretest(
        module,
        req,
        odds_multiplier=prep.effective_pretest_odds_multiplier,
    )
    combined_lr_value = combined_lr(module.items, prep.analysis_findings)
    posttest_probability = post_test_prob(adjusted_pretest, combined_lr_value)

    harms = resolve_harms(module, req, states_override=prep.harm_findings)
    thresholds = derive_decision_thresholds(harms)
    recommendation = recommendation_for_probability(posttest_probability, thresholds)
    confidence = confidence_from_thresholds(posttest_probability, thresholds)

    applied_findings = applied_finding_summaries(module, prep.analysis_findings)
    stepwise = build_stepwise_path(
        pretest_probability=adjusted_pretest,
        module=module,
        findings=prep.analysis_findings,
        ordered_ids=req.ordered_finding_ids,
    )

    reasons = _build_reasons(
        module=module,
        base_pretest=base_pretest,
        adjusted_pretest=adjusted_pretest,
        preset_id=preset_id,
        combined_lr_value=combined_lr_value,
        thresholds=thresholds,
        recommendation=recommendation,
        applied_findings=applied_findings,
        prep_notes=prep.notes,
    )
    if req.harms is None and module.default_harms is None:
        harm_estimate = estimate_harms(module.id, prep.harm_findings)
        if harm_estimate.rationale:
            reasons.append("Harm model: " + " ".join(harm_estimate.rationale))
    risk_flags = _build_risk_flags(
        posttest_probability=posttest_probability,
        thresholds=thresholds,
        applied_count=len(applied_findings),
    )
    recommendation_summary, recommended_next_steps = _build_recommendation_summary(
        module=module,
        recommendation=recommendation,
        prep_findings=prep.analysis_findings,
        preset_id=preset_id,
    )

    response = AnalyzeResponse(
        moduleId=module.id,
        moduleName=module.name,
        pretest=PretestSummary(baseProbability=base_pretest, adjustedProbability=adjusted_pretest, presetId=preset_id),
        combinedLR=combined_lr_value,
        posttestProbability=posttest_probability,
        thresholds=thresholds,
        recommendation=recommendation,  # type: ignore[arg-type]
        recommendationSummary=recommendation_summary,
        recommendedNextSteps=recommended_next_steps,
        confidence=confidence,
        appliedFindings=applied_findings,
        stepwise=stepwise,
        reasons=reasons,
        riskFlags=risk_flags,
        explanationForUser=None,
    )
    if req.include_explanation:
        response.explanation_for_user = _build_probid_consult_message(module, response)
    return response


@app.post("/v1/analyze-text", response_model=TextAnalyzeResponse)
def analyze_text(req: TextAnalyzeRequest) -> TextAnalyzeResponse:
    warnings: List[str] = []
    parser_fallback_used = False

    parsed = None
    if req.parser_strategy == "rule":
        parsed = parse_text_to_request(
            store=store,
            text=req.text,
            module_hint=req.module_hint,
            preset_hint=req.preset_hint,
            include_explanation=req.include_explanation,
        )
    elif req.parser_strategy == "local":
        try:
            parsed = parse_text_with_local_model(
                store=store,
                text=req.text,
                module_hint=req.module_hint,
                preset_hint=req.preset_hint,
                include_explanation=req.include_explanation,
            )
        except LocalParserError as exc:
            if not req.allow_fallback:
                raise HTTPException(status_code=422, detail=f"Local parser failed: {exc}")
            parsed = parse_text_to_request(
                store=store,
                text=req.text,
                module_hint=req.module_hint,
                preset_hint=req.preset_hint,
                include_explanation=req.include_explanation,
            )
            parser_fallback_used = True
            warnings.append(f"Local parser unavailable/failed, used rule parser fallback: {exc}")
    elif req.parser_strategy == "openai":
        try:
            parsed = parse_text_with_openai(
                store=store,
                text=req.text,
                module_hint=req.module_hint,
                preset_hint=req.preset_hint,
                include_explanation=req.include_explanation,
                parser_model=req.parser_model,
            )
        except LLMParserError as exc:
            if not req.allow_fallback:
                raise HTTPException(status_code=502, detail=f"OpenAI parser failed: {exc}")
            parsed = parse_text_to_request(
                store=store,
                text=req.text,
                module_hint=req.module_hint,
                preset_hint=req.preset_hint,
                include_explanation=req.include_explanation,
            )
            parser_fallback_used = True
            warnings.append(f"OpenAI parser unavailable/failed, used rule parser fallback: {exc}")
    else:  # auto
        local_err: str | None = None
        openai_err: str | None = None
        openai_available = bool((os.getenv("OPENAI_API_KEY") or "").strip())

        if openai_available:
            try:
                parsed = parse_text_with_openai(
                    store=store,
                    text=req.text,
                    module_hint=req.module_hint,
                    preset_hint=req.preset_hint,
                    include_explanation=req.include_explanation,
                    parser_model=req.parser_model,
                )
            except LLMParserError as exc:
                openai_err = str(exc)

        if parsed is None:
            try:
                parsed = parse_text_with_local_model(
                    store=store,
                    text=req.text,
                    module_hint=req.module_hint,
                    preset_hint=req.preset_hint,
                    include_explanation=req.include_explanation,
                )
            except LocalParserError as exc:
                local_err = str(exc)

        if parsed is None:
            if not req.allow_fallback:
                detail_parts = []
                if openai_err:
                    detail_parts.append(f"OpenAI parser failed: {openai_err}")
                if local_err:
                    detail_parts.append(f"Local parser failed: {local_err}")
                raise HTTPException(status_code=502, detail="; ".join(detail_parts) or "No parser available")
            parsed = parse_text_to_request(
                store=store,
                text=req.text,
                module_hint=req.module_hint,
                preset_hint=req.preset_hint,
                include_explanation=req.include_explanation,
            )
            parser_fallback_used = True
            if openai_err:
                warnings.append(f"OpenAI parser unavailable/failed: {openai_err}")
            if local_err:
                warnings.append(f"Local parser unavailable/failed: {local_err}")
            warnings.append("Used rule parser fallback.")
        elif openai_err:
            warnings.append(f"OpenAI parser unavailable/failed, used local parser: {openai_err}")

    if parsed is None:
        raise HTTPException(status_code=500, detail="Text parser failed to produce a result.")

    analysis: AnalyzeResponse | None = None
    if req.run_analyze and parsed.parsed_request is not None:
        try:
            analysis = _analyze_internal(parsed.parsed_request)
        except HTTPException as exc:
            warnings.append(f"Parsed request could not be analyzed yet: {exc.detail}")
            parsed.requires_confirmation = True

    response = TextAnalyzeResponse(
        parser=parsed.parser_name,
        text=req.text,
        parserFallbackUsed=parser_fallback_used,
        parsedRequest=parsed.parsed_request,
        understood=parsed.understood,
        warnings=[*warnings, *parsed.warnings],
        requiresConfirmation=parsed.requires_confirmation,
        analysis=analysis,
    )
    _sync_text_result_references(text_result=response)
    return response


def _assistant_module_options() -> List[AssistantOption]:
    options: List[AssistantOption] = [
        AssistantOption(
            value=MECHID_ASSISTANT_ID,
            label=MECHID_ASSISTANT_LABEL,
            description=MECHID_ASSISTANT_DESCRIPTION,
        )
    ]
    for summary in store.list_summaries():
        module = store.get(summary.id)
        options.append(
            AssistantOption(
                value=summary.id,
                label=_assistant_module_label(module) if module else summary.name,
                description=(module.description[:120] + "...") if module and module.description and len(module.description) > 120 else (module.description if module else None),
            )
        )
    return options


def _assistant_consult_focus_options() -> List[AssistantOption]:
    return [
        AssistantOption(
            value="focus_resistance",
            label="Resistance first",
            description="Start with the isolate, mechanism, and therapy interpretation.",
        ),
        AssistantOption(
            value="focus_syndrome",
            label="Syndrome first",
            description="Start with the diagnostic syndrome workup and post-test probability.",
        ),
        AssistantOption(
            value="focus_both",
            label="Both",
            description="Work through both, one step at a time.",
        ),
    ]


def _assistant_message_explicitly_mentions_module(message_text: str, module: SyndromeModule) -> bool:
    text = _normalize_choice(message_text)
    candidates = {
        module.name.lower(),
        module.id.replace("_", " ").lower(),
        _assistant_module_label(module).lower(),
    }
    return any(candidate and candidate in text for candidate in candidates)


def _assistant_module_label(module: SyndromeModule) -> str:
    return ASSISTANT_MODULE_LABELS.get(module.id, module.name)


def _assistant_review_options() -> List[AssistantOption]:
    return [
        AssistantOption(value="run_assessment", label="Give consultant impression"),
        AssistantOption(value="add_more_details", label="Add case detail"),
        AssistantOption(value="restart", label="Start new consult"),
    ]


def _assistant_single_ast_follow_up(label: str) -> str | None:
    cleaned = (label or "").strip()
    if not cleaned:
        return None
    normalized = re.sub(r"\s+(positive|negative|present|absent)$", "", cleaned, flags=re.IGNORECASE)
    return (
        f"The next thing I want to know is {cleaned}. "
        f"You can answer in one line, for example: '{normalized} susceptible' or '{normalized} resistant.'"
    )


def _assistant_single_case_follow_up(
    module: SyndromeModule,
    parsed_request: AnalyzeRequest | None,
    *,
    state: AssistantState | None = None,
) -> str | None:
    next_items = _top_missing_item_specs(module, parsed_request, limit=1, state=state)
    if not next_items:
        return None
    item_id, label = next_items[0]
    item = _assistant_case_item_by_id(module, item_id)
    if item is None:
        return None
    present_text, absent_text = _assistant_case_item_text(item, module)
    return (
        f"The next thing I want to know is {label}. "
        f"You can answer in one line, for example: '{present_text}' or '{absent_text}'."
    )


def _assistant_populate_case_review_analysis(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
) -> None:
    if text_result.analysis is not None or text_result.parsed_request is None:
        return
    try:
        text_result.analysis = _analyze_internal(text_result.parsed_request)
    except HTTPException:
        return


def _assistant_case_is_consult_ready(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> bool:
    if text_result.parsed_request is None:
        return False

    _assistant_populate_case_review_analysis(module, text_result)
    if text_result.analysis is None:
        return False

    item_lookup = {item.id: item for item in module.items}
    informative_count = 0
    for item_id, finding_state in (text_result.parsed_request.findings or {}).items():
        if finding_state == "unknown":
            continue
        item = item_lookup.get(item_id)
        if item is None:
            continue
        if any(token in item.id for token in ("not_done", "unknown", "not_used", "na")):
            continue
        informative_count += 1

    if len(text_result.analysis.applied_findings) >= 2:
        return True
    if informative_count >= 3:
        return True
    if (
        informative_count >= 2
        and text_result.analysis.combined_lr is not None
        and abs(text_result.analysis.combined_lr - 1.0) >= 0.25
    ):
        return True
    return not bool(_top_missing_item_specs(module, text_result.parsed_request, limit=1, state=state))


def _assistant_ready_for_consult_message(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> str:
    next_items = _top_missing_item_specs(module, text_result.parsed_request, limit=1, state=state)
    message = (
        "I have enough to run the consult with what you gave me. "
        "If you want to add more detail, just keep typing. If not, ask for my consultant impression."
    )
    if next_items:
        message += f" If you want to sharpen it further, the next useful detail would be {next_items[0][1]}."
    return message


def _assistant_probability_change_sentence(
    previous_analysis: AnalyzeResponse | None,
    updated_analysis: AnalyzeResponse | None,
) -> str | None:
    if previous_analysis is None or updated_analysis is None:
        return None
    previous = previous_analysis.posttest_probability
    updated = updated_analysis.posttest_probability
    delta = updated - previous
    if abs(delta) < 0.001:
        return f"The post-test probability is essentially unchanged at {updated:.1%}."
    direction = "up" if delta > 0 else "down"
    return (
        f"The post-test probability moved {direction} from {previous:.1%} to {updated:.1%}."
    )


def _assistant_concise_probid_follow_up(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
    *,
    lead: str = "That helps.",
) -> str:
    pieces = [lead]
    if _assistant_case_is_consult_ready(module, text_result, state):
        pieces.append(_assistant_ready_for_consult_message(module, text_result, state))
    else:
        follow_up = _assistant_single_case_follow_up(module, text_result.parsed_request, state=state)
        if follow_up:
            pieces.append(follow_up)
        elif text_result.requires_confirmation:
            pieces.append("If anything looks off, correct it or add another detail.")
        else:
            pieces.append("If this extraction matches the case, ask for my consultant impression.")
    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def _assistant_concise_mechid_follow_up(
    result: MechIDTextAnalyzeResponse,
    *,
    lead: str = "That helps.",
    latest_message: str | None = None,
) -> str:
    parsed = result.parsed_request
    if parsed is None:
        message = (
            "I still could not confidently identify the organism and AST pattern. "
            "Please add the organism plus a few susceptibility calls."
        )
        if result.warnings:
            message += " " + result.warnings[0]
        return message

    pieces = [lead]
    if result.provisional_advice is not None and result.provisional_advice.missing_susceptibilities:
        next_label = result.provisional_advice.missing_susceptibilities[0]
        if latest_message:
            latest_norm = " ".join(_normalize_choice(latest_message).split())
            for candidate in result.provisional_advice.missing_susceptibilities:
                candidate_norm = " ".join(_normalize_choice(candidate.replace(":", " ")).split())
                if candidate_norm and candidate_norm in latest_norm:
                    continue
                next_label = candidate
                break
        follow_up = _assistant_single_ast_follow_up(next_label)
        if follow_up:
            pieces.append(follow_up)
    elif result.analysis is None:
        pieces.append("Please add a few susceptibility calls so I can tighten the interpretation.")
    else:
        pieces.append("If this extraction matches the case, ask for my consultant impression.")
    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def _join_readable(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _friendly_probid_bottom_line(module: SyndromeModule, analysis: AnalyzeResponse) -> str:
    probability = round(analysis.posttest_probability * 100)
    syndrome_name = module.name.lower()
    if analysis.recommendation == "treat":
        return (
            f"The estimated probability of {module.name} is about {probability}%, which is high enough that I would treat this as likely {syndrome_name} while confirming the diagnosis."
        )
    if analysis.recommendation == "test":
        return (
            f"The estimated probability of {module.name} is about {probability}%, which keeps {syndrome_name} in an intermediate zone. I would keep working it up before treating it as established."
        )
    return (
        f"The estimated probability of {module.name} is about {probability}%, which puts {syndrome_name} below the action threshold for now unless new findings shift the picture."
    )


def _friendly_probid_drivers(analysis: AnalyzeResponse) -> str:
    if not analysis.applied_findings:
        return (
            "I do not yet have strong discriminating findings, so the estimate is still being driven mostly by the starting pretest probability."
        )

    top_findings = [
        f"{finding.label} ({finding.state}, LR {finding.lr_used:.2f})"
        for finding in analysis.applied_findings[:3]
    ]
    return f"The biggest drivers here are {_join_readable(top_findings)}."


def _friendly_probid_probability_and_harm(module: SyndromeModule, analysis: AnalyzeResponse) -> str:
    preset_label = None
    if analysis.pretest.preset_id:
        preset_label = next(
            (preset.label for preset in module.pretest_presets if preset.id == analysis.pretest.preset_id),
            analysis.pretest.preset_id,
        )
    probability = analysis.posttest_probability
    threshold_sentence = (
        f"Post-test probability is {probability:.1%} after a combined LR of {analysis.combined_lr:.2f}. "
        f"The current action thresholds are observe at or below {analysis.thresholds.observe_probability:.1%} "
        f"and treat at or above {analysis.thresholds.treat_probability:.1%}."
    )
    if preset_label:
        threshold_sentence = (
            f"The starting pretest came from preset '{preset_label}' at {analysis.pretest.base_probability:.1%}. "
            + threshold_sentence
        )
    harm_reason = next((reason for reason in analysis.reasons if reason.startswith("Harm model: ")), None)
    if harm_reason:
        return threshold_sentence + " " + harm_reason.removeprefix("Harm model: ").strip()
    return (
        threshold_sentence
        + " Those thresholds reflect the current harm tradeoff between unnecessary treatment and a missed diagnosis."
    )


def _friendly_probid_next_steps(analysis: AnalyzeResponse) -> str:
    if analysis.recommended_next_steps:
        return _join_readable(analysis.recommended_next_steps[:3]) + "."
    if analysis.recommendation == "treat":
        return "Start syndrome-directed treatment and close the highest-yield diagnostic gaps in parallel."
    if analysis.recommendation == "test":
        return "Get the highest-yield next test, imaging study, or microbiology result that would move this estimate one way or the other."
    return "Keep watching the clinical trajectory and only reopen this workup if new objective findings increase concern."


def _friendly_probid_change_mind(
    analysis: AnalyzeResponse,
    missing_suggestions: List[str] | None = None,
) -> str:
    if missing_suggestions:
        return f"The results most likely to move this estimate are {_join_readable(missing_suggestions[:3])}."
    if analysis.recommendation == "treat":
        return "High-quality negative objective data or a stronger competing diagnosis would lower my confidence."
    if analysis.recommendation == "test":
        return "A discriminating microbiology, imaging, or bedside finding could push this either toward treatment or away from the syndrome."
    return "New focal findings, supportive microbiology, or compatible imaging could raise concern quickly."


def _build_probid_consult_message(
    module: SyndromeModule,
    analysis: AnalyzeResponse,
    *,
    missing_suggestions: List[str] | None = None,
    include_panel_note: bool = False,
) -> str:
    lines = [
        "My impression:",
        f"Bottom line: {_friendly_probid_bottom_line(module, analysis)}",
        f"Probability and harm: {_friendly_probid_probability_and_harm(module, analysis)}",
        f"Why I think that: {_friendly_probid_drivers(analysis)}",
        f"What I would do next: {_friendly_probid_next_steps(analysis)}",
        f"What would change my mind: {_friendly_probid_change_mind(analysis, missing_suggestions)}",
    ]
    if include_panel_note:
        lines.append("I left the LR breakdown and structured findings in the analysis panel below.")
    return "\n".join(lines)


def _format_mechid_results(results: Dict[str, str]) -> str:
    if not results:
        return "no susceptibility calls yet"
    ordered = [f"{antibiotic} {result.lower()}" for antibiotic, result in sorted(results.items())]
    return _join_readable(ordered)


def _format_mechid_organism_list(organisms: List[str]) -> str:
    if not organisms:
        return "no organism identified yet"
    return _join_readable(organisms)


def _build_mechid_provisional_advice(parsed: MechIDTextParsedRequest | None) -> MechIDProvisionalAdvice | None:
    if parsed is None:
        return None

    organisms = list(dict.fromkeys(parsed.mentioned_organisms or ([parsed.organism] if parsed.organism else [])))
    syndrome = parsed.tx_context.syndrome
    focus_detail = parsed.tx_context.focus_detail
    severity = parsed.tx_context.severity
    oral_preference = parsed.tx_context.oral_preference
    phenotype_hints = set(parsed.resistance_phenotypes)
    has_ast = bool(parsed.susceptibility_results)

    if has_ast:
        return None

    has_mrsa = "MRSA" in phenotype_hints
    has_staph_aureus = "Staphylococcus aureus" in organisms
    has_beta_strep = "β-hemolytic Streptococcus (GAS/GBS)" in organisms
    has_enterococcus = any(org.startswith("Enterococcus") for org in organisms)
    has_gnr = any(
        org in {
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Pseudomonas aeruginosa",
            "Acinetobacter baumannii complex",
            "Enterobacter cloacae complex",
            "Serratia marcescens",
            "Proteus mirabilis",
        }
        for org in organisms
    )
    is_deep_wound = syndrome == "Other deep-seated / high-inoculum focus"
    is_bone_joint = syndrome == "Bone/joint infection"
    is_pneumonia = syndrome == "Pneumonia (HAP/VAP or severe CAP)"
    is_intra_abdominal = syndrome == "Intra-abdominal infection"
    is_severe = severity == "Severe / septic shock"

    def _base_missing(*items: str) -> List[str]:
        return list(dict.fromkeys(items))

    if has_mrsa and has_staph_aureus and has_beta_strep and (is_deep_wound or is_bone_joint):
        recommended = ["Vancomycin", "Linezolid", "Daptomycin for non-pulmonary infection"]
        notes = [
            "For MRSA, doxycycline or trimethoprim/sulfamethoxazole alone would not be my preferred answer here because streptococcal coverage is less reliable.",
        ]
        if focus_detail == "Diabetic foot infection":
            notes.insert(
                0,
                "If this diabetic foot infection is severe, ischemic, malodorous, or clearly polymicrobial, I would usually add gram-negative and anaerobic coverage until the full culture picture is clearer.",
            )
        if focus_detail == "Osteomyelitis":
            notes.insert(0, "Debridement and bone source control matter as much as the antibiotic choice in osteomyelitis.")
        if focus_detail == "Septic arthritis":
            notes.insert(0, "Joint drainage and clinical response matter just as much as the isolate list in septic arthritis.")
        oral_options = ["Linezolid"] if oral_preference else []
        if oral_preference:
            oral_options.append("Clindamycin if both isolates are susceptible")
        return MechIDProvisionalAdvice(
            summary=(
                f"For {focus_detail.lower() if focus_detail != 'Not specified' else 'this infection'} growing MRSA plus group B streptococcus, I would choose an agent that reliably covers both organisms while susceptibilities are pending."
            ),
            recommendedOptions=recommended,
            oralOptions=oral_options,
            missingSusceptibilities=_base_missing(
                "MRSA: clindamycin",
                "MRSA: linezolid",
                "MRSA: doxycycline",
                "MRSA: trimethoprim/sulfamethoxazole",
                "Group B streptococcus: clindamycin",
            ),
            notes=notes,
        )

    if has_mrsa and has_staph_aureus:
        oral_options = ["Linezolid"] if oral_preference else []
        notes = [
            "Without susceptibilities, I would default to a reliable anti-MRSA option and then narrow once the AST returns."
        ]
        if oral_preference:
            notes.append(
                "If you want an oral option, linezolid is the cleanest empiric oral MRSA agent before the AST comes back."
            )
        return MechIDProvisionalAdvice(
            summary="This sounds like an MRSA-driven infection, so I would start with dependable MRSA coverage and then narrow once susceptibilities are available.",
            recommendedOptions=["Vancomycin", "Linezolid", "Daptomycin for non-pulmonary infection"],
            oralOptions=oral_options,
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
            ),
            notes=notes,
        )

    if focus_detail == "Diabetic foot infection" or (organisms and is_deep_wound):
        return MechIDProvisionalAdvice(
            summary=(
                "For a diabetic foot or deep wound infection, I would match therapy to severity, depth, and whether the picture looks polymicrobial rather than relying only on the first culture names."
            ),
            recommendedOptions=[
                "If the culture is mainly gram-positive and the patient is stable: choose focused gram-positive coverage.",
                "If the wound is severe, deep, limb-threatening, or clearly polymicrobial: add gram-negative and anaerobic coverage."
            ],
            oralOptions=["Oral step-down may be possible later if the patient is improving and the susceptibilities support it."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
                "fluoroquinolone or other reported gram-negative agents if gram-negatives are present",
            ),
            notes=[
                "Source control, debridement, and depth of infection matter as much as the isolate list in diabetic foot infections.",
                "If osteomyelitis is also present, definitive oral step-down usually depends on reliable AST plus clinical improvement.",
            ],
        )

    if focus_detail == "Osteomyelitis":
        return MechIDProvisionalAdvice(
            summary="For osteomyelitis, I would choose therapy that reliably covers the recovered organisms and then narrow once susceptibilities and source-control plans are clear.",
            recommendedOptions=[
                "For MRSA concern: Vancomycin, Linezolid, or Daptomycin for non-pulmonary infection",
                "For streptococcal-only infection: a beta-lactam is often preferred once confirmed susceptible",
            ],
            oralOptions=["Linezolid can be a bridge oral option in selected cases.", "Other oral step-down options depend heavily on susceptibilities and source control."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
                "beta-lactam susceptibilities for the non-MRSA isolate",
            ),
            notes=[
                "Debridement, hardware considerations, and the ability to achieve source control are central in osteomyelitis.",
            ],
        )

    if focus_detail == "Septic arthritis":
        return MechIDProvisionalAdvice(
            summary="For septic arthritis, I would start with dependable coverage for the recovered organisms and then narrow quickly once susceptibilities return and the joint has been drained.",
            recommendedOptions=[
                "For MRSA concern: Vancomycin or Linezolid",
                "For streptococcal-only infection: a beta-lactam is often preferred once susceptibility is known",
            ],
            oralOptions=["Linezolid may be a temporary oral bridge in selected stable patients, but most early septic arthritis treatment starts IV."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "beta-lactam susceptibilities for the streptococcal isolate",
            ),
            notes=[
                "Drainage and serial clinical response are core parts of septic arthritis management, not just antibiotic selection.",
            ],
        )

    if is_pneumonia:
        recommended = []
        notes = []
        if has_mrsa:
            recommended.append("Vancomycin or Linezolid for MRSA coverage")
        if has_gnr or is_severe:
            recommended.append("A beta-lactam with strong pneumonia activity, and add antipseudomonal coverage if the organism list or setting supports it")
        if not recommended:
            recommended.append("Choose empiric therapy based on whether this behaves like CAP versus HAP/VAP and then narrow to the isolated organism(s)")
        notes.append("For pneumonia, daptomycin is not useful.")
        if oral_preference:
            notes.append("I would not anchor on oral therapy early in pneumonia unless the patient is clearly improving and the isolates support it.")
        return MechIDProvisionalAdvice(
            summary="For pneumonia, I would choose therapy based on the likely setting and whether MRSA or resistant gram-negatives truly need to be covered, then narrow once susceptibilities return.",
            recommendedOptions=recommended,
            oralOptions=["Oral step-down is sometimes possible later, but not usually the starting move for severe pneumonia."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "oxacillin or cefoxitin if Staphylococcus aureus is present",
                "ceftriaxone",
                "cefepime",
                "piperacillin/tazobactam",
                "levofloxacin",
            ),
            notes=notes,
        )

    if is_intra_abdominal:
        recommended = [
            "Broad intra-abdominal coverage that addresses enteric gram-negatives and anaerobes",
        ]
        if has_enterococcus:
            recommended.append("Add Enterococcus-active therapy when the source and patient context justify it")
        if has_mrsa:
            recommended.append("Add anti-MRSA therapy only if the culture and syndrome truly make MRSA clinically relevant")
        return MechIDProvisionalAdvice(
            summary="For intra-abdominal infection, I would treat the source first and use a regimen that covers enteric gram-negatives plus anaerobes, then narrow once the clinical source and susceptibilities are clearer.",
            recommendedOptions=recommended,
            oralOptions=["Oral step-down is sometimes possible later after source control and clinical improvement."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "ceftriaxone",
                "cefepime",
                "piperacillin/tazobactam",
                "ertapenem or meropenem if a resistant gram-negative is present",
                "fluoroquinolone or trimethoprim/sulfamethoxazole if an oral step-down is being considered",
            ),
            notes=[
                "Drainage or source control usually matters more than trying to pick the final narrowest antibiotic before the source is defined.",
            ],
        )

    if organisms and is_bone_joint:
        return MechIDProvisionalAdvice(
            summary="For a bone or joint infection, I can give a provisional site-based recommendation now, but definitive therapy still depends on susceptibilities and source control.",
            recommendedOptions=[
                "Use dependable coverage for the listed organisms first, then narrow once susceptibilities return.",
            ],
            oralOptions=["Oral step-down may be possible later if the patient is improving and the susceptibilities support it."] if oral_preference else [],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
                "beta-lactam susceptibilities for the non-MRSA isolate",
            ),
            notes=[
                "Source control matters as much as the isolate list in bone and joint infections.",
            ],
        )

    return None


def _assistant_mechid_review_options(result: MechIDTextAnalyzeResponse) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    if result.analysis is not None or result.provisional_advice is not None:
        options.append(AssistantOption(value="run_assessment", label="Give consultant impression"))
    options.append(AssistantOption(value="add_more_details", label="Add case detail"))
    options.append(AssistantOption(value="restart", label="Start new consult"))
    return options


def _clean_mechid_text(text: str) -> str:
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    cleaned = cleaned.replace("→", ": ").replace("β", "beta")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _friendly_mechid_mechanism(result: MechIDAnalyzeResponse) -> str:
    if not result.mechanisms:
        return "I do not see a single dominant resistance mechanism from the submitted pattern."

    cleaned = [_clean_mechid_text(item).rstrip(".") for item in result.mechanisms[:2]]
    first = cleaned[0]
    if "ESBL pattern" in first:
        summary = "This looks most like an ESBL-producing isolate."
    elif "Fluoroquinolone resistance" in first:
        summary = "This looks most like a fluoroquinolone-resistant isolate."
    elif "Methicillin" in first:
        summary = "This looks most like methicillin resistance."
    else:
        summary = f"The leading resistance explanation is: {first}."

    if len(cleaned) > 1:
        second = cleaned[1]
        if "Fluoroquinolone resistance" in second:
            summary += " I also see clear fluoroquinolone resistance."
        else:
            summary += f" I also see a second resistance signal: {second}."
    return summary


def _friendly_mechid_therapy(
    result: MechIDAnalyzeResponse,
    parsed: MechIDTextParsedRequest | None,
) -> str:
    def _option_overview(
        syndrome_local: str,
        focus_local: str,
        susceptible_agents: list[str],
    ) -> str | None:
        susceptible = set(susceptible_agents)

        if syndrome_local == "Uncomplicated cystitis":
            oral_choices = [
                agent
                for agent in (
                    "Nitrofurantoin",
                    "Trimethoprim/Sulfamethoxazole",
                    "Fosfomycin",
                    "Ciprofloxacin",
                    "Levofloxacin",
                )
                if agent in susceptible
            ]
            if oral_choices:
                return (
                    "For cystitis, the practical options from the susceptibilities you gave me are "
                    f"{_join_readable(oral_choices)}."
                )
            return (
                "For cystitis, I would mainly look for a susceptible oral lower-tract option such as "
                "nitrofurantoin, trimethoprim/sulfamethoxazole, or fosfomycin before defaulting to a broad IV agent."
            )

        if syndrome_local == "Complicated UTI / pyelonephritis":
            oral_choices = [
                agent
                for agent in (
                    "Trimethoprim/Sulfamethoxazole",
                    "Ciprofloxacin",
                    "Levofloxacin",
                )
                if agent in susceptible
            ]
            if oral_choices:
                return (
                    "For pyelonephritis or complicated UTI, I would think in terms of a reliably active upfront agent, "
                    f"with possible oral step-down later using {_join_readable(oral_choices)} if the patient improves."
                )
            return (
                "For pyelonephritis or complicated UTI, I would usually start with a reliably active agent and only think about oral step-down later if the isolate leaves a clearly active oral choice."
            )

        if syndrome_local == "Bloodstream infection":
            if focus_local == "Endocarditis":
                return (
                    "For endocarditis, I would treat this as an IV-first problem and would not usually frame the answer around oral options unless there is a very specific validated step-down plan."
                )
            if susceptible_agents:
                return (
                    "For bacteremia, I would usually start with dependable IV therapy. "
                    f"The susceptible drugs you gave me that look potentially usable are {_join_readable(susceptible_agents[:3])}."
                )
            return "For bacteremia, I would usually start with dependable IV therapy and only think about oral step-down much later in selected uncomplicated cases."

        if syndrome_local == "Bone/joint infection":
            oral_choices = [
                agent
                for agent in (
                    "Linezolid",
                    "Trimethoprim/Sulfamethoxazole",
                    "Doxycycline",
                    "Ciprofloxacin",
                    "Levofloxacin",
                )
                if agent in susceptible
            ]
            if oral_choices:
                return (
                    "For osteomyelitis or septic arthritis, I would mainly think about reliable upfront therapy, "
                    f"with oral step-down later sometimes possible using {_join_readable(oral_choices)} once source control is addressed."
                )
            return (
                "For osteomyelitis or septic arthritis, I would usually start with dependable therapy and only think about oral step-down later if source control is in place and the isolate leaves a clearly reliable oral option."
            )

        if syndrome_local == "Other deep-seated / high-inoculum focus":
            oral_choices = [
                agent
                for agent in (
                    "Linezolid",
                    "Trimethoprim/Sulfamethoxazole",
                    "Doxycycline",
                    "Ciprofloxacin",
                    "Levofloxacin",
                )
                if agent in susceptible
            ]
            if oral_choices:
                return (
                    "For a diabetic foot or other deep wound infection, I would first make sure drainage or debridement is adequate, "
                    f"then think about step-down options such as {_join_readable(oral_choices)} if the patient is improving."
                )
            return (
                "For a diabetic foot or other deep wound infection, I would usually start with reliable therapy for a deep or severe infection and then reassess after source control and AST review."
            )

        if syndrome_local == "Pneumonia (HAP/VAP or severe CAP)":
            return "For pneumonia, I would usually prioritize a reliably active IV option first and only think about narrowing once the isolate, site, and clinical trajectory are clearer."

        if syndrome_local == "Intra-abdominal infection":
            return "For intra-abdominal infection, I would think about a reliably active regimen plus source control rather than chasing an oral option up front."

        if syndrome_local == "CNS infection":
            return "For CNS infection, I would prioritize an agent with reliable CNS activity rather than relying only on the susceptibility label."

        return None

    def _select_preferred_agent(susceptible_agents: list[str]) -> str | None:
        for candidate in ("Meropenem", "Imipenem", "Ertapenem"):
            if candidate in susceptible_agents:
                return candidate
        return susceptible_agents[0] if susceptible_agents else None

    def _recommendation_sentence(
        selected_note_local: str | None,
        preferred_local: str | None,
        syndrome_local: str,
        severity_local: str,
        resistant_agents_local: list[str],
    ) -> str | None:
        if selected_note_local is not None:
            lead, sep, rest = selected_note_local.partition(":")
            if sep and rest.strip():
                return (
                    f"Based on the susceptibilities you gave me, this fits {lead.strip().lower()}, "
                    f"so I would {rest.strip().rstrip('.')}."
                )
            return (
                "Based on the susceptibilities you gave me, "
                f"{selected_note_local[0].lower() + selected_note_local[1:].rstrip('.') }."
            )
        if preferred_local is not None:
            if severity_local == "Severe / septic shock":
                return f"Based on this severity, I would lean toward {preferred_local} if it fits the infection source and patient factors."
            if syndrome_local == "CNS infection":
                return (
                    f"Based on the drugs you gave me, {preferred_local} looks active, but I would still choose based on CNS penetration rather than the susceptibility label alone."
                )
            return f"Based on the susceptibilities you gave me, I would lean toward {preferred_local}."
        if resistant_agents_local:
            avoid = _join_readable(resistant_agents_local[:3])
            return f"Based on this pattern, I would avoid {avoid} unless there is additional data that changes the interpretation."
        return None

    def _select_mechid_therapy_note() -> str | None:
        if not result.therapy_notes:
            return None
        cleaned_notes = [_clean_mechid_text(note).rstrip(".") for note in result.therapy_notes]
        syndrome_local = parsed.tx_context.syndrome if parsed is not None else "Not specified"
        severity_local = parsed.tx_context.severity if parsed is not None else "Not specified"
        carbapenemase_result_local = parsed.tx_context.carbapenemase_result if parsed is not None else "Not specified"
        carbapenemase_class_local = parsed.tx_context.carbapenemase_class if parsed is not None else "Not specified"

        best_note = cleaned_notes[0]
        best_score = -1
        for note in cleaned_notes:
            note_lower = note.lower()
            score = 0
            if any(token in note_lower for token in ("preferred", "prefer", "use ", "appropriate", "standard")):
                score += 3
            if any(token in note_lower for token in ("avoid", "do not rely", "not preferred")):
                score += 1
            if severity_local == "Severe / septic shock" and any(
                token in note_lower
                for token in ("serious infections", "high-risk syndrome", "invasive disease", "severe sepsis", "bacteremia", "pneumonia")
            ):
                score += 4
            if syndrome_local == "Uncomplicated cystitis" and any(token in note_lower for token in ("cystitis", "urinary", "oral option")):
                score += 4
            if syndrome_local == "Complicated UTI / pyelonephritis" and any(
                token in note_lower for token in ("pyelonephritis", "urinary", "oral option")
            ):
                score += 4
            if syndrome_local == "Pneumonia (HAP/VAP or severe CAP)" and "pneumonia" in note_lower:
                score += 4
            if carbapenemase_result_local == "Positive" and any(
                token in note_lower
                for token in (
                    "meropenem/vaborbactam",
                    "ceftazidime/avibactam",
                    "imipenem/cilastatin/relebactam",
                    "aztreonam",
                    "cefiderocol",
                    "carbapenemase pattern",
                )
            ):
                score += 5
            if carbapenemase_class_local != "Not specified" and carbapenemase_class_local.lower() in note_lower:
                score += 4
            if score > best_score:
                best_score = score
                best_note = note
        return best_note

    provided_results = parsed.susceptibility_results if parsed is not None else {}
    susceptible_agents = [
        antibiotic
        for antibiotic, call in (provided_results or result.final_results).items()
        if call == "Susceptible"
    ]
    resistant_agents = [
        antibiotic
        for antibiotic, call in (provided_results or result.final_results).items()
        if call == "Resistant"
    ]
    syndrome = parsed.tx_context.syndrome if parsed is not None else "Not specified"
    severity = parsed.tx_context.severity if parsed is not None else "Not specified"
    focus = parsed.tx_context.focus_detail if parsed is not None else "Not specified"
    selected_note = _select_mechid_therapy_note()
    preferred = _select_preferred_agent(susceptible_agents)
    option_overview = _option_overview(syndrome, focus, susceptible_agents)
    recommendation = _recommendation_sentence(
        selected_note,
        preferred,
        syndrome,
        severity,
        resistant_agents,
    )

    lines: List[str] = []
    if option_overview is not None:
        lines.append(option_overview)
    if recommendation is not None:
        lines.append(recommendation)

    return " ".join(lines) if lines else "I would match therapy to the susceptible agents and infection source."


def _friendly_mechid_oral_options(
    result: MechIDAnalyzeResponse,
    parsed: MechIDTextParsedRequest | None,
) -> str | None:
    if parsed is None:
        return None

    syndrome = parsed.tx_context.syndrome
    oral_preference = parsed.tx_context.oral_preference
    provided_results = parsed.susceptibility_results or result.final_results
    susceptible_agents = {
        antibiotic
        for antibiotic, call in provided_results.items()
        if call == "Susceptible"
    }

    if syndrome == "Uncomplicated cystitis":
        oral_choices = [
            agent
            for agent in (
                "Nitrofurantoin",
                "Trimethoprim/Sulfamethoxazole",
                "Fosfomycin",
            )
            if agent in susceptible_agents
        ]
        if oral_choices:
            return (
                f"For uncomplicated cystitis, oral options that could be considered from your submitted AST are "
                f"{_join_readable(oral_choices)} if there are no patient-specific contraindications."
            )
        return "For uncomplicated cystitis, I would look for a susceptible oral lower-tract option rather than defaulting to a broad IV agent."

    if syndrome == "Complicated UTI / pyelonephritis":
        oral_choices = [
            agent
            for agent in (
                "Trimethoprim/Sulfamethoxazole",
                "Ciprofloxacin",
                "Levofloxacin",
            )
            if agent in susceptible_agents
        ]
        if oral_choices:
            return (
                f"For pyelonephritis or complicated UTI, oral step-down could be considered with "
                f"{_join_readable(oral_choices)} if the patient is improving and source control is adequate."
            )
        return "For pyelonephritis or complicated UTI, I would be more cautious about oral step-down unless a clearly active oral option is available."

    if syndrome == "Bone/joint infection" and oral_preference:
        oral_choices = [
            agent
            for agent in (
                "Trimethoprim/Sulfamethoxazole",
                "Ciprofloxacin",
                "Levofloxacin",
                "Linezolid",
                "Doxycycline",
            )
            if agent in susceptible_agents
        ]
        if oral_choices:
            return (
                f"For osteomyelitis, septic arthritis, or another bone/joint infection, oral therapy can sometimes be used with "
                f"{_join_readable(oral_choices)} once the patient is stable and source control is addressed."
            )
        return "For bone or joint infection, I would be cautious about promising an oral option unless the isolate leaves you with a clearly reliable oral agent."

    if syndrome == "Other deep-seated / high-inoculum focus" and oral_preference:
        oral_choices = [
            agent
            for agent in (
                "Trimethoprim/Sulfamethoxazole",
                "Ciprofloxacin",
                "Levofloxacin",
                "Linezolid",
                "Doxycycline",
            )
            if agent in susceptible_agents
        ]
        if oral_choices:
            return (
                f"For a diabetic foot or deep wound-type infection, oral step-down might be possible with "
                f"{_join_readable(oral_choices)} if the patient is improving and the wound has been adequately drained or debrided."
            )
        return "For a diabetic foot or other deep wound infection, I would be cautious about oral therapy unless you have a clearly active oral option plus good source control."

    if oral_preference:
        oral_choices = [
            agent
            for agent in (
                "Trimethoprim/Sulfamethoxazole",
                "Ciprofloxacin",
                "Levofloxacin",
                "Nitrofurantoin",
                "Fosfomycin",
                "Linezolid",
                "Doxycycline",
            )
            if agent in susceptible_agents
        ]
        if oral_choices:
            return f"If you are specifically looking for an oral option, the submitted AST suggests {_join_readable(oral_choices)} could be considered depending on source and severity."
        return "You asked about an oral option, but from the submitted AST I do not see an obvious oral choice I would feel comfortable recommending without more context."

    return None


def _friendly_mechid_caution(result: MechIDAnalyzeResponse) -> str | None:
    if not result.cautions:
        return None
    return _clean_mechid_text(result.cautions[0]).rstrip(".") + "."


def _friendly_mechid_bottom_line(
    result: MechIDAnalyzeResponse,
    parsed: MechIDTextParsedRequest | None,
) -> str:
    therapy_line = _friendly_mechid_therapy(result, parsed)
    if therapy_line and therapy_line != "I would match therapy to the susceptible agents and infection source.":
        sentences = [part.strip() for part in re.split(r"(?<=[.?!])\s+", therapy_line) if part.strip()]
        if sentences:
            return sentences[-1]
        return therapy_line

    provided_results = parsed.susceptibility_results if parsed is not None else {}
    susceptible_agents = [
        antibiotic
        for antibiotic, call in (provided_results or result.final_results).items()
        if call == "Susceptible"
    ]

    if result.mechanisms:
        first = _clean_mechid_text(result.mechanisms[0]).rstrip(".")
        if susceptible_agents:
            return f"The main takeaway is {first.lower()}, with {_join_readable(susceptible_agents[:2])} looking like the most usable active option(s) from the submitted AST."
        return f"The main takeaway is {first.lower()}."

    if susceptible_agents:
        return f"The main takeaway is that {_join_readable(susceptible_agents[:2])} appears active from the submitted AST."

    return "The main takeaway is that this pattern needs organism-specific interpretation before choosing therapy."


def _build_mechid_review_message(result: MechIDTextAnalyzeResponse, *, final: bool = False) -> str:
    parsed = result.parsed_request
    if parsed is None:
        message = (
            "I could not confidently identify the organism yet. "
            "Paste the organism plus a few susceptibility calls, for example: "
            "'E. coli resistant to ceftriaxone and ciprofloxacin, susceptible to meropenem.'"
        )
        if result.warnings:
            message += " " + result.warnings[0]
        return message

    extracted_target = parsed.organism or _format_mechid_organism_list(parsed.mentioned_organisms)
    summary = f"I extracted {extracted_target} with {_format_mechid_results(parsed.susceptibility_results)}."
    if parsed.resistance_phenotypes:
        summary += f" I also noted {_join_readable(parsed.resistance_phenotypes)}."
    context_bits: List[str] = []
    if parsed.tx_context.focus_detail != "Not specified":
        context_bits.append(parsed.tx_context.focus_detail)
    if parsed.tx_context.syndrome != "Not specified":
        context_bits.append(parsed.tx_context.syndrome)
    if parsed.tx_context.severity != "Not specified":
        context_bits.append(parsed.tx_context.severity)
    if context_bits:
        summary += f" Clinical context: {_join_readable(context_bits)}."
    if parsed.tx_context.carbapenemase_result != "Not specified":
        carbapenemase_summary = parsed.tx_context.carbapenemase_result.lower()
        if parsed.tx_context.carbapenemase_class != "Not specified":
            carbapenemase_summary += f" ({parsed.tx_context.carbapenemase_class})"
        summary += f" Carbapenemase testing: {carbapenemase_summary}."

    if not final:
        if result.analysis is not None:
            summary += "\n\nWhat this supports so far:"
            summary += f"\nLikely resistance pattern captured: {_friendly_mechid_mechanism(result.analysis)}"
            summary += f"\nTreatment-relevant signal captured: {_friendly_mechid_therapy(result.analysis, parsed)}"
            oral_options = _friendly_mechid_oral_options(result.analysis, parsed)
            if oral_options:
                summary += f"\nPossible oral options captured: {oral_options}"
            caution = _friendly_mechid_caution(result.analysis)
            if caution:
                summary += f"\nImportant caution captured: {caution}"
            summary += "\nIf this extraction matches the case, ask for my consultant impression. Otherwise add or correct details."
            return summary

        if result.provisional_advice is not None:
            advice = result.provisional_advice
            summary += "\n\nWhat this supports so far:"
            summary += f"\nCurrent treatment direction captured: {advice.summary}"
            if advice.recommended_options:
                summary += f"\nOptions already supported by the current data: {_join_readable(advice.recommended_options)}."
            if advice.oral_options:
                summary += f"\nPossible oral options already supported: {_join_readable(advice.oral_options)}."
            if advice.notes:
                summary += f"\nImportant context captured: {advice.notes[0]}"
            if advice.missing_susceptibilities:
                follow_up = _assistant_single_ast_follow_up(advice.missing_susceptibilities[0])
                if follow_up:
                    summary += f"\n{follow_up}"
            summary += "\nIf this extraction matches the case, ask for my consultant impression. Otherwise add or correct details."
            return summary

        if result.warnings:
            summary += " " + _join_readable(result.warnings[:2])
        summary += " Add or correct susceptibility details so I can infer a mechanism and therapy plan."
        return summary

    if result.analysis is not None:
        summary += "\n\nMy impression:"
        summary += f"\nBottom line: {_friendly_mechid_bottom_line(result.analysis, parsed)}"
        summary += f"\nPattern: {_friendly_mechid_mechanism(result.analysis)}"
        summary += f"\nTreatment approach: {_friendly_mechid_therapy(result.analysis, parsed)}"
        oral_options = _friendly_mechid_oral_options(result.analysis, parsed)
        if oral_options:
            summary += f"\nPossible oral options: {oral_options}"
        caution = _friendly_mechid_caution(result.analysis)
        if caution:
            summary += f"\nWhat I would watch out for: {caution}"
        if final:
            summary += "\nI left the more technical mechanism details in the analysis panel below."
        else:
            summary += "\nIf that summary matches the case, run the interpretation. Otherwise add or correct details."
        return summary

    if result.provisional_advice is not None:
        advice = result.provisional_advice
        summary += "\n\nMy impression:"
        summary += f"\nBottom line: {advice.summary}"
        if advice.recommended_options:
            summary += f"\nOptions I would consider now: {_join_readable(advice.recommended_options)}."
        if advice.oral_options:
            summary += f"\nPossible oral options: {_join_readable(advice.oral_options)}."
        if advice.notes:
            summary += f"\nWhat I would watch out for: {advice.notes[0]}"
        if advice.missing_susceptibilities:
            summary += (
                f"\nWhat I still need: If you give susceptibilities for {_join_readable(advice.missing_susceptibilities[:5])}, "
                "I can narrow this to a more specific treatment plan."
            )
        if final:
            summary += "\nI can make this more definitive once you add the isolate susceptibilities."
        else:
            summary += "\nAdd susceptibilities if you want me to narrow this to a more specific treatment plan."
        return summary

    if result.warnings:
        summary += " " + _join_readable(result.warnings[:2])
    summary += " Add or correct susceptibility details so I can infer a mechanism and therapy plan."
    return summary


def _assistant_mechid_review_message(
    result: MechIDTextAnalyzeResponse,
    *,
    final: bool = False,
    transient_examples: List[Dict[str, str]] | None = None,
) -> tuple[str, bool]:
    fallback = _build_mechid_review_message(result, final=final)
    if final:
        return narrate_mechid_assistant_message(
            mechid_result=result,
            fallback_message=fallback,
            transient_examples=transient_examples,
        )
    return narrate_mechid_review_message(
        mechid_result=result,
        fallback_message=fallback,
        transient_examples=transient_examples,
    )


def _assistant_review_options_for_case(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    items_by_id = {item.id: item for item in module.items}
    score_options = _assistant_missing_endo_score_options(state)
    missing_item_specs = _top_missing_item_specs(
        module,
        text_result.parsed_request,
        limit=3,
        state=state,
    )
    if score_options:
        options.append(
            AssistantOption(
                value="section:score_review",
                label=f"{(_assistant_selected_endo_score_id(state) or 'Score').upper()} Components To Confirm",
            )
        )
        options.extend(score_options)
    if missing_item_specs:
        if score_options:
            options.append(AssistantOption(value="section:missing_findings", label="Important Missing Findings"))
    for item_id, label in missing_item_specs:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        present_text, absent_text = _assistant_case_item_text(item, module)
        options.append(
            AssistantOption(
                value=f"insert_text:{item_id}",
                label=label,
                description="Use Present or Absent, then click to add this clarification to the draft.",
                insertText=present_text,
                absentText=absent_text,
            )
        )
    options.extend(_assistant_review_options())
    return options


def _assistant_endo_blood_culture_options() -> List[AssistantOption]:
    return [
        AssistantOption(
            value=value,
            label=entry["label"],
            description=entry["description"],
        )
        for value, entry in ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES.items()
    ]


def _assistant_selected_endo_score_id(state: AssistantState | None) -> str | None:
    if state is None or state.module_id != "endo":
        return None
    context = state.endo_blood_culture_context
    if not context:
        return None
    choice = ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES.get(context)
    if choice is None:
        return None
    score_id = choice.get("score_id")
    return str(score_id) if score_id else None


def _assistant_merge_endo_score_factor_ids(state: AssistantState, factor_ids: List[str]) -> None:
    if state.module_id != "endo" or not factor_ids:
        return
    current = list(state.endo_score_factor_ids)
    for factor_id in factor_ids:
        if factor_id in current:
            continue
        for exclusive_group in ENDO_ASSISTANT_EXCLUSIVE_SCORE_GROUPS:
            if factor_id in exclusive_group:
                current = [item for item in current if item not in exclusive_group]
        current.append(factor_id)
    state.endo_score_factor_ids = current


def _assistant_endo_score_component_entries(state: AssistantState | None) -> List[tuple[str, str]]:
    score_id = _assistant_selected_endo_score_id(state)
    if not score_id:
        return []
    return list(ENDO_ASSISTANT_SCORE_COMPONENTS.get(score_id, ()))


def _assistant_missing_endo_score_options(state: AssistantState | None) -> List[AssistantOption]:
    if state is None or state.module_id != "endo":
        return []
    selected = set(state.endo_score_factor_ids)
    score_name = _assistant_selected_endo_score_id(state)
    if not score_name:
        return []
    options: List[AssistantOption] = []
    for score_factor_id, label in _assistant_endo_score_component_entries(state):
        if score_factor_id in selected:
            continue
        options.append(
            AssistantOption(
                value=f"add_score:{score_factor_id}",
                label=label,
                description=f"Click if this {score_name.upper()} component applies.",
            )
        )
    return options


def _assistant_infer_endo_score_factor_ids_from_text(state: AssistantState, text: str) -> List[str]:
    if state.module_id != "endo":
        return []
    score_id = _assistant_selected_endo_score_id(state)
    if not score_id:
        return []
    text_norm = _normalize_choice(text)
    if not text_norm:
        return []
    inferred: List[str] = []
    for score_factor_id, aliases in ENDO_ASSISTANT_SCORE_TEXT_ALIASES.get(score_id, {}).items():
        if any(alias in text_norm for alias in aliases):
            inferred.append(score_factor_id)
    return inferred


def _assistant_pretest_factor_entries(
    module: SyndromeModule,
    state: AssistantState | None = None,
) -> List[tuple[str, str, float]]:
    specs = resolve_pretest_factor_specs(module)
    if module.id == "endo":
        specs = [spec for spec in specs if spec.context_group != "score_overlap"]
    return [(spec.id, spec.label, spec.weight) for spec in specs]


def _module_supports_pretest_factors(module: SyndromeModule | None, state: AssistantState | None = None) -> bool:
    if not module:
        return False
    return bool(_assistant_pretest_factor_entries(module, state) or _assistant_endo_score_component_entries(state))


def _pretest_factor_label(module: SyndromeModule, factor_id: str) -> str:
    spec = next((entry for entry in resolve_pretest_factor_specs(module) if entry.id == factor_id), None)
    if spec is not None:
        return spec.label
    return factor_id.replace("_", " ").title()


def _assistant_pretest_factor_options(
    module: SyndromeModule,
    selected_ids: List[str],
    state: AssistantState | None = None,
) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    selected_set = set(selected_ids)
    available_ids = {factor_id for factor_id, _, _ in _assistant_pretest_factor_entries(module, state)}
    if available_ids:
        if module.id == "endo":
            options.append(AssistantOption(value="section:baseline", label="Baseline Modifiers"))
        for spec in resolve_pretest_factor_specs(module):
            if spec.id in selected_set or spec.id not in available_ids:
                continue
            factor_label = "Baseline host/context factor" if module.id == "endo" else "Baseline risk factor"
            options.append(
                AssistantOption(
                    value=spec.id,
                    label=spec.label,
                    description=f"{factor_label} (OR-like weight {spec.weight:.2f}). Source: {spec.source_note}",
                )
            )
    score_selected_set = set(state.endo_score_factor_ids if state is not None else [])
    score_entries = _assistant_endo_score_component_entries(state)
    if score_entries:
        score_name = (_assistant_selected_endo_score_id(state) or "").upper()
        options.append(AssistantOption(value="section:score", label=f"{score_name} Components"))
        for score_factor_id, label in score_entries:
            if score_factor_id in score_selected_set:
                continue
            options.append(
                AssistantOption(
                    value=score_factor_id,
                    label=label,
                    description=f"Used to auto-calculate the {score_name} score.",
                )
            )
    options.extend(
        [
            AssistantOption(value="continue_to_case", label="Continue consult"),
            AssistantOption(value="restart", label="Start new consult"),
        ]
    )
    return options


def _assistant_selected_factor_labels(module: SyndromeModule, selected_ids: List[str]) -> List[str]:
    return [_pretest_factor_label(module, factor_id) for factor_id in selected_ids]


def _sync_pretest_factor_labels(state: AssistantState, module: SyndromeModule | None) -> None:
    state.pretest_factor_labels = _assistant_selected_factor_labels(module, state.pretest_factor_ids) if module else []


def _assistant_pretest_factor_prompt(
    module: SyndromeModule,
    selected_ids: List[str],
    state: AssistantState | None = None,
) -> str:
    if module.id == "endo":
        context = state.endo_blood_culture_context if state is not None else None
        choice = ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES.get(context or "")
        context_label = choice["label"] if choice else "this blood-culture context"
        score_selected = len(state.endo_score_factor_ids) if state is not None else 0
        if selected_ids or score_selected:
            return (
                "Add more baseline modifiers or score components, or continue to the case details that will update the probability of "
                f"{_assistant_module_label(module)}."
            )
        return (
            "Before we describe the case, let’s capture baseline modifiers and the relevant score components for "
            f"{_assistant_module_label(module)} in {context_label}. Add any that apply, then click Next when you’re ready."
        )
    if selected_ids:
        return "Add more risk factors or continue to tests that have been shown to affect the probability of this syndrome."
    return (
        f"Before we describe the case, let’s capture baseline factors that can raise the pretest probability for "
        f"{_assistant_module_label(module)}. Add any that apply, then click Next when you’re ready."
    )


def _assistant_preset_options(module: SyndromeModule) -> List[AssistantOption]:
    label_map = {
        "Emergency department": "Emergency Dept",
        "Primary care": "Primary Care",
        "ICU, mechanically ventilated >48 hours": "ICU >48h",
        "ICU, mechanically ventilated ≥5 days": "ICU >=5 days",
    }
    return [
        AssistantOption(
            value=p.id,
            label=label_map.get(p.label, p.label),
            description=f"Pretest {round(p.p * 100)}%",
        )
        for p in module.pretest_presets
    ]


def _assistant_lay_preset_prompt(module: SyndromeModule) -> str:
    labels = [preset.label for preset in module.pretest_presets]
    readable = _join_readable(labels[:3]) if labels else "the current setting"
    return (
        f"Before I set the starting probability for {_assistant_module_label(module)}, "
        f"where is the patient being evaluated? In plain terms, is this more like {readable}?"
    )


def _assistant_text_explicitly_supports_preset(
    message_text: str,
    module: SyndromeModule,
) -> bool:
    parsed = parse_text_to_request(
        store=store,
        text=message_text,
        module_hint=module.id,
        include_explanation=False,
    )
    return bool(parsed.parsed_request and parsed.parsed_request.preset_id)


def _select_endo_blood_culture_context_from_turn(req: AssistantTurnRequest) -> str | None:
    selection = (req.selection or "").strip()
    if selection in ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES:
        return selection

    message = _normalize_choice(req.message)
    if not message:
        return None

    keyword_map = {
        "staph": {"staph", "staphylococcus", "staph aureus", "staphylococcus aureus", "s aureus", "saureus", "sab"},
        "strep": {"strep", "streptococcus", "viridans", "viridans group", "vgs", "nbhs", "s gallolyticus"},
        "enterococcus": {"enterococcus", "enterococcal", "e faecalis", "enterococcus faecalis", "efaecalis"},
        "other_unknown_pending": {"other", "unknown", "pending", "not sure", "not yet", "no cultures"},
    }
    for context_id, keywords in keyword_map.items():
        if message == context_id:
            return context_id
        if any(keyword in message for keyword in keywords):
            return context_id
    return None


def _select_endo_score_component_from_turn(state: AssistantState, req: AssistantTurnRequest) -> str | None:
    valid_ids = {score_factor_id for score_factor_id, _ in _assistant_endo_score_component_entries(state)}
    selection = (req.selection or "").strip()
    if selection in valid_ids:
        return selection

    message = _normalize_choice(req.message)
    if not message:
        return None

    for score_factor_id, label in _assistant_endo_score_component_entries(state):
        label_normalized = label.lower()
        if message == score_factor_id.lower() or message == label_normalized or message in label_normalized or label_normalized in message:
            return score_factor_id
    return None


def _assistant_initial_state(req: AssistantTurnRequest) -> AssistantState:
    if req.state is not None:
        state = req.state
    else:
        state = AssistantState()
    if req.parser_strategy is not None:
        state.parser_strategy = req.parser_strategy
    if req.parser_model is not None:
        state.parser_model = req.parser_model.strip() or None
    if req.allow_fallback is not None:
        state.allow_fallback = req.allow_fallback
    _sync_pretest_factor_labels(state, store.get(state.module_id or ""))
    return state


def _normalize_choice(value: str | None) -> str:
    return (value or "").strip().lower()


def _append_case_text(existing: str | None, addition: str | None) -> str:
    current = (existing or "").strip()
    extra = (addition or "").strip()
    if not current:
        return extra
    if not extra:
        return current
    return f"{current}\n{extra}"


def _assistant_probid_controls(state: AssistantState) -> ProbIDControlsInput | None:
    if not state.module_id:
        return None
    if state.module_id == "vap":
        if not state.pretest_factor_ids:
            return None
        return ProbIDControlsInput(
            vapRiskModifiers={
                "enabled": True,
                "selectedIds": state.pretest_factor_ids,
            }
        )
    if state.module_id == "endo":
        selected_score_id = _assistant_selected_endo_score_id(state)
        endo_scores = None
        if selected_score_id == "virsta":
            selected = set(state.endo_score_factor_ids)
            endo_scores = {
                "virsta": {
                    "enabled": True,
                    "emboli": "virsta_emboli" in selected,
                    "meningitis": "virsta_meningitis" in selected,
                    "intracardiacDevice": "virsta_intracardiac_device" in selected,
                    "priorEndocarditis": "virsta_prior_endocarditis" in selected,
                    "nativeValveDisease": "virsta_native_valve_disease" in selected,
                    "ivdu": "virsta_ivdu" in selected,
                    "persistentBacteremia48h": "virsta_persistent_bacteremia_48h" in selected,
                    "vertebralOsteomyelitis": "virsta_vertebral_osteomyelitis" in selected,
                    "acquisition": "community_or_nhca" if "virsta_community_or_nhca" in selected else "nosocomial",
                    "severeSepsisShock": "virsta_severe_sepsis_shock" in selected,
                    "crpGt190": "virsta_crp_gt_190" in selected,
                }
            }
        elif selected_score_id == "denova":
            selected = set(state.endo_score_factor_ids)
            endo_scores = {
                "denova": {
                    "enabled": True,
                    "duration7d": "denova_duration_7d" in selected,
                    "embolization": "denova_embolization" in selected,
                    "numPositive2": "denova_num_positive_2" in selected,
                    "originUnknown": "denova_origin_unknown" in selected,
                    "valveDisease": "denova_valve_disease" in selected,
                    "auscultationMurmur": "denova_auscultation_murmur" in selected,
                }
            }
        elif selected_score_id == "handoc":
            selected = set(state.endo_score_factor_ids)
            species = "unspecified_other"
            if "handoc_species_high_risk" in selected:
                species = "s_gallolyticus_bovis_group"
            elif "handoc_species_anginosus" in selected:
                species = "s_anginosus_group"
            endo_scores = {
                "handoc": {
                    "enabled": True,
                    "heartMurmurValve": "handoc_heart_murmur_valve" in selected,
                    "species": species,
                    "numPositive2": "handoc_num_positive_2" in selected,
                    "duration7d": "handoc_duration_7d" in selected,
                    "onlyOneSpecies": "handoc_only_one_species" in selected,
                    "communityAcquired": "handoc_community_acquired" in selected,
                }
            }
        if not state.pretest_factor_ids and endo_scores is None:
            return None
        payload = {}
        if state.pretest_factor_ids:
            payload["endoRiskModifiers"] = {
                "enabled": True,
                "selectedIds": state.pretest_factor_ids,
            }
        if endo_scores is not None:
            payload["endoScores"] = endo_scores
        return ProbIDControlsInput(**payload)
    return None


def _apply_pretest_factors_to_parsed_request(
    *,
    module: SyndromeModule,
    state: AssistantState,
    parsed_request: AnalyzeRequest | None,
) -> None:
    if parsed_request is None:
        return

    if module.id in {"vap", "endo"}:
        parsed_request.probid_controls = _assistant_probid_controls(state)
        return

    if not state.pretest_factor_ids:
        return

    pretest_factor_specs = {spec.id: spec for spec in resolve_pretest_factor_specs(module)}
    selected_ids = [factor_id for factor_id in state.pretest_factor_ids if factor_id in pretest_factor_specs]
    if not selected_ids:
        return

    raw_multiplier = 1.0
    for factor_id in selected_ids:
        raw_multiplier *= pretest_factor_specs[factor_id].weight
    tuning = get_pretest_factor_tuning(module.id)
    applied_multiplier = min(
        max(pow(raw_multiplier, tuning.shrink_exponent), 1.0),
        tuning.max_multiplier,
    )
    parsed_request.pretest_odds_multiplier *= applied_multiplier

    selected_set = set(selected_ids)
    parsed_request.findings = {
        factor_id: state_value
        for factor_id, state_value in parsed_request.findings.items()
        if factor_id not in selected_set
    }
    parsed_request.ordered_finding_ids = [
        factor_id for factor_id in parsed_request.ordered_finding_ids if factor_id not in selected_set
    ]


def _is_ready_to_assess(req: AssistantTurnRequest) -> bool:
    if (req.selection or "").strip() == "run_assessment":
        return True
    normalized = _normalize_choice(req.message)
    return normalized in {
        "yes",
        "yes run it",
        "run it",
        "analyze",
        "analyse",
        "looks good",
        "thats it",
        "that's it",
        "nothing else",
        "no",
        "nope",
    }


def _top_missing_tests(
    module: SyndromeModule,
    parsed_request: AnalyzeRequest | None,
    limit: int = 3,
    *,
    state: AssistantState | None = None,
) -> List[str]:
    return [label for _, label in _top_missing_item_specs(module, parsed_request, limit=limit, state=state)]


ASSISTANT_MISSING_PRIORITY_BY_MODULE: Dict[str, List[str]] = {
    "cap": [
        "cap_cxr_consolidation",
        "cap_wbc_ge15",
        "cap_procal_high",
        "cap_rvp_pos",
    ],
    "vap": [
        "vap_cxr_infiltrate",
        "vap_bal_qcx",
        "vap_eta_qcx",
        "vap_psb_qcx",
        "vap_leukocytosis",
        "vap_hypoxemia_pf240",
        "vap_pct_elevated",
    ],
    "cdi": [
        "cdi_naat_pos_tox_pos",
        "cdi_naat_pos_tox_neg",
        "cdi_naat_pos_tox_na",
        "cdi_naat_neg",
        "cdi_wbc15",
        "cdi_cr",
    ],
    "uti": [
        "ua_pyuria_pos",
        "ua_le_pos",
        "ua_nit_pos",
        "ua_bact_pos",
        "uti_cx_pos",
    ],
    "active_tb": [
        "tb_cxr_suggestive",
        "tb_ct_suggestive",
        "tb_mtbpcr_sputum",
        "tb_afb_smear_sputum",
        "tb_culture_sputum",
        "tb_mtbpcr_bal",
        "tb_culture_bal",
    ],
    "pjp": [
        "pjp_ct_typical",
        "pjp_cxr_typical",
        "pjp_pcr_bal",
        "pjp_dfa",
        "pjp_bdg_serum",
        "pjp_ldh_high",
    ],
    "inv_candida": [
        "icand_culture_positive",
        "icand_t2candida",
        "icand_bdg_serum",
        "icand_mannan_antimannan",
        "icand_pcr_blood",
    ],
    "inv_mold": [
        "imi_ct_halo_sign",
        "imi_serum_gm_odi10",
        "imi_bal_gm_odi10",
        "imi_aspergillus_pcr_bal",
        "imi_serum_bdg",
        "imi_mucorales_pcr_bal",
        "imi_mucorales_pcr_blood",
    ],
    "pji": [
        "pji_crp",
        "pji_esr",
        "pji_synovial_fluid_culture",
        "pji_alpha_defensin_elisa",
        "pji_alpha_defensin_lateral_flow",
        "pji_xray_supportive",
        "pji_synovial_pcr",
    ],
    "septic_arthritis": [
        "sa_synovial_wbc_ge50k",
        "sa_gram_stain",
        "sa_synovial_culture",
        "sa_crp",
        "sa_synovial_pmn_ge90",
        "sa_blood_culture_positive",
        "sa_ultrasound_effusion",
    ],
    "bacterial_meningitis": [
        "bm_csf_gram_stain",
        "bm_csf_culture",
        "bm_csf_bacterial_pcr",
        "bm_csf_glucose_ratio_low",
        "bm_csf_wbc_ge1000",
        "bm_csf_lactate_high",
        "bm_blood_culture_positive",
    ],
    "encephalitis": [
        "enc_hsv_pcr",
        "enc_mri_temporal",
        "enc_csf_pleocytosis",
        "enc_csf_rbc_high",
        "enc_csf_viral_pcr",
        "enc_eeg_temporal",
    ],
    "spinal_epidural_abscess": [
        "sea_mri_positive",
        "sea_esr_high",
        "sea_crp_high",
        "sea_blood_culture_positive",
        "sea_exam_neuro_deficit",
        "sea_discitis_osteo",
    ],
    "brain_abscess": [
        "ba_mri_dwi_positive",
        "ba_aspirate_culture_positive",
        "ba_ct_ring_enhancing",
        "ba_blood_culture_positive",
        "ba_exam_focal_deficit",
        "ba_imaging_multifocal",
    ],
    "necrotizing_soft_tissue_infection": [
        "nsti_ct_positive",
        "nsti_operative_findings",
        "nsti_vital_hypotension",
        "nsti_exam_bullae_or_necrosis",
        "nsti_crp_high",
        "nsti_lactate_high",
    ],
}


def _assistant_missing_priority_ids(
    module: SyndromeModule,
    state: AssistantState | None = None,
) -> List[str]:
    if module.id != "endo":
        return ASSISTANT_MISSING_PRIORITY_BY_MODULE.get(module.id, [])

    context = state.endo_blood_culture_context if state is not None else None
    if context == "staph":
        micro_priority = [
            "endo_bcx_saureus_multi",
            "endo_bcx_major_persistent",
            "endo_bcx_pos_not_major",
            "endo_bcx_negative",
        ]
    elif context == "strep":
        micro_priority = [
            "endo_bcx_nbhs_multi",
            "endo_bcx_pos_not_major",
            "endo_bcx_negative",
        ]
    elif context == "enterococcus":
        micro_priority = [
            "endo_bcx_efaecalis_multi",
            "endo_bcx_pos_not_major",
            "endo_bcx_negative",
        ]
    else:
        micro_priority = [
            "endo_bcx_major_typical",
            "endo_bcx_major_persistent",
            "endo_bcx_pos_not_major",
            "endo_bcx_negative",
            "endo_coxiella_major",
        ]

    return [
        *micro_priority,
        "endo_tte",
        "endo_tee",
        "endo_pet",
        "endo_esr_crp",
        "endo_anemia",
    ]


def _top_missing_item_specs(
    module: SyndromeModule,
    parsed_request: AnalyzeRequest | None,
    limit: int = 3,
    *,
    state: AssistantState | None = None,
) -> List[tuple[str, str]]:
    seen_ids = set((parsed_request.findings or {}).keys()) if parsed_request is not None else set()
    seen_groups: set[str] = set()
    ranked: List[tuple[float, str, str, str | None]] = []
    items_by_id = {item.id: item for item in module.items}
    suggestions: List[tuple[str, str]] = []
    conflicting_ids = _assistant_conflicting_missing_ids(module, parsed_request)

    for item_id in _assistant_missing_priority_ids(module, state):
        item = items_by_id.get(item_id)
        if item is None or item.id in seen_ids or item.id in conflicting_ids:
            continue
        if item.category not in {"lab", "imaging", "micro"}:
            continue
        if not _assistant_case_item_allowed(module, item, state):
            continue
        if item.group and item.group in seen_groups:
            continue
        suggestions.append((item.id, item.label))
        if item.group:
            seen_groups.add(item.group)
        if len(suggestions) >= limit:
            return suggestions

    for item in module.items:
        if item.id in seen_ids or item.id in conflicting_ids:
            continue
        if item.category not in {"lab", "imaging", "micro"}:
            continue
        if not _assistant_case_item_allowed(module, item, state):
            continue

        strength = 1.0
        if item.lr_pos is not None and item.lr_pos > 0:
            strength = max(strength, item.lr_pos if item.lr_pos >= 1 else (1 / item.lr_pos))
        if item.lr_neg is not None and item.lr_neg > 0:
            strength = max(strength, (1 / item.lr_neg) if item.lr_neg < 1 else item.lr_neg)
        if strength <= 1.1:
            continue
        ranked.append((strength, item.id, item.label, item.group))

    ranked.sort(key=lambda value: value[0], reverse=True)

    for _, item_id, label, group in ranked:
        if group and group in seen_groups:
            continue
        if any(existing_id == item_id for existing_id, _ in suggestions):
            continue
        suggestions.append((item_id, label))
        if group:
            seen_groups.add(group)
        if len(suggestions) >= limit:
            break
    return suggestions


def _assistant_conflicting_missing_ids(
    module: SyndromeModule,
    parsed_request: AnalyzeRequest | None,
) -> set[str]:
    if parsed_request is None or not parsed_request.findings:
        return set()

    present_ids = {
        item_id
        for item_id, state in parsed_request.findings.items()
        if state == "present"
    }
    if not present_ids:
        return set()

    conflicts: set[str] = set()

    if module.id == "endo":
        endo_major_positive_ids = {
            "endo_bcx_major_typical",
            "endo_bcx_major_persistent",
            "endo_bcx_saureus_multi",
            "endo_bcx_efaecalis_multi",
            "endo_bcx_nbhs_multi",
            "endo_coxiella_major",
        }
        endo_nonmajor_or_negative_ids = {
            "endo_bcx_pos_not_major",
            "endo_bcx_negative",
            "endo_micro_na",
        }
        if present_ids & endo_major_positive_ids:
            conflicts.update(endo_nonmajor_or_negative_ids)
        if "endo_bcx_pos_not_major" in present_ids:
            conflicts.update((endo_major_positive_ids | {"endo_bcx_negative", "endo_micro_na"}) - {"endo_bcx_pos_not_major"})
        if "endo_bcx_negative" in present_ids:
            conflicts.update(endo_major_positive_ids | {"endo_bcx_pos_not_major", "endo_micro_na"})

    if module.id == "cdi":
        cdi_test_ids = {
            "cdi_naat_pos_tox_pos",
            "cdi_naat_pos_tox_neg",
            "cdi_naat_pos_tox_na",
            "cdi_naat_neg",
            "cdi_test_na",
        }
        chosen_cdi_test_ids = present_ids & cdi_test_ids
        if chosen_cdi_test_ids:
            conflicts.update(cdi_test_ids - chosen_cdi_test_ids)

    return conflicts


def _lr_strength(item) -> float:
    strength = 1.0
    if item.lr_pos is not None and item.lr_pos > 0:
        strength = max(strength, item.lr_pos if item.lr_pos >= 1 else (1 / item.lr_pos))
    if item.lr_neg is not None and item.lr_neg > 0:
        strength = max(strength, (1 / item.lr_neg) if item.lr_neg < 1 else item.lr_neg)
    return strength


def _assistant_case_item_text(item, module: SyndromeModule | None = None) -> tuple[str, str]:
    alias_groups = COMMON_FINDING_ALIASES.get(item.id, {})
    explicit_present = [phrase for phrase in alias_groups.get("present", []) if phrase and phrase.strip()]
    explicit_absent = [phrase for phrase in alias_groups.get("absent", []) if phrase and phrase.strip()]
    if explicit_present:
        present_text = explicit_present[0]
        absent_text = explicit_absent[0] if explicit_absent else f"No {present_text}"
        return present_text, absent_text
    if module is not None:
        module_overrides = ASSISTANT_CASE_TEXT_OVERRIDES.get(module.id, {})
        override = module_overrides.get(item.id)
        if override is not None:
            return override
    return item.label, f"No {item.label}"


def _assistant_case_item_by_id(module: SyndromeModule, item_id: str):
    return next((item for item in module.items if item.id == item_id), None)


def _assistant_case_item_allowed(module: SyndromeModule, item, state: AssistantState | None = None) -> bool:
    label_lower = item.label.lower()
    if "not done" in label_lower or "unknown" in label_lower:
        return False
    if module.id == "endo":
        if item.id in ENDO_ASSISTANT_SCORE_ITEM_IDS:
            return False
        if item.category == "micro":
            context = state.endo_blood_culture_context if state is not None else None
            allowed_micro_ids = {
                "staph": {
                    "endo_bcx_saureus_multi",
                    "endo_bcx_major_persistent",
                    "endo_bcx_pos_not_major",
                    "endo_bcx_negative",
                },
                "strep": {"endo_bcx_nbhs_multi", "endo_bcx_pos_not_major", "endo_bcx_negative"},
                "enterococcus": {"endo_bcx_efaecalis_multi", "endo_bcx_pos_not_major", "endo_bcx_negative"},
                "other_unknown_pending": {
                    "endo_bcx_major_typical",
                    "endo_bcx_major_persistent",
                    "endo_bcx_pos_not_major",
                    "endo_bcx_negative",
                    "endo_coxiella_major",
                },
            }
            allowed = allowed_micro_ids.get(context or "", set())
            if allowed and item.id not in allowed:
                return False
    return True


def _assistant_case_sections_for_module(module: SyndromeModule, state: AssistantState | None = None) -> List[str]:
    sections: List[str] = []
    for section in ENDO_CASE_SECTION_ORDER:
        if _assistant_case_prompt_options(module, state, section_override=section):
            sections.append(section)
    return sections


def _assistant_next_case_section(module: SyndromeModule, current: str | None, state: AssistantState | None = None) -> str | None:
    sections = _assistant_case_sections_for_module(module, state)
    if not sections:
        return None
    if current not in sections:
        return sections[0]
    idx = sections.index(current)
    if idx + 1 >= len(sections):
        return None
    return sections[idx + 1]


def _assistant_case_section_prompt(module: SyndromeModule, section: str | None) -> tuple[str, List[str]]:
    if section == "exam_vitals":
        return (
            "Start with vital signs and physical examination findings. Add what is present or absent, then click Next to move to laboratory findings.",
            [
                "This section is for symptoms, exam findings, and bedside clinical features.",
                "When you are done with this section, click Next to move forward.",
            ],
        )
    if section == "lab":
        if module.id == "endo":
            return (
                "Now add laboratory findings. The organism-specific score is already handled separately, so only enter non-score lab findings here.",
                [
                    "Use this section for non-score labs such as ESR/CRP or anemia.",
                    "Click Next when you are ready to move to microbiology.",
                ],
            )
        return (
            "Now add laboratory findings. Then click Next to move to microbiology.",
            [
                "Use this section for laboratory findings relevant to this syndrome.",
                "Click Next when you are ready to move to microbiology.",
            ],
        )
    if section == "micro":
        if module.id == "endo":
            return (
                "Now add microbiology findings for the selected organism pathway. Then click Next to move to radiology.",
                [
                    "Use the filtered microbiology options for blood-culture pattern and related microbiology.",
                    "Click Next when you are ready to move to radiology.",
                ],
            )
        return (
            "Now add microbiology findings. Then click Next to move to radiology.",
            [
                "Use this section for culture, PCR, antigen, or other microbiology data.",
                "Click Next when you are ready to move to radiology.",
            ],
        )
    if section == "imaging":
        return (
            "Now add radiology and advanced imaging findings. After this section, I will review the full case before running the assessment.",
            [
                    "Use this section for imaging, TTE/TEE/PET-CT where applicable, and other radiology results.",
                    "Click Next when you are ready for the review screen.",
                ],
        )
    return (
        "Describe the case in plain language.",
        ["Add details, then continue."],
    )


def _assistant_case_prompt_options(
    module: SyndromeModule,
    state: AssistantState | None = None,
    *,
    section_override: str | None = None,
) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    if section_override:
        if section_override == "exam_vitals":
            group_specs = [("Symptoms", {"symptom"}), ("Vital Signs", {"vital"}), ("Physical Exam", {"exam"})]
        elif section_override == "lab":
            group_specs = [("Laboratory Tests", {"lab"})]
        elif section_override == "micro":
            group_specs = [("Microbiology", {"micro"})]
        elif section_override == "imaging":
            group_specs = [("Radiology", {"imaging"})]
        else:
            group_specs = []
    else:
        group_specs = [
            ("Symptoms", {"symptom"}),
            ("Vital Signs", {"vital"}),
            ("Physical Exam", {"exam"}),
            ("Laboratory Tests", {"lab"}),
            ("Microbiology", {"micro"}),
            ("Radiology", {"imaging"}),
        ]

    for group_label, categories in group_specs:
        candidates = []
        for item in module.items:
            if item.category not in categories:
                continue
            if not _assistant_case_item_allowed(module, item, state):
                continue
            candidates.append(item)
        candidates.sort(key=lambda item: (-_lr_strength(item), item.label))
        if not candidates:
            continue
        options.append(
            AssistantOption(
                value=f"section:{group_label}",
                label=group_label,
                description=None,
            )
        )
        if module.id == "active_tb" and group_label == "Symptoms":
            for helper_id, helper_label, insert_text, absent_text in ACTIVE_TB_WHO_SYMPTOM_HELPERS:
                options.append(
                    AssistantOption(
                        value=f"insert_text:{helper_id}",
                        label=helper_label,
                        description="WHO TB symptom shortcut. Click to add this symptom to the draft.",
                        insertText=insert_text,
                        absentText=absent_text,
                    )
                )
        for item in candidates:
            present_text, absent_text = _assistant_case_item_text(item, module)
            options.append(
                AssistantOption(
                    value=f"insert_text:{item.id}",
                    label=item.label,
                    description="Click to add this to your draft.",
                    insertText=present_text,
                    absentText=absent_text,
                )
            )
    options.append(
        AssistantOption(
            value="continue_case_draft",
            label="Continue consult",
            description="Move on with the case details you have drafted.",
        )
    )
    return options


def _build_case_review_message(module: SyndromeModule, text_result: TextAnalyzeResponse, state: AssistantState) -> str:
    understood = text_result.understood
    placeholder_tokens = {"none", "none.", "no", "n/a", "na"}

    present_findings = [
        finding for finding in understood.findings_present if finding and finding.strip().lower() not in placeholder_tokens
    ]
    absent_findings = [
        finding for finding in understood.findings_absent if finding and finding.strip().lower() not in placeholder_tokens
    ]
    present_summary = _join_readable(present_findings) if present_findings else "no clear supporting findings yet"
    summary = f"What I extracted so far for {_assistant_module_label(module)}: {present_summary}."
    if absent_findings:
        summary = (
            f"What I extracted so far for {_assistant_module_label(module)}: {present_summary}. "
            f"I also captured {_join_readable(absent_findings)} as absent or negative."
        )

    if _assistant_case_is_consult_ready(module, text_result, state):
        summary += " " + _assistant_ready_for_consult_message(module, text_result, state)
    else:
        single_follow_up = _assistant_single_case_follow_up(module, text_result.parsed_request, state=state)
        if single_follow_up:
            summary += " " + single_follow_up
    score_name = _assistant_selected_endo_score_id(state)
    if module.id == "endo" and score_name and _assistant_missing_endo_score_options(state):
        summary += f" I also listed the remaining {score_name.upper()} components below so you can tighten that score before I run the assessment."

    if _assistant_case_is_consult_ready(module, text_result, state):
        return summary
    if text_result.requires_confirmation:
        summary += " If anything looks off, correct it or add more case detail. If this extraction matches the case, ask for my consultant impression."
    else:
        summary += " If this extraction matches the case, ask for my consultant impression. Otherwise, add more case detail."
    return summary


def _assistant_probid_review_message(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
    *,
    prefix: str | None = None,
) -> tuple[str, bool]:
    fallback = (prefix or "") + _build_case_review_message(module, text_result, state)
    return narrate_probid_review_message(
        text_result=text_result,
        fallback_message=fallback,
        module_label=_assistant_module_label(module),
    )


def _assistant_parse_case_text(module: SyndromeModule, state: AssistantState) -> TextAnalyzeResponse:
    text_result = analyze_text(
        TextAnalyzeRequest(
            text=state.case_text or "",
            moduleHint=state.module_id,
            presetHint=state.preset_id,
            parserStrategy=state.parser_strategy,
            parserModel=state.parser_model,
            allowFallback=state.allow_fallback,
            runAnalyze=False,
            includeExplanation=True,
        )
    )
    _apply_pretest_factors_to_parsed_request(module=module, state=state, parsed_request=text_result.parsed_request)
    _sync_text_result_references(
        text_result=text_result,
        module=module,
        selected_pretest_factor_ids=state.pretest_factor_ids,
    )
    return text_result


def _assistant_infer_endo_blood_culture_context(
    text_result: TextAnalyzeResponse,
    message_text: str,
) -> str:
    findings = set((text_result.parsed_request.findings or {}).keys()) if text_result.parsed_request is not None else set()
    if "endo_bcx_saureus_multi" in findings:
        return "staph"
    if "endo_bcx_nbhs_multi" in findings:
        return "strep"
    if "endo_bcx_efaecalis_multi" in findings:
        return "enterococcus"

    message_norm = _normalize_choice(message_text)
    if any(token in message_norm for token in {"staphylococcus aureus", "s aureus", "staph aureus", "mrsa", "mssa"}):
        return "staph"
    if any(token in message_norm for token in {"viridans", "strep sanguinis", "strep mitis", "strep gordonii"}):
        return "strep"
    if "enterococcus" in message_norm or "e faecalis" in message_norm:
        return "enterococcus"
    return "other_unknown_pending"


def _assistant_intake_case_from_text(
    req: AssistantTurnRequest,
    state: AssistantState,
    *,
    module_hint: str | None = None,
    preset_hint: str | None = None,
) -> AssistantTurnResponse | None:
    return _assistant_start_case_from_text(
        (req.message or "").strip(),
        state,
        module_hint=module_hint,
        preset_hint=preset_hint,
    )


def _assistant_preview_case_from_text(
    message_text: str,
    state: AssistantState,
    *,
    module_hint: str | None = None,
    preset_hint: str | None = None,
    require_high_confidence: bool = False,
) -> tuple[TextAnalyzeResponse, SyndromeModule, str | None, List[str]] | None:
    if not message_text:
        return None

    text_result = analyze_text(
        TextAnalyzeRequest(
            text=message_text,
            moduleHint=module_hint,
            presetHint=preset_hint,
            parserStrategy=state.parser_strategy,
            parserModel=state.parser_model,
            allowFallback=state.allow_fallback,
            runAnalyze=False,
            includeExplanation=True,
        )
    )
    if text_result.parsed_request is None:
        return None

    module = store.get(text_result.parsed_request.module_id or module_hint or "")
    if module is None:
        return None

    inferred_context: str | None = None
    inferred_score_factor_ids: List[str] = []
    if module.id == "endo":
        inferred_context = state.endo_blood_culture_context or _assistant_infer_endo_blood_culture_context(text_result, message_text)
        preview_state = state.model_copy(deep=True)
        preview_state.module_id = "endo"
        preview_state.endo_blood_culture_context = inferred_context
        inferred_score_factor_ids = _assistant_infer_endo_score_factor_ids_from_text(preview_state, message_text)

    if not text_result.parsed_request.findings and not inferred_score_factor_ids:
        return None

    if require_high_confidence:
        finding_count = len(text_result.parsed_request.findings or {})
        if not _assistant_message_explicitly_mentions_module(message_text, module) and finding_count < 2 and not inferred_score_factor_ids:
            return None

    return text_result, module, inferred_context, inferred_score_factor_ids


def _assistant_start_case_from_text(
    message_text: str,
    state: AssistantState,
    *,
    module_hint: str | None = None,
    preset_hint: str | None = None,
) -> AssistantTurnResponse | None:
    preview = _assistant_preview_case_from_text(
        message_text,
        state,
        module_hint=module_hint,
        preset_hint=preset_hint,
    )
    if preview is None:
        return None

    text_result, module, inferred_context, inferred_score_factor_ids = preview
    explicit_preset_supported = bool(preset_hint) or _assistant_text_explicitly_supports_preset(message_text, module)
    if module.pretest_presets and (not text_result.parsed_request.preset_id or not explicit_preset_supported):
        state.module_id = module.id
        state.workflow = "probid"
        state.preset_id = None
        state.case_text = message_text
        state.mechid_text = None
        state.case_section = None
        state.stage = "select_preset"
        if module.id == "endo":
            state.endo_blood_culture_context = inferred_context
            _assistant_merge_endo_score_factor_ids(state, inferred_score_factor_ids)
        else:
            state.endo_blood_culture_context = None
            state.endo_score_factor_ids = []
        _sync_pretest_factor_labels(state, module)
        return AssistantTurnResponse(
            assistantMessage=_assistant_lay_preset_prompt(module),
            assistantNarrationRefined=False,
            state=state,
            options=_assistant_preset_options(module),
            analysis=text_result,
            tips=[
                "You can answer in plain language, for example 'this is in the ED' or 'already inpatient on the floor'.",
                "Once I have the setting, I will keep the rest of the case details you already gave me.",
            ],
        )

    state.module_id = module.id
    state.workflow = "probid"
    state.preset_id = text_result.parsed_request.preset_id or state.preset_id
    state.pending_intake_text = None
    state.case_text = message_text
    state.mechid_text = None
    state.case_section = None
    if module.id == "endo":
        if not state.endo_blood_culture_context:
            state.endo_blood_culture_context = inferred_context
        _assistant_merge_endo_score_factor_ids(state, inferred_score_factor_ids)
    else:
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
    _sync_pretest_factor_labels(state, module)
    _apply_pretest_factors_to_parsed_request(module=module, state=state, parsed_request=text_result.parsed_request)
    _sync_text_result_references(
        text_result=text_result,
        module=module,
        selected_pretest_factor_ids=state.pretest_factor_ids,
    )
    state.stage = "confirm_case"
    review_message, narration_refined = _assistant_probid_review_message(
        module,
        text_result,
        state,
        prefix="I parsed your case description and pre-populated the calculator inputs. ",
    )
    return AssistantTurnResponse(
        assistantMessage=review_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=_assistant_review_options_for_case(module, text_result, state),
        analysis=text_result,
        tips=[
            "Reply with the single follow-up detail I asked for, in normal words, and I will keep the case moving.",
            "If the extraction already looks right, ask for my consultant impression.",
        ],
    )


def _assistant_is_mechid_intent(message: str | None) -> bool:
    text = _normalize_choice(message)
    if not text:
        return False
    try:
        parsed = parse_mechid_text(message or "")
    except MechIDEngineError:
        parsed = None
    if parsed is not None and bool(
        parsed.get("organism")
        or parsed.get("mentionedOrganisms")
        or parsed.get("resistancePhenotypes")
        or parsed.get("susceptibilityResults")
    ):
        return True
    if any(token in text for token in MECHID_INTENT_TOKENS):
        return True
    if any(token in text for token in MECHID_THERAPY_INTENT_TOKENS):
        return True
    return False


def _assistant_intake_mechid_from_text(req: AssistantTurnRequest, state: AssistantState) -> AssistantTurnResponse | None:
    return _assistant_start_mechid_from_text((req.message or "").strip(), state)


def _assistant_preview_mechid_from_text(
    message_text: str,
    state: AssistantState,
) -> MechIDTextAnalyzeResponse | None:
    if not message_text or not _assistant_is_mechid_intent(message_text):
        return None

    mechid_result = _build_mechid_text_response(
        message_text,
        parser_strategy=state.parser_strategy,
        parser_model=state.parser_model,
        allow_fallback=state.allow_fallback,
    )
    parsed = mechid_result.parsed_request
    if parsed is None:
        return None
    if not (
        parsed.organism
        or parsed.mentioned_organisms
        or parsed.resistance_phenotypes
        or parsed.susceptibility_results
    ):
        return None
    return mechid_result


def _assistant_start_mechid_from_text(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    mechid_result = _assistant_preview_mechid_from_text(message_text, state)
    if mechid_result is None:
        return None

    state.workflow = "mechid"
    state.stage = "mechid_confirm"
    state.module_id = None
    state.preset_id = None
    state.pending_intake_text = None
    state.case_section = None
    state.case_text = None
    state.pretest_factor_ids = []
    state.pretest_factor_labels = []
    state.endo_blood_culture_context = None
    state.endo_score_factor_ids = []
    state.mechid_text = message_text
    review_message, narration_refined = _assistant_mechid_review_message(mechid_result)
    return AssistantTurnResponse(
        assistantMessage=review_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=_assistant_mechid_review_options(mechid_result),
        mechidAnalysis=mechid_result,
        tips=[
            "Reply with the next susceptibility or context detail I asked for, in normal words, and I will update the case.",
            "If the extraction already looks right, ask for my consultant impression.",
        ],
    )


def _select_module_from_turn(req: AssistantTurnRequest) -> str | None:
    sel = (req.selection or "").strip()
    if sel == MECHID_ASSISTANT_ID:
        return sel
    if sel and store.get(sel):
        return sel

    msg = (req.message or "").strip()
    if not msg:
        return None
    if _assistant_is_mechid_intent(msg):
        return MECHID_ASSISTANT_ID

    # Reuse text parser module inference for typed natural-language syndrome selection.
    parsed = parse_text_to_request(
        store=store,
        text=msg,
        include_explanation=False,
    )
    if parsed.understood.module_id:
        return parsed.understood.module_id
    return None


def _select_consult_focus_from_turn(req: AssistantTurnRequest) -> str | None:
    selection = (req.selection or "").strip()
    if selection in {"focus_resistance", "focus_syndrome", "focus_both"}:
        return selection

    message = _normalize_choice(req.message)
    if not message:
        return None
    if message in {"resistance", "mechid", "isolate", "ast", "mechanism"}:
        return "focus_resistance"
    if message in {"syndrome", "probid", "probability", "diagnosis"}:
        return "focus_syndrome"
    if message in {"both", "both please", "do both", "both of them"}:
        return "focus_both"
    return None


def _assistant_start_pending_followup(state: AssistantState) -> AssistantTurnResponse | None:
    pending_text = (state.pending_followup_text or "").strip()
    pending_workflow = state.pending_followup_workflow
    if not pending_text or pending_workflow is None:
        return None

    state.pending_followup_workflow = None
    state.pending_followup_text = None
    if pending_workflow == "mechid":
        response = _assistant_start_mechid_from_text(pending_text, state)
        if response is not None:
            response.assistant_message = (
                "I carried the same case into the resistance lane. " + response.assistant_message
            )
        return response

    response = _assistant_start_case_from_text(pending_text, state)
    if response is not None:
        response.assistant_message = (
            "I carried the same case into the syndrome lane. " + response.assistant_message
        )
    return response


def _select_preset_from_turn(module: SyndromeModule, req: AssistantTurnRequest) -> str | None:
    candidates = {p.id: p for p in module.pretest_presets}
    sel = (req.selection or "").strip()
    if sel in candidates:
        return sel

    msg = _normalize_choice(req.message)
    if not msg:
        return None

    for p in module.pretest_presets:
        if msg == p.id.lower():
            return p.id
        if p.label.lower() in msg or msg in p.label.lower():
            return p.id

    parsed = parse_text_to_request(
        store=store,
        text=req.message or "",
        module_hint=module.id,
        include_explanation=False,
    )
    if parsed.parsed_request and parsed.parsed_request.preset_id:
        return parsed.parsed_request.preset_id
    return None


def _select_pretest_factor_from_turn(module: SyndromeModule, req: AssistantTurnRequest) -> str | None:
    selection = (req.selection or "").strip()
    state = req.state
    valid_ids = {factor_id for factor_id, _, _ in _assistant_pretest_factor_entries(module, state)}
    if selection in valid_ids:
        return selection

    message = _normalize_choice(req.message)
    if not message:
        return None

    for factor_id, label, _ in _assistant_pretest_factor_entries(module, state):
        label = label.lower()
        if message == factor_id.lower() or message == label or message in label or label in message:
            return factor_id
    return None


@app.post("/v1/assistant/turn", response_model=AssistantTurnResponse)
def assistant_turn(req: AssistantTurnRequest) -> AssistantTurnResponse:
    state = _assistant_initial_state(req)
    user_text = _normalize_choice(req.message or req.selection)
    restart_requested = user_text in {"restart", "start over", "reset", "new case"}

    if restart_requested:
        state = AssistantState(
            workflow="probid",
            caseText=None,
            mechidText=None,
            pendingIntakeText=None,
            pendingFollowupWorkflow=None,
            pendingFollowupText=None,
            endoScoreFactorIds=[],
            caseSection=None,
            pretestFactorIds=[],
            pretestFactorLabels=[],
            parserStrategy=state.parser_strategy,
            parserModel=state.parser_model,
            allowFallback=state.allow_fallback,
        )

    if state.stage == "select_module":
        message_text = (req.message or "").strip()
        if message_text:
            probid_preview = _assistant_preview_case_from_text(message_text, state, require_high_confidence=True)
            mechid_preview = _assistant_preview_mechid_from_text(message_text, state)
            if probid_preview is not None and mechid_preview is not None:
                _, module, _, _ = probid_preview
                state.stage = "select_consult_focus"
                state.pending_intake_text = message_text
                state.module_id = module.id
                return AssistantTurnResponse(
                    assistantMessage=(
                        f"I extracted both an isolate/resistance pattern and a syndrome signal for {_assistant_module_label(module)}. "
                        "Do you want me to start with resistance, with the syndrome, or work through both step by step?"
                    ),
                    state=state,
                    options=_assistant_consult_focus_options(),
                    analysis=probid_preview[0],
                    mechidAnalysis=mechid_preview,
                    tips=[
                        "Choose resistance, syndrome, or both.",
                        "If you choose both, I will keep the same case text and carry it from one lane to the other.",
                    ],
                )

        direct_mechid_response = _assistant_intake_mechid_from_text(req, state)
        if direct_mechid_response is not None:
            return direct_mechid_response

        direct_case_response = _assistant_intake_case_from_text(req, state)
        if direct_case_response is not None:
            return direct_case_response

        chosen_module_id = _select_module_from_turn(req)
        if chosen_module_id:
            if chosen_module_id == MECHID_ASSISTANT_ID:
                state.workflow = "mechid"
                state.stage = "mechid_describe"
                state.module_id = None
                state.preset_id = None
                state.case_section = None
                state.case_text = None
                state.mechid_text = None
                state.pretest_factor_ids = []
                state.pretest_factor_labels = []
                state.endo_blood_culture_context = None
                state.endo_score_factor_ids = []
                return AssistantTurnResponse(
                    assistantMessage=(
                        "Paste the organism and susceptibility pattern in plain language. "
                        "For example: 'E. coli resistant to ceftriaxone and ciprofloxacin, susceptible to meropenem, bloodstream infection in septic shock.'"
                    ),
                    state=state,
                    options=[AssistantOption(value="restart", label="Start new consult")],
                    tips=[
                        "I can extract the organism, AST pattern, and basic treatment context from free text.",
                        "Ask for likely resistance mechanism, therapy, or both.",
                    ],
                )

            state.workflow = "probid"
            state.module_id = chosen_module_id
            state.mechid_text = None
            state.endo_blood_culture_context = None
            state.endo_score_factor_ids = []
            state.case_section = None
            module = store.get(chosen_module_id)
            if module is None:
                raise HTTPException(status_code=400, detail=f"Selected module '{chosen_module_id}' not found")
            state.stage = "select_preset"
            return AssistantTurnResponse(
                assistantMessage=(
                    f"Great, we’ll work on {_assistant_module_label(module)}. "
                    "Which setting/pretest context fits this case best?"
                ),
                state=state,
                options=_assistant_preset_options(module),
                tips=[
                    "You can click an option or type something like 'ED', 'ICU', or the preset name.",
                    "Type 'restart' anytime to begin a new consult.",
                ],
            )

        return AssistantTurnResponse(
            assistantMessage=(
                "I’m your ID Consultant Assistant. You can describe a syndrome case in plain language, or choose the resistance mechanism pathway if you want organism plus AST interpretation."
            ),
            state=state,
            options=_assistant_module_options(),
            tips=[
                "I can either run a ProbID syndrome workup or a MechID resistance-mechanism interpretation.",
            ],
        )

    if state.stage == "select_consult_focus":
        selected_focus = _select_consult_focus_from_turn(req)
        pending_text = (state.pending_intake_text or "").strip()
        if not pending_text:
            state.stage = "select_module"
            state.module_id = None
            return AssistantTurnResponse(
                assistantMessage="I lost the original case text. Paste the case again and I will sort out resistance, syndrome, or both.",
                state=state,
                options=_assistant_module_options(),
            )

        if selected_focus == "focus_resistance":
            state.pending_followup_workflow = None
            state.pending_followup_text = None
            return _assistant_start_mechid_from_text(pending_text, state) or AssistantTurnResponse(
                assistantMessage="I could not start the resistance lane from that text yet. Add the organism plus a few susceptibility calls.",
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
            )

        if selected_focus == "focus_syndrome":
            state.pending_followup_workflow = None
            state.pending_followup_text = None
            return _assistant_start_case_from_text(pending_text, state) or AssistantTurnResponse(
                assistantMessage="I could not start the syndrome lane from that text yet. Add a few syndrome-defining findings or test results.",
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
            )

        if selected_focus == "focus_both":
            state.pending_followup_workflow = "probid"
            state.pending_followup_text = pending_text
            response = _assistant_start_mechid_from_text(pending_text, state)
            if response is not None:
                response.assistant_message = (
                    "I’ll start with the resistance side first, then we can carry the same case into the syndrome workup. "
                    + response.assistant_message
                )
                return response
            return AssistantTurnResponse(
                assistantMessage="I could not start the resistance side yet. Add the organism plus a few susceptibility calls, or choose syndrome first.",
                state=state,
                options=_assistant_consult_focus_options(),
            )

        return AssistantTurnResponse(
            assistantMessage="I found both an isolate/resistance pattern and a syndrome signal. Do you want resistance, syndrome, or both?",
            state=state,
            options=_assistant_consult_focus_options(),
            tips=[
                "Choose resistance, syndrome, or both.",
                "If you choose both, I will carry the same text forward so you do not need to paste it again.",
            ],
        )

    if state.stage == "mechid_describe":
        message_text = (req.message or "").strip()
        if not message_text:
            return AssistantTurnResponse(
                assistantMessage=(
                    "Paste the organism and susceptibility pattern in plain language. "
                    "For example: 'Klebsiella pneumoniae resistant to ceftriaxone, susceptible to meropenem and amikacin.'"
                ),
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
                tips=[
                    "Mention the organism plus at least a few antibiotics.",
                    "You can also include the syndrome or severity, such as pneumonia or septic shock.",
                ],
            )

        state.workflow = "mechid"
        state.mechid_text = _append_case_text(state.mechid_text, message_text)
        state.stage = "mechid_confirm"
        mechid_result = _build_mechid_text_response(
            state.mechid_text,
            parser_strategy=state.parser_strategy,
            parser_model=state.parser_model,
            allow_fallback=state.allow_fallback,
        )
        review_message, narration_refined = _assistant_mechid_review_message(mechid_result)
        return AssistantTurnResponse(
            assistantMessage=review_message,
            assistantNarrationRefined=narration_refined,
            state=state,
            options=_assistant_mechid_review_options(mechid_result),
            mechidAnalysis=mechid_result,
            tips=[
                "Answer the single follow-up question in one line if that is faster than using the buttons.",
                "Add more AST details, syndrome context, or severity if anything is missing or wrong.",
            ],
        )

    if state.stage == "mechid_confirm":
        if not state.mechid_text:
            state.stage = "mechid_describe"
            return AssistantTurnResponse(
                assistantMessage="Paste the organism and susceptibility pattern, and I’ll interpret the likely mechanism and therapy implications.",
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
                tips=["Include the organism plus resistant or susceptible calls for named antibiotics."],
            )

        if req.message and req.message.strip():
            state.mechid_text = _append_case_text(state.mechid_text, req.message)

        mechid_result = _build_mechid_text_response(
            state.mechid_text,
            parser_strategy=state.parser_strategy,
            parser_model=state.parser_model,
            allow_fallback=state.allow_fallback,
        )

        if req.selection == "add_more_details" and not (req.message and req.message.strip()):
            state.stage = "mechid_describe"
            return AssistantTurnResponse(
                assistantMessage=(
                    "Add any other susceptibility details, organism clarifications, or treatment context you want me to factor in."
                ),
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
                mechidAnalysis=mechid_result,
                tips=[
                    "Useful additions are more AST calls, syndrome context, severity, or source information.",
                ],
            )

        if _is_ready_to_assess(req):
            if mechid_result.analysis is None and mechid_result.provisional_advice is None:
                review_message, narration_refined = _assistant_mechid_review_message(mechid_result)
                return AssistantTurnResponse(
                    assistantMessage=review_message,
                    assistantNarrationRefined=narration_refined,
                    state=state,
                    options=_assistant_mechid_review_options(mechid_result),
                    mechidAnalysis=mechid_result,
                    tips=[
                        "I still need a clearer organism and susceptibility pattern before I can finalize the interpretation.",
                    ],
                )
            state.stage = "done"
            narrated_message, narration_refined = _assistant_mechid_review_message(mechid_result, final=True)
            done_options = [
                AssistantOption(value="add_more_details", label="Update this case"),
                AssistantOption(value="restart", label="Start new consult"),
            ]
            done_tips = [
                "Add another susceptibility, test result, or clinical detail anytime and I will update the same case.",
                "Review the mechanism, cautions, therapy notes, and references in the analysis panel.",
            ]
            if state.pending_followup_workflow == "probid" and (state.pending_followup_text or "").strip():
                done_options.insert(0, AssistantOption(value="continue_to_syndrome", label="Continue to syndrome"))
                done_tips.insert(0, "If you want, I can carry the same case into the syndrome workup next.")
            return AssistantTurnResponse(
                assistantMessage=narrated_message,
                assistantNarrationRefined=narration_refined,
                state=state,
                options=done_options,
                mechidAnalysis=mechid_result,
                tips=done_tips,
            )

        return AssistantTurnResponse(
            assistantMessage=_assistant_concise_mechid_follow_up(
                mechid_result,
                latest_message=req.message,
            ),
            assistantNarrationRefined=False,
            state=state,
            options=_assistant_mechid_review_options(mechid_result),
            mechidAnalysis=mechid_result,
            tips=[
                "Keep replying in normal words and I will keep the case moving one question at a time.",
                "If the extraction looks right already, ask for my consultant impression.",
            ],
        )

    if state.stage == "select_preset":
        if not state.module_id:
            state.stage = "select_module"
            return AssistantTurnResponse(
                assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                state=state,
                options=_assistant_module_options(),
            )

        module = store.get(state.module_id)
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        direct_case_response = _assistant_intake_case_from_text(
            req,
            state,
            module_hint=state.module_id,
            preset_hint=state.preset_id,
        )
        if direct_case_response is not None:
            return direct_case_response

        chosen_preset_id = _select_preset_from_turn(module, req)
        if chosen_preset_id:
            state.preset_id = chosen_preset_id
            state.endo_blood_culture_context = None
            state.endo_score_factor_ids = []
            state.case_section = None
            state.pretest_factor_ids = []
            state.pretest_factor_labels = []
            preset = next((p for p in module.pretest_presets if p.id == chosen_preset_id), None)
            existing_case_text = (state.case_text or "").strip()
            if existing_case_text:
                return _assistant_start_case_from_text(
                    existing_case_text,
                    state,
                    module_hint=state.module_id,
                    preset_hint=chosen_preset_id,
                ) or AssistantTurnResponse(
                    assistantMessage=(
                        f"I’ll use {preset.label if preset else chosen_preset_id}. "
                        "I still need a little more clinical detail before I can continue."
                    ),
                    state=state,
                    options=[AssistantOption(value="restart", label="Start new consult")],
                )

            state.case_text = None
            if module.id == "endo":
                state.stage = "select_endo_blood_culture_context"
                return AssistantTurnResponse(
                    assistantMessage=(
                        f"Perfect. I’ll use {_assistant_module_label(module)} with {preset.label if preset else chosen_preset_id}. "
                        "Before we continue, which blood-culture track best matches what you want to assess?"
                    ),
                    state=state,
                    options=_assistant_endo_blood_culture_options(),
                    tips=[
                        "Choose Staphylococcus aureus, viridans-group streptococci, Enterococcus, or other/unknown/pending.",
                        "This lets me show the most relevant organism-specific pretest modifiers next.",
                    ],
                )

            if _module_supports_pretest_factors(module, state):
                state.stage = "select_pretest_factors"
                return AssistantTurnResponse(
                    assistantMessage=(
                        f"Perfect. I’ll use {_assistant_module_label(module)} with {preset.label if preset else chosen_preset_id}. "
                        + _assistant_pretest_factor_prompt(module, state.pretest_factor_ids, state)
                    ),
                    state=state,
                    options=_assistant_pretest_factor_options(module, state.pretest_factor_ids, state),
                    tips=[
                        "These are optional baseline risk modifiers that can increase the pretest probability.",
                        "Select any that apply, then continue to the case description.",
                    ],
                )

            state.stage = "describe_case"
            state.case_section = _assistant_next_case_section(module, None, state)
            prompt, tips = _assistant_case_section_prompt(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage=(
                    f"Perfect. I’ll use {_assistant_module_label(module)} with {preset.label if preset else chosen_preset_id}. "
                    + prompt
                ),
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                tips=tips,
            )

        return AssistantTurnResponse(
            assistantMessage=_assistant_lay_preset_prompt(module),
            state=state,
            options=_assistant_preset_options(module),
            tips=[
                "You can type it in normal words, for example 'this is in the ED' or 'already inpatient'.",
                "Or click the preset that fits best.",
            ],
        )

    if state.stage == "select_endo_blood_culture_context":
        if state.module_id != "endo":
            state.stage = "select_preset"
            module = store.get(state.module_id or "")
            if module is None:
                state.stage = "select_module"
                return AssistantTurnResponse(
                    assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                    state=state,
                    options=_assistant_module_options(),
                )
            return AssistantTurnResponse(
                assistantMessage="Which setting/pretest context fits this case best?",
                state=state,
                options=_assistant_preset_options(module),
            )

        module = store.get(state.module_id)
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        direct_case_response = _assistant_intake_case_from_text(
            req,
            state,
            module_hint=state.module_id,
            preset_hint=state.preset_id,
        )
        if direct_case_response is not None:
            return direct_case_response

        chosen_context = _select_endo_blood_culture_context_from_turn(req)
        if chosen_context:
            state.endo_blood_culture_context = chosen_context
            state.endo_score_factor_ids = []
            state.case_section = None
            state.pretest_factor_ids = []
            _sync_pretest_factor_labels(state, module)
            choice = ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES[chosen_context]
            context_label = choice["label"]
            if _module_supports_pretest_factors(module, state):
                state.stage = "select_pretest_factors"
                return AssistantTurnResponse(
                    assistantMessage=(
                        f"Understood. We’ll use the {context_label} blood-culture pathway. "
                        + _assistant_pretest_factor_prompt(module, state.pretest_factor_ids, state)
                    ),
                    state=state,
                    options=_assistant_pretest_factor_options(module, state.pretest_factor_ids, state),
                    tips=[
                        "These are optional baseline host/context modifiers before the diagnostic findings.",
                        "Pick any that apply, then continue to the case details.",
                    ],
                )

            state.stage = "describe_case"
            state.case_section = _assistant_next_case_section(module, None, state)
            prompt, tips = _assistant_case_section_prompt(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage=(
                    f"Understood. We’ll use the {context_label} blood-culture pathway. "
                    + prompt
                ),
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                tips=tips,
            )

        return AssistantTurnResponse(
            assistantMessage=(
                "Before we continue, which blood-culture track best matches what you want to assess for endocarditis?"
            ),
            state=state,
            options=_assistant_endo_blood_culture_options(),
            tips=[
                "Choose Staphylococcus aureus, viridans-group streptococci, Enterococcus, or other/unknown/pending.",
                "If cultures are pending or not one of those groups, choose other/unknown/pending.",
            ],
        )

    if state.stage == "select_pretest_factors":
        if not state.module_id:
            state.stage = "select_module"
            return AssistantTurnResponse(
                assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                state=state,
                options=_assistant_module_options(),
            )

        module = store.get(state.module_id)
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        selection = (req.selection or "").strip()
        user_choice = _normalize_choice(req.message)

        direct_case_response = _assistant_intake_case_from_text(
            req,
            state,
            module_hint=state.module_id,
            preset_hint=state.preset_id,
        )
        if direct_case_response is not None:
            return direct_case_response

        if selection == "skip_factors" or user_choice in {"skip", "none", "none apply", "no factors"}:
            state.pretest_factor_ids = []
            state.endo_score_factor_ids = []
            _sync_pretest_factor_labels(state, module)
            state.stage = "describe_case"
            state.case_section = _assistant_next_case_section(module, None, state)
            skip_message = "Understood. We’ll skip the baseline pretest modifiers. "
            if module.id == "endo":
                skip_message = "Understood. We’ll skip the baseline modifiers and leave the organism-specific score at its default starting state. "
            prompt, tips = _assistant_case_section_prompt(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage=(
                    skip_message
                    + prompt
                ),
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                tips=tips,
            )

        if selection == "continue_to_case" or user_choice in {"continue", "next", "done", "thats all", "that's all"}:
            state.stage = "describe_case"
            state.case_section = _assistant_next_case_section(module, None, state)
            prefix = ""
            if state.pretest_factor_labels:
                prefix = "I’ll carry forward your selected pretest-risk factors. "
            if module.id == "endo" and _assistant_selected_endo_score_id(state):
                prefix += f"I’ll auto-calculate {_assistant_selected_endo_score_id(state).upper()} from the score components you selected. "
            prompt, tips = _assistant_case_section_prompt(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage=(
                    prefix
                    + prompt
                ),
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                tips=tips,
            )

        selected_factor_id = _select_pretest_factor_from_turn(module, req)
        if selected_factor_id:
            if selected_factor_id in state.pretest_factor_ids:
                state.pretest_factor_ids = [item for item in state.pretest_factor_ids if item != selected_factor_id]
                action = "Okay, removed"
            else:
                state.pretest_factor_ids.append(selected_factor_id)
                action = "Great! Added"
            _sync_pretest_factor_labels(state, module)
            label = _pretest_factor_label(module, selected_factor_id)
            return AssistantTurnResponse(
                assistantMessage=(
                    f"{action} pretest-risk factor: {label}. "
                    + _assistant_pretest_factor_prompt(module, state.pretest_factor_ids, state)
                ),
                state=state,
                options=_assistant_pretest_factor_options(module, state.pretest_factor_ids, state),
                tips=[
                    "Click another factor to add or remove it.",
                    "When you’re done, continue to the case description.",
                ],
            )

        selected_score_factor_id = _select_endo_score_component_from_turn(state, req) if module.id == "endo" else None
        if selected_score_factor_id:
            if selected_score_factor_id in state.endo_score_factor_ids:
                state.endo_score_factor_ids = [item for item in state.endo_score_factor_ids if item != selected_score_factor_id]
                action = "Okay, removed"
            else:
                for exclusive_group in ENDO_ASSISTANT_EXCLUSIVE_SCORE_GROUPS:
                    if selected_score_factor_id in exclusive_group:
                        state.endo_score_factor_ids = [item for item in state.endo_score_factor_ids if item not in exclusive_group]
                state.endo_score_factor_ids.append(selected_score_factor_id)
                action = "Great! Added"
            label = next(
                (entry_label for entry_id, entry_label in _assistant_endo_score_component_entries(state) if entry_id == selected_score_factor_id),
                selected_score_factor_id.replace("_", " "),
            )
            score_name = (_assistant_selected_endo_score_id(state) or "score").upper()
            return AssistantTurnResponse(
                assistantMessage=(
                    f"{action} {score_name} component: {label}. "
                    + _assistant_pretest_factor_prompt(module, state.pretest_factor_ids, state)
                ),
                state=state,
                options=_assistant_pretest_factor_options(module, state.pretest_factor_ids, state),
                tips=[
                    "Add any other baseline modifiers or score components that apply.",
                    "When you’re done, continue to the case description.",
                ],
            )

        return AssistantTurnResponse(
            assistantMessage=_assistant_pretest_factor_prompt(module, state.pretest_factor_ids, state),
            state=state,
            options=_assistant_pretest_factor_options(module, state.pretest_factor_ids, state),
            tips=[
                "Click any factor or score component that applies, then continue.",
                "Click Next when you’re ready to move on.",
            ],
        )

    if state.stage == "describe_case":
        module = store.get(state.module_id or "")
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        selection = (req.selection or "").strip()
        message_text = (req.message or "").strip()
        if not message_text and not (state.case_section and selection == "continue_case_draft"):
            if state.case_section:
                prompt, tips = _assistant_case_section_prompt(module, state.case_section)
                return AssistantTurnResponse(
                    assistantMessage=prompt,
                    state=state,
                    options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                    tips=tips,
                )
            return AssistantTurnResponse(
                assistantMessage=(
                    "Describe the case in plain language. Start with vital signs, then physical examination findings, then laboratory, microbiology, and radiographic tests below. Use the Present/Absent toggle if a finding is negative."
                ),
                state=state,
                options=_assistant_case_prompt_options(module, state),
                tips=[
                    "Use the Present/Absent toggle, then click the suggested findings below to build the case more quickly.",
                    "You can still type the full case naturally if you prefer.",
                ],
            )

        already_appended = False
        if state.case_section:
            current_section = state.case_section
            if selection == "continue_case_draft":
                if message_text:
                    state.case_text = _append_case_text(state.case_text, message_text)
                    already_appended = True
                next_section = _assistant_next_case_section(module, current_section, state)
                if next_section is not None:
                    state.case_section = next_section
                    prompt, tips = _assistant_case_section_prompt(module, next_section)
                    return AssistantTurnResponse(
                        assistantMessage=prompt,
                        state=state,
                        options=_assistant_case_prompt_options(module, state, section_override=next_section),
                        tips=tips,
                    )
                if not (state.case_text or "").strip():
                    prompt, tips = _assistant_case_section_prompt(module, current_section)
                    return AssistantTurnResponse(
                        assistantMessage=(
                            "I still need at least one clinical detail before I can review the case. " + prompt
                        ),
                        state=state,
                        options=_assistant_case_prompt_options(module, state, section_override=current_section),
                        tips=tips,
                    )
                state.case_section = None
            elif message_text:
                state.case_text = _append_case_text(state.case_text, message_text)
                already_appended = True
                next_section = _assistant_next_case_section(module, current_section, state)
                next_label = ENDO_CASE_SECTION_LABELS.get(next_section or "", "review")
                current_label = ENDO_CASE_SECTION_LABELS.get(current_section, current_section.replace("_", " "))
                return AssistantTurnResponse(
                    assistantMessage=(
                        f"Added that to {current_label}. Add more for this section, or click Next to continue to {next_label}."
                    ),
                    state=state,
                    options=_assistant_case_prompt_options(module, state, section_override=current_section),
                    tips=[
                        "You can keep adding details for this section, or click Next when you are ready.",
                        "The review screen will appear after the final imaging section.",
                    ],
                )

        if not already_appended:
            state.case_text = _append_case_text(state.case_text, req.message)
        text_result = _assistant_parse_case_text(module, state)
        if text_result.parsed_request is None:
            state.stage = "describe_case"
            return AssistantTurnResponse(
                assistantMessage=(
                    "I parsed your description, but I need confirmation before I can complete the analysis. "
                    "Please review the parsed request and warnings."
                ),
                state=state,
                options=[
                    AssistantOption(value="restart", label="Start new consult"),
                    AssistantOption(value="describe_more", label="Add case detail"),
                ],
                analysis=text_result,
                tips=[
                    "Look at `parsedRequest` and `warnings` in the response.",
                    "You can submit another case detail or begin a new consult.",
                ],
            )

        state.stage = "confirm_case"
        review_message, narration_refined = _assistant_probid_review_message(module, text_result, state)
        return AssistantTurnResponse(
            assistantMessage=review_message,
            assistantNarrationRefined=narration_refined,
            state=state,
            options=_assistant_review_options_for_case(module, text_result, state),
            analysis=text_result,
            tips=[
                "Answer the single next question in normal words if that is faster than using the Add buttons.",
                "If the extraction looks right already, ask for my consultant impression.",
            ],
        )

    if state.stage == "confirm_case":
        if not state.case_text:
            state.stage = "describe_case"
            return AssistantTurnResponse(
                assistantMessage="Describe the case in plain language and I’ll translate it into ProbID inputs.",
                state=state,
                options=[],
                tips=["Include syndrome details, tests, and negatives if you have them."],
            )

        module = store.get(state.module_id or "")
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        selection = (req.selection or "").strip()
        if selection.startswith("add_missing:"):
            item_id = selection.split(":", 1)[1]
            item = _assistant_case_item_by_id(module, item_id)
            if item is None or not _assistant_case_item_allowed(module, item, state):
                text_result = _assistant_parse_case_text(module, state)
                return AssistantTurnResponse(
                    assistantMessage=(
                        "That suggested finding is no longer available for this pathway. "
                        + _build_case_review_message(module, text_result, state)
                    ),
                    state=state,
                    options=_assistant_review_options_for_case(module, text_result, state),
                    analysis=text_result,
                    tips=[
                        "Use the remaining Add buttons or type another detail.",
                        "Ask for my consultant impression when the parsed request looks complete.",
                    ],
                )
            present_text, _ = _assistant_case_item_text(item, module)
            existing_lines = {line.strip().lower() for line in (state.case_text or "").splitlines() if line.strip()}
            if present_text.strip().lower() not in existing_lines:
                state.case_text = _append_case_text(state.case_text, present_text)
            text_result = _assistant_parse_case_text(module, state)
            return AssistantTurnResponse(
                assistantMessage=_assistant_concise_probid_follow_up(
                    module,
                    text_result,
                    state,
                    lead=f"Okay, I added {item.label}.",
                ),
                assistantNarrationRefined=False,
                state=state,
                options=_assistant_review_options_for_case(module, text_result, state),
                analysis=text_result,
                tips=[
                    "Keep replying in normal words and I will keep the case moving one question at a time.",
                    "If the extraction looks right already, ask for my consultant impression.",
                ],
            )

        if selection.startswith("add_score:"):
            score_factor_id = selection.split(":", 1)[1]
            valid_ids = {entry_id for entry_id, _ in _assistant_endo_score_component_entries(state)}
            if score_factor_id in valid_ids:
                _assistant_merge_endo_score_factor_ids(state, [score_factor_id])
            text_result = _assistant_parse_case_text(module, state)
            score_label = next(
                (entry_label for entry_id, entry_label in _assistant_endo_score_component_entries(state) if entry_id == score_factor_id),
                score_factor_id.replace("_", " "),
            )
            return AssistantTurnResponse(
                assistantMessage=_assistant_concise_probid_follow_up(
                    module,
                    text_result,
                    state,
                    lead=f"Okay, I added the score component {score_label}.",
                ),
                assistantNarrationRefined=False,
                state=state,
                options=_assistant_review_options_for_case(module, text_result, state),
                analysis=text_result,
                tips=[
                    "Keep replying in normal words and I will keep the case moving one question at a time.",
                    "If the extraction looks right already, ask for my consultant impression.",
                ],
            )

        if selection == "add_more_details" and not (req.message and req.message.strip()):
            state.stage = "describe_case"
            state.case_section = _assistant_next_case_section(module, None, state)
            prompt, tips = _assistant_case_section_prompt(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage="Add anything else you want me to factor in. " + prompt,
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                tips=tips,
            )

        if _is_ready_to_assess(req):
            text_result = _assistant_parse_case_text(module, state)
            if text_result.parsed_request is not None:
                try:
                    text_result.analysis = _analyze_internal(text_result.parsed_request)
                except HTTPException as exc:
                    text_result.warnings.append(f"Parsed request could not be analyzed yet: {exc.detail}")
                    text_result.requires_confirmation = True
            if text_result.analysis is None:
                return AssistantTurnResponse(
                    assistantMessage=(
                        "I still need clarification before I can complete the assessment. "
                        "Please review the parsed request and add any missing details."
                    ),
                    state=state,
                    options=_assistant_review_options_for_case(module, text_result, state),
                    analysis=text_result,
                    tips=[
                        "Use the parsed request, warnings, and Add buttons as the checklist.",
                        "You can add another case detail or begin a new consult.",
                    ],
                )

            state.stage = "done"
            missing_suggestions = _top_missing_tests(module, text_result.parsed_request, limit=3, state=state)
            final_message = _build_probid_consult_message(
                module,
                text_result.analysis,
                missing_suggestions=missing_suggestions,
                include_panel_note=True,
            )
            narrated_message, narration_refined = narrate_probid_assistant_message(
                text_result=text_result,
                fallback_message=final_message,
                module_label=_assistant_module_label(module),
            )
            done_options = [
                AssistantOption(value="add_more_details", label="Update this case"),
                AssistantOption(value="restart", label="Start new consult"),
            ]
            done_tips = [
                "Add another test result or case detail anytime and I will update the same consult.",
                "Review `understood` to confirm what I extracted from your text.",
            ]
            if state.pending_followup_workflow == "mechid" and (state.pending_followup_text or "").strip():
                done_options.insert(0, AssistantOption(value="continue_to_resistance", label="Continue to resistance"))
                done_tips.insert(0, "If you want, I can carry the same case into the isolate/resistance interpretation next.")
            return AssistantTurnResponse(
                assistantMessage=narrated_message,
                assistantNarrationRefined=narration_refined,
                state=state,
                options=done_options,
                analysis=text_result,
                tips=done_tips,
            )

        if req.message and req.message.strip():
            state.case_text = _append_case_text(state.case_text, req.message)
            if module.id == "endo":
                _assistant_merge_endo_score_factor_ids(state, _assistant_infer_endo_score_factor_ids_from_text(state, req.message))
            text_result = _assistant_parse_case_text(module, state)
            return AssistantTurnResponse(
                assistantMessage=_assistant_concise_probid_follow_up(module, text_result, state),
                assistantNarrationRefined=False,
                state=state,
                options=_assistant_review_options_for_case(module, text_result, state),
                analysis=text_result,
                tips=[
                    "Keep replying in normal words and I will keep the case moving one question at a time.",
                    "If the extraction looks right already, ask for my consultant impression.",
                ],
            )

        text_result = _assistant_parse_case_text(module, state)
        review_message, narration_refined = _assistant_probid_review_message(module, text_result, state)
        return AssistantTurnResponse(
            assistantMessage=review_message,
            assistantNarrationRefined=narration_refined,
            state=state,
            options=_assistant_review_options_for_case(module, text_result, state),
            analysis=text_result,
            tips=[
                "Ask for my consultant impression if the parsed request looks right.",
                "Otherwise, use Present or Absent with the suggestion buttons, or add more case detail.",
            ],
        )

    if state.stage == "done":
        selection = (req.selection or "").strip()
        if selection in {"continue_to_syndrome", "continue_to_resistance"}:
            followup_response = _assistant_start_pending_followup(state)
            if followup_response is not None:
                return followup_response
        if selection == "add_more_details" and not (req.message and req.message.strip()):
            if state.workflow == "mechid":
                return AssistantTurnResponse(
                    assistantMessage=(
                        "Add the new organism detail, susceptibility result, or treatment context in plain language, "
                        "and I will update the same isolate consult."
                    ),
                    state=state,
                    options=[AssistantOption(value="restart", label="Start new consult")],
                    tips=["For example: 'cefepime resistant' or 'this is bacteremia rather than cystitis'."],
                )
            return AssistantTurnResponse(
                assistantMessage=(
                    "Add the new test result or case detail in plain language, and I will update the same consult."
                ),
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
                tips=["For example: 'TEE negative', 'blood cultures cleared', or 'CSF Gram stain positive'."],
            )
        if req.message and req.message.strip():
            if state.workflow == "mechid" and state.mechid_text:
                previous_result = _build_mechid_text_response(
                    state.mechid_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                state.mechid_text = _append_case_text(state.mechid_text, req.message)
                updated_result = _build_mechid_text_response(
                    state.mechid_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                updated_message, narration_refined = _assistant_mechid_review_message(updated_result, final=True)
                prefix = "I updated the isolate consult with the new information."
                if previous_result.analysis is not None and updated_result.analysis is not None:
                    if previous_result.analysis.final_results != updated_result.analysis.final_results:
                        prefix += " The susceptibility interpretation changed."
                    elif previous_result.analysis.mechanisms != updated_result.analysis.mechanisms:
                        prefix += " The mechanism interpretation changed."
                done_options = [
                    AssistantOption(value="add_more_details", label="Update this case"),
                    AssistantOption(value="restart", label="Start new consult"),
                ]
                done_tips = [
                    "Keep adding AST details or context if you want me to keep refining the same case.",
                ]
                if state.pending_followup_workflow == "probid" and (state.pending_followup_text or "").strip():
                    done_options.insert(0, AssistantOption(value="continue_to_syndrome", label="Continue to syndrome"))
                    done_tips.insert(0, "If you want, I can carry the same case into the syndrome workup next.")
                return AssistantTurnResponse(
                    assistantMessage=f"{prefix} {updated_message}",
                    assistantNarrationRefined=narration_refined,
                    state=state,
                    options=done_options,
                    mechidAnalysis=updated_result,
                    tips=done_tips,
                )
            if state.workflow == "probid" and state.case_text and state.module_id:
                module = store.get(state.module_id or "")
                if module is not None:
                    previous_result = _assistant_parse_case_text(module, state)
                    _assistant_populate_case_review_analysis(module, previous_result)
                    state.case_text = _append_case_text(state.case_text, req.message)
                    updated_result = _assistant_parse_case_text(module, state)
                    _assistant_populate_case_review_analysis(module, updated_result)
                    if updated_result.analysis is None:
                        state.stage = "confirm_case"
                        review_message, narration_refined = _assistant_probid_review_message(
                            module,
                            updated_result,
                            state,
                            prefix="I added the new detail, but I still need a bit more clarification before I rerun the consult. ",
                        )
                        return AssistantTurnResponse(
                            assistantMessage=review_message,
                            assistantNarrationRefined=narration_refined,
                            state=state,
                            options=_assistant_review_options_for_case(module, updated_result, state),
                            analysis=updated_result,
                            tips=[
                                "Keep replying in normal words and I will keep the case moving one question at a time.",
                                "If the extraction looks right already, ask for my consultant impression.",
                            ],
                        )
                    missing_suggestions = _top_missing_tests(module, updated_result.parsed_request, limit=3, state=state)
                    final_message = _build_probid_consult_message(
                        module,
                        updated_result.analysis,
                        missing_suggestions=missing_suggestions,
                        include_panel_note=True,
                    )
                    narrated_message, narration_refined = narrate_probid_assistant_message(
                        text_result=updated_result,
                        fallback_message=final_message,
                        module_label=_assistant_module_label(module),
                    )
                    probability_change = _assistant_probability_change_sentence(
                        previous_result.analysis,
                        updated_result.analysis,
                    )
                    lead = "I updated the consult with the new information."
                    if probability_change:
                        lead += " " + probability_change
                    done_options = [
                        AssistantOption(value="add_more_details", label="Update this case"),
                        AssistantOption(value="restart", label="Start new consult"),
                    ]
                    done_tips = [
                        "Add another test result anytime if you want to see the probability update again.",
                        "Review `understood` to confirm what I extracted from your text.",
                    ]
                    if state.pending_followup_workflow == "mechid" and (state.pending_followup_text or "").strip():
                        done_options.insert(0, AssistantOption(value="continue_to_resistance", label="Continue to resistance"))
                        done_tips.insert(0, "If you want, I can carry the same case into the isolate/resistance interpretation next.")
                    return AssistantTurnResponse(
                        assistantMessage=f"{lead} {narrated_message}",
                        assistantNarrationRefined=narration_refined,
                        state=state,
                        options=done_options,
                        analysis=updated_result,
                        tips=done_tips,
                    )

    # done / fallback
    if restart_requested:
        state = AssistantState(
            workflow="probid",
            caseText=None,
            mechidText=None,
            pendingIntakeText=None,
            pendingFollowupWorkflow=None,
            pendingFollowupText=None,
            endoScoreFactorIds=[],
            caseSection=None,
            parserStrategy=state.parser_strategy,
            parserModel=state.parser_model,
            allowFallback=state.allow_fallback,
        )
    else:
        state.stage = "select_module"
        state.workflow = "probid"
        state.module_id = None
        state.preset_id = None
        state.pending_intake_text = None
        state.pending_followup_workflow = None
        state.pending_followup_text = None
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []

    return AssistantTurnResponse(
        assistantMessage=(
            "Ready for another case. You can start a syndrome workup or a resistance mechanism interpretation."
        ),
        state=state,
        options=_assistant_module_options(),
        tips=["Type 'restart' anytime to reset the conversation."],
    )
