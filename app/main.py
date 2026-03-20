from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
    DoseIDAssistantAnalysis,
    DoseIDAssistantPatientContext,
    AntibioticAllergyAnalyzeRequest,
    AntibioticAllergyAnalyzeResponse,
    AntibioticAllergyTextAnalyzeRequest,
    AntibioticAllergyTextAnalyzeResponse,
    AssistantState,
    AssistantTurnRequest,
    AssistantTurnResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    DecisionThresholds,
    DoseIDCalculateRequest,
    DoseIDCalculateResponse,
    DoseIDCatalogResponse,
    DoseIDDoseRecommendation,
    DoseIDFollowUpQuestion,
    DoseIDMedicationCatalogEntry,
    DoseIDIndicationOption,
    DoseIDTextAnalyzeRequest,
    DoseIDTextAnalyzeResponse,
    DoseIDTextParsedRequest,
    ImmunoAgentListResponse,
    ImmunoAnalyzeRequest,
    ImmunoAnalyzeResponse,
    ImmunoRegimenListResponse,
    MechIDAnalyzeRequest,
    MechIDAnalyzeResponse,
    AntibiogramUploadRequest,
    MechIDImageAnalyzeRequest,
    MechIDImageAnalyzeResponse,
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
    SessionPatientContext,
    SyndromeModule,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
)
from .services.module_store import InMemoryModuleStore
from .services.antibiotic_allergy_service import AGENT_ALIASES, analyze_antibiotic_allergy, parse_antibiotic_allergy_text
from .services.consult_narrator import (
    narrate_allergyid_assistant_message,
    narrate_consult_summary,
    narrate_discharge_counselling_answer,
    narrate_doseid_assistant_message,
    narrate_allergy_delabeling_answer,
    narrate_biomarker_interpretation_answer,
    narrate_cns_infection_answer,
    narrate_drug_interaction_answer,
    narrate_duration_answer,
    narrate_empiric_therapy_answer,
    narrate_fluid_interpretation_answer,
    narrate_followup_tests_answer,
    narrate_fungal_management_answer,
    narrate_general_id_answer,
    narrate_mycobacterial_answer,
    narrate_pregnancy_antibiotics_answer,
    narrate_sepsis_management_answer,
    narrate_travel_medicine_answer,
    narrate_immunoid_assistant_message,
    narrate_iv_to_oral_answer,
    narrate_mechid_assistant_message,
    narrate_mechid_review_message,
    narrate_opat_answer,
    narrate_oral_therapy_answer,
    narrate_probid_assistant_message,
    narrate_probid_review_message,
    narrate_prophylaxis_dose_answer,
    narrate_source_control_answer,
    narrate_stewardship_answer,
    narrate_treatment_failure_answer,
    narrate_stewardship_review_answer,
    narrate_impression_plan,
    narrate_duke_criteria_answer,
    narrate_ast_clinical_meaning_answer,
    narrate_complexity_flag_answer,
    narrate_course_tracker_answer,
)
from .services.doseid_llm_parser import parse_doseid_text_with_openai
from .services.doseid_service import (
    DoseIDError,
    calculate_medication,
    default_indication_id,
    list_medications,
    normalize_patient,
    normalize_patient_from_available_inputs,
    suggest_mechid_doses,
)
from .services.local_text_parser import LocalParserError, parse_text_with_local_model
from .services.immunoid_engine import analyze_immunoid, list_immunoid_agents, list_immunoid_regimens
from .services.immunoid_regimens import IMMUNOID_REGIMENS
from .services.mechid_engine import MechIDEngineError, analyze_mechid, list_mechid_organisms
from .services.mechid_eval import EvalStats, evaluate_mechid_case
from .services.antibiogram_image_parser import antibiogram_to_prompt_block, parse_antibiogram_image_with_openai
from .services.mechid_image_parser import parse_mechid_image_with_openai
from .services.mechid_llm_parser import parse_mechid_text_with_openai
from .services.mechid_text_parser import parse_mechid_text
from .services.mechid_trainer_guidance import MechIDTrainerGuidanceError, generate_mechid_trainer_targets
from .services.mechid_trainer_parser import MechIDTrainerParseError, parse_mechid_trainer_correction
from .services.llm_text_parser import LLMParserError, parse_text_with_openai
from .services.text_parser import COMMON_FINDING_ALIASES, MODULE_ALIASES, parse_text_to_request, summarize_parsed_request
from .services.tb_uveitis_cots import analyze_tb_uveitis


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
PROBID_ASSISTANT_ID = "probid"
PROBID_ASSISTANT_LABEL = "Clinical syndrome probability"
PROBID_ASSISTANT_DESCRIPTION = "Estimate syndrome probability from clinical findings and test results."

ASSISTANT_MODULE_LABELS = {
    "cap": "Community-acquired pneumonia (CAP)",
    "vap": "Ventilator-associated pneumonia (VAP)",
    "cdi": "Clostridioides difficile infection (CDI)",
    "uti": "Urinary tract infection (UTI)",
    "endo": "Infective endocarditis",
    "active_tb": "Active tuberculosis (TB)",
    "tb_uveitis": "Tuberculous uveitis",
    "pjp": "Pneumocystis jirovecii pneumonia (PJP)",
    "inv_candida": "Invasive candidiasis",
    "inv_mold": "Invasive mold infection",
    "septic_arthritis": "Septic arthritis",
    "bacterial_meningitis": "Bacterial meningitis",
    "encephalitis": "Encephalitis",
    "spinal_epidural_abscess": "Spinal epidural abscess",
    "brain_abscess": "Brain abscess",
    "necrotizing_soft_tissue_infection": "Necrotizing soft tissue infection",
    "diabetic_foot_infection": "Diabetic foot infection / osteomyelitis",
    "pji": "Prosthetic joint infection (PJI)",
}

MECHID_ASSISTANT_ID = "mechid"
MECHID_ASSISTANT_LABEL = "Resistance mechanism + therapy"
MECHID_ASSISTANT_DESCRIPTION = (
    "Interpret an organism plus susceptibility pattern to estimate likely resistance mechanisms and therapy options."
)
DOSEID_ASSISTANT_ID = "doseid"
DOSEID_ASSISTANT_LABEL = "Antimicrobial dosing"
DOSEID_ASSISTANT_DESCRIPTION = (
    "Estimate antimicrobial dosing from the drug, weight, renal function, and dialysis context."
)
EXPLICIT_SYNDROME_REQUEST_TOKENS = (
    "assess",
    "assessment",
    "can you help with",
    "concerned about",
    "concern for",
    "diagnose",
    "diagnosis",
    "evaluate",
    "evaluation",
    "go into",
    "go to",
    "help me with",
    "help with",
    "likelihood",
    "need help with",
    "open",
    "pathway",
    "please help with",
    "probability",
    "question of",
    "right lane",
    "route me to",
    "route to",
    "rule out",
    "r/o",
    "screen for",
    "show me",
    "start",
    "suspicion for",
    "suspect",
    "suspected",
    "take me to",
    "worried about",
    "think about",
    "walk me through",
    "work up",
    "workup",
)
EXPLICIT_NON_SYNDROME_WORKFLOW_ALIASES: Dict[str, tuple[str, ...]] = {
    "mechid": (
        "mechid",
        "resistance mechanism",
        "resistance interpretation",
        "susceptibility interpretation",
        "ast interpretation",
        "antibiogram interpretation",
        "isolate interpretation",
        "resistance pathway",
    ),
    "doseid": (
        "doseid",
        "antimicrobial dosing",
        "antibiotic dosing",
        "dosing help",
        "dosing pathway",
        "dose this antibiotic",
        "dose this regimen",
        "renal dosing",
    ),
    "immunoid": (
        "immunoid",
        "immunosuppression",
        "immunosuppression prophylaxis",
        "screening before immunosuppression",
        "prophylaxis before immunosuppression",
        "chemotherapy prophylaxis",
        "biologic prophylaxis",
        "reactivation screening",
    ),
    "allergyid": (
        "allergyid",
        "allergy compatibility",
        "antibiotic allergy",
        "beta lactam allergy",
        "beta lactam compatibility",
        "cross reactivity",
        "cross reactivity check",
        "cross reactivity review",
    ),
}
CONSULT_INTENT_TREATMENT_START_TOKENS = (
    "should i start",
    "should we start",
    "whether to start",
    "whether we should start",
    "should id start",
    "should i treat",
    "should we treat",
    "would you treat",
    "treat now",
    "start treatment",
    "start therapy",
    "start empiric",
    "begin treatment",
    "begin therapy",
    "empiric treatment",
    "empiric therapy",
    "empiric coverage",
    "need treatment",
    "need therapy",
    "need empiric",
    "warrant treatment",
    "warrant therapy",
    "indicated treatment",
    "indicated therapy",
    "hold treatment",
    "hold therapy",
    "hold off on treatment",
    "hold off on therapy",
    "hold off on antifungal",
    "hold off on antibiotics",
    "defer treatment",
    "defer therapy",
    "wait on treatment",
    "wait on therapy",
    "can i hold off",
    "can we hold off",
    "can i hold",
    "can we hold",
    "can i wait",
    "can we wait",
)
CONSULT_INTENT_THERAPY_SELECTION_TOKENS = (
    "what antibiotics would you start",
    "what antibiotics you would start",
    "what would you use",
    "what would you reach for",
    "what should i use",
    "what should we use",
    "what would you start",
    "which drug would you reach for",
    "what should i start",
    "what regimen would you use",
    "what drug would you reach for",
    "which treatment would you use",
    "which therapy would you use",
    "which antifungal would you use",
    "which antibiotic would you use",
    "what antifungal would you use",
    "what antibiotic would you use",
    "what would id use",
)
CONSULT_INTENT_ANTIMICROBIAL_TOKENS = (
    "treatment",
    "therapy",
    "cover",
    "coverage",
    "antibiotic",
    "antibiotics",
    "antimicrobial",
    "antifungal",
    "mold active",
    "mold-active",
)
CONSULT_INTENT_FUNGAL_TOKENS = (
    "fungal",
    "fungus",
    "antifungal",
    "yeast",
    "candida",
    "candidemia",
    "fungemia",
    "mold",
    "aspergillus",
    "aspergillosis",
)
CONSULT_INTENT_CANDIDA_TOKENS = (
    "candida",
    "candidemia",
    "fungemia",
    "yeast",
    "t2candida",
)
CONSULT_INTENT_MOLD_TOKENS = (
    "mold",
    "mold active",
    "mold-active",
    "aspergillus",
    "aspergillosis",
    "galactomannan",
    "halo sign",
    "reverse halo",
    "air crescent",
    "nodular lung lesion",
    "nodular lung lesions",
    "pulmonary nodules",
    "pulmonary nodule",
)
IMMUNOID_ASSISTANT_ID = "immunoid"
IMMUNOID_ASSISTANT_LABEL = "Immunosuppression screening + prophylaxis"
IMMUNOID_ASSISTANT_DESCRIPTION = (
    "Review chemotherapy, steroids, biologics, or transplant immunosuppression for infection screening, prophylaxis, and geography-sensitive follow-up."
)
ALLERGYID_ASSISTANT_ID = "allergyid"
ALLERGYID_ASSISTANT_LABEL = "Antibiotic allergy compatibility"
ALLERGYID_ASSISTANT_DESCRIPTION = (
    "Interpret antibiotic allergy labels, reaction phenotype, and cross-reactivity so the safest preferred therapy stays in play."
)
DOSEID_INTENT_TOKENS = (
    "dose",
    "dosing",
    "dosage",
    "hemodialysis",
    "dialysis",
    "hd",
    "crrt",
    "creatinine clearance",
    "crcl",
    "renal dose",
    "renal dosing",
    "ripe",
    "rhze",
)
IMMUNOID_INTENT_TOKENS = (
    "chemotherapy prophylaxis",
    "biologic prophylaxis",
    "before rituximab",
    "before infliximab",
    "before chemotherapy",
    "before steroids",
    "immunosuppression prophylaxis",
    "screen before immunosuppression",
    "screening before immunosuppression",
    "hbv reactivation",
    "tb screening",
    "strongyloides",
    "pjp prophylaxis",
    "prophylaxis",
)
IMMUNOID_COMMON_AGENT_IDS = (
    "prednisone_20",
    "rituximab",
    "infliximab",
    "tofacitinib",
    "cyclophosphamide",
    "tacrolimus",
    "mycophenolate_mofetil",
    "eculizumab",
)
IMMUNOID_COMMON_REGIMEN_IDS = (
    "r_chop",
    "da_r_epoch",
    "br",
    "fcr",
    "seven_plus_three",
    "flag_ida",
    "vrd",
    "dara_vrd",
)
ALLERGYID_INTENT_TOKENS = (
    "allergy",
    "allergic",
    "anaphylaxis",
    "hives",
    "urticaria",
    "angioedema",
    "sjs",
    "ten",
    "dress",
    "cross reactivity",
    "cross-reactivity",
    "penicillin allergy",
    "cephalosporin allergy",
    "beta lactam allergy",
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
    "what antibiotic",
    "what would you treat with",
    "how would you treat",
    "how do i treat",
    "how should i treat",
    "how to treat",
    "treat this",
    "how do i manage",
    "how should i manage",
    "what should i use",
    "what can i use",
    "what do i use",
    "what is the treatment",
    "treatment for this isolate",
    "which therapy",
    "which treatment",
    "antibiotic choice",
    "antibiotic choices",
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
    "diabetic_foot_infection": [
        {
            "context": "Evidence base: Diabetic foot infection and osteomyelitis",
            "citation": "IWGDF/IDSA diabetic foot infection guideline (2023)",
            "url": "https://doi.org/10.1093/cid/ciad527",
        },
        {
            "context": "Evidence base: Diabetic foot osteomyelitis imaging",
            "citation": "Dinh et al. Diagnostic accuracy of exam and imaging tests for diabetic foot osteomyelitis (2008)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/18611152/",
        },
        {
            "context": "Evidence base: Probe-to-bone test",
            "citation": "Lam et al. Diagnostic accuracy of probe to bone to detect osteomyelitis in the diabetic foot (2016)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/27369321/",
        },
        {
            "context": "Evidence base: Inflammatory markers in infected diabetic foot ulcers",
            "citation": "Suwanwongse et al. Inflammatory blood laboratory markers meta-analysis (2024)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39667886/",
        },
    ],
    "tb_uveitis": [
        {
            "context": "Evidence base: Tuberculous uveitis consensus calculator",
            "citation": "COTS Calculator",
            "url": "https://www.oculartb.net/cots-calc",
        },
        {
            "context": "Evidence base: Tuberculous choroiditis ATT initiation consensus",
            "citation": "Agrawal et al. Collaborative Ocular Tuberculosis Study Report 1 (2021)",
            "url": "https://doi.org/10.1016/j.ophtha.2020.01.008",
        },
        {
            "context": "Evidence base: Anterior/intermediate/panuveitis/retinal vasculitis ATT initiation consensus",
            "citation": "Agrawal et al. Collaborative Ocular Tuberculosis Study Report 2 (2021)",
            "url": "https://doi.org/10.1016/j.ophtha.2020.06.052",
        },
    ],
}

ENDO_ASSISTANT_BLOOD_CULTURE_CHOICES = {
    "staph": {
        "label": "Staphylococci",
        "description": "Use the staphylococcal pathway and show VIRSTA-overlap baseline modifiers; in prosthetic valve/TAVI/device cases this also supports CoNS prosthetic-material bacteremia.",
        "score_id": "virsta",
    },
    "strep": {
        "label": "Viridans group streptococci",
        "description": "Use the viridans/NBHS pathway and show HANDOC-overlap baseline modifiers.",
        "score_id": "handoc",
    },
    "enterococcus": {
        "label": "Enterococcus",
        "description": "Use the enterococcal pathway and show DENOVA-overlap baseline modifiers; in prosthetic valve/TAVI/device cases this also supports prosthetic-material enterococcal bacteremia.",
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


def _assistant_case_section_categories(module: SyndromeModule, section: str | None) -> set[str]:
    section_map = {
        "exam_vitals": {"symptom", "vital", "exam"},
        "lab": {"lab"},
        "micro": {"micro"},
        "imaging": {"imaging"},
    }
    categories = section_map.get(section or "", set())
    return {item.category for item in module.items if item.category in categories}


def _assistant_case_section_label(module: SyndromeModule, section: str | None) -> str:
    if section != "exam_vitals":
        return ENDO_CASE_SECTION_LABELS.get(section or "", (section or "review").replace("_", " "))
    categories = _assistant_case_section_categories(module, section)
    has_symptom = "symptom" in categories
    has_vital = "vital" in categories
    has_exam = "exam" in categories
    if has_symptom and not has_vital and not has_exam:
        return "clinical symptoms"
    if has_symptom and has_vital and not has_exam:
        return "symptoms and vital signs"
    if has_symptom and has_exam and not has_vital:
        return "symptoms and physical exam"
    if has_symptom and has_vital and has_exam:
        return "vital signs and physical exam"
    if has_vital and has_exam:
        return "vital signs and physical exam"
    if has_symptom:
        return "clinical symptoms"
    return "bedside clinical findings"

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
    "endo": {
        "endo_bcx_saureus_multi": (
            "Staphylococcus aureus in at least 2 blood culture sets",
            "No Staphylococcus aureus in at least 2 blood culture sets",
        ),
        "endo_bcx_cons_prosthetic_multi": (
            "Coagulase-negative staphylococci in at least 2 blood culture sets with prosthetic valve, TAVI, or intracardiac device context",
            "No coagulase-negative staphylococcal prosthetic/device blood culture pattern",
        ),
        "endo_bcx_major_persistent": (
            "Persistent positive blood cultures",
            "Blood cultures cleared without persistent positivity",
        ),
        "endo_bcx_major_typical": (
            "Typical endocarditis blood cultures in at least 2 sets",
            "No typical endocarditis blood cultures in at least 2 sets",
        ),
        "endo_bcx_enterococcus_prosthetic_multi": (
            "Enterococcal bacteremia in at least 2 blood culture sets with prosthetic valve, TAVI, or intracardiac device context",
            "No enterococcal prosthetic/device blood culture pattern",
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
    "tb_uveitis": {
        "tbu_phenotype_au_first": (
            "Anterior uveitis, first episode",
            "Anterior uveitis, first episode not selected",
        ),
        "tbu_phenotype_au_recurrent": (
            "Anterior uveitis, recurrent episode",
            "Anterior uveitis, recurrent episode not selected",
        ),
        "tbu_phenotype_intermediate": (
            "Intermediate uveitis",
            "Intermediate uveitis not selected",
        ),
        "tbu_phenotype_panuveitis": (
            "Panuveitis",
            "Panuveitis not selected",
        ),
        "tbu_phenotype_rv_active": (
            "Active retinal vasculitis",
            "Active retinal vasculitis not selected",
        ),
        "tbu_phenotype_rv_inactive": (
            "Inactive retinal vasculitis",
            "Inactive retinal vasculitis not selected",
        ),
        "tbu_phenotype_choroiditis_serpiginoid": (
            "Serpiginoid choroiditis",
            "Serpiginoid choroiditis not selected",
        ),
        "tbu_phenotype_choroiditis_multifocal": (
            "Multifocal or non-serpiginoid choroiditis",
            "Multifocal or non-serpiginoid choroiditis not selected",
        ),
        "tbu_phenotype_choroiditis_tuberculoma": (
            "Choroidal tuberculoma or nodule",
            "Choroidal tuberculoma or nodule not selected",
        ),
        "tbu_endemicity_endemic": (
            "Patient from TB-endemic region",
            "Patient not from TB-endemic region",
        ),
        "tbu_endemicity_non_endemic": (
            "Patient from TB-non-endemic region",
            "Patient not from TB-non-endemic region",
        ),
        "tbu_tst_positive": (
            "Tuberculin skin test positive",
            "Tuberculin skin test not positive",
        ),
        "tbu_tst_negative": (
            "Tuberculin skin test negative",
            "Tuberculin skin test not negative",
        ),
        "tbu_tst_na": (
            "Tuberculin skin test not done",
            "Tuberculin skin test completed",
        ),
        "tbu_igra_positive": (
            "IGRA or QuantiFERON positive",
            "IGRA or QuantiFERON not positive",
        ),
        "tbu_igra_negative": (
            "IGRA or QuantiFERON negative",
            "IGRA or QuantiFERON not negative",
        ),
        "tbu_igra_na": (
            "IGRA or QuantiFERON not done",
            "IGRA or QuantiFERON completed",
        ),
        "tbu_chest_imaging_positive": (
            "Chest X-ray or CT positive for healed or active TB signs",
            "Chest X-ray or CT not positive for TB signs",
        ),
        "tbu_chest_imaging_negative": (
            "Chest X-ray or CT negative for healed or active TB signs",
            "Chest X-ray or CT not negative for TB signs",
        ),
        "tbu_chest_imaging_na": (
            "Chest X-ray or CT not done",
            "Chest X-ray or CT completed",
        ),
        "tbu_pretest_prior_tb_or_ltbi": (
            "Prior TB disease or known latent TB infection",
            "Prior TB disease or known latent TB infection not selected",
        ),
        "tbu_pretest_close_tb_contact": (
            "Close TB contact or household TB exposure",
            "Close TB contact or household TB exposure not selected",
        ),
        "tbu_harm_macular_or_vision_threatening_lesion": (
            "Macular or vision-threatening lesion",
            "Macular or vision-threatening lesion not selected",
        ),
        "tbu_harm_bilateral_or_only_seeing_eye": (
            "Bilateral disease or only-seeing eye",
            "Bilateral disease or only-seeing eye not selected",
        ),
        "tbu_harm_progressive_vision_loss_or_severe_inflammation": (
            "Progressive vision loss or severe inflammation",
            "Progressive vision loss or severe inflammation not selected",
        ),
        "tbu_harm_immunosuppressed": (
            "Immunosuppressed host",
            "Immunosuppressed host not selected",
        ),
        "tbu_harm_hepatotoxicity_risk": (
            "Major ATT hepatotoxicity risk",
            "Major ATT hepatotoxicity risk not selected",
        ),
        "tbu_harm_cld_mild": (
            "Mild chronic liver disease risk",
            "Mild chronic liver disease risk not selected",
        ),
        "tbu_harm_cld_moderate": (
            "Moderate chronic liver disease risk",
            "Moderate chronic liver disease risk not selected",
        ),
        "tbu_harm_cld_severe": (
            "Severe chronic liver disease risk",
            "Severe chronic liver disease risk not selected",
        ),
        "tbu_harm_ethambutol_ocular_risk": (
            "Higher ethambutol ocular-toxicity risk",
            "Higher ethambutol ocular-toxicity risk not selected",
        ),
        "tbu_harm_major_drug_interaction_or_intolerance": (
            "Major rifamycin interaction, prior ATT intolerance, or resistance concern",
            "Major rifamycin interaction, prior ATT intolerance, or resistance concern not selected",
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
            "Serum galactomannan >0.5 positive",
            "Serum galactomannan >0.5 negative",
        ),
        "imi_bal_gm_odi10": (
            "BAL galactomannan >1.0 positive",
            "BAL galactomannan >1.0 negative",
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
        "imi_aspergillus_culture_resp": (
            "Respiratory culture positive for Aspergillus",
            "Respiratory culture negative for Aspergillus",
        ),
        "imi_aspergillus_culture_na": (
            "Respiratory fungal culture not done",
            "Respiratory fungal culture completed",
        ),
        "imi_aspergillus_pcr_plasma": (
            "Aspergillus PCR positive from plasma",
            "Aspergillus PCR negative from plasma",
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
    "diabetic_foot_infection": {
        "dfi_local_inflammation_2plus": (
            "At least 2 local signs of diabetic foot infection",
            "No convincing local diabetic foot infection signs",
        ),
        "dfi_purulence": (
            "Purulent drainage or pus from the ulcer",
            "No purulence",
        ),
        "dfi_erythema_ge2cm_or_deep": (
            "Erythema greater than 2 cm or infection deeper than skin/subcutaneous tissue",
            "No erythema greater than 2 cm and no obvious deeper infection",
        ),
        "dfi_systemic_toxicity": (
            "Systemic toxicity, hemodynamic instability, or shock",
            "Hemodynamically stable without systemic toxicity",
        ),
        "dfi_deep_abscess_or_gangrene": (
            "Deep abscess, gangrene, or other limb-threatening destructive feature",
            "No deep abscess, gangrene, or destructive limb-threatening feature",
        ),
        "dfi_probe_to_bone_positive": (
            "Probe-to-bone positive",
            "Probe-to-bone negative",
        ),
        "dfi_probe_to_bone_na": (
            "Probe-to-bone not done",
            "Probe-to-bone completed",
        ),
        "dfi_exposed_bone": (
            "Exposed or visible bone",
            "No exposed or visible bone",
        ),
        "dfi_forefoot_only": (
            "Forefoot-only infection or osteomyelitis",
            "Not limited to the forefoot",
        ),
        "dfi_esr_high": (
            "ESR markedly elevated",
            "ESR not markedly elevated",
        ),
        "dfi_crp_high": (
            "CRP elevated",
            "CRP not elevated",
        ),
        "dfi_wbc_high": (
            "Peripheral WBC elevated",
            "Peripheral WBC not elevated",
        ),
        "dfi_xray_osteomyelitis": (
            "Plain radiograph suggests osteomyelitis",
            "Plain radiograph does not suggest osteomyelitis",
        ),
        "dfi_xray_na": (
            "Plain radiographs not done",
            "Plain radiographs completed",
        ),
        "dfi_mri_osteomyelitis_or_abscess": (
            "MRI compatible with osteomyelitis or deep abscess",
            "MRI not compatible with osteomyelitis or deep abscess",
        ),
        "dfi_mri_na": (
            "MRI not done",
            "MRI completed",
        ),
        "dfi_deep_tissue_culture_pos": (
            "Deep tissue or operative culture positive",
            "Deep tissue or operative culture negative",
        ),
        "dfi_deep_tissue_culture_na": (
            "Deep tissue or operative culture not done",
            "Deep tissue or operative culture completed",
        ),
        "dfi_bone_biopsy_culture_pos": (
            "Bone biopsy culture positive",
            "Bone biopsy culture negative",
        ),
        "dfi_bone_histology_pos": (
            "Bone histology positive for osteomyelitis",
            "Bone histology negative for osteomyelitis",
        ),
        "dfi_bone_biopsy_na": (
            "Bone biopsy or histology not done",
            "Bone biopsy or histology completed",
        ),
        "dfi_surgery_debridement_done": (
            "Surgical debridement or source control already performed",
            "No surgical debridement or source control yet",
        ),
        "dfi_minor_amputation_done": (
            "Minor amputation or bone resection already performed",
            "No minor amputation or bone resection",
        ),
        "dfi_positive_bone_margin": (
            "Positive residual bone margin after resection",
            "Negative residual bone margin after resection",
        ),
    },
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/assistant")
def assistant_web() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "assistant.html")


@app.get("/doseid")
def doseid_web() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "doseid.html")


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
    from fastapi.testclient import TestClient

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


def _mechid_canonical_text(parsed_request: MechIDTextParsedRequest | None) -> str:
    if parsed_request is None:
        return ""

    parts: List[str] = []
    target = parsed_request.organism or _format_mechid_organism_list(parsed_request.mentioned_organisms)
    if target:
        parts.append(target)

    grouped_results: Dict[str, List[str]] = {"Resistant": [], "Intermediate": [], "Susceptible": []}
    for antibiotic, call in (parsed_request.susceptibility_results or {}).items():
        if call in grouped_results:
            grouped_results[call].append(antibiotic)

    for state_label, prefix in (
        ("Resistant", "resistant to"),
        ("Intermediate", "intermediate to"),
        ("Susceptible", "susceptible to"),
    ):
        antibiotics = grouped_results[state_label]
        if antibiotics:
            parts.append(f"{prefix} {_join_readable(antibiotics)}")

    if parsed_request.resistance_phenotypes:
        parts.append(f"phenotype {_join_readable(parsed_request.resistance_phenotypes)}")

    tx_context = parsed_request.tx_context
    context_bits: List[str] = []
    if tx_context.focus_detail != "Not specified":
        context_bits.append(tx_context.focus_detail)
    elif tx_context.syndrome != "Not specified":
        context_bits.append(tx_context.syndrome)
    if tx_context.severity != "Not specified":
        context_bits.append(tx_context.severity)
    if context_bits:
        parts.append(f"context {_join_readable(context_bits)}")
    if tx_context.carbapenemase_result != "Not specified":
        carbapenemase_text = tx_context.carbapenemase_result
        if tx_context.carbapenemase_class != "Not specified":
            carbapenemase_text += f" ({tx_context.carbapenemase_class})"
        parts.append(f"carbapenemase {carbapenemase_text}")

    return ". ".join(part.strip().rstrip(".") for part in parts if part and part.strip()).strip() + ("" if not parts else ".")


def _build_mechid_response_from_parsed(
    *,
    text: str,
    parsed: Dict[str, Any],
    parser_name: str,
    parser_fallback_used: bool = False,
    warnings: List[str] | None = None,
) -> MechIDTextAnalyzeResponse:
    warning_list = list(warnings or [])
    parsed_request = None
    analysis = None
    if (
        parsed.get("organism") is not None
        or parsed.get("mentionedOrganisms")
        or parsed.get("susceptibilityResults")
        or parsed.get("resistancePhenotypes")
    ):
        parsed_request = MechIDTextParsedRequest(
            organism=parsed.get("organism"),
            mentionedOrganisms=parsed.get("mentionedOrganisms", []),
            resistancePhenotypes=parsed.get("resistancePhenotypes", []),
            susceptibilityResults=parsed.get("susceptibilityResults", {}),
            txContext=parsed.get("txContext", {}),
        )
        if parsed.get("organism") is not None and parsed.get("susceptibilityResults"):
            try:
                analyzed = analyze_mechid(
                    organism=parsed["organism"],
                    susceptibility_results=parsed["susceptibilityResults"],
                    tx_context=parsed["txContext"],
                )
                if parsed_request.tx_context.syndrome == "Uncomplicated cystitis":
                    susceptible_results = parsed.get("susceptibilityResults") or analyzed["final_results"]
                    analyzed["therapy_notes"] = _prioritize_cystitis_therapy_notes(
                        analyzed.get("therapy_notes", []),
                        susceptible_results,
                    )
                treatment_duration_guidance, monitoring_recommendations = _build_mechid_duration_monitoring_guidance(
                    organism=analyzed["organism"],
                    final_results=analyzed["final_results"],
                    tx_context=parsed.get("txContext"),
                    therapy_notes=analyzed["therapy_notes"],
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
                    treatmentDurationGuidance=treatment_duration_guidance,
                    monitoringRecommendations=monitoring_recommendations,
                    references=analyzed["references"],
                    warnings=list(parsed.get("warnings", [])),
                )
            except MechIDEngineError as exc:
                warning_list.append(str(exc))

    warning_list.extend(parsed.get("warnings", []))
    provisional_advice = _build_mechid_provisional_advice(parsed_request)
    return MechIDTextAnalyzeResponse(
        parser=parser_name,
        text=text,
        parsedRequest=parsed_request,
        warnings=warning_list,
        requiresConfirmation=bool(parsed.get("requiresConfirmation") or analysis is None),
        parserFallbackUsed=parser_fallback_used,
        analysis=analysis,
        provisionalAdvice=provisional_advice,
    )


def _prioritize_cystitis_therapy_notes(
    therapy_notes: List[str],
    susceptibility_results: Dict[str, str | None],
) -> List[str]:
    susceptible_agents = {
        antibiotic
        for antibiotic, call in (susceptibility_results or {}).items()
        if call == "Susceptible"
    }
    oral_choices = [
        agent
        for agent in (
            "Nitrofurantoin",
            "Trimethoprim/Sulfamethoxazole",
            "Fosfomycin",
            "Ciprofloxacin",
            "Levofloxacin",
        )
        if agent in susceptible_agents
    ]
    prioritized_note = (
        "**Lower-tract cystitis pattern** → for uncomplicated cystitis I would prefer a susceptible oral option such as "
        f"{_join_readable(oral_choices)} rather than an IV regimen or carbapenem."
        if oral_choices
        else "**Lower-tract cystitis pattern** → for uncomplicated cystitis I would look first for a susceptible oral lower-tract option rather than an IV regimen or carbapenem."
    )

    filtered_notes: List[str] = []
    suppress_tokens = (
        "ceftriaxone",
        "cefepime",
        "piperacillin/tazobactam",
        "pip/tazo",
        "meropenem",
        "imipenem",
        "ertapenem",
        "doripenem",
        "carbapenem",
        "iv ",
        "intravenous",
    )
    for note in therapy_notes:
        note_lower = note.lower()
        if any(token in note_lower for token in suppress_tokens):
            continue
        if note not in filtered_notes:
            filtered_notes.append(note)

    return [prioritized_note, *filtered_notes]


def _doseid_catalog_response() -> DoseIDCatalogResponse:
    medications = [
        DoseIDMedicationCatalogEntry(
            id=med.id,
            name=med.name,
            category=med.category,
            indications=[DoseIDIndicationOption(id=item.id, label=item.label) for item in med.indications],
            sourcePages=med.source_pages,
        )
        for med in list_medications()
    ]
    return DoseIDCatalogResponse(medications=medications)


def _doseid_recommendation_model(payload: Dict[str, Any]) -> DoseIDDoseRecommendation:
    return DoseIDDoseRecommendation(
        medicationId=payload["medication_id"],
        medicationName=payload["medication_name"],
        category=payload["category"],
        indicationId=payload["indication_id"],
        indicationLabel=payload["indication_label"],
        regimen=payload["regimen"],
        renalBucket=payload["renal_bucket"],
        notes=payload.get("notes", []),
        sourcePages=payload["source_pages"],
        doseWeight=payload.get("dose_weight"),
    )


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

    return _build_mechid_response_from_parsed(
        text=text,
        parsed=parsed,
        parser_name=parser_name,
        parser_fallback_used=parser_fallback_used,
        warnings=warnings,
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


@app.get("/v1/doseid/medications", response_model=DoseIDCatalogResponse)
def list_doseid_medications() -> DoseIDCatalogResponse:
    return _doseid_catalog_response()


@app.post("/v1/allergyid/analyze", response_model=AntibioticAllergyAnalyzeResponse)
def analyze_antibiotic_allergy_endpoint(req: AntibioticAllergyAnalyzeRequest) -> AntibioticAllergyAnalyzeResponse:
    return analyze_antibiotic_allergy(req)


@app.post("/v1/allergyid/analyze-text", response_model=AntibioticAllergyTextAnalyzeResponse)
def analyze_antibiotic_allergy_text_endpoint(
    req: AntibioticAllergyTextAnalyzeRequest,
) -> AntibioticAllergyTextAnalyzeResponse:
    return parse_antibiotic_allergy_text(req)


def _assistant_allergyid_message(result: AntibioticAllergyAnalyzeResponse) -> str:
    parts: List[str] = []
    if result.recommendations:
        avoids = [item.agent for item in result.recommendations if item.recommendation == "avoid"]
        cautions = [item.agent for item in result.recommendations if item.recommendation == "caution"]
        preferred = [item.agent for item in result.recommendations if item.recommendation == "preferred"]
        if avoids:
            parts.append(f"I would avoid {', '.join(avoids[:3])} based on the current allergy history.")
        elif cautions:
            parts.append(f"I would treat {', '.join(cautions[:3])} as caution choices rather than routine substitutions.")
        elif preferred:
            parts.append(f"The best-supported options from the current allergy profile are {', '.join(preferred[:3])}.")
        top_recommendation = next((item for item in result.recommendations if item.rationale), None)
        if top_recommendation is not None:
            parts.append(top_recommendation.rationale)
    parts.append(result.summary)
    if result.general_advice:
        parts.append(result.general_advice[0])
    if result.follow_up_questions:
        parts.append(result.follow_up_questions[0].prompt)
    elif result.delabeling_opportunities:
        parts.append(result.delabeling_opportunities[0])
    return " ".join(part.strip() for part in parts if part and part.strip())


def _assistant_allergyid_entry_phrase(entry) -> str:
    reaction_map = {
        "anaphylaxis": "anaphylaxis",
        "angioedema": "angioedema",
        "urticaria": "urticaria",
        "benign_delayed_rash": "a delayed rash",
        "isolated_gi": "GI upset",
        "headache": "headache",
        "family_history_only": "family history only",
        "scar": "SJS/TEN or another severe cutaneous reaction",
        "organ_injury": "organ injury",
        "serum_sickness_like": "a serum-sickness-like reaction",
        "hemolytic_anemia": "immune hemolysis",
        "unknown": "an unclear reaction",
        "intolerance": "intolerance",
    }
    phrase = reaction_map.get(getattr(entry, "reaction_type", "unknown"), "an unclear reaction")
    return f"{entry.reported_agent} caused {phrase}"


def _assistant_reply_introduces_candidate_agents(reply: str) -> bool:
    normalized = _normalize_choice(reply)
    candidate_markers = (
        "can i use",
        "can i still use",
        "could i use",
        "could i still use",
        "can we use",
        "what can i use",
        "what about",
        "would you use",
        "is it safe to use",
        "is it okay to use",
        "is it ok to use",
        "best antibiotic",
        "candidate antibiotic",
        "candidate antibiotics",
        "which antibiotic",
        "which antibiotics",
        "should i use",
        "considering",
        "instead use",
        "safe to give",
        "still fine",
        "still okay",
        "still ok",
        "thinking about",
        "we are thinking about",
        "may need",
        "now needs",
        "needs",
    )
    return any(marker in normalized for marker in candidate_markers)


def _assistant_extract_allergy_culprit_override(reply: str) -> str | None:
    normalized = _normalize_choice(reply)
    alias_pattern = "|".join(sorted((re.escape(alias) for alias in AGENT_ALIASES.keys()), key=len, reverse=True))
    patterns = (
        rf"(?:it was|it was actually|actually it was|the culprit was|actually the culprit was)\s+(?P<drug>{alias_pattern})(?:\s+not\s+(?:the\s+)?(?P<old>{alias_pattern}))?",
        rf"not\s+(?P<old>{alias_pattern})\s*,?\s*(?:it was|it was actually|actually it was|the culprit was)\s+(?P<drug>{alias_pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group("drug").strip()
    return None


def _assistant_reply_updates_reaction_details(reply: str) -> bool:
    normalized = _normalize_choice(reply)
    markers = (
        "anaphylaxis",
        "hives",
        "urticaria",
        "angioedema",
        "rash",
        "sjs",
        "ten",
        "dress",
        "serum sickness",
        "hemolytic",
        "hepatitis",
        "liver injury",
        "kidney injury",
        "nephritis",
        "nausea",
        "vomiting",
        "diarrhea",
        "gi",
        "headache",
        "family history",
        "reaction was",
        "only",
    )
    return any(marker in normalized for marker in markers)


def _assistant_clean_allergy_reaction_reply(reply: str) -> str:
    cleaned = reply.strip()
    substitutions = (
        r"^(?:actually|well)\s+",
        r"^(?:the\s+)?reaction\s+was\s+",
        r"^(?:it|this)\s+was\s+",
        r"^(?:only|just)\s+",
    )
    for pattern in substitutions:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .")
    return cleaned or reply.strip()


def _assistant_merge_allergyid_followup_text(existing_text: str | None, reply: str) -> str:
    existing_text = (existing_text or "").strip()
    previous = (
        parse_antibiotic_allergy_text(AntibioticAllergyTextAnalyzeRequest(text=existing_text))
        if existing_text
        else None
    )
    update = parse_antibiotic_allergy_text(AntibioticAllergyTextAnalyzeRequest(text=reply))
    previous_parsed = previous.parsed_request if previous is not None else None
    update_parsed = update.parsed_request

    candidate_agents: List[str] = []
    tolerated_agents: List[str] = list(update_parsed.tolerated_agents if update_parsed else [])
    if update_parsed and update_parsed.candidate_agents and _assistant_reply_introduces_candidate_agents(reply):
        candidate_agents = list(update_parsed.candidate_agents)
    if not candidate_agents and previous_parsed:
        candidate_agents = list(previous_parsed.candidate_agents)
    if not tolerated_agents and previous_parsed:
        tolerated_agents = list(previous_parsed.tolerated_agents)

    allergy_entries = list(update_parsed.allergy_entries if update_parsed else [])
    if not allergy_entries and previous_parsed and previous_parsed.allergy_entries:
        culprit_override = _assistant_extract_allergy_culprit_override(reply)
        culprit_name = culprit_override or previous_parsed.allergy_entries[0].reported_agent
        if _assistant_reply_updates_reaction_details(reply):
            cleaned_reply = _assistant_clean_allergy_reaction_reply(reply)
            synthetic_text = f"{culprit_name} caused {cleaned_reply}"
            synthetic = parse_antibiotic_allergy_text(AntibioticAllergyTextAnalyzeRequest(text=synthetic_text))
            allergy_entries = list(synthetic.parsed_request.allergy_entries if synthetic.parsed_request else [])
        elif culprit_override:
            previous_entry = previous_parsed.allergy_entries[0]
            allergy_entries = [
                type(previous_entry)(
                    reportedAgent=culprit_name,
                    reactionType=previous_entry.reaction_type,
                    timing=previous_entry.timing,
                )
            ]
        if not allergy_entries:
            allergy_entries = list(previous_parsed.allergy_entries)

    pieces: List[str] = []
    for entry in allergy_entries:
        pieces.append(_assistant_allergyid_entry_phrase(entry) + ".")
    for agent in tolerated_agents:
        pieces.append(f"Previously tolerated {agent}.")
    if candidate_agents:
        pieces.append("Candidate antibiotics: " + ", ".join(candidate_agents) + ".")
    if not pieces:
        return reply
    return " ".join(pieces)


def _assistant_allergyid_response(
    state: AssistantState,
    *,
    message_text: str,
    prefix: str = "",
) -> AssistantTurnResponse:
    state.workflow = "allergyid"
    state.stage = "done"
    state.module_id = None
    state.preset_id = None
    state.case_section = None
    state.case_text = None
    state.mechid_text = None
    state.doseid_text = None
    state.allergyid_text = message_text
    state.pretest_factor_ids = []
    state.pretest_factor_labels = []
    state.endo_blood_culture_context = None
    state.endo_score_factor_ids = []
    _assistant_reset_immunoid_state(state)
    parsed = parse_antibiotic_allergy_text(AntibioticAllergyTextAnalyzeRequest(text=message_text))
    result = parsed.analysis or analyze_antibiotic_allergy(AntibioticAllergyAnalyzeRequest(infectionContext=message_text))
    _snapshot_allergy_result(state, result)
    fallback_message = ((prefix or "") + _assistant_allergyid_message(result)).strip()
    message, narration_refined = narrate_allergyid_assistant_message(
        allergy_result=result,
        fallback_message=fallback_message,
        established_syndrome=state.established_syndrome,
        prior_context_summary=_consult_prior_context_summary(state),
    )
    doseid_options = _allergyid_doseid_options(result)
    options: List[AssistantOption] = [
        *doseid_options,
        AssistantOption(value="add_more_details", label="Update allergy details"),
        AssistantOption(value="probid", label="Run syndrome workup"),
        AssistantOption(value="restart", label="Start new consult"),
    ]
    tips = [
        "A useful reply would be: 'the reaction was hives within 1 hour' or 'the culprit was ceftriaxone, not amoxicillin'.",
        "If MechID already gave you candidate antibiotics, paste them here and I can sort out which ones stay preferred despite the allergy label.",
    ]
    if doseid_options:
        tips.insert(0, "If you want, I can calculate the renal-adjusted dose for the preferred safe alternative.")
    # Proactive bridging: if allergy rules out the primary agent and MechID organisms are known, offer empiric alternative
    allergy_verdict = getattr(result, "verdict", None) or ""
    if allergy_verdict.lower() in {"avoid", "caution"} and state.consult_organisms and state.established_syndrome:
        options.insert(0, AssistantOption(
            value="empiric_therapy",
            label="Find safe empiric alternative",
            description="I'll suggest an agent that covers the same organism/syndrome while respecting this allergy.",
        ))
        tips.insert(0, "I can suggest an empiric alternative that covers the same pathogen while avoiding this allergy.")
    return AssistantTurnResponse(
        assistantMessage=message,
        assistantNarrationRefined=narration_refined,
        state=state,
        allergyidAnalysis=result,
        options=options,
        tips=tips,
    )


def _assistant_start_allergyid_from_text(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    if not message_text or not _assistant_is_allergyid_intent(message_text):
        return None
    return _assistant_allergyid_response(state, message_text=message_text)


@app.post("/v1/doseid/parse-text", response_model=DoseIDTextAnalyzeResponse)
def parse_doseid_text_endpoint(req: DoseIDTextAnalyzeRequest) -> DoseIDTextAnalyzeResponse:
    return _build_doseid_text_response(
        req.text,
        parser_strategy=req.parser_strategy,
        parser_model=req.parser_model,
        allow_fallback=req.allow_fallback,
    )


@app.post("/v1/doseid/calculate", response_model=DoseIDCalculateResponse)
def calculate_doseid_endpoint(req: DoseIDCalculateRequest) -> DoseIDCalculateResponse:
    try:
        patient_context = _doseid_patient_context_from_partial_input(req.patient, renal_mode=req.renal_mode)
        medication_ids = [selection.medication_id for selection in req.selections]
        indication_ids = {
            selection.medication_id: selection.indication_id or default_indication_id(selection.medication_id)
            for selection in req.selections
        }
        follow_up_questions = _doseid_missing_input_questions(
            medication_ids=medication_ids,
            indication_ids=indication_ids,
            patient_context=patient_context,
        )
        if follow_up_questions:
            return DoseIDCalculateResponse(
                status="needs_more_info",
                recommendations=[],
                patientContext=patient_context,
                assumptions=[],
                warnings=[],
                missingInputs=[DOSEID_FIELD_LABELS[item.id] for item in follow_up_questions],
                followUpQuestions=follow_up_questions,
            )
        recommendations, assumptions, warnings = _doseid_recommendations_ready(
            medication_ids=medication_ids,
            indication_ids=indication_ids,
            patient_context=patient_context,
        )
    except DoseIDError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return DoseIDCalculateResponse(
        status="ready",
        recommendations=recommendations,
        patientContext=patient_context,
        assumptions=assumptions,
        warnings=warnings,
        missingInputs=[],
        followUpQuestions=[],
    )


@app.get("/v1/immunoid/agents", response_model=ImmunoAgentListResponse)
def list_immunoid_supported_agents() -> ImmunoAgentListResponse:
    return ImmunoAgentListResponse(agents=list_immunoid_agents())


@app.get("/v1/immunoid/regimens", response_model=ImmunoRegimenListResponse)
def list_immunoid_supported_regimens() -> ImmunoRegimenListResponse:
    return ImmunoRegimenListResponse(regimens=list_immunoid_regimens())


@app.post("/v1/immunoid/analyze", response_model=ImmunoAnalyzeResponse)
def analyze_immunoid_endpoint(req: ImmunoAnalyzeRequest) -> ImmunoAnalyzeResponse:
    return ImmunoAnalyzeResponse(**analyze_immunoid(req.model_dump()))


@app.post("/v1/mechid/analyze", response_model=MechIDAnalyzeResponse)
def analyze_mechid_endpoint(req: MechIDAnalyzeRequest) -> MechIDAnalyzeResponse:
    warnings: List[str] = []
    try:
        payload = analyze_mechid(
            organism=req.organism,
            susceptibility_results=req.susceptibility_results,
            tx_context=req.tx_context.model_dump() if req.tx_context is not None else None,
        )
    except MechIDEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    dosing_recommendations: List[DoseIDDoseRecommendation] = []
    treatment_duration_guidance, monitoring_recommendations = _build_mechid_duration_monitoring_guidance(
        organism=payload["organism"],
        final_results=payload["final_results"],
        tx_context=req.tx_context.model_dump(by_alias=True) if req.tx_context is not None else None,
        therapy_notes=payload["therapy_notes"],
    )
    if req.dose_context is not None:
        try:
            patient = normalize_patient(
                age_years=req.dose_context.patient.age_years,
                sex=req.dose_context.patient.sex,
                total_body_weight_kg=req.dose_context.patient.total_body_weight_kg,
                height_cm=req.dose_context.patient.height_cm,
                serum_creatinine_mg_dl=req.dose_context.patient.serum_creatinine_mg_dl,
            )
            dosing_recommendations = [
                _doseid_recommendation_model(item)
                for item in suggest_mechid_doses(
                    organism=payload["organism"],
                    final_results=payload["final_results"],
                    therapy_notes=payload["therapy_notes"],
                    tx_context=req.tx_context.model_dump(by_alias=True) if req.tx_context is not None else None,
                    patient=patient,
                    renal_mode=req.dose_context.renal_mode,
                    max_suggestions=req.dose_context.max_suggestions,
                )
            ]
        except DoseIDError as exc:
            warnings.append(str(exc))

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
        treatmentDurationGuidance=treatment_duration_guidance,
        monitoringRecommendations=monitoring_recommendations,
        dosingRecommendations=dosing_recommendations,
        references=payload["references"],
        warnings=warnings,
    )


@app.post("/v1/mechid/analyze-text", response_model=MechIDTextAnalyzeResponse)
def analyze_mechid_text_endpoint(req: MechIDTextAnalyzeRequest) -> MechIDTextAnalyzeResponse:
    return _build_mechid_text_response(
        req.text,
        parser_strategy=req.parser_strategy,
        parser_model=req.parser_model,
        allow_fallback=req.allow_fallback,
    )


@app.post("/v1/mechid/analyze-image", response_model=MechIDImageAnalyzeResponse)
def analyze_mechid_image_endpoint(req: MechIDImageAnalyzeRequest) -> MechIDImageAnalyzeResponse:
    try:
        parsed = parse_mechid_image_with_openai(
            image_data_url=req.image_data_url,
            filename=req.filename,
            parser_model=req.parser_model,
        )
    except LLMParserError as exc:
        raise HTTPException(status_code=502, detail=f"Image extraction failed: {exc}") from exc

    mechid_result = _build_mechid_response_from_parsed(
        text=_mechid_canonical_text(
            MechIDTextParsedRequest(
                organism=parsed.get("organism"),
                mentionedOrganisms=parsed.get("mentionedOrganisms", []),
                resistancePhenotypes=parsed.get("resistancePhenotypes", []),
                susceptibilityResults=parsed.get("susceptibilityResults", {}),
                txContext=parsed.get("txContext", {}),
            )
        ),
        parsed=parsed,
        parser_name=str(parsed.get("parser") or "openai-mechid-image"),
    )
    return MechIDImageAnalyzeResponse(
        parser=str(parsed.get("parser") or "openai-mechid-image"),
        imageFilename=req.filename,
        sourceSummary="Uploaded antimicrobial susceptibility test image.",
        mechidResult=mechid_result,
    )


@app.post("/v1/assistant/mechid-image", response_model=AssistantTurnResponse)
def assistant_mechid_image(req: MechIDImageAnalyzeRequest) -> AssistantTurnResponse:
    try:
        parsed = parse_mechid_image_with_openai(
            image_data_url=req.image_data_url,
            filename=req.filename,
            parser_model=req.parser_model,
        )
    except LLMParserError as exc:
        raise HTTPException(status_code=502, detail=f"Image extraction failed: {exc}") from exc

    parsed_request = MechIDTextParsedRequest(
        organism=parsed.get("organism"),
        mentionedOrganisms=parsed.get("mentionedOrganisms", []),
        resistancePhenotypes=parsed.get("resistancePhenotypes", []),
        susceptibilityResults=parsed.get("susceptibilityResults", {}),
        txContext=parsed.get("txContext", {}),
    )
    canonical_text = _mechid_canonical_text(parsed_request) or "Uploaded antimicrobial susceptibility test image."
    mechid_result = _build_mechid_response_from_parsed(
        text=canonical_text,
        parsed=parsed,
        parser_name=str(parsed.get("parser") or "openai-mechid-image"),
    )

    state = AssistantState(
        workflow="mechid",
        stage="mechid_confirm",
        mechidText=canonical_text,
    )
    review_message, narration_refined = _assistant_mechid_review_message(mechid_result)
    assistant_message = (
        "I extracted this isolate report from the uploaded image. Please confirm or correct anything that looks off before I run the interpretation. "
        + review_message
    )
    return AssistantTurnResponse(
        assistantMessage=assistant_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=_assistant_mechid_review_options(mechid_result, established_syndrome=state.established_syndrome),
        mechidAnalysis=mechid_result,
        tips=[
            "If the extraction looks right, select the clinical syndrome so I can tailor the therapy recommendation.",
            "If anything is off, type the correction in plain language, for example 'meropenem resistant' or 'this is Klebsiella, not E. coli'.",
        ],
    )


@app.post("/v1/assistant/antibiogram-upload", response_model=AssistantTurnResponse)
def assistant_antibiogram_upload(req: AntibiogramUploadRequest) -> AssistantTurnResponse:
    """Load an institutional antibiogram image into the session state for antibiogram-aware empiric therapy."""
    state = req.state if req.state is not None else AssistantState()
    try:
        antibiogram = parse_antibiogram_image_with_openai(
            image_data_url=req.image_data_url,
            filename=req.filename,
        )
    except LLMParserError as exc:
        raise HTTPException(status_code=502, detail=f"Antibiogram extraction failed: {exc}") from exc

    state.institutional_antibiogram = antibiogram

    institution = antibiogram.get("institution") or "your institution"
    year = antibiogram.get("year")
    year_str = f" ({year})" if year else ""
    organisms: dict = antibiogram.get("organisms", {})
    org_count = len(organisms)
    org_names_list = list(organisms.keys())
    if org_count <= 5:
        org_names = ", ".join(org_names_list)
    else:
        org_names = ", ".join(org_names_list[:5]) + f" and {org_count - 5} more"
    confidence = antibiogram.get("confidence", "medium")
    confidence_note = " (extraction confidence: low — please review)" if confidence == "low" else ""
    ambiguities = antibiogram.get("ambiguities", [])
    ambiguity_note = f" Flagged ambiguities: {'; '.join(ambiguities[:3])}." if ambiguities else ""

    # Build context-aware message — mention the active syndrome if one is established
    syndrome_note = ""
    if state.established_syndrome:
        syndrome_note = (
            f" Since we're working up **{state.established_syndrome}**, "
            "I can immediately give you empiric coverage recommendations using your local resistance data."
        )
    elif not state.consult_organisms:
        syndrome_note = (
            " Ask me 'What's the best empiric coverage for HAP?' or start a syndrome workup "
            "and I'll tailor recommendations to your local data."
        )

    assistant_message = (
        f"Antibiogram loaded — {institution}{year_str}, {org_count} organism{'s' if org_count != 1 else ''}{confidence_note}. "
        f"Organisms on file: {org_names}.{ambiguity_note}"
        f"{syndrome_note} "
        "I'll flag any agent where local susceptibility is below 80% and suggest alternatives with better coverage at your institution."
    )

    # Offer empiric therapy first if syndrome is known; otherwise offer workup options
    antibiogram_options: list[AssistantOption] = []
    if state.established_syndrome:
        antibiogram_options.append(AssistantOption(value="empiric_therapy", label=f"Empiric therapy for {state.established_syndrome}"))
    else:
        antibiogram_options.append(AssistantOption(value="empiric_therapy", label="Ask about empiric therapy"))
    if not state.consult_organisms:
        antibiogram_options.append(AssistantOption(value="mechid", label="Paste culture results"))
    if not state.established_syndrome:
        antibiogram_options.append(AssistantOption(value="probid", label="Start syndrome workup"))
    if _is_mid_consult(state):
        antibiogram_options.append(AssistantOption(value="consult_summary", label="Full consult summary"))

    tips = [
        "I'll flag agents with <80% local susceptibility and recommend alternatives — ask about any syndrome or organism.",
        "You can upload a new antibiogram at any time to update the local resistance data.",
    ]
    return AssistantTurnResponse(
        assistantMessage=assistant_message,
        assistantNarrationRefined=False,
        state=state,
        options=antibiogram_options,
        tips=tips,
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
    if module.id == "inv_mold":
        action_text = {
            "observe": "Current action zone favors broadening the differential over anchoring on mold.",
            "test": "Current action zone favors targeted mycologic or tissue confirmation while keeping mold on the differential.",
            "treat": "Current action zone supports mold-active therapy while diagnostic confirmation continues.",
        }.get(recommendation, f"Current action zone: {recommendation}.")
        reasons.append(action_text)
    else:
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


def _endo_recommendation_floor(
    *,
    recommendation: str,
    prep_findings: dict[str, str],
    harm_findings: dict[str, str],
) -> tuple[str, str | None]:
    if recommendation != "observe":
        return recommendation, None

    prosthetic_or_device = any(
        harm_findings.get(item_id) == "present"
        for item_id in {"endo_prosthetic_valve", "endo_cied"}
    )
    if not prosthetic_or_device:
        return recommendation, None

    major_microbiology_present = any(
        prep_findings.get(item_id) == "present"
        for item_id in {
            "endo_bcx_major_typical",
            "endo_bcx_major_persistent",
            "endo_bcx_saureus_multi",
            "endo_bcx_cons_prosthetic_multi",
            "endo_bcx_efaecalis_multi",
            "endo_bcx_enterococcus_prosthetic_multi",
            "endo_bcx_nbhs_multi",
            "endo_coxiella_major",
        }
    )
    if not major_microbiology_present:
        return recommendation, None

    return (
        "test",
        "Because prosthetic valve or intracardiac device risk is present alongside major endocarditis microbiology, I raised the action floor to diagnostic testing even though the raw probability remained just below the default testing threshold.",
    )


def _dfi_recommendation_floor(
    *,
    recommendation: str,
    prep_findings: dict[str, str],
) -> tuple[str, str | None]:
    if recommendation == "treat":
        return recommendation, None

    has_local_infection = any(
        prep_findings.get(item_id) == "present"
        for item_id in {
            "dfi_local_inflammation_2plus",
            "dfi_purulence",
            "dfi_erythema_ge2cm_or_deep",
        }
    )
    has_severe_feature = any(
        prep_findings.get(item_id) == "present"
        for item_id in {
            "dfi_systemic_toxicity",
            "dfi_deep_abscess_or_gangrene",
        }
    )
    if has_local_infection and has_severe_feature:
        return (
            "treat",
            "Because bedside findings already suggest a severe diabetic foot infection with systemic toxicity or destructive deep-tissue features, I raised the action floor to treatment rather than waiting for more testing to justify starting therapy.",
        )

    if recommendation == "observe" and any(
        prep_findings.get(item_id) == "present"
        for item_id in {
            "dfi_probe_to_bone_positive",
            "dfi_xray_osteomyelitis",
            "dfi_mri_osteomyelitis_or_abscess",
            "dfi_bone_biopsy_culture_pos",
            "dfi_bone_histology_pos",
        }
    ):
        return (
            "test",
            "Because there is already meaningful diabetic foot osteomyelitis signal, I raised the action floor to further testing or directed management even though the raw probability remained below the generic testing threshold.",
        )

    return recommendation, None


def _build_recommendation_summary(
    *,
    module: SyndromeModule,
    recommendation: str,
    prep_findings: dict[str, str],
    preset_id: str | None = None,
) -> tuple[str | None, List[str]]:
    if module.id == "inv_mold":
        next_steps: List[str] = []
        bal_based_micro_done = any(
            item_id in prep_findings
            for item_id in {
                "imi_bal_gm_odi10",
                "imi_aspergillus_pcr_bal",
                "imi_aspergillus_culture_resp",
                "imi_mucorales_pcr_bal",
            }
        )
        if recommendation == "observe":
            summary = (
                "Invasive mold infection is not strongly supported by the current data, so I would broaden the differential rather than anchor on mold right now."
            )
            next_steps.append(
                "Reassess alternative diagnoses such as bacterial pneumonia, other fungal infection, nocardiosis, malignancy, drug toxicity, or other noninfectious inflammatory lung disease."
            )
            return summary, next_steps

        if recommendation == "test":
            summary = (
                "Invasive mold infection remains on the differential, but I would still want better microbiologic or tissue confirmation before calling this established disease."
            )
            if not bal_based_micro_done:
                next_steps.append(
                    "If feasible, obtain BAL and send fungal culture/cytology plus galactomannan and Aspergillus PCR."
                )
            next_steps.append(
                "If the lesion is accessible and the procedure is safe, pursue tissue biopsy for histopathology and fungal culture."
            )
            next_steps.append(
                "Keep competing diagnoses in play rather than assuming all compatible imaging is mold."
            )
            return summary, next_steps

        if recommendation == "treat":
            summary = (
                "Invasive mold infection is concerning enough that empiric mold-active therapy is reasonable while the diagnosis is being confirmed and competing explanations are still being reassessed."
            )
            if not bal_based_micro_done:
                next_steps.append(
                    "If feasible, obtain BAL and send fungal culture/cytology plus galactomannan and Aspergillus PCR."
                )
            next_steps.append(
                "Start mold-active therapy and reassess with follow-up chest CT to make sure the radiographic trajectory is improving rather than worsening."
            )
            next_steps.append(
                "If the patient is not responding clinically or the CT chest is worsening despite treatment, pursue tissue biopsy for histopathology and fungal culture if the lesion is accessible and the procedure is safe."
            )
            next_steps.append(
                "Continue to reassess alternative diagnoses and the possibility of a non-Aspergillus mold process."
            )
            return summary, next_steps

        return None, []

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

    if module.id == "diabetic_foot_infection":
        next_steps: List[str] = []
        severe_features = any(
            prep_findings.get(item_id) == "present"
            for item_id in {"dfi_systemic_toxicity", "dfi_deep_abscess_or_gangrene"}
        )
        local_infection = any(
            prep_findings.get(item_id) == "present"
            for item_id in {"dfi_local_inflammation_2plus", "dfi_purulence", "dfi_erythema_ge2cm_or_deep"}
        )
        pad_or_ischemia = prep_findings.get("dfi_host_pad_or_ischemia") == "present"
        osteomyelitis_signal = any(
            prep_findings.get(item_id) == "present"
            for item_id in {
                "dfi_probe_to_bone_positive",
                "dfi_exposed_bone",
                "dfi_xray_osteomyelitis",
                "dfi_mri_osteomyelitis_or_abscess",
                "dfi_bone_biopsy_culture_pos",
                "dfi_bone_histology_pos",
            }
        )
        xray_done = prep_findings.get("dfi_xray_na", "unknown") != "present"
        mri_done = prep_findings.get("dfi_mri_na", "unknown") != "present"
        ptb_done = prep_findings.get("dfi_probe_to_bone_na", "unknown") != "present"
        bone_sampling_done = prep_findings.get("dfi_bone_biopsy_na", "unknown") != "present"

        if recommendation == "observe":
            summary = (
                "The current data do not support an infected diabetic foot ulcer strongly enough to justify antibiotics right now."
            )
            if not local_infection:
                summary += (
                    " For a clinically uninfected ulcer, I would focus on wound care, off-loading, perfusion assessment, and local follow-up rather than empiric antibiotics."
                )
            if osteomyelitis_signal and not severe_features:
                summary += (
                    " Because there is still some osteomyelitis signal, I would at least complete the initial bone-oriented workup rather than dismissing the concern."
                )
            if not ptb_done:
                next_steps.append("Perform a bedside probe-to-bone test if it has not been done yet.")
            if not xray_done:
                next_steps.append("Obtain plain foot radiographs as part of the initial osteomyelitis screen.")
            if prep_findings.get("dfi_esr_high", "unknown") == "unknown" and prep_findings.get("dfi_crp_high", "unknown") == "unknown":
                next_steps.append("Send ESR and/or CRP if bone infection is still on the differential.")
            return summary, next_steps

        if recommendation == "test":
            summary = (
                "This diabetic foot case still sits in a zone where better diagnostic definition should guide whether prolonged antibiotic therapy or surgery is needed."
            )
            if severe_features:
                summary += (
                    " Even while testing continues, I would manage this with the urgency of a potentially limb-threatening infection."
                )
            else:
                summary += (
                    " In a hemodynamically stable patient, this is the range where I would usually gather the next highest-yield data before committing to a long treatment course."
                )
            if not ptb_done:
                next_steps.append("Perform a bedside probe-to-bone test if it has not been done yet.")
            if not xray_done:
                next_steps.append("Obtain plain foot radiographs as part of the initial workup.")
            if prep_findings.get("dfi_esr_high", "unknown") == "unknown":
                next_steps.append("Check ESR because a marked elevation can materially raise concern for osteomyelitis.")
            if prep_findings.get("dfi_crp_high", "unknown") == "unknown":
                next_steps.append("Check CRP as an adjunct inflammatory marker.")
            if osteomyelitis_signal and not mri_done and prep_findings.get("dfi_mri_osteomyelitis_or_abscess", "unknown") == "unknown":
                next_steps.append("If uncertainty persists after bedside exam, plain films, and inflammatory markers, obtain MRI.")
            if osteomyelitis_signal and not bone_sampling_done:
                next_steps.append("If osteomyelitis remains likely and the patient is stable enough, obtain bone culture or histology when feasible rather than relying on superficial swabs.")
            if severe_features or pad_or_ischemia:
                next_steps.append("Get urgent surgical and vascular input if there is gangrene, deep abscess, or ischemia.")
            return summary, next_steps

        if recommendation == "treat":
            if severe_features and local_infection:
                summary = (
                    "This looks severe enough that I would start antibiotics now and involve surgery urgently rather than waiting for more tests to justify treatment."
                )
            elif osteomyelitis_signal:
                summary = (
                    "The probability of diabetic foot osteomyelitis is high enough that treatment is justified, but I would still try to anchor therapy to deep tissue or bone sampling when feasible, especially if a long course is anticipated."
                )
            else:
                summary = (
                    "Diabetic foot infection is likely enough that syndrome-directed treatment is justified now."
                )
            if local_infection and not severe_features:
                summary += (
                    " If this is a milder infection without ischemia or polymicrobial concern, empiric therapy can often focus on gram-positive pathogens first; broader coverage makes more sense for deeper, ischemic, necrotic, or severe presentations."
                )
            if severe_features:
                next_steps.append("Start antibiotics now and obtain urgent surgical assessment for drainage, debridement, or amputation planning as needed.")
            if pad_or_ischemia:
                next_steps.append("Reassess perfusion urgently and involve vascular surgery because ischemia changes both healing and source-control decisions.")
            if osteomyelitis_signal and not bone_sampling_done:
                next_steps.append("If feasible, obtain bone culture or histology before or early in therapy, especially if prolonged osteomyelitis treatment is expected.")
            if prep_findings.get("dfi_forefoot_only") == "present" and not pad_or_ischemia and prep_findings.get("dfi_exposed_bone") != "present" and not severe_features:
                next_steps.append("A nonsurgical antibiotic-first strategy can be reasonable for selected forefoot osteomyelitis cases without PAD, exposed bone, or an immediate drainage need.")
            return summary, next_steps

        return None, []

    if module.id != "endo":
        return None, []

    tee_done = prep_findings.get("endo_tee", "unknown") != "unknown"
    pet_done = prep_findings.get("endo_pet", "unknown") != "unknown"
    advanced_imaging_done = tee_done or pet_done
    next_steps: List[str] = []

    if recommendation == "observe":
        summary = (
            "Endocarditis probability is below the observation threshold, so treating this as complicated bacteremia is reasonable."
        )
        if not advanced_imaging_done:
            summary += " TEE is probably not necessary unless new features increase concern for endocarditis."
        return summary, next_steps

    if recommendation == "test":
        summary = (
            "Endocarditis probability remains in an intermediate range, so further diagnostic testing is appropriate before escalating to full endocarditis-directed treatment. "
            "If the main question is whether more imaging is needed, this is a range where TEE is appropriate."
        )
        if not tee_done:
            next_steps.append("Consider TEE if it has not been performed yet.")
        if not pet_done:
            next_steps.append(
                "Consider FDG PET/CT if prosthetic valve/device infection remains a concern and it has not been performed yet."
            )
        next_steps.append(
            "Consider gated CTA when peri-annular extension, root abscess, or other structural complications are the main concern."
        )
        return summary, next_steps

    if recommendation == "treat":
        summary = (
            "Endocarditis probability is high enough that endocarditis-directed treatment is justified based on the current risk-benefit balance. "
            "I would not wait for another imaging test to justify treatment, although TEE can still help define complications and guide surgery."
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


def _build_duration_monitoring_guidance(
    *,
    module: SyndromeModule,
    recommendation: str,
    prep_findings: dict[str, str],
) -> tuple[List[str], List[str]]:
    if recommendation == "observe":
        return [], []

    duration: List[str] = []
    monitoring: List[str] = []

    if module.id == "endo":
        duration = [
            "Most infective endocarditis courses are measured in weeks, not days. Native-valve disease is often treated for about 4 to 6 weeks, while Staphylococcus aureus, Enterococcus, prosthetic-valve infection, or other complicated disease often needs 6 weeks or longer.",
            "If bacteremia clears slowly, source control is incomplete, or there are metastatic foci, the effective treatment clock usually follows the first truly negative blood culture and adequate source control rather than the admission date.",
        ]
        monitoring = [
            "Repeat blood cultures every 24 to 48 hours until bloodstream clearance is documented.",
            "Reassess for heart failure, conduction changes, embolic complications, persistent bacteremia, and whether repeat TEE or surgical input is needed.",
            "Monitor drug-specific toxicity during the prolonged course, especially renal function and vancomycin exposure if vancomycin is used, or CBC/CMP for other long-course regimens.",
        ]
        return duration, monitoring

    if module.id == "inv_candida":
        duration = [
            "Uncomplicated candidemia is typically treated for at least 14 days after the first negative blood culture and clinical improvement.",
            "If there is endocarditis, endophthalmitis, deep organ involvement, an infected device, or another uncontrolled focus, duration is usually longer and tied to source control plus site-specific follow-up.",
        ]
        monitoring = [
            "Repeat blood cultures daily or every other day until candidemia has clearly cleared.",
            "Confirm source control, including line removal when appropriate and reassessment for deep foci.",
            "Follow CBC/CMP and liver tests during therapy, and reassess ocular symptoms or ophthalmologic evaluation when clinically indicated.",
        ]
        return duration, monitoring

    if module.id == "inv_mold":
        duration = [
            "Invasive aspergillosis is usually treated for at least 6 to 12 weeks, and often longer when immunosuppression persists or radiographic lesions have not clearly improved.",
            "If the syndrome is more consistent with mucormycosis or another non-Aspergillus mold, therapy commonly extends for weeks to months and depends heavily on immune recovery and serial imaging response.",
        ]
        monitoring = [
            "Reassess with follow-up chest CT, often after about 2 weeks or sooner if the patient worsens clinically.",
            "Follow drug-specific toxicity closely: CBC/CMP and liver tests for azoles, azole drug levels when relevant, and creatinine plus potassium/magnesium for amphotericin formulations.",
            "Keep reassessing whether tissue diagnosis, BAL-based microbiology, or revision of the differential is needed if the trajectory is not improving.",
        ]
        return duration, monitoring

    if module.id == "active_tb":
        duration = [
            "For drug-susceptible pulmonary TB, the usual starting framework is 2 months of RIPE followed by 4 additional months of isoniazid plus rifampin if the clinical and microbiologic response is appropriate.",
            "Duration often needs to be extended when there is slow culture conversion, cavitary disease with persistent positive cultures, CNS involvement, major drug resistance, or another complicated extrapulmonary focus.",
        ]
        monitoring = [
            "Follow sputum smear or culture and clinical response through treatment, with repeat respiratory sampling until microbiologic improvement is documented.",
            "Monitor liver tests and symptoms of hepatitis during RIPE therapy, especially in patients with baseline liver risk or symptoms.",
            "Track vision and color discrimination if ethambutol is being continued, and review rifamycin drug interactions throughout the course.",
        ]
        return duration, monitoring

    if module.id == "diabetic_foot_infection":
        osteomyelitis_signal = any(
            prep_findings.get(item_id) == "present"
            for item_id in {
                "dfi_probe_to_bone_positive",
                "dfi_exposed_bone",
                "dfi_xray_osteomyelitis",
                "dfi_mri_osteomyelitis_or_abscess",
                "dfi_bone_biopsy_culture_pos",
                "dfi_bone_histology_pos",
            }
        )
        surgery_done = prep_findings.get("dfi_surgery_debridement_done") == "present"
        amputation_done = prep_findings.get("dfi_minor_amputation_done") == "present"
        positive_margin = prep_findings.get("dfi_positive_bone_margin") == "present"
        pad_or_ischemia = prep_findings.get("dfi_host_pad_or_ischemia") == "present"

        if osteomyelitis_signal:
            if amputation_done and positive_margin:
                duration = [
                    "After minor amputation or bone resection with a positive residual bone margin, the usual framework is up to about 3 weeks of antibiotic therapy.",
                    "If the margin status is uncertain, the final course should still follow the operative findings, pathology or culture results, and whether additional infected bone remains in place.",
                ]
            elif amputation_done:
                duration = [
                    "After bone resection or minor amputation with clean margins, duration is often much shorter than a full nonsurgical osteomyelitis course and should be tied to the residual soft-tissue infection burden, if any.",
                    "If all infected bone appears to have been removed, prolonged osteomyelitis-duration therapy is often unnecessary."
                ]
            else:
                duration = [
                    "For diabetic foot osteomyelitis managed without bone resection or amputation, the usual framework is about 6 weeks of antibiotic therapy.",
                    "Selected forefoot osteomyelitis cases without PAD, exposed bone, or an immediate need for drainage can sometimes be managed nonsurgically, but that decision depends on source control and clinical stability.",
                ]
            monitoring = [
                "Follow serial wound appearance, drainage, erythema, pain, and the need for additional debridement or off-loading.",
                "Reassess perfusion and healing potential, especially if PAD or ischemia is present.",
                "For osteomyelitis, judge remission over months rather than days; guideline-based follow-up often looks at outcomes at 6 months or more after antibiotics end.",
            ]
            return duration, monitoring

        duration = [
            "For a soft-tissue diabetic foot infection without convincing osteomyelitis, the usual antibiotic course is about 1 to 2 weeks.",
            "If the infection is extensive, slow to improve, or complicated by severe PAD or delayed source control, treatment may need to continue for up to about 3 to 4 weeks."
        ]
        if surgery_done:
            duration.append(
                "After surgical debridement for a moderate or severe soft-tissue diabetic foot infection, some patients can be treated with a shorter roughly 10-day postoperative course if the trajectory is improving."
            )
        monitoring = [
            "Reassess the wound frequently for shrinking erythema, less drainage, less pain, and a clean granulating base rather than relying on labs alone.",
            "Keep reviewing whether more debridement, drainage, off-loading, or vascular optimization is needed."
        ]
        if pad_or_ischemia:
            monitoring.append("PAD or ischemia should trigger close perfusion follow-up because it slows healing and raises the risk of treatment failure.")
        return duration, monitoring

    return duration, monitoring


def _build_mechid_duration_monitoring_guidance(
    *,
    organism: str,
    final_results: Dict[str, str],
    tx_context: Dict[str, str] | None,
    therapy_notes: List[str],
) -> tuple[List[str], List[str]]:
    syndrome = (tx_context or {}).get("syndrome", "Not specified")
    focus_detail = (tx_context or {}).get("focusDetail", "Not specified")
    notes_text = " ".join(therapy_notes).lower()
    duration: List[str] = []
    monitoring: List[str] = []

    if focus_detail == "Prosthetic joint infection":
        duration = [
            "For prosthetic joint infection, duration depends heavily on the operative plan, whether hardware is being retained, and the organism, so I would not force this into the same simple timeline as native osteomyelitis."
        ]
        if organism in {"Staphylococcus aureus", "Staphylococcus lugdunensis"}:
            duration.append(
                "If this is staphylococcal PJI with retained hardware, I would usually think in terms of a prolonged companion-drug strategy rather than a short native-joint-style course."
            )
        monitoring = [
            "Keep the antibiotic plan aligned with the surgical pathway, such as DAIR versus one-stage or two-stage exchange, because that changes both duration and what counts as adequate source control.",
            "Follow wound drainage, pain, range of motion or function, inflammatory markers when useful, and weekly CBC/CMP plus drug-specific toxicity labs during prolonged therapy.",
        ]
        if organism in {"Staphylococcus aureus", "Staphylococcus lugdunensis"}:
            monitoring.append(
                "If a rifampin-based companion strategy is being used for retained staphylococcal hardware, make sure it is paired with another active agent and watch closely for drug interactions and hepatotoxicity."
            )
        return duration, monitoring

    if organism == "Staphylococcus aureus":
        if syndrome == "Bloodstream infection" or focus_detail == "Endocarditis":
            duration = [
                "If this is uncomplicated S. aureus bacteremia with rapid clearance and no metastatic focus, treatment is usually at least 14 days after the first negative blood culture; true endovascular infection or endocarditis usually needs 4 to 6 weeks or longer.",
                "The final duration should follow the first documented culture clearance plus whether endocarditis, osteomyelitis, epidural infection, hardware infection, or another metastatic focus is present.",
            ]
            monitoring = [
                "Repeat blood cultures every 24 to 48 hours until clearance is documented.",
                "If vancomycin is used for MRSA, follow renal function and AUC or vancomycin levels; if daptomycin is used, follow CK at least weekly and watch for eosinophilic pneumonia or myopathy.",
                "Keep reassessing for endocarditis, metastatic foci, removable lines, drainable collections, and source control.",
            ]
            return duration, monitoring
        if syndrome == "Bone/joint infection":
            duration = [
                "For staphylococcal osteomyelitis or other deep bone and joint infection, treatment commonly extends for about 6 weeks, and sometimes longer when hardware remains or debridement is incomplete."
            ]
            monitoring = [
                "Follow source control, wound or joint drainage status, inflammatory markers when useful, and weekly CBC/CMP plus drug-specific toxicity labs during prolonged therapy."
            ]
            return duration, monitoring

    if organism.startswith("Enterococcus"):
        if syndrome == "Bloodstream infection" or focus_detail == "Endocarditis":
            if organism == "Enterococcus faecalis" and focus_detail == "Endocarditis" and final_results.get("Ampicillin") == "Susceptible":
                duration = [
                    "Ampicillin-susceptible Enterococcus faecalis endocarditis is usually treated with a prolonged beta-lactam synergy course, commonly about 6 weeks, with the final clock anchored to culture clearance and valve complexity."
                ]
                monitoring = [
                    "Repeat blood cultures until clearance and keep the echo strategy and source-control review active throughout the case.",
                    "If using Ampicillin plus Ceftriaxone, follow CBC, renal function, and hepatic chemistry during the prolonged dual beta-lactam course rather than centering monitoring on salvage-agent toxicities.",
                ]
                return duration, monitoring
            duration = [
                "Uncomplicated enterococcal bacteremia is often treated for at least 14 days after clearance, while enterococcal endocarditis usually requires a prolonged course such as 4 to 6 weeks and sometimes synergy-based treatment depending on the regimen."
            ]
            monitoring = [
                "Repeat blood cultures until clearance and reassess whether endocarditis or another deep focus is present.",
                "If daptomycin is used for VRE, follow CK at least weekly; if linezolid is used, follow CBC and watch for neuropathy or serotonin-toxicity issues during longer courses.",
            ]
            if organism == "Enterococcus faecium":
                duration.insert(
                    0,
                    "Enterococcus faecium endovascular infection can be especially difficult to sterilize, so I would be cautious about calling this uncomplicated bacteremia until clearance is documented and endocarditis has been addressed.",
                )
                monitoring.append(
                    "For E. faecium bacteremia or endocarditis, keep a low threshold for repeat echocardiography, source-control review, and expert input because persistent bacteremia can be difficult to clear."
                )
                if final_results.get("Vancomycin") == "Resistant" and focus_detail == "Endocarditis":
                    duration.insert(
                        1,
                        "For VRE Enterococcus faecium endocarditis, I would usually think in terms of a high-dose daptomycin-based combination for at least 8 weeks rather than a shorter standard enterococcal course.",
                    )
                    monitoring.append(
                        "With high-dose daptomycin-based combination therapy, follow CK, CBC, renal function, and electrolytes closely, especially if a beta-lactam or fosfomycin partner is added."
                    )
            return duration, monitoring
        if syndrome == "Bone/joint infection":
            duration = [
                "Enterococcal bone or joint infection often needs a prolonged course, commonly around 6 weeks, adjusted to source control and hardware status."
            ]
            monitoring = [
                "Follow CBC/CMP and drug-specific toxicity, plus clinical response and any required orthopedic or source-control reassessment."
            ]
            return duration, monitoring

    if organism == "Mycobacterium tuberculosis complex":
        duration = [
            "Drug-susceptible pulmonary TB usually follows a 2-month RIPE intensive phase followed by 4 months of isoniazid plus rifampin if the response is appropriate.",
            "Longer courses are common when there is slow culture conversion, cavitary disease with persistent positive cultures, CNS disease, or other complicated extrapulmonary infection."
        ]
        monitoring = [
            "Track sputum culture conversion and clinical response through treatment.",
            "Follow liver tests and hepatitis symptoms during RIPE therapy, and monitor vision if ethambutol continues.",
            "Review rifamycin drug interactions throughout the course."
        ]
        return duration, monitoring

    if syndrome == "Bloodstream infection":
        duration = [
            "For many uncomplicated Gram-negative bloodstream infections with good source control and clinical response, total therapy is often measured in about 7 to 14 days, but longer courses are used when there is uncontrolled source or another deep focus."
        ]
        monitoring = [
            "Make sure source control is adequate and that the patient is truly improving before shortening treatment.",
            "Follow renal function and other drug-specific toxicity labs for the chosen regimen."
        ]
        return duration, monitoring

    return duration, monitoring


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return _analyze_internal(req)


def _analyze_internal(req: AnalyzeRequest) -> AnalyzeResponse:
    module = _resolve_module(req)
    if module.id == "tb_uveitis":
        response = analyze_tb_uveitis(module, req)
        if req.include_explanation:
            response.explanation_for_user = _build_probid_consult_message(module, response)
        return response

    module = module.model_copy(deep=True)
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
    recommendation, endo_floor_note = _endo_recommendation_floor(
        recommendation=recommendation,
        prep_findings=prep.analysis_findings,
        harm_findings=prep.harm_findings,
    )
    dfi_floor_note = None
    if module.id == "diabetic_foot_infection":
        recommendation, dfi_floor_note = _dfi_recommendation_floor(
            recommendation=recommendation,
            prep_findings=prep.analysis_findings,
        )
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
    if endo_floor_note:
        reasons.append(endo_floor_note)
    if dfi_floor_note:
        reasons.append(dfi_floor_note)
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
    treatment_duration_guidance, monitoring_recommendations = _build_duration_monitoring_guidance(
        module=module,
        recommendation=recommendation,
        prep_findings=prep.analysis_findings,
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
        treatmentDurationGuidance=treatment_duration_guidance,
        monitoringRecommendations=monitoring_recommendations,
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
    return [
        AssistantOption(
            value=MECHID_ASSISTANT_ID,
            label=MECHID_ASSISTANT_LABEL,
            description=MECHID_ASSISTANT_DESCRIPTION,
        ),
        AssistantOption(
            value=DOSEID_ASSISTANT_ID,
            label=DOSEID_ASSISTANT_LABEL,
            description=DOSEID_ASSISTANT_DESCRIPTION,
        ),
        AssistantOption(
            value=PROBID_ASSISTANT_ID,
            label=PROBID_ASSISTANT_LABEL,
            description=PROBID_ASSISTANT_DESCRIPTION,
        ),
        AssistantOption(
            value=IMMUNOID_ASSISTANT_ID,
            label=IMMUNOID_ASSISTANT_LABEL,
            description=IMMUNOID_ASSISTANT_DESCRIPTION,
        ),
        AssistantOption(
            value=ALLERGYID_ASSISTANT_ID,
            label=ALLERGYID_ASSISTANT_LABEL,
            description=ALLERGYID_ASSISTANT_DESCRIPTION,
        ),
    ]


def _assistant_syndrome_module_options() -> List[AssistantOption]:
    options: List[AssistantOption] = []
    for summary in store.list_summaries():
        module = store.get(summary.id)
        options.append(
            AssistantOption(
                value=summary.id,
                label=_assistant_module_label(module) if module else summary.name,
                description=(module.description[:120] + "...") if module and module.description and len(module.description) > 120 else (module.description if module else None),
            )
        )
    options.append(AssistantOption(value="restart", label="Start new consult"))
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
        _normalize_choice(module.name),
        _normalize_choice(module.id.replace("_", " ")),
        _normalize_choice(_assistant_module_label(module)),
    }
    candidates.update(_normalize_choice(alias) for alias in MODULE_ALIASES.get(module.id, []))
    return any(
        candidate and re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", text) is not None
        for candidate in candidates
    )


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


def _assistant_is_endo_imaging_question(text: str | None) -> bool:
    if not text:
        return False
    normalized = _normalize_choice(text)
    if not re.search(
        r"\b(?:tee|transesophageal echocardiogram|transesophageal echo|pet/?ct|pet ct|fdg pet/?ct|gated cta|cardiac cta|cta)\b",
        normalized,
    ):
        return False
    return any(
        token in normalized
        for token in (
            "should i order",
            "should we order",
            "should i get",
            "should we get",
            "do i need",
            "does this patient need",
            "does the patient need",
            "need a",
            "need an",
            "needs a",
            "needs an",
            "require a",
            "requires a",
            "necessary",
            "indicated",
            "is tee needed",
            "is pet ct needed",
            "is gated cta needed",
            "would you order",
            "would you get",
        )
    )


def _assistant_endo_imaging_guidance(
    analysis: AnalyzeResponse,
    *,
    case_text: str | None = None,
) -> str | None:
    if analysis.module_id != "endo":
        return None
    if not _assistant_is_endo_imaging_question(case_text):
        return None
    if analysis.recommendation == "observe":
        return (
            "Regarding TEE or advanced endocarditis imaging: with the current probability, I would not routinely order TEE right now. "
            "I would treat this as lower-risk bacteremia unless new features raise concern."
        )
    if analysis.recommendation == "test":
        return (
            "Regarding TEE or advanced endocarditis imaging: the current probability is high enough that I would pursue more endocarditis-directed imaging now. "
            "TEE is usually the next study I would prioritize. FDG PET/CT or gated CTA can help when prosthetic valve, device, or peri-annular extension is the concern."
        )
    if analysis.recommendation == "treat":
        return (
            "Regarding TEE or advanced endocarditis imaging: the current probability is high enough that I would treat this as endocarditis now rather than waiting for another test to justify therapy. "
            "TEE can still help define valve complications or surgical planning, and FDG PET/CT or gated CTA can be useful in prosthetic valve, device, or peri-annular disease."
        )
    return None


def _assistant_has_explicit_endo_imaging_result(text: str | None, modality: str) -> bool:
    if not text:
        return False
    normalized = _normalize_choice(text)
    modality_tokens = {
        "tee": ("tee", "transesophageal echocardiogram", "transesophageal echo"),
        "tte": ("tte", "transthoracic echocardiogram", "transthoracic echo"),
        "pet": ("pet/ct", "pet ct", "fdg pet/ct", "fdg pet ct"),
    }
    patterns = {
        "tee": r"\b(?:tee|transesophageal echocardiogram|transesophageal echo)\b.{0,24}\b(?:positive|negative|showing|shows|done|performed)\b|\b(?:positive|negative)\b.{0,12}\b(?:tee|transesophageal echocardiogram|transesophageal echo)\b|\b(?:vegetation|regurgitation|perforation)\b.{0,12}\b(?:on|seen on)\s+(?:tee|transesophageal echocardiogram|transesophageal echo)\b|\b(?:tee|transesophageal echocardiogram|transesophageal echo)\b.{0,16}\bwithout vegetation\b",
        "tte": r"\b(?:tte|transthoracic echocardiogram|transthoracic echo)\b.{0,24}\b(?:positive|negative|showing|shows|done|performed)\b|\b(?:positive|negative)\b.{0,12}\b(?:tte|transthoracic echocardiogram|transthoracic echo)\b|\b(?:vegetation|regurgitation|perforation)\b.{0,12}\b(?:on|seen on)\s+(?:tte|transthoracic echocardiogram|transthoracic echo)\b|\b(?:tte|transthoracic echocardiogram|transthoracic echo)\b.{0,16}\bwithout vegetation\b",
        "pet": r"\b(?:pet/?ct|pet ct|fdg pet/?ct)\b.{0,24}\b(?:positive|negative|showing|shows|done|performed)\b|\b(?:positive|negative)\b.{0,12}\b(?:pet/?ct|pet ct|fdg pet/?ct)\b|\b(?:uptake|avid)\b.{0,12}\b(?:on|seen on)\s+(?:pet/?ct|pet ct|fdg pet/?ct)\b|\b(?:pet/?ct|pet ct|fdg pet/?ct)\b.{0,16}\bwithout uptake\b",
    }
    pattern = patterns.get(modality)
    tokens = modality_tokens.get(modality)
    if pattern is None or tokens is None:
        return False
    clauses = [segment.strip() for segment in re.split(r"[.\n;]+", normalized) if segment.strip()]
    return any(any(token in clause for token in tokens) and re.search(pattern, clause) for clause in clauses)


def _assistant_sanitize_endo_imaging_question_parse(
    text_result: TextAnalyzeResponse,
    message_text: str | None,
) -> None:
    if text_result.parsed_request is None or text_result.parsed_request.module_id != "endo":
        return
    if not _assistant_is_endo_imaging_question(message_text):
        return
    findings = dict(text_result.parsed_request.findings or {})
    if "endo_tee" in findings and not _assistant_has_explicit_endo_imaging_result(message_text, "tee"):
        findings.pop("endo_tee", None)
        text_result.understood.findings_present = [
            label
            for label in text_result.understood.findings_present
            if "transesophageal echo" not in label.lower() and "tee" not in label.lower()
        ]
        text_result.understood.findings_absent = [
            label
            for label in text_result.understood.findings_absent
            if "transesophageal echo" not in label.lower() and "tee" not in label.lower()
        ]
    if "endo_pet" in findings and not _assistant_has_explicit_endo_imaging_result(message_text, "pet"):
        findings.pop("endo_pet", None)
        text_result.understood.findings_present = [
            label
            for label in text_result.understood.findings_present
            if "pet/ct" not in label.lower() and "pet ct" not in label.lower() and "fdg pet" not in label.lower()
        ]
        text_result.understood.findings_absent = [
            label
            for label in text_result.understood.findings_absent
            if "pet/ct" not in label.lower() and "pet ct" not in label.lower() and "fdg pet" not in label.lower()
        ]
    text_result.parsed_request.findings = findings


def _assistant_case_text_for_parser(
    module: SyndromeModule,
    case_text: str | None,
) -> str:
    text = (case_text or "").strip()
    if not text:
        return ""

    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "
    augmented = text

    if module.id == "spinal_epidural_abscess":
        mentions_shorthand_mri_positive = any(
            token in normalized
            for token in (
                " mri positive ",
                " spine mri positive ",
            )
        )
        has_explicit_sea_mri_result = any(
            token in normalized
            for token in (
                " epidural abscess ",
                " epidural phlegmon ",
                " spinal epidural abscess ",
                " mri shows epidural abscess ",
            )
        )
        if mentions_shorthand_mri_positive and not has_explicit_sea_mri_result:
            augmented = _append_case_text(augmented, "MRI spine shows epidural abscess or phlegmon")

    return augmented


def _assistant_append_unique_warning(text_result: TextAnalyzeResponse, warning: str) -> None:
    if warning and warning not in text_result.warnings:
        text_result.warnings.append(warning)


def _assistant_anchor_guided_case_parse(
    module: SyndromeModule,
    state: AssistantState,
    text_result: TextAnalyzeResponse,
) -> None:
    parsed_request = text_result.parsed_request
    if parsed_request is None:
        return

    valid_item_ids = {item.id for item in module.items}
    findings = dict(parsed_request.findings or {})
    filtered_findings = {item_id: item_state for item_id, item_state in findings.items() if item_id in valid_item_ids}

    module_changed = parsed_request.module_id != module.id
    preset_changed = bool(state.preset_id and parsed_request.preset_id != state.preset_id)
    filtered_any = len(filtered_findings) != len(findings)

    parsed_request.module_id = module.id
    parsed_request.module = None
    if state.preset_id:
        parsed_request.preset_id = state.preset_id
    parsed_request.findings = filtered_findings

    ordered_ids = [item_id for item_id in parsed_request.ordered_finding_ids if item_id in filtered_findings]
    for item_id in filtered_findings:
        if item_id not in ordered_ids:
            ordered_ids.append(item_id)
    parsed_request.ordered_finding_ids = ordered_ids

    if module_changed:
        _assistant_append_unique_warning(
            text_result,
            "Kept the guided consult anchored to the syndrome you already selected.",
        )
    if preset_changed:
        _assistant_append_unique_warning(
            text_result,
            "Kept the guided consult anchored to the pretest setting you already selected.",
        )
    if filtered_any:
        _assistant_append_unique_warning(
            text_result,
            "Ignored extracted findings that do not belong to the selected syndrome.",
        )


def _assistant_backfill_guided_case_rule_findings(
    module: SyndromeModule,
    state: AssistantState,
    text_result: TextAnalyzeResponse,
    case_text: str,
) -> None:
    if not case_text.strip():
        return

    rule_result = parse_text_to_request(
        store=store,
        text=case_text,
        module_hint=module.id,
        preset_hint=state.preset_id,
        include_explanation=True,
    )
    rule_request = rule_result.parsed_request
    if rule_request is None:
        return

    if text_result.parsed_request is None:
        text_result.parsed_request = rule_request
        text_result.requires_confirmation = text_result.requires_confirmation or rule_result.requires_confirmation
        for warning in rule_result.warnings:
            _assistant_append_unique_warning(text_result, warning)
        _assistant_append_unique_warning(
            text_result,
            "Recovered the guided consult parse using deterministic rule-based extraction.",
        )
        return

    parsed_request = text_result.parsed_request
    parsed_findings = dict(parsed_request.findings or {})
    added_ids: List[str] = []
    for item_id, item_state in (rule_request.findings or {}).items():
        existing_state = parsed_findings.get(item_id)
        if existing_state in {"present", "absent"}:
            continue
        parsed_findings[item_id] = item_state
        added_ids.append(item_id)

    if not added_ids:
        return

    parsed_request.findings = parsed_findings
    for item_id in rule_request.ordered_finding_ids:
        if item_id in added_ids and item_id not in parsed_request.ordered_finding_ids:
            parsed_request.ordered_finding_ids.append(item_id)

    added_labels = [item.label for item in module.items if item.id in added_ids]
    if added_labels:
        _assistant_append_unique_warning(
            text_result,
            "Added deterministic rule-based clues from the guided case text: " + ", ".join(added_labels),
        )


def _assistant_refresh_case_parse_summary(text_result: TextAnalyzeResponse) -> None:
    parsed_request = text_result.parsed_request
    if parsed_request is None:
        return

    understood, summary_warnings, summary_requires_confirmation = summarize_parsed_request(store, parsed_request)
    text_result.understood = understood
    text_result.requires_confirmation = text_result.requires_confirmation or summary_requires_confirmation
    for warning in summary_warnings:
        _assistant_append_unique_warning(text_result, warning)


def _assistant_cache_probid_case_result(
    state: AssistantState,
    text_result: TextAnalyzeResponse,
) -> None:
    if state.workflow != "probid" or not state.module_id or not (state.case_text or "").strip():
        state.probid_cached_case_result = None
        return

    state.probid_cached_case_result = {
        "moduleId": state.module_id,
        "presetId": state.preset_id,
        "caseText": state.case_text,
        "pretestFactorIds": list(state.pretest_factor_ids),
        "endoScoreFactorIds": list(state.endo_score_factor_ids),
        "textResult": {
            "parser": text_result.parser,
            "text": text_result.text,
            "parserFallbackUsed": text_result.parser_fallback_used,
            "parsedRequest": (
                text_result.parsed_request.model_dump(by_alias=True, mode="json")
                if text_result.parsed_request is not None
                else None
            ),
            "understood": text_result.understood.model_dump(by_alias=True, mode="json"),
            "warnings": list(text_result.warnings),
            "requiresConfirmation": text_result.requires_confirmation,
            "references": [reference.model_dump(by_alias=True, mode="json") for reference in text_result.references],
            "analysis": (
                text_result.analysis.model_dump(by_alias=True, mode="json")
                if text_result.analysis is not None
                else None
            ),
        },
    }


def _assistant_cached_probid_case_result(state: AssistantState) -> TextAnalyzeResponse | None:
    payload = state.probid_cached_case_result
    if not isinstance(payload, dict):
        return None
    if payload.get("moduleId") != state.module_id:
        return None
    if payload.get("presetId") != state.preset_id:
        return None
    if payload.get("caseText") != state.case_text:
        return None
    if payload.get("pretestFactorIds") != list(state.pretest_factor_ids):
        return None
    if payload.get("endoScoreFactorIds") != list(state.endo_score_factor_ids):
        return None

    text_result_payload = payload.get("textResult")
    if not isinstance(text_result_payload, dict):
        return None
    try:
        return TextAnalyzeResponse.model_validate(text_result_payload)
    except Exception:
        return None


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

    if module.id == "endo":
        if len(text_result.analysis.applied_findings) >= 1:
            return True
        if _assistant_is_endo_imaging_question(state.case_text) and informative_count >= 1:
            return True

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
    if module.id == "endo" and _assistant_is_endo_imaging_question(state.case_text):
        message = (
            "I have enough to answer the imaging question from what you gave me. "
            "If you want to add more detail, just keep typing. If not, select the clinical syndrome to get the therapy recommendation."
        )
    else:
        message = (
            "I have enough to run the consult with what you gave me. "
            "If you want to add more detail, just keep typing. If not, select the clinical syndrome to get the therapy recommendation."
        )
    if next_items:
        message += f" If you want to sharpen it further, the next useful detail would be {next_items[0][1]}."
    return message


def _assistant_case_can_run_provisional_consult(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> bool:
    if _assistant_case_is_consult_ready(module, text_result, state):
        return False
    _assistant_populate_case_review_analysis(module, text_result)
    return text_result.parsed_request is not None and text_result.analysis is not None


def _assistant_provisional_consult_message(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> str:
    next_items = _top_missing_item_specs(module, text_result.parsed_request, limit=1, state=state)
    message = (
        "I can still run a best-effort consult with the data you already gave me, but it will be more provisional because some high-yield details are still missing."
    )
    if next_items:
        message += f" The next detail most likely to change the estimate would be {next_items[0][1]}."
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
    elif _assistant_case_can_run_provisional_consult(module, text_result, state):
        pieces.append(_assistant_provisional_consult_message(module, text_result, state))
    else:
        follow_up = _assistant_single_case_follow_up(module, text_result.parsed_request, state=state)
        if follow_up:
            pieces.append(follow_up)
        elif text_result.requires_confirmation:
            pieces.append("If anything looks off, correct it or add another detail.")
        else:
            pieces.append("If this extraction matches the case, select the clinical syndrome to get the therapy recommendation.")
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
        pieces.append("If this extraction matches the case, select the clinical syndrome to get the therapy recommendation.")
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
    if module.id == "inv_mold":
        if analysis.recommendation == "treat":
            return (
                f"The estimated probability of {module.name} is about {probability}%, which makes invasive mold disease plausible enough that I would start mold-active therapy while still trying to secure microbiologic or tissue confirmation."
            )
        if analysis.recommendation == "test":
            return (
                f"The estimated probability of {module.name} is about {probability}%, which keeps invasive mold disease meaningfully on the differential and worth pursuing further mycologic or tissue confirmation."
            )
        return (
            f"The estimated probability of {module.name} is about {probability}%, which makes invasive mold disease less compelling right now than competing explanations unless new host, imaging, or microbiology data emerge."
        )
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
    if module.id == "tb_uveitis":
        intro = (
            f"The mapped COTS post-test probability is {probability:.1%}."
            f" The current action thresholds are observe at or below {analysis.thresholds.observe_probability:.1%}"
            f" and treat at or above {analysis.thresholds.treat_probability:.1%}."
        )
        if preset_label:
            intro = (
                f"The fallback preset is '{preset_label}', but the starting pretest is {analysis.pretest.base_probability:.1%} "
                "after deriving the baseline from the selected phenotype/endemicity context when available. "
                + intro
            )
        return (
            intro
            + " In this module, the displayed combined LR and stepwise LR values are back-calculated scenario effects from mapped COTS probabilities, not independent pooled diagnostic LRs."
        )
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
    if analysis.module_id == "inv_mold":
        if analysis.recommendation == "treat":
            return "Start mold-active therapy, obtain BAL-based mycology if it has not been done, repeat chest CT to assess trajectory, and consider tissue biopsy if the patient is not responding or imaging worsens."
        if analysis.recommendation == "test":
            return "Get the next highest-yield mycologic data, ideally BAL-based testing or tissue diagnosis if feasible, while keeping the differential broad."
        return "Revisit the differential and only re-escalate the mold workup if new host-risk, imaging, or mycologic signals appear."
    if analysis.recommendation == "treat":
        return "Start syndrome-directed treatment and close the highest-yield diagnostic gaps in parallel."
    if analysis.recommendation == "test":
        return "Get the highest-yield next test, imaging study, or microbiology result that would move this estimate one way or the other."
    return "Keep watching the clinical trajectory and only reopen this workup if new objective findings increase concern."


def _friendly_probid_change_mind(
    analysis: AnalyzeResponse,
    missing_suggestions: List[str] | None = None,
) -> str:
    if analysis.module_id == "inv_mold" and missing_suggestions:
        return (
            f"The results most likely to move this estimate are {_join_readable(missing_suggestions[:3])}, and tissue diagnosis would be especially helpful when feasible."
        )
    if analysis.module_id == "inv_mold":
        if analysis.recommendation == "treat":
            return "Convincing alternative pathology, nondiagnostic tissue or BAL workup, or high-quality negative mold biomarkers would lower my confidence."
        if analysis.recommendation == "test":
            return "A positive tissue diagnosis or more specific mold microbiology could push this toward treatment, while a stronger competing diagnosis could move it away from mold."
        return "New compatible CT findings, supportive mold microbiology, or tissue evidence could raise concern quickly."
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
    case_text: str | None = None,
) -> str:
    lines = [
        _friendly_probid_bottom_line(module, analysis),
        f"Probability and harm: {_friendly_probid_probability_and_harm(module, analysis)}",
        f"Why I think that: {_friendly_probid_drivers(analysis)}",
        f"What I would do next: {_friendly_probid_next_steps(analysis)}",
        f"What would change my mind: {_friendly_probid_change_mind(analysis, missing_suggestions)}",
    ]
    imaging_guidance = _assistant_endo_imaging_guidance(analysis, case_text=case_text)
    if imaging_guidance:
        lines.append(imaging_guidance)
    if analysis.treatment_duration_guidance:
        lines.append(f"How long I would usually treat: {_join_readable(analysis.treatment_duration_guidance[:2])}.")
    if analysis.monitoring_recommendations:
        lines.append(f"What I would monitor: {_join_readable(analysis.monitoring_recommendations[:3])}.")
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
                "If the patient is clinically stable, source control is adequate, and the AST supports it: I would actively look first for a high-bioavailability oral option rather than defaulting to prolonged IV therapy.",
                "If the culture is mainly gram-positive and the patient is stable: choose focused gram-positive coverage.",
                "If the wound is severe, deep, limb-threatening, or clearly polymicrobial: add gram-negative and anaerobic coverage."
            ],
            oralOptions=["High-bioavailability oral therapy can be reasonable in selected stable diabetic foot cases once the AST and source-control plan are clear."],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
                "fluoroquinolone or other reported gram-negative agents if gram-negatives are present",
            ),
            notes=[
                "Source control, debridement, and depth of infection matter as much as the isolate list in diabetic foot infections.",
                "In clinically stable diabetic foot osteomyelitis, high-bioavailability oral therapy can be a reasonable primary strategy when the isolate is susceptible and source control is adequate.",
            ],
        )

    if focus_detail == "Prosthetic joint infection":
        return MechIDProvisionalAdvice(
            summary=(
                "For prosthetic joint infection, the antibiotic plan depends on the organism, the operative strategy, and whether hardware is being retained rather than treating this exactly like native osteomyelitis."
            ),
            recommendedOptions=[
                "Build the plan around the surgical pathway first, such as DAIR versus one-stage or two-stage exchange, because that changes how aggressive and how long therapy needs to be.",
                "If the patient is clinically stable, the joint has been drained or revised, and the AST supports it, a high-bioavailability oral option can still be reasonable rather than assuming the whole course must stay IV.",
                "If this is staphylococcal PJI with retained hardware, a rifampin-based companion strategy may become important, but only after active companion therapy is in place and the wound is stable rather than as standalone treatment.",
            ],
            oralOptions=[
                "High-bioavailability oral therapy can be reasonable in selected stable PJI cases, but it should be aligned with the hardware plan and source control."
            ],
            missingSusceptibilities=_base_missing(
                "clindamycin",
                "linezolid",
                "doxycycline",
                "trimethoprim/sulfamethoxazole",
                "fluoroquinolone susceptibilities when gram-negatives are present",
            ),
            notes=[
                "PJI should be framed around hardware retention versus exchange and whether surgical source control is adequate.",
                "If staphylococcal hardware-associated infection is being treated with retained hardware, rifampin is generally a companion drug discussion rather than the first or only agent."
            ],
        )

    if focus_detail == "Osteomyelitis":
        return MechIDProvisionalAdvice(
            summary="For osteomyelitis, I would choose therapy that reliably covers the recovered organisms and then narrow once susceptibilities and source-control plans are clear.",
            recommendedOptions=[
                "If the patient is clinically stable and source control is in place, I would look first for a high-bioavailability oral option supported by the AST rather than assuming prolonged IV therapy is required.",
                "For MRSA concern: Vancomycin, Linezolid, or Daptomycin for non-pulmonary infection",
                "For streptococcal-only infection: a beta-lactam is often preferred once confirmed susceptible",
            ],
            oralOptions=["High-bioavailability oral therapy is often reasonable in selected stable osteomyelitis cases once the AST and source-control plan are clear."],
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
                "If the patient is clinically stable after drainage and source control, a high-bioavailability oral option can be reasonable when the AST supports it.",
                "For MRSA concern: Vancomycin or Linezolid",
                "For streptococcal-only infection: a beta-lactam is often preferred once susceptibility is known",
            ],
            oralOptions=["High-bioavailability oral therapy can be reasonable in selected stable patients after drainage/source control."],
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
                "If the patient is clinically stable and source control is adequate, I would first look for a high-bioavailability oral option supported by the AST.",
                "Use dependable coverage for the listed organisms first, then narrow once susceptibilities return.",
            ],
            oralOptions=["High-bioavailability oral therapy may be reasonable when the AST supports it and source control is adequate."],
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


_MECHID_SYNDROME_CHIPS: List[Dict[str, str]] = [
    {"value": "mechid_set_syndrome:Bacteraemia", "label": "Bacteraemia"},
    {"value": "mechid_set_syndrome:Urinary tract infection", "label": "UTI"},
    {"value": "mechid_set_syndrome:Pneumonia", "label": "Pneumonia"},
    {"value": "mechid_set_syndrome:Infective endocarditis", "label": "Endocarditis"},
    {"value": "mechid_set_syndrome:Intra-abdominal infection", "label": "Intra-abdominal"},
    {"value": "mechid_set_syndrome:Skin/soft tissue infection", "label": "Skin/soft tissue"},
    {"value": "mechid_set_syndrome:Bone/joint infection", "label": "Bone/joint"},
    {"value": "mechid_set_syndrome:CNS infection", "label": "CNS infection"},
    {"value": "mechid_set_syndrome:Other", "label": "Other / not specified"},
]


def _assistant_mechid_review_options(
    result: MechIDTextAnalyzeResponse,
    *,
    established_syndrome: str | None = None,
) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    has_result = result.analysis is not None or result.provisional_advice is not None
    if has_result and not established_syndrome:
        # Ask syndrome first — no "Give consultant impression" button
        for chip in _MECHID_SYNDROME_CHIPS:
            options.append(AssistantOption(value=chip["value"], label=chip["label"]))
    elif has_result and established_syndrome:
        # Syndrome already known — offer to proceed directly
        options.append(AssistantOption(value="run_assessment", label="Get therapy recommendation"))
    options.append(AssistantOption(value="add_more_details", label="Add case detail"))
    options.append(AssistantOption(value="restart", label="Start new consult"))
    return options


def _assistant_apply_established_syndrome_to_mechid_parsed_request(
    parsed_request: MechIDTextParsedRequest | None,
    established_syndrome: str | None,
) -> MechIDTextParsedRequest | None:
    if parsed_request is None:
        return None

    label = (established_syndrome or "").strip()
    if not label or label == "Other":
        return parsed_request

    tx_updates: Dict[str, Any] = {}
    focus_detail = parsed_request.tx_context.focus_detail
    syndrome = parsed_request.tx_context.syndrome

    if label == "Bacteraemia":
        tx_updates["syndrome"] = "Bloodstream infection"
        tx_updates["focus_detail"] = "Not specified"
    elif label == "Urinary tract infection":
        tx_updates["syndrome"] = (
            syndrome if syndrome in {"Uncomplicated cystitis", "Complicated UTI / pyelonephritis"} else "Complicated UTI / pyelonephritis"
        )
        tx_updates["focus_detail"] = "Not specified"
    elif label == "Pneumonia":
        tx_updates["syndrome"] = "Pneumonia (HAP/VAP or severe CAP)"
        tx_updates["focus_detail"] = "Not specified"
    elif label == "Infective endocarditis":
        tx_updates["syndrome"] = "Bloodstream infection"
        tx_updates["focus_detail"] = "Endocarditis"
    elif label == "Intra-abdominal infection":
        tx_updates["syndrome"] = "Intra-abdominal infection"
        tx_updates["focus_detail"] = "Not specified"
    elif label == "Skin/soft tissue infection":
        tx_updates["syndrome"] = "Other deep-seated / high-inoculum focus"
        tx_updates["focus_detail"] = focus_detail if focus_detail != "Not specified" else "Skin/soft tissue infection"
    elif label == "Bone/joint infection":
        tx_updates["syndrome"] = "Bone/joint infection"
        tx_updates["focus_detail"] = "Not specified"
    elif label == "CNS infection":
        tx_updates["syndrome"] = "CNS infection"
        tx_updates["focus_detail"] = "Not specified"
    else:
        return parsed_request

    updated_tx_context = parsed_request.tx_context.model_copy(update=tx_updates)
    if updated_tx_context == parsed_request.tx_context:
        return parsed_request
    return parsed_request.model_copy(update={"tx_context": updated_tx_context})


def _assistant_effective_mechid_result(
    result: MechIDTextAnalyzeResponse,
    *,
    established_syndrome: str | None = None,
) -> MechIDTextAnalyzeResponse:
    effective_parsed = _assistant_apply_established_syndrome_to_mechid_parsed_request(
        result.parsed_request,
        established_syndrome,
    )
    if effective_parsed is None or effective_parsed == result.parsed_request:
        return result

    parsed_payload = effective_parsed.model_dump(by_alias=True, mode="json")
    return _build_mechid_response_from_parsed(
        text=result.text,
        parsed=parsed_payload,
        parser_name=result.parser,
        parser_fallback_used=result.parser_fallback_used,
        warnings=result.warnings,
    )


def _clean_mechid_text(text: str) -> str:
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    cleaned = cleaned.replace("→", ": ").replace("β", "beta")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _mechid_high_bioavailability_oral_choices(
    result: MechIDAnalyzeResponse,
    parsed: MechIDTextParsedRequest | None,
) -> List[str]:
    if parsed is None:
        return []

    syndrome = parsed.tx_context.syndrome
    focus_detail = parsed.tx_context.focus_detail
    severity = parsed.tx_context.severity
    provided_results = parsed.susceptibility_results or result.final_results
    susceptible_agents = {
        antibiotic
        for antibiotic, call in provided_results.items()
        if call == "Susceptible"
    }
    organism = parsed.organism or ""
    is_bone_joint = syndrome == "Bone/joint infection"
    is_diabetic_foot = focus_detail == "Diabetic foot infection" or (
        syndrome == "Other deep-seated / high-inoculum focus" and "diabetic foot" in focus_detail.lower()
    )
    is_streptococcal = organism in {
        "β-hemolytic Streptococcus (GAS/GBS)",
        "Viridans group streptococci (VGS)",
        "Streptococcus pneumoniae",
    }
    if not (is_bone_joint or is_diabetic_foot):
        return []
    if severity == "Severe / septic shock":
        return []

    staphylococcal_hardware_case = (
        organism.startswith("Staphylococcus")
        and is_bone_joint
        and focus_detail == "Prosthetic joint infection"
    )

    oral_choices: List[str] = []

    def _add(choice: str) -> None:
        if choice not in oral_choices:
            oral_choices.append(choice)

    if "Trimethoprim/Sulfamethoxazole" in susceptible_agents:
        _add("Trimethoprim/Sulfamethoxazole")
    if "Linezolid" in susceptible_agents:
        _add("Linezolid")
    if "Clindamycin" in susceptible_agents:
        _add("Clindamycin")
    if "Metronidazole" in susceptible_agents:
        _add("Metronidazole")
    if "Levofloxacin" in susceptible_agents or "Moxifloxacin" in susceptible_agents:
        if staphylococcal_hardware_case:
            _add("Levofloxacin plus Rifampin")
        else:
            _add("Levofloxacin")
    if "Ciprofloxacin" in susceptible_agents:
        _add("Ciprofloxacin")
    if "Doxycycline" in susceptible_agents or "Tetracycline/Doxycycline" in susceptible_agents:
        _add("Doxycycline")

    if (organism.startswith("Enterococcus") or is_streptococcal) and (
        "Ampicillin" in susceptible_agents or "Penicillin" in susceptible_agents
    ):
        _add("Amoxicillin")
        _add("Amoxicillin/Clavulanate")

    return oral_choices


def _assistant_mechid_should_pair_levo_with_rifampin(result: MechIDTextAnalyzeResponse) -> bool:
    parsed = result.parsed_request
    analysis = result.analysis
    if parsed is None or analysis is None:
        return False
    if parsed.tx_context.syndrome != "Bone/joint infection":
        return False
    organism = parsed.organism or ""
    if not organism.startswith("Staphylococcus"):
        return False
    normalized_text = _normalize_choice(result.text)
    hardware_case = (
        parsed.tx_context.focus_detail == "Prosthetic joint infection"
        or any(
            token in normalized_text
            for token in ("prosthetic", "hardware", "retained", "implant", "biofilm", "dair")
        )
    )
    if not hardware_case:
        return False
    final_results = parsed.susceptibility_results or analysis.final_results
    return (
        final_results.get("Levofloxacin") == "Susceptible"
        or final_results.get("Moxifloxacin") == "Susceptible"
    )


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
    organism_local = parsed.organism if parsed is not None else None
    organism_norm = (organism_local or "").lower()

    def _is_enterococcus_faecium() -> bool:
        return organism_norm == "enterococcus faecium"

    def _is_ampicillin_susceptible_e_faecalis_endocarditis(
        syndrome_local: str,
        focus_local: str,
        susceptible_agents_local: list[str],
    ) -> bool:
        return (
            organism_norm == "enterococcus faecalis"
            and syndrome_local == "Bloodstream infection"
            and focus_local == "Endocarditis"
            and "Ampicillin" in susceptible_agents_local
        )

    def _is_vre_faecium_endocarditis(
        syndrome_local: str,
        focus_local: str,
        resistant_agents_local: list[str],
        susceptible_agents_local: list[str],
    ) -> bool:
        return (
            _is_enterococcus_faecium()
            and syndrome_local == "Bloodstream infection"
            and focus_local == "Endocarditis"
            and "Vancomycin" in resistant_agents_local
            and "Daptomycin" in susceptible_agents_local
        )

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
            if _is_ampicillin_susceptible_e_faecalis_endocarditis(
                syndrome_local,
                focus_local,
                susceptible_agents,
            ):
                return (
                    "For ampicillin-susceptible Enterococcus faecalis endocarditis, I would keep the plan narrow and beta-lactam based rather than surfacing broader fallback agents up front."
                )
            if _is_enterococcus_faecium():
                provided_map = parsed.susceptibility_results if parsed is not None else result.final_results
                if _is_vre_faecium_endocarditis(
                    syndrome_local,
                    focus_local,
                    ["Vancomycin"] if provided_map.get("Vancomycin") == "Resistant" else [],
                    susceptible_agents,
                ):
                    return (
                        "For VRE Enterococcus faecium endocarditis, I would treat this as a salvage-level endovascular infection. "
                        "An ESC-style approach is high-dose Daptomycin 10 to 12 mg/kg IV every 24 hours for at least 8 weeks, usually combined with a synergy partner rather than using Daptomycin alone."
                    )
                if focus_local == "Endocarditis":
                    return (
                        "For Enterococcus faecium endocarditis, I would treat this as a hard-to-treat endovascular infection rather than routine bacteremia. "
                        "I would usually think in terms of high-exposure IV therapy, serial blood-culture clearance, and early endocarditis-focused management."
                    )
                return (
                    "For Enterococcus faecium bacteremia, I would treat this as a potentially endovascular problem until blood cultures clear and the endocarditis workup is settled. "
                    "This is not a syndrome where I would be casual about oral therapy up front."
                )
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
            oral_choices = _mechid_high_bioavailability_oral_choices(result, parsed)
            if focus_local == "Prosthetic joint infection":
                if oral_choices:
                    return (
                        "For prosthetic joint infection, I would anchor the antibiotic plan to the hardware strategy first. "
                        f"If the patient is clinically stable, the joint has been drained or revised, and the AST supports it, a high-bioavailability oral option such as {_join_readable(oral_choices)} can still be reasonable rather than assuming prolonged IV therapy is mandatory."
                        + (
                            " If this is staphylococcal PJI with retained hardware, I would also think explicitly about a rifampin-based companion strategy once effective therapy is established and the wound is stable."
                            if organism_local in {"Staphylococcus aureus", "Staphylococcus lugdunensis"}
                            else ""
                        )
                    )
                return (
                    "For prosthetic joint infection, I would not treat this exactly like native osteomyelitis. The hardware plan, whether DAIR or exchange is being used, and whether there is a clearly reliable oral option matter as much as the susceptibility list itself."
                )
            if oral_choices:
                return (
                    "For osteomyelitis or septic arthritis, if the patient is clinically stable and source control is adequate, "
                    f"I would now look first for a high-bioavailability oral option such as {_join_readable(oral_choices)} rather than defaulting to prolonged IV therapy."
                )
            return (
                "For osteomyelitis or septic arthritis, I would still actively look for a high-bioavailability oral option first, but only if the isolate leaves a clearly reliable oral choice and source control is in place."
            )

        if syndrome_local == "Other deep-seated / high-inoculum focus":
            oral_choices = _mechid_high_bioavailability_oral_choices(result, parsed)
            if oral_choices:
                return (
                    "For a diabetic foot or other deep wound infection, if the patient is clinically stable and debridement or drainage is adequate, "
                    f"I would look first for a high-bioavailability oral option such as {_join_readable(oral_choices)}."
                )
            return (
                "For a diabetic foot or other deep wound infection, I would still think about oral-first treatment when possible, but only if the AST gives a clearly reliable option and the wound has adequate source control."
            )

        if syndrome_local == "Pneumonia (HAP/VAP or severe CAP)":
            return "For pneumonia, I would usually prioritize a reliably active IV option first and only think about narrowing once the isolate, site, and clinical trajectory are clearer."

        if syndrome_local == "Intra-abdominal infection":
            return "For intra-abdominal infection, I would think about a reliably active regimen plus source control rather than chasing an oral option up front."

        if syndrome_local == "CNS infection":
            return "For CNS infection, I would prioritize an agent with reliable CNS activity rather than relying only on the susceptibility label."

        return None

    def _narrow_agent_preference(
        susceptible_agents: list[str],
        syndrome_local: str,
        focus_local: str,
    ) -> str | None:
        susceptible = list(dict.fromkeys(susceptible_agents))
        susceptible_set = set(susceptible)
        if not susceptible:
            return None

        if syndrome_local == "Uncomplicated cystitis":
            for candidate in (
                "Nitrofurantoin",
                "Trimethoprim/Sulfamethoxazole",
                "Fosfomycin",
                "Ciprofloxacin",
                "Levofloxacin",
            ):
                if candidate in susceptible_set:
                    return candidate

        if organism_local == "Staphylococcus aureus":
            for candidate in ("Nafcillin/Oxacillin", "Cefazolin", "Linezolid", "Daptomycin", "Vancomycin"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local == "Staphylococcus lugdunensis":
            for candidate in ("Nafcillin/Oxacillin", "Cefazolin", "Vancomycin", "Linezolid"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local and organism_local.startswith("Enterococcus"):
            if organism_local == "Enterococcus faecium":
                if focus_local == "Endocarditis":
                    for candidate in ("Daptomycin", "Linezolid", "Vancomycin"):
                        if candidate in susceptible_set:
                            return candidate
                for candidate in ("Vancomycin", "Daptomycin", "Linezolid", "Ampicillin", "Penicillin"):
                    if candidate in susceptible_set:
                        return candidate
            for candidate in ("Ampicillin", "Penicillin", "Vancomycin", "Linezolid", "Daptomycin"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local == "Streptococcus pneumoniae":
            if syndrome_local == "CNS infection":
                for candidate in ("Ceftriaxone", "Cefotaxime", "Vancomycin"):
                    if candidate in susceptible_set:
                        return candidate
            for candidate in ("Penicillin", "Ceftriaxone", "Cefotaxime", "Levofloxacin"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local in {"β-hemolytic Streptococcus (GAS/GBS)", "Viridans group streptococci (VGS)"}:
            for candidate in ("Penicillin", "Ceftriaxone", "Vancomycin"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local in {
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Klebsiella oxytoca",
            "Citrobacter koseri",
            "Proteus mirabilis",
            "Salmonella enterica",
        }:
            if syndrome_local == "Complicated UTI / pyelonephritis":
                for candidate in ("Ceftriaxone", "Trimethoprim/Sulfamethoxazole", "Ciprofloxacin", "Levofloxacin", "Cefepime"):
                    if candidate in susceptible_set:
                        return candidate
            if syndrome_local == "Bloodstream infection" and focus_local != "Endocarditis":
                for candidate in ("Ceftriaxone", "Cefepime", "Piperacillin/Tazobactam"):
                    if candidate in susceptible_set:
                        return candidate
            for candidate in ("Ceftriaxone", "Cefepime", "Piperacillin/Tazobactam", "Aztreonam"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local in {
            "Enterobacter cloacae complex",
            "Klebsiella aerogenes",
            "Citrobacter freundii complex",
            "Serratia marcescens",
            "Morganella morganii",
            "Proteus vulgaris group",
        }:
            for candidate in ("Cefepime", "Meropenem", "Imipenem", "Ertapenem", "Piperacillin/Tazobactam"):
                if candidate in susceptible_set:
                    return candidate

        if organism_local == "Pseudomonas aeruginosa":
            for candidate in ("Cefepime", "Piperacillin/Tazobactam", "Ceftazidime", "Aztreonam", "Meropenem", "Imipenem"):
                if candidate in susceptible_set:
                    return candidate

        carbapenem_agents = {"Meropenem", "Imipenem", "Ertapenem", "Doripenem"}
        non_carbapenems = [agent for agent in susceptible if agent not in carbapenem_agents]
        if non_carbapenems:
            return non_carbapenems[0]
        return susceptible[0]

    def _select_preferred_agent(susceptible_agents: list[str], syndrome_local: str, focus_local: str) -> str | None:
        carbapenem_agents = {"Meropenem", "Imipenem", "Ertapenem", "Doripenem"}
        carbapenem_resistance_present = any(
            (provided_results or result.final_results).get(agent) == "Resistant"
            for agent in carbapenem_agents
        )
        if carbapenem_resistance_present:
            return _narrow_agent_preference(susceptible_agents, syndrome_local, focus_local)
        return _narrow_agent_preference(susceptible_agents, syndrome_local, focus_local)

    def _recommendation_sentence(
        selected_note_local: str | None,
        preferred_local: str | None,
        syndrome_local: str,
        focus_local: str,
        severity_local: str,
        resistant_agents_local: list[str],
        susceptible_agents_local: list[str],
    ) -> str | None:
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
                if agent in susceptible_agents_local
            ]
            if oral_choices:
                return (
                    "For uncomplicated cystitis, I would favor a susceptible oral lower-tract option such as "
                    f"{_join_readable(oral_choices)} rather than leading with an IV regimen."
                )
            return (
                "For uncomplicated cystitis, I would look first for a susceptible oral lower-tract option rather than defaulting to a broad IV agent."
            )

        if _is_ampicillin_susceptible_e_faecalis_endocarditis(
            syndrome_local,
            focus_local,
            susceptible_agents_local,
        ):
            return (
                "Because this is ampicillin-susceptible Enterococcus faecalis endocarditis, I would use Ampicillin plus Ceftriaxone as the preferred narrow synergy regimen and reserve Vancomycin or Linezolid for beta-lactam intolerance or other fallback situations."
            )

        if _is_enterococcus_faecium():
            if syndrome_local == "Bloodstream infection" and focus_local == "Endocarditis":
                if _is_vre_faecium_endocarditis(
                    syndrome_local,
                    focus_local,
                    resistant_agents_local,
                    susceptible_agents_local,
                ):
                    return (
                        "Because this looks like VRE Enterococcus faecium endocarditis, I would frame the regimen around high-dose Daptomycin 10 to 12 mg/kg IV every 24 hours for at least 8 weeks, "
                        "plus one synergy partner such as Ampicillin 300 mg/kg per 24 hours IV in 4 to 6 divided doses, Ertapenem 2 g IV once daily, Ceftaroline 1800 mg/day IV in 3 divided doses, or Fosfomycin 12 g/day IV in 4 divided doses. "
                        "If Daptomycin cannot be used, Linezolid can still be active, but I would treat that as a fallback discussion rather than the preferred ESC-style backbone."
                    )
                if preferred_local is not None:
                    return (
                        "Because this looks like Enterococcus faecium endocarditis, I would treat it as a hard-to-treat endovascular infection and would lean on "
                        f"{preferred_local} only in the context of a full endocarditis plan rather than as routine bacteremia therapy."
                    )
            if syndrome_local == "Bloodstream infection":
                if (
                    "Vancomycin" in resistant_agents_local
                    and "Daptomycin" in susceptible_agents_local
                    and "Linezolid" in susceptible_agents_local
                ):
                    return (
                        "For Enterococcus faecium bacteremia, I would usually think in terms of Daptomycin or Linezolid when the isolate is vancomycin-resistant, "
                        "while continuing to reassess whether this is really uncomplicated bacteremia versus endovascular infection."
                    )
                if preferred_local is not None:
                    return (
                        "For Enterococcus faecium bacteremia, I would lean toward "
                        f"{preferred_local}, but I would keep a low threshold to treat this as an endovascular problem if clearance is delayed or the echo strategy is incomplete."
                    )

        if selected_note_local is not None:
            lead, sep, rest = selected_note_local.partition(":")
            if sep and rest.strip():
                rest_text = rest.strip().rstrip(".")
                if re.match(r"^(use|avoid|prefer|treat|consider|choose|lean toward)\b", rest_text.lower()):
                    return (
                        f"Based on the susceptibilities you gave me, {lead.strip().lower()} means I would {rest_text}."
                    )
                return (
                    f"Based on the susceptibilities you gave me, {lead.strip().lower()} suggests {rest_text}."
                )
            return (
                "Based on the susceptibilities you gave me, "
                f"{selected_note_local[0].lower() + selected_note_local[1:].rstrip('.') }."
            )
        carbapenem_agents = {"Meropenem", "Imipenem", "Ertapenem", "Doripenem"}
        carbapenem_resistant = [
            agent
            for agent in carbapenem_agents
            if (provided_results or result.final_results).get(agent) == "Resistant"
        ]
        carbapenem_susceptible = [
            agent
            for agent in carbapenem_agents
            if (provided_results or result.final_results).get(agent) == "Susceptible"
        ]
        if carbapenem_resistant and carbapenem_susceptible:
            return (
                "Because there is discordant carbapenem susceptibility, I would avoid relying on imipenem or meropenem alone "
                "and would favor another confirmed active agent or a newer CRE-directed option if the mechanism supports it."
            )
        if carbapenem_resistant and not preferred_local:
            return (
                "Because a carbapenem is resistant here, I would avoid defaulting to meropenem or imipenem and would look for another confirmed active agent."
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
            if any(token in note_lower for token in ("narrower cephalosporin", "ceftriaxone is preferred", "narrowest dependable iv")):
                score += 3
            if "broadly susceptible enterobacterales pattern" in note_lower:
                score += 5
            if syndrome_local == "Uncomplicated cystitis" and any(token in note_lower for token in ("cystitis", "urinary", "oral option")):
                score += 4
            if syndrome_local == "Complicated UTI / pyelonephritis" and any(
                token in note_lower for token in ("pyelonephritis", "urinary", "oral option")
            ):
                score += 4
            if syndrome_local == "Pneumonia (HAP/VAP or severe CAP)" and "pneumonia" in note_lower:
                score += 4
            if carbapenemase_result_local != "Positive" and any(token in note_lower for token in ("meropenem", "imipenem", "ertapenem", "carbapenem")):
                score -= 2
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
    preferred = _select_preferred_agent(susceptible_agents, syndrome, focus)
    option_overview = _option_overview(syndrome, focus, susceptible_agents)
    recommendation = _recommendation_sentence(
        selected_note,
        preferred,
        syndrome,
        focus,
        severity,
        resistant_agents,
        susceptible_agents,
    )

    lines: List[str] = []
    if option_overview is not None:
        lines.append(option_overview)
    if recommendation is not None and not (
        option_overview is not None
        and syndrome in {"Bone/joint infection", "Other deep-seated / high-inoculum focus"}
        and recommendation.startswith("Based on the susceptibilities you gave me")
    ):
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
                "Ciprofloxacin",
                "Levofloxacin",
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

    if syndrome == "Bone/joint infection":
        oral_choices = _mechid_high_bioavailability_oral_choices(result, parsed)
        if parsed is not None and parsed.tx_context.focus_detail == "Prosthetic joint infection":
            if oral_choices:
                return (
                    f"For prosthetic joint infection, high-bioavailability oral therapy can still be used with {_join_readable(oral_choices)} in selected stable cases, but it should be matched to the hardware plan, debridement status, and whether implant retention is being attempted."
                )
            return "For prosthetic joint infection, I would only frame this as oral-first if the hardware strategy and source control are both clear enough to support that plan."
        if oral_choices:
            return (
                f"For osteomyelitis, septic arthritis, or another bone/joint infection, high-bioavailability oral therapy can often be used with "
                f"{_join_readable(oral_choices)} when the patient is clinically stable and source control is addressed."
            )
        return "For bone or joint infection, I would still look for an oral-first option when possible, but only if the isolate leaves a clearly reliable high-bioavailability agent."

    if syndrome == "Other deep-seated / high-inoculum focus":
        oral_choices = _mechid_high_bioavailability_oral_choices(result, parsed)
        if oral_choices:
            return (
                f"For a diabetic foot or deep wound-type infection, I would prefer a high-bioavailability oral option such as "
                f"{_join_readable(oral_choices)} when the patient is stable and the wound has been adequately drained or debrided."
            )
        return "For a diabetic foot or other deep wound infection, I would still think oral-first when possible, but only if you have a clearly active oral option plus good source control."

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
                "Clindamycin",
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


# ---------------------------------------------------------------------------
# Antibiogram organism lookup helpers
# ---------------------------------------------------------------------------

_ANTIBIOGRAM_ORG_ALIASES: Dict[str, List[str]] = {
    "e. coli": ["escherichia coli", "e coli", "ecoli"],
    "k. pneumoniae": ["klebsiella pneumoniae", "klebsiella", "k pneumoniae", "kpneumoniae"],
    "k. oxytoca": ["klebsiella oxytoca"],
    "p. aeruginosa": ["pseudomonas aeruginosa", "pseudomonas", "p aeruginosa"],
    "s. aureus": ["staphylococcus aureus", "s aureus", "staph aureus"],
    "s. aureus (mssa)": ["mssa", "methicillin-susceptible s. aureus"],
    "s. aureus (mrsa)": ["mrsa", "methicillin-resistant s. aureus"],
    "e. faecalis": ["enterococcus faecalis", "enterococcus"],
    "e. faecium": ["enterococcus faecium", "vre"],
    "s. pneumoniae": ["streptococcus pneumoniae", "strep pneumoniae", "pneumococcus"],
    "acinetobacter baumannii": ["acinetobacter", "a. baumannii"],
    "enterobacter cloacae": ["enterobacter", "e. cloacae"],
    "proteus mirabilis": ["proteus", "p. mirabilis"],
    "serratia marcescens": ["serratia", "s. marcescens"],
    "candida albicans": ["c. albicans"],
    "candida glabrata": ["c. glabrata", "nakaseomyces glabrata"],
    "candida tropicalis": ["c. tropicalis"],
    "candida parapsilosis": ["c. parapsilosis"],
    "candida krusei": ["c. krusei", "pichia kudriavzevii"],
}


def _antibiogram_lookup_organism(
    antibiogram: Dict[str, Any],
    organism_name: str,
) -> Dict[str, float] | None:
    """Fuzzy-match organism_name against antibiogram organism keys.

    Returns the antibiotic→%susceptible dict for the best match, or None.
    Matching is case-insensitive with alias expansion.
    """
    if not antibiogram:
        return None
    organisms: Dict[str, Dict[str, float]] = antibiogram.get("organisms") or {}
    if not organisms:
        return None

    needle = organism_name.lower().strip()

    # Direct match first
    for key, abx_map in organisms.items():
        if key.lower() == needle:
            return abx_map

    # Alias expansion: needle → canonical key
    for canonical, aliases in _ANTIBIOGRAM_ORG_ALIASES.items():
        if needle == canonical or needle in aliases:
            for key, abx_map in organisms.items():
                if key.lower() == canonical:
                    return abx_map
            # also check if any alias matches a key
            for key, abx_map in organisms.items():
                if key.lower() in aliases:
                    return abx_map

    # Partial substring match (e.g. "klebsiella" matches "K. pneumoniae")
    for key, abx_map in organisms.items():
        key_lower = key.lower()
        if needle in key_lower or key_lower in needle:
            return abx_map
        # genus match (first word)
        if needle.split()[0] in key_lower.split()[0] if needle.split() else False:
            return abx_map

    return None


def _antibiogram_provisional_note(
    antibiogram: Dict[str, Any],
    organism_name: str,
    top_n: int = 8,
) -> str | None:
    """Return a one-sentence summary of local susceptibility data for organism_name.

    Returns None if organism not found in antibiogram.
    """
    abx_map = _antibiogram_lookup_organism(antibiogram, organism_name)
    if not abx_map:
        return None
    # Sort descending by susceptibility so best options are first
    sorted_abx = sorted(abx_map.items(), key=lambda x: -x[1])[:top_n]
    institution = antibiogram.get("institution") or "your institution"
    year = antibiogram.get("year")
    src = f"{institution}{', ' + year if year else ''}"
    pairs = ", ".join(f"{abx} {int(pct)}%" for abx, pct in sorted_abx)
    good = [abx for abx, pct in sorted_abx if pct >= 80]
    poor = [abx for abx, pct in sorted_abx if pct < 60]
    note = f"Local antibiogram ({src}) for {organism_name}: {pairs}."
    if good:
        note += f" Reliable empirically (≥80%): {', '.join(good[:3])}."
    if poor:
        note += f" Unreliable empirically (<60%): {', '.join(poor[:3])}."
    return note


def _build_mechid_review_message(result: MechIDTextAnalyzeResponse, *, final: bool = False, established_syndrome: str | None = None, institutional_antibiogram: Dict[str, Any] | None = None) -> str:
    result = _assistant_effective_mechid_result(result, established_syndrome=established_syndrome)
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
    if established_syndrome:
        context_bits.append(established_syndrome)
    if parsed.tx_context.focus_detail != "Not specified":
        context_bits.append(parsed.tx_context.focus_detail)
    if parsed.tx_context.syndrome != "Not specified" and parsed.tx_context.syndrome != established_syndrome:
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

    # Inject local antibiogram data when organism is identified but AST is absent/sparse
    has_no_ast = not parsed.susceptibility_results
    if has_no_ast and institutional_antibiogram and parsed.organism:
        local_note = _antibiogram_provisional_note(institutional_antibiogram, parsed.organism)
        if local_note:
            summary += f"\n\nLocal susceptibility data (no AST provided): {local_note}"

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
            summary += "\nIf this extraction matches the case, select the clinical syndrome to get the therapy recommendation. Otherwise add or correct details."
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
                summary += "\nAlternatively, if you already have a specific antibiotic in mind, name it and I can calculate the renal-adjusted dose."
            summary += "\nIf this extraction matches the case, select the clinical syndrome to get the therapy recommendation. Otherwise add or correct details."
            return summary

        if result.warnings:
            summary += " " + _join_readable(result.warnings[:2])
        if has_no_ast and institutional_antibiogram and parsed.organism:
            summary += " Select the clinical syndrome and I'll use your local antibiogram data to guide empiric therapy, or paste the actual AST results for targeted therapy."
        else:
            summary += " Add susceptibilities for targeted therapy, or select the clinical syndrome for guideline-based empiric recommendations."
        return summary

    if result.analysis is not None:
        summary += "\n\nMy impression:"
        summary += f"\nBottom line: {_friendly_mechid_bottom_line(result.analysis, parsed)}"
        summary += f"\nPattern: {_friendly_mechid_mechanism(result.analysis)}"
        summary += f"\nTreatment approach: {_friendly_mechid_therapy(result.analysis, parsed)}"
        if result.analysis.treatment_duration_guidance:
            summary += f"\nTypical duration frame: {_join_readable(result.analysis.treatment_duration_guidance[:2])}."
        if result.analysis.monitoring_recommendations:
            summary += f"\nMonitoring during therapy: {_join_readable(result.analysis.monitoring_recommendations[:3])}."
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
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    polymicrobial_analyses: List[Dict[str, Any]] | None = None,
    institutional_antibiogram: Dict[str, Any] | None = None,
) -> tuple[str, bool]:
    fallback = _build_mechid_review_message(result, final=final, established_syndrome=established_syndrome, institutional_antibiogram=institutional_antibiogram)
    if final:
        return narrate_mechid_assistant_message(
            mechid_result=result,
            fallback_message=fallback,
            transient_examples=transient_examples,
            established_syndrome=established_syndrome,
            consult_organisms=consult_organisms,
            polymicrobial_analyses=polymicrobial_analyses,
        )
    return narrate_mechid_review_message(
        mechid_result=result,
        fallback_message=fallback,
        transient_examples=transient_examples,
        established_syndrome=established_syndrome,
        consult_organisms=consult_organisms,
    )


def _assistant_review_options_for_case(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    items_by_id = {item.id: item for item in module.items}
    score_options = _assistant_missing_endo_score_options(state, limit=3)
    missing_item_specs = _top_missing_item_specs(
        module,
        text_result.parsed_request,
        limit=2 if _assistant_case_is_consult_ready(module, text_result, state) else 3,
        state=state,
    )
    if score_options:
        options.extend(score_options)
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
    run_label = "Give consultant impression"
    run_description = "Run the consult with the currently available data."
    if _assistant_case_can_run_provisional_consult(module, text_result, state):
        run_label = "Run provisional consult"
        run_description = "Run a best-effort consult now and keep the missing details visible as provisional gaps."
    options.extend(
        [
            AssistantOption(value="run_assessment", label=run_label, description=run_description),
            AssistantOption(value="add_more_details", label="Add case detail"),
            AssistantOption(value="restart", label="Start new consult"),
        ]
    )
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


def _assistant_endo_score_applicable(state: AssistantState | None, score_id: str | None) -> bool:
    if state is None or state.module_id != "endo" or not score_id:
        return False
    case_norm = _normalize_choice(state.case_text)
    prosthetic_material_context = _assistant_endo_has_prosthetic_material_risk(state)
    if score_id == "virsta" and any(
        token in case_norm
        for token in (
            "coagulase negative staph",
            "coagulase-negative staph",
            "coagulase negative staphylococci",
            "coagulase-negative staphylococci",
            "staphylococcus epidermidis",
            "s epidermidis",
            "cons",
        )
    ):
        return False
    if score_id == "denova":
        if any(token in case_norm for token in ("enterococcus faecium", "e faecium")):
            return False
        if prosthetic_material_context and not any(token in case_norm for token in ("enterococcus faecalis", "e faecalis")):
            return False
    return True


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
    score_id = str(score_id) if score_id else None
    return score_id if _assistant_endo_score_applicable(state, score_id) else None


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


def _assistant_merge_pretest_factor_ids(state: AssistantState, factor_ids: List[str]) -> None:
    if not factor_ids:
        return
    current = list(state.pretest_factor_ids)
    for factor_id in factor_ids:
        if factor_id not in current:
            current.append(factor_id)
    state.pretest_factor_ids = current


def _assistant_endo_score_component_entries(state: AssistantState | None) -> List[tuple[str, str]]:
    score_id = _assistant_selected_endo_score_id(state)
    if not score_id:
        return []
    return list(ENDO_ASSISTANT_SCORE_COMPONENTS.get(score_id, ()))


def _assistant_missing_endo_score_options(
    state: AssistantState | None,
    *,
    limit: int | None = None,
) -> List[AssistantOption]:
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
        if limit is not None and len(options) >= limit:
            break
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


def _assistant_infer_pretest_factor_ids_from_text(
    module: SyndromeModule,
    text: str,
    state: AssistantState | None = None,
) -> List[str]:
    text_norm = _normalize_choice(text)
    if not text_norm:
        return []

    if module.id == "endo":
        inferred: List[str] = []
        available_ids = {factor_id for factor_id, _, _ in _assistant_pretest_factor_entries(module, state)}
        for factor_id, aliases in ENDO_PRETEST_FACTOR_TEXT_ALIASES.items():
            if factor_id not in available_ids:
                continue
            if any(alias in text_norm for alias in aliases):
                inferred.append(factor_id)
        return inferred

    return []


def _assistant_pretest_factor_entries(
    module: SyndromeModule,
    state: AssistantState | None = None,
) -> List[tuple[str, str, float]]:
    specs = resolve_pretest_factor_specs(module)
    if module.id == "endo":
        specs = [spec for spec in specs if spec.context_group != "score_overlap"]
    return [(spec.id, spec.label, spec.weight) for spec in specs]


ENDO_PRETEST_FACTOR_TEXT_ALIASES: Dict[str, tuple[str, ...]] = {
    "prosthetic_valve": (
        "prosthetic valve",
        "prosthetic aortic valve",
        "prosthetic mitral valve",
        "mechanical valve",
        "bioprosthetic valve",
        "prosthetic mitral prosthesis",
        "prosthetic aortic prosthesis",
        "tavr",
        "tavi",
        "transcatheter aortic valve",
        "transcatheter valve",
    ),
    "chd": ("congenital heart disease", "congenital valve disease", "repaired congenital heart disease"),
    "hemodialysis": ("hemodialysis", "haemodialysis", "dialysis", "esrd", "hd", "ihd"),
    "central_venous_catheter": ("central venous catheter", "central line", "cvc", "picc", "port", "hickman"),
    "immunosuppression": ("immunosuppressed", "immunocompromised", "on chemotherapy", "transplant recipient", "high dose steroids"),
    "recent_healthcare_or_invasive_exposure": (
        "healthcare associated",
        "recent hospitalization",
        "recent healthcare exposure",
        "recent invasive procedure",
        "recent surgery",
        "nosocomial",
    ),
    "cied": ("pacemaker", "icd", "cied", "cardiac device", "intracardiac device"),
}


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
        score_name = (_assistant_selected_endo_score_id(state) or "score").upper()
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


def _assistant_endo_has_prosthetic_material_risk(state: AssistantState | None) -> bool:
    if state is None:
        return False
    if {"prosthetic_valve", "cied"} & set(state.pretest_factor_ids):
        return True
    case_norm = _normalize_choice(state.case_text)
    return any(
        token in case_norm
        for token in (
            "prosthetic valve",
            "mechanical valve",
            "bioprosthetic valve",
            "tavr",
            "tavi",
            "transcatheter aortic valve",
            "pacemaker",
            "icd",
            "cied",
            "cardiac device",
            "intracardiac device",
        )
    )


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
        "staph": {
            "staph",
            "staphylococcus",
            "staph aureus",
            "staphylococcus aureus",
            "s aureus",
            "saureus",
            "sab",
            "coagulase negative staph",
            "coagulase negative staphylococci",
            "coagulase-negative staph",
            "coagulase-negative staphylococci",
            "cons",
            "s epidermidis",
            "staphylococcus epidermidis",
        },
        "strep": {"strep", "streptococcus", "viridans", "viridans group", "vgs", "nbhs", "s gallolyticus"},
        "enterococcus": {
            "enterococcus",
            "enterococcal",
            "e faecalis",
            "enterococcus faecalis",
            "efaecalis",
            "e faecium",
            "enterococcus faecium",
        },
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
        "give consultant impression",
        "consultant impression",
        "give consult impression",
        "consult impression",
        "give my consultant impression",
        "give impression",
        "my consultant impression",
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
        "imi_aspergillus_culture_resp",
        "imi_aspergillus_pcr_bal",
        "imi_aspergillus_pcr_plasma",
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
    "diabetic_foot_infection": [
        "dfi_local_inflammation_2plus",
        "dfi_probe_to_bone_positive",
        "dfi_xray_osteomyelitis",
        "dfi_esr_high",
        "dfi_mri_osteomyelitis_or_abscess",
        "dfi_bone_biopsy_culture_pos",
        "dfi_bone_histology_pos",
        "dfi_systemic_toxicity",
    ],
}


def _assistant_missing_priority_ids(
    module: SyndromeModule,
    state: AssistantState | None = None,
) -> List[str]:
    if module.id != "endo":
        return ASSISTANT_MISSING_PRIORITY_BY_MODULE.get(module.id, [])

    context = state.endo_blood_culture_context if state is not None else None
    prosthetic_material_context = _assistant_endo_has_prosthetic_material_risk(state)
    if context == "staph":
        micro_priority = [
            *(
                ["endo_bcx_cons_prosthetic_multi"]
                if prosthetic_material_context
                else []
            ),
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
            *(
                ["endo_bcx_enterococcus_prosthetic_multi"]
                if prosthetic_material_context
                else []
            ),
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
        if (
            module.id == "endo"
            and state is not None
            and _assistant_is_endo_imaging_question(state.case_text)
            and item.id in {"endo_tee", "endo_pet"}
        ):
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
        if (
            module.id == "endo"
            and state is not None
            and _assistant_is_endo_imaging_question(state.case_text)
            and item.id in {"endo_tee", "endo_pet"}
        ):
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
            "endo_bcx_cons_prosthetic_multi",
            "endo_bcx_efaecalis_multi",
            "endo_bcx_enterococcus_prosthetic_multi",
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
        prosthetic_material_context = _assistant_endo_has_prosthetic_material_risk(state)
        if item.id in ENDO_ASSISTANT_SCORE_ITEM_IDS:
            return False
        if item.id in {
            "endo_bcx_cons_prosthetic_multi",
            "endo_bcx_enterococcus_prosthetic_multi",
        } and not prosthetic_material_context:
            return False
        if item.category == "micro":
            context = state.endo_blood_culture_context if state is not None else None
            allowed_micro_ids = {
                "staph": {
                    "endo_bcx_saureus_multi",
                    "endo_bcx_cons_prosthetic_multi",
                    "endo_bcx_major_persistent",
                    "endo_bcx_pos_not_major",
                    "endo_bcx_negative",
                },
                "strep": {"endo_bcx_nbhs_multi", "endo_bcx_pos_not_major", "endo_bcx_negative"},
                "enterococcus": {
                    "endo_bcx_efaecalis_multi",
                    "endo_bcx_enterococcus_prosthetic_multi",
                    "endo_bcx_pos_not_major",
                    "endo_bcx_negative",
                },
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
        section_label = _assistant_case_section_label(module, section)
        return (
            f"Start with {section_label}. Add what is present or absent, then click Next to move to laboratory findings.",
            [
                f"This section is for {section_label}.",
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
    extraction_summary = f"For {_assistant_module_label(module)}, I extracted {present_summary}."
    if absent_findings:
        extraction_summary += f" I also captured {_join_readable(absent_findings)} as absent or negative."

    next_items = _top_missing_item_specs(module, text_result.parsed_request, limit=1, state=state)
    next_detail = next_items[0][1] if next_items else None
    score_name = _assistant_selected_endo_score_id(state)
    score_note = (
        f" I also listed the remaining {score_name.upper()} components below so you can tighten that score before I run the assessment."
        if module.id == "endo" and score_name and _assistant_missing_endo_score_options(state)
        else ""
    )
    if _assistant_case_is_consult_ready(module, text_result, state):
        return _assistant_consult_style_message(
            bottom_line=extraction_summary,
            why="I have enough structured information to run the consult now.",
            what_i_still_need=(
                (
                    f"If you want to sharpen the estimate further, the next highest-yield detail would be {next_detail}.{score_note}"
                    if next_detail
                    else f"No single missing item is blocking the consult at this point.{score_note}"
                )
            ),
            what_i_would_do_now="If this extraction matches the case, select the clinical syndrome to get the therapy recommendation now. Otherwise, add another case detail.",
            what_could_change_management=(
                f"The next detail most likely to shift the estimate is {next_detail}."
                if next_detail
                else "A strong new microbiology, imaging, or bedside finding could still move the estimate."
            ),
        )
    if _assistant_case_can_run_provisional_consult(module, text_result, state):
        return _assistant_consult_style_message(
            bottom_line=extraction_summary,
            why="I can already give a best-effort provisional answer, but some high-yield details are still missing.",
            what_i_still_need=(
                (
                    f"The next detail most likely to change the estimate would be {next_detail}.{score_note}"
                    if next_detail
                    else f"A few high-yield details are still missing, but none prevents a provisional answer.{score_note}"
                )
            ),
            what_i_would_do_now="If this looks right, you can still select the clinical syndrome to get the therapy recommendation now, or add more detail first.",
            what_could_change_management=(
                f"The provisional answer could change meaningfully if you add {next_detail}."
                if next_detail
                else "A discriminating microbiology, imaging, or bedside finding could still change management."
            ),
        )
    single_follow_up = _assistant_single_case_follow_up(module, text_result.parsed_request, state=state)
    if single_follow_up:
        return _assistant_consult_style_message(
            bottom_line=extraction_summary,
            why="I have the right syndrome lane, but I still need one more high-yield detail before the consult will feel solid.",
            what_i_still_need=single_follow_up + score_note,
            what_i_would_do_now="Reply with that one detail in plain language and I’ll keep the consult moving.",
            what_could_change_management=(
                f"The answer could move meaningfully depending on {next_detail}."
                if next_detail
                else "The next objective detail could still move the estimate."
            ),
        )
    if text_result.requires_confirmation:
        return _assistant_consult_style_message(
            bottom_line=extraction_summary,
            why="I may still have a few extraction gaps or ambiguities to clean up.",
            what_i_still_need="Any correction or missing case detail that would make the parsed case more accurate." + score_note,
            what_i_would_do_now="If anything looks off, correct it or add more case detail. If it looks right, select the clinical syndrome to get the therapy recommendation.",
            what_could_change_management="A correction to the parsed findings, setting, or microbiology could change the direction of the consult.",
        )

    summary = extraction_summary
    if text_result.requires_confirmation:
        summary += " If anything looks off, correct it or add more case detail."
    else:
        summary += " If this extraction matches the case, select the clinical syndrome to get the therapy recommendation. Otherwise, add more case detail."
    summary += score_note
    return _assistant_consult_style_message(
        bottom_line=summary,
        what_i_would_do_now="Keep replying in plain language and I will keep the consult moving one question at a time.",
    )


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
    case_text_for_parser = _assistant_case_text_for_parser(module, state.case_text)
    text_result = analyze_text(
        TextAnalyzeRequest(
            text=case_text_for_parser,
            moduleHint=state.module_id,
            presetHint=state.preset_id,
            parserStrategy=state.parser_strategy,
            parserModel=state.parser_model,
            allowFallback=state.allow_fallback,
            runAnalyze=False,
            includeExplanation=True,
        )
    )
    _assistant_anchor_guided_case_parse(module, state, text_result)
    _assistant_backfill_guided_case_rule_findings(module, state, text_result, case_text_for_parser)
    _assistant_sanitize_endo_imaging_question_parse(text_result, state.case_text)
    _assistant_refresh_case_parse_summary(text_result)
    _apply_pretest_factors_to_parsed_request(module=module, state=state, parsed_request=text_result.parsed_request)
    _sync_text_result_references(
        text_result=text_result,
        module=module,
        selected_pretest_factor_ids=state.pretest_factor_ids,
    )
    _assistant_cache_probid_case_result(state, text_result)
    return text_result


def _assistant_infer_endo_blood_culture_context(
    text_result: TextAnalyzeResponse,
    message_text: str,
) -> str:
    findings = set((text_result.parsed_request.findings or {}).keys()) if text_result.parsed_request is not None else set()
    if "endo_bcx_saureus_multi" in findings:
        return "staph"
    if "endo_bcx_cons_prosthetic_multi" in findings:
        return "staph"
    if "endo_bcx_nbhs_multi" in findings:
        return "strep"
    if "endo_bcx_efaecalis_multi" in findings:
        return "enterococcus"
    if "endo_bcx_enterococcus_prosthetic_multi" in findings:
        return "enterococcus"

    message_norm = _normalize_choice(message_text)
    if any(
        token in message_norm
        for token in {
            "staphylococcus aureus",
            "s aureus",
            "staph aureus",
            "mrsa",
            "mssa",
            "coagulase negative staph",
            "coagulase negative staphylococci",
            "coagulase-negative staph",
            "coagulase-negative staphylococci",
            "cons",
            "staphylococcus epidermidis",
            "s epidermidis",
        }
    ):
        return "staph"
    if any(token in message_norm for token in {"viridans", "strep sanguinis", "strep mitis", "strep gordonii"}):
        return "strep"
    if any(token in message_norm for token in {"enterococcus", "enterococcal", "e faecalis", "e faecium", "enterococcus faecium"}):
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


def _assistant_explicit_syndrome_module_request(message_text: str) -> str | None:
    normalized = _normalize_choice(message_text)
    if not normalized:
        return None
    if not any(_assistant_text_has_phrase(normalized, token) for token in EXPLICIT_SYNDROME_REQUEST_TOKENS):
        return None
    if any(token in normalized for token in CONSULT_INTENT_FUNGAL_TOKENS) and _assistant_has_ambiguous_fungal_lane_request(normalized):
        return None
    if _assistant_explicit_non_syndrome_workflow_request(message_text) is not None:
        return None
    if (
        _assistant_is_doseid_intent(message_text)
        or _assistant_is_mechid_intent(message_text)
        or _assistant_is_immunoid_intent(message_text)
        or _assistant_is_allergyid_intent(message_text)
    ):
        return None

    parsed = parse_text_to_request(
        store=store,
        text=message_text,
        include_explanation=False,
    )
    module_id = parsed.understood.module_id
    if not module_id or store.get(module_id) is None:
        return None
    return module_id


def _assistant_text_has_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_choice(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text) is not None


def _assistant_consult_style_message(
    *,
    bottom_line: str,
    why: str | None = None,
    what_i_still_need: str | None = None,
    what_i_would_do_now: str | None = None,
    what_could_change_management: str | None = None,
) -> str:
    lines = [f"Bottom line: {bottom_line.strip()}"]
    if why and why.strip():
        lines.append(f"Why: {why.strip()}")
    if what_i_still_need and what_i_still_need.strip():
        lines.append(f"What I still need: {what_i_still_need.strip()}")
    if what_i_would_do_now and what_i_would_do_now.strip():
        lines.append(f"What I would do now: {what_i_would_do_now.strip()}")
    if what_could_change_management and what_could_change_management.strip():
        lines.append(f"What could change management: {what_could_change_management.strip()}")
    return "\n".join(lines)


def _assistant_detect_consult_intent(message_text: str) -> str | None:
    normalized = _normalize_choice(message_text)
    if not normalized:
        return None
    parsed = parse_text_to_request(
        store=store,
        text=message_text,
        include_explanation=False,
    )
    has_syndrome_signal = bool(parsed.understood.module_id and store.get(parsed.understood.module_id or ""))
    has_medication_signal = bool(_assistant_detect_doseid_medication_ids(message_text))
    has_treatment_start_signal = any(token in normalized for token in CONSULT_INTENT_TREATMENT_START_TOKENS)
    has_therapy_selection_signal = any(token in normalized for token in CONSULT_INTENT_THERAPY_SELECTION_TOKENS)
    if not has_treatment_start_signal:
        has_treatment_start_signal = bool(
            re.search(r"\bwhether\b.*\bstart\b", normalized)
            or re.search(r"\bshould\b.*\bstart\b", normalized)
            or re.search(r"\bhold\b.*\bantibiotic", normalized)
        )
    if not has_therapy_selection_signal:
        has_therapy_selection_signal = bool(
            re.search(r"\b(what|which)\b.*\breach for\b", normalized)
            or re.search(r"\b(what|which)\b.*\bwould you use\b", normalized)
            or re.search(r"\b(what|which)\b.*\bwould you start\b", normalized)
        )
    has_antimicrobial_signal = any(token in normalized for token in CONSULT_INTENT_ANTIMICROBIAL_TOKENS)
    if has_therapy_selection_signal and (has_antimicrobial_signal or has_syndrome_signal or has_medication_signal):
        return "therapy_selection"
    if has_treatment_start_signal and (has_antimicrobial_signal or has_syndrome_signal or has_medication_signal):
        return "treatment_decision"
    return None


def _assistant_has_ambiguous_fungal_lane_request(normalized: str) -> bool:
    candida_signal = any(token in normalized for token in CONSULT_INTENT_CANDIDA_TOKENS)
    mold_signal = any(token in normalized for token in CONSULT_INTENT_MOLD_TOKENS)
    fungal_uncertainty = any(
        token in normalized
        for token in (
            "not sure",
            "not even sure",
            "uncertain",
            "candida or mold",
            "mold or candida",
            "whether this is candida or mold",
        )
    )
    return candida_signal and mold_signal and fungal_uncertainty


def _assistant_consult_treatment_module_hint(message_text: str) -> str | None:
    normalized = _normalize_choice(message_text)
    if any(token in normalized for token in CONSULT_INTENT_FUNGAL_TOKENS) and _assistant_has_ambiguous_fungal_lane_request(normalized):
        return None

    parsed = parse_text_to_request(
        store=store,
        text=message_text,
        include_explanation=False,
    )
    module_id = parsed.understood.module_id
    if module_id and store.get(module_id) is not None:
        return module_id

    if not any(token in normalized for token in CONSULT_INTENT_FUNGAL_TOKENS):
        return None
    if any(token in normalized for token in CONSULT_INTENT_MOLD_TOKENS):
        return "inv_mold"
    if any(token in normalized for token in CONSULT_INTENT_CANDIDA_TOKENS):
        return "inv_candida"
    return None


def _assistant_consult_treatment_intro(module: SyndromeModule) -> str:
    if module.id == "inv_mold":
        return (
            "This reads like a treatment-start consult. "
            "I’ll help decide whether empiric mold-active therapy is warranted, what data would most change that decision, "
            "and whether it is reasonable to act before the workup is complete. "
        )
    if module.id == "inv_candida":
        return (
            "This reads like a treatment-start consult. "
            "I’ll help decide whether empiric antifungal therapy is warranted for invasive candidiasis or candidemia, "
            "what missing data matters most, and whether a best-effort answer is still reasonable now. "
        )
    return (
        "This reads like a treatment-start consult. "
        "I’ll help decide whether syndrome-directed therapy is warranted now, what I still need, and what could change management. "
    )


def _assistant_consult_therapy_selection_intro(module: SyndromeModule) -> str:
    if module.id == "inv_mold":
        return (
            "This reads like a therapy-selection consult. "
            "I’ll help decide whether mold-active treatment is warranted, which antifungal lane fits best, and what data would most change the recommendation. "
        )
    if module.id == "inv_candida":
        return (
            "This reads like a therapy-selection consult. "
            "I’ll help decide whether empiric antifungal therapy is warranted for invasive candidiasis or candidemia and what treatment direction fits the current data best. "
        )
    return (
        "This reads like a therapy-selection consult. "
        "I’ll help decide whether treatment is warranted, what direction I would lean, and what data would most change that recommendation. "
    )


def _assistant_consult_fungal_clarification_response(state: AssistantState) -> AssistantTurnResponse:
    response = _assistant_begin_selected_workflow(state, PROBID_ASSISTANT_ID)
    response.assistant_message = _assistant_consult_style_message(
        bottom_line=(
            "This reads like a fungal treatment consult, but I need to know whether the question is mainly about invasive candidiasis or invasive mold disease before I can give the most useful ID-style answer."
        ),
        why=(
            "Those two fungal pathways ask different host-risk, microbiology, and imaging questions and can lead to different empiric-treatment thresholds."
        ),
        what_i_still_need=(
            "Whether this behaves more like candidemia / yeast infection or more like invasive mold disease."
        ),
        what_i_would_do_now=(
            "Choose the fungal syndrome below or reply with a short clarification like 'candidemia from a central line' or 'neutropenic with pulmonary nodules'."
        ),
        what_could_change_management=(
            "Blood-culture yeast, line-related candidemia clues, pulmonary nodules, galactomannan, sinus disease, or major neutropenia would immediately push the consult in different directions."
        ),
    )
    response.options = [
        AssistantOption(
            value="inv_candida",
            label="Possible candidemia / yeast",
            description="Use the invasive candidiasis pathway for candidemia, yeast in blood, line-related yeast concern, or abdominal Candida syndromes.",
        ),
        AssistantOption(
            value="inv_mold",
            label="Possible mold infection",
            description="Use the invasive mold pathway for pulmonary nodules, halo signs, galactomannan, sinus disease, or Aspergillus-style host-risk patterns.",
        ),
        AssistantOption(value="restart", label="Start new consult"),
    ]
    response.tips = [
        "If you are not sure, a useful reply would be: 'neutropenic with pulmonary nodules' or 'candidemia from a central line'.",
        "Once I know the fungal syndrome lane, I can still give a provisional best-effort opinion if some tests are missing.",
    ]
    return response


def _assistant_consult_generic_antimicrobial_clarification_response(
    state: AssistantState,
    *,
    message_text: str,
    intent: str,
) -> AssistantTurnResponse:
    response = _assistant_begin_selected_workflow(state, PROBID_ASSISTANT_ID)
    detected_medication_ids = _assistant_detect_doseid_medication_ids(message_text)
    meds_by_id = _assistant_doseid_medications_by_id()
    medication_names = [meds_by_id[item].name for item in detected_medication_ids if item in meds_by_id]
    if medication_names:
        medication_phrase = _join_readable(medication_names[:3])
        response.assistant_message = _assistant_consult_style_message(
            bottom_line=(
                f"This reads like a treatment consult about {medication_phrase}, but I still need the main syndrome, source, or organism before I can give the most useful ID-style recommendation."
            ),
            why=(
                f"The right answer for {medication_phrase} depends heavily on whether this is pneumonia, bacteremia, endocarditis, skin/soft tissue infection, meningitis, or something else."
            ),
            what_i_still_need="The main syndrome, source, or organism you are treating.",
            what_i_would_do_now=(
                "Reply with a short framing line such as 'possible MRSA pneumonia', 'staph bacteremia with endocarditis concern', or 'cellulitis without purulence'."
            ),
            what_could_change_management=(
                "A different syndrome, source-control issue, organism, or illness severity could completely change whether I would start it, hold it, or choose something else."
            ),
        )
        response.tips = [
            "A useful reply would be: 'possible MRSA pneumonia', 'staph bacteremia with concern for endocarditis', or 'cellulitis without purulence'.",
            "Once I know the syndrome lane, I can still give a provisional best-effort opinion if some tests are missing.",
        ]
        return response

    response.assistant_message = _assistant_consult_style_message(
        bottom_line="This reads like an antimicrobial treatment consult, but I still need the main syndrome, source, or organism to answer it well.",
        why=(
            "The threshold to start, hold, or choose therapy depends much more on the syndrome and source than on the drug name alone."
        ),
        what_i_still_need="The main syndrome, source, or organism you are worried about.",
        what_i_would_do_now=(
            "Reply with a short framing line such as 'possible pneumonia', 'endocarditis concern after Staph aureus bacteremia', or 'febrile neutropenia with pulmonary nodules'."
        ),
        what_could_change_management=(
            "Once I know the syndrome lane, I can tell you whether I would start treatment now, hold it, or favor a different regimen."
        ),
    )
    response.tips = [
        "A useful reply would be: 'possible pneumonia', 'endocarditis concern after Staph aureus bacteremia', or 'febrile neutropenia with pulmonary nodules'.",
        "If you already know the syndrome, you can also click it below and I’ll keep the question framed like an ID consult.",
    ]
    return response


def _assistant_llm_triage_intent(message_text: str, state: "AssistantState | None" = None) -> "dict | None":
    """
    Use the LLM to classify a free-text message into a known workflow intent AND extract
    any clinical data embedded in the message (age, weight, creatinine, organisms, allergy, etc.).

    Context-aware: passes the current consult state so routing accounts for what is already known.

    Returns a dict with:
      - 'intent': one of the known intent strings
      - 'extracted': dict of clinical data found in the message (may be partially filled)
    Returns None if the LLM is unavailable or the call fails, so the caller can fall through.
    """
    from .services.consult_narrator import consult_narration_enabled
    from .services.llm_text_parser import _try_import_openai

    if not consult_narration_enabled():
        return None

    try:
        OpenAI = _try_import_openai()
        import json as _json
        import os as _os

        api_key = _os.getenv("OPENAI_API_KEY")
        client_kwargs: dict = {"api_key": api_key}
        base_url = _os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        model = _os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        _VALID_INTENTS = {
            "probid", "mechid", "doseid", "immunoid", "allergyid",
            "empiric_therapy", "iv_to_oral", "duration", "followup_tests",
            "stewardship", "stewardship_review", "opat", "oral_therapy",
            "discharge_counselling", "drug_interaction", "prophylaxis",
            "source_control", "treatment_failure", "biomarker_interpretation",
            "fluid_interpretation", "allergy_delabeling", "fungal_management",
            "sepsis_management", "cns_infection", "mycobacterial",
            "pregnancy_antibiotics", "travel_medicine", "impression_plan",
            "duke_criteria", "ast_meaning", "complexity_flag", "course_tracker",
            "general_id", "consult_summary",
            "hiv_initial_art", "hiv_monitoring",
            "hiv_prep", "hiv_pep", "hiv_pregnancy", "hiv_oi_art_timing",
            "hiv_treatment_failure", "hiv_resistance", "hiv_switch",
            "unclear",
        }

        # Build a compact context block from current consult state
        context_block = ""
        if state is not None:
            ctx = _consult_prior_context_summary(state)
            if ctx:
                context_block = f"CURRENT CONSULT CONTEXT (already known): {ctx}\n\n"

        triage_prompt = (
            "You are the intake layer of an infectious diseases clinical AI assistant.\n"
            "Your job is TWO things simultaneously:\n"
            "1. Identify what the clinician is ACTUALLY ASKING (the clinical question), "
            "not just what clinical data they are providing.\n"
            "2. Extract any patient or clinical data embedded in the message.\n\n"
            + context_block
            + "AVAILABLE INTENTS:\n"
            "- probid: Syndrome workup — describes patient findings suggesting an infectious syndrome\n"
            "- mechid: Resistance/AST interpretation — organism + susceptibility results or resistance mechanisms\n"
            "- doseid: Antimicrobial dosing — dose, renal adjustment, dialysis dosing, weight-based calculation\n"
            "- immunoid: Prophylaxis/immunosuppression — biologic therapy, chemotherapy, steroids, transplant, pre-treatment screening\n"
            "- allergyid: Antibiotic allergy or cross-reactivity — allergy history + what is safe to use\n"
            "- empiric_therapy: What antibiotic to start empirically before cultures return\n"
            "- iv_to_oral: Whether/how to switch from IV to oral antibiotics\n"
            "- duration: How long to treat or when to stop antibiotics\n"
            "- followup_tests: Follow-up investigations — TEE, repeat cultures, drug levels, imaging\n"
            "- stewardship: De-escalation, narrowing, streamlining after cultures return\n"
            "- stewardship_review: Review a named list of current antibiotics and advise stop/narrow/continue\n"
            "- opat: OPAT eligibility, home IV therapy, PICC line, discharge on IV antibiotics\n"
            "- oral_therapy: Oral antibiotic options for a syndrome — OVIVA/POET applicability\n"
            "- discharge_counselling: What to tell the patient at discharge\n"
            "- drug_interaction: Drug-drug interaction involving an antimicrobial\n"
            "- prophylaxis: Antimicrobial prophylaxis dosing for an immunosuppressed patient\n"
            "- source_control: Line removal, abscess drainage, debridement, prosthetic joint/device decision\n"
            "- treatment_failure: Patient not improving — why is treatment failing, persistent fever/bacteraemia\n"
            "- biomarker_interpretation: What a specific lab value means (PCT, BDG, galactomannan, IGRA, CrAg)\n"
            "- fluid_interpretation: CSF, pleural, ascitic, or synovial fluid result interpretation\n"
            "- allergy_delabeling: Is a reported antibiotic allergy genuine, rechallenge decision, oral challenge\n"
            "- fungal_management: Candidaemia, invasive aspergillosis, cryptococcal meningitis, mucormycosis\n"
            "- sepsis_management: Sepsis bundle, Hour-1, vasopressors, PCT stopping rule\n"
            "- cns_infection: Bacterial meningitis, HSV encephalitis, brain abscess, Listeria coverage\n"
            "- mycobacterial: TB treatment, LTBI, MAC pulmonary, MDR-TB\n"
            "- pregnancy_antibiotics: Antibiotic safety in pregnancy, trimester-specific guidance, GBS\n"
            "- travel_medicine: Returned traveller with fever — malaria, dengue, typhoid, tropical infections\n"
            "- impression_plan: Write a structured ID consult note with impression and numbered plan\n"
            "- duke_criteria: Apply Modified Duke Criteria for endocarditis classification\n"
            "- ast_meaning: What a resistance phenotype means clinically (ESBL, AmpC, MRSA, hVISA, D-zone)\n"
            "- complexity_flag: Whether a case is complex, whether to escalate or needs MDT review\n"
            "- course_tracker: Day-of-therapy milestones, what to check on day X, when treatment is complete\n"
            "- hiv_initial_art: Start ART, new HIV diagnosis, recommend antiretroviral regimen, Biktarvy, Dovato, dolutegravir\n"
            "- hiv_monitoring: HIV lab monitoring schedule, when to check viral load, CD4 monitoring, ART labs\n"
            "- hiv_prep: PrEP, pre-exposure prophylaxis, HIV prevention, Truvada, Descovy, cabotegravir, Apretude, doxyPEP\n"
            "- hiv_pep: PEP, post-exposure prophylaxis, needlestick, HIV exposure, occupational exposure\n"
            "- hiv_pregnancy: HIV in pregnancy, ART in pregnancy, PMTCT, neonatal prophylaxis, HIV delivery planning\n"
            "- hiv_oi_art_timing: When to start ART with an active OI, IRIS risk, defer ART for crypto/TB meningitis\n"
            "- hiv_treatment_failure: HIV virological failure, detectable viral load on ART, adherence assessment, ART not working\n"
            "- hiv_resistance: HIV resistance mutations, genotype interpretation, drug resistance testing, M184V, K65R, INSTI mutations\n"
            "- hiv_switch: Simplify ART, switch regimen, 2-drug regimen eligibility, injectable ART, Cabenuva, lenacapavir\n"
            "- general_id: General ID knowledge question without specific patient findings for formal analysis\n"
            "- consult_summary: Summary, full picture, or recap of this entire consult\n"
            "- unclear: Cannot be classified as an infectious diseases question\n\n"
            "CRITICAL ROUTING RULES — read carefully:\n"
            "- Classify by the QUESTION being asked, not by the clinical data being provided. "
            "A message that gives age/weight/creatinine + asks about a dose → doseid. "
            "A message that gives organism/AST + asks 'what do I treat with?' → mechid. "
            "A message that gives all patient info + asks 'what to start?' with no cultures → empiric_therapy.\n"
            "- If consult context shows an established syndrome AND the message says 'still febrile', "
            "'not improving', 'cultures still positive after 48h', 'persistent bacteraemia' → treatment_failure.\n"
            "- If the message lists current antibiotics by name and asks to review or optimise → stewardship_review.\n"
            "- If cultures are back (organism mentioned) but question is 'what's the best drug?' → mechid.\n"
            "- If no organism yet and question is about what to start → empiric_therapy.\n"
            "- If patient demographics + drug name + renal data → doseid.\n\n"
            "Reply with ONLY a JSON object on one line — no markdown, no explanation:\n"
            "{\"intent\": \"<one of the above>\", \"extracted\": {\"age_years\": <int|null>, "
            "\"sex\": <\"male\"|\"female\"|null>, \"weight_kg\": <float|null>, "
            "\"height_cm\": <float|null>, \"creatinine\": <float|null>, "
            "\"renal_mode\": <\"ihd\"|\"crrt\"|null>, \"organisms\": [<strings>], "
            "\"allergy\": <str|null>, \"syndrome_hint\": <str|null>}}\n"
            "Set null for any extracted field not mentioned. organisms is [] if none mentioned."
        )

        response = client.responses.create(
            model=model,
            instructions=triage_prompt,
            input=message_text,
        )
        output_text = getattr(response, "output_text", None) or ""
        data = _json.loads(output_text.strip())
        intent = data.get("intent", "unclear")
        if intent not in _VALID_INTENTS:
            intent = "unclear"
        data["intent"] = intent
        if "extracted" not in data or not isinstance(data["extracted"], dict):
            data["extracted"] = {}
        return data
    except Exception:
        return None


def _apply_triage_extracted_context(state: AssistantState, extracted: dict) -> None:
    """
    Apply clinical data extracted by LLM triage to the session state.
    Only fills fields that are not yet known — never overwrites existing data.
    Called before routing so all downstream response functions benefit from the richer context.
    """
    if not extracted:
        return

    # Ensure patient_context exists
    if state.patient_context is None:
        from .schemas import SessionPatientContext as _SPC
        state.patient_context = _SPC()

    pc = state.patient_context

    if pc.age_years is None and extracted.get("age_years") is not None:
        try:
            pc.age_years = int(extracted["age_years"])
        except (TypeError, ValueError):
            pass

    if pc.sex is None and extracted.get("sex") in ("male", "female"):
        pc.sex = extracted["sex"]

    if pc.total_body_weight_kg is None and extracted.get("weight_kg") is not None:
        try:
            pc.total_body_weight_kg = float(extracted["weight_kg"])
        except (TypeError, ValueError):
            pass

    if pc.height_cm is None and extracted.get("height_cm") is not None:
        try:
            pc.height_cm = float(extracted["height_cm"])
        except (TypeError, ValueError):
            pass

    if pc.serum_creatinine_mg_dl is None and extracted.get("creatinine") is not None:
        try:
            pc.serum_creatinine_mg_dl = float(extracted["creatinine"])
        except (TypeError, ValueError):
            pass

    if pc.renal_mode == "standard" and extracted.get("renal_mode") in ("ihd", "crrt"):
        pc.renal_mode = extracted["renal_mode"]

    if not pc.allergy_text and extracted.get("allergy"):
        pc.allergy_text = str(extracted["allergy"])

    # Accumulate organisms — add new ones not already tracked
    new_orgs = [str(o).strip() for o in (extracted.get("organisms") or []) if o and str(o).strip()]
    if new_orgs:
        existing_lower = {o.lower() for o in (state.consult_organisms or [])}
        for org in new_orgs:
            if org.lower() not in existing_lower:
                state.consult_organisms = list(state.consult_organisms or []) + [org]
                existing_lower.add(org.lower())


def _assistant_consult_summary_response(state: AssistantState) -> AssistantTurnResponse:
    """Build a unified verbal summary of everything established in the current consult."""
    pc = state.patient_context
    patient_summary_parts: List[str] = []
    if pc:
        if pc.age_years is not None:
            patient_summary_parts.append(f"{pc.age_years}yo")
        if pc.sex:
            patient_summary_parts.append(pc.sex)
        if pc.total_body_weight_kg is not None:
            patient_summary_parts.append(f"{pc.total_body_weight_kg}kg")
        if pc.serum_creatinine_mg_dl is not None:
            patient_summary_parts.append(f"SCr {pc.serum_creatinine_mg_dl}")
        if pc.renal_mode != "standard":
            patient_summary_parts.append(pc.renal_mode.upper())
        if pc.allergy_text:
            patient_summary_parts.append(f"allergy: {pc.allergy_text}")
    patient_summary = ", ".join(patient_summary_parts) if patient_summary_parts else None

    # Build compact fallback text
    fallback_parts: List[str] = []
    if state.established_syndrome:
        fallback_parts.append(f"Syndrome: {state.established_syndrome}.")
    if state.consult_organisms:
        fallback_parts.append(f"Organisms: {', '.join(state.consult_organisms)}.")
    if patient_summary:
        fallback_parts.append(f"Patient: {patient_summary}.")
    if not fallback_parts:
        fallback_parts.append(
            "I don't have enough consult data to summarise yet. "
            "Start with a syndrome description or paste culture results and I'll build the picture."
        )
    fallback_message = " ".join(fallback_parts)

    message, narration_refined = narrate_consult_summary(
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        probid_payload=state.last_probid_summary,
        mechid_payload=state.last_mechid_summary,
        doseid_payload=state.last_doseid_summary,
        allergy_payload=state.last_allergy_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="add_more_details", label="Add more details"),
            AssistantOption(value="mechid", label="Paste culture results"),
            AssistantOption(value="doseid", label="Calculate dosing"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "Ask for a summary anytime to get the full picture of what we've established.",
            "Add another module result and ask again — the summary will update.",
        ],
    )


_CONSULT_SUMMARY_TRIGGERS: tuple[str, ...] = (
    "summary",
    "summarize",
    "summarise",
    "full picture",
    "give me the full",
    "what have we established",
    "what do we know",
    "recap",
    "recapitulate",
    "sign out",
    "sign-out",
    "overall impression",
    "pull it together",
    "put it together",
    "what's the plan",
)


def _is_consult_summary_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _CONSULT_SUMMARY_TRIGGERS)


# ---------------------------------------------------------------------------
# Empiric therapy, IV-to-oral, duration, and follow-up test intents
# ---------------------------------------------------------------------------

_EMPIRIC_THERAPY_TRIGGERS: tuple[str, ...] = (
    "empiric",
    "empirical",
    "start empiric",
    "empiric treatment",
    "empiric therapy",
    "empiric coverage",
    "what do i start",
    "what should i start",
    "what to start",
    "before cultures",
    "cultures pending",
    "culture pending",
    "waiting for cultures",
    "awaiting cultures",
    "best empiric",
    "broad coverage",
    "cover empirically",
)

_IV_TO_ORAL_TRIGGERS: tuple[str, ...] = (
    "iv to oral",
    "iv-to-oral",
    "switch to oral",
    "oral step-down",
    "oral stepdown",
    "step down to oral",
    "transition to oral",
    "po step",
    "convert to po",
    "switch to po",
    "oral equivalent",
    "oral option",
    "oral therapy",
    "oral alternative",
    "can i switch",
    "ready for oral",
    "eligible for oral",
)

_DURATION_TRIGGERS: tuple[str, ...] = (
    "how long",
    "duration",
    "length of therapy",
    "length of treatment",
    "how many days",
    "how many weeks",
    "treatment duration",
    "antibiotic duration",
    "course of antibiotics",
    "finish antibiotics",
    "complete therapy",
    "when to stop",
    "stop antibiotics",
    "when can i stop",
    "total course",
    "total duration",
)

_FOLLOWUP_TEST_TRIGGERS: tuple[str, ...] = (
    "tee",
    "transesophageal",
    "echocardiogram",
    "echo",
    "repeat blood culture",
    "repeat cultures",
    "follow-up culture",
    "follow up culture",
    "clearance culture",
    "document clearance",
    "inflammatory markers",
    "crp",
    "esr",
    "procalcitonin",
    "drug level",
    "vancomycin level",
    "vanco level",
    "trough",
    "auc",
    "lung biopsy",
    "bronchoscopy",
    "bal",
    "bronchoalveolar",
    "pet scan",
    "pet ct",
    "mri spine",
    "follow-up imaging",
    "what tests",
    "what investigations",
    "what workup",
    "further workup",
    "additional testing",
    "do i need a tee",
    "should i get a tee",
    "need imaging",
)


_STEWARDSHIP_TRIGGERS: tuple[str, ...] = (
    "de-escalate",
    "de-escalation",
    "deescalate",
    "deescalation",
    "narrow",
    "narrowing",
    "narrow down",
    "streamline",
    "streamlining",
    "stop antibiotics",
    "stop the antibiotics",
    "cultures came back",
    "cultures are back",
    "cultures returned",
    "what to narrow",
    "can i stop",
    "when to stop",
    "can we stop",
    "discontinue",
    "stewardship",
    "antibiotic stewardship",
    "de escalate",
    "switch from vancomycin",
    "switch from vanco",
    "off vanco",
)

_OPAT_TRIGGERS: tuple[str, ...] = (
    "opat",
    "outpatient iv",
    "outpatient parenteral",
    "outpatient antibiotic",
    "home iv",
    "home antibiotics",
    "send home on iv",
    "discharge on iv",
    "discharge on antibiotics",
    "iv at home",
    "home therapy",
    "home treatment",
    "eligible for opat",
    "candidate for opat",
    "picc",
    "outpatient treatment",
    "complete treatment at home",
    "finish antibiotics at home",
)


def _is_empiric_therapy_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _EMPIRIC_THERAPY_TRIGGERS)


def _is_iv_to_oral_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _IV_TO_ORAL_TRIGGERS)


def _is_duration_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _DURATION_TRIGGERS)


def _is_followup_test_request(text: str) -> bool:
    normalized = _normalize_choice(text)
    return any(_assistant_text_has_phrase(normalized, trigger) for trigger in _FOLLOWUP_TEST_TRIGGERS)


_ORAL_THERAPY_TRIGGERS: tuple[str, ...] = (
    "oral antibiotics",
    "oral antibiotic",
    "oral therapy",
    "oral treatment",
    "oral option",
    "treat with oral",
    "treated orally",
    "oral for osteomyelitis",
    "oral for bone",
    "oral for joint",
    "oral for uti",
    "oral for pneumonia",
    "oral for cellulitis",
    "oviva",
    "poet trial",
    "high bioavailability",
    "can oral treat",
    "can i use oral",
    "is oral enough",
    "po antibiotics",
    "po therapy",
    "oral bone",
    "oral osteomyelitis",
)

_DISCHARGE_COUNSELLING_TRIGGERS: tuple[str, ...] = (
    "discharge counselling",
    "discharge counseling",
    "what do i tell the patient",
    "what to tell the patient",
    "patient education",
    "going home",
    "ready for discharge",
    "discharge instructions",
    "red flags",
    "red flag symptoms",
    "monitoring at home",
    "follow up instructions",
    "what should the patient watch for",
    "patient information",
    "discharge plan",
    "discharge summary",
    "what should i tell",
)

_STEWARDSHIP_REVIEW_TRIGGERS: tuple[str, ...] = (
    "review my antibiotics",
    "review the antibiotics",
    "antibiotic review",
    "review my regimen",
    "antibiotic list",
    "current antibiotics",
    "my current antibiotics",
    "list of antibiotics",
    "running on",
    "currently on antibiotics",
    "which antibiotics can i stop",
    "what can i stop",
    "what should i stop",
    "antibiotics to stop",
    "antibiotic checklist",
)


def _is_stewardship_request(text: str) -> bool:
    normalized = _normalize_choice(text)
    return any(_assistant_text_has_phrase(normalized, trigger) for trigger in _STEWARDSHIP_TRIGGERS)


def _is_opat_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _OPAT_TRIGGERS)


def _is_oral_therapy_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _ORAL_THERAPY_TRIGGERS)


def _is_discharge_counselling_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _DISCHARGE_COUNSELLING_TRIGGERS)


def _is_stewardship_review_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _STEWARDSHIP_REVIEW_TRIGGERS)


_DRUG_INTERACTION_TRIGGERS: tuple[str, ...] = (
    "drug interaction",
    "drug-drug interaction",
    "interaction between",
    "safe to give",
    "safe with",
    "can i give",
    "can i use",
    "interact with",
    "interacts with",
    "combination of",
    "rifampin and",
    "rifampicin and",
    "fluconazole and",
    "voriconazole and",
    "tacrolimus and",
    "warfarin and",
    "linezolid and",
    "metronidazole and",
    "vancomycin and",
    "serotonin syndrome",
    "cytochrome",
    "cyp3a4",
    "drug level",
    "drug levels",
)


def _is_drug_interaction_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _DRUG_INTERACTION_TRIGGERS)


_PROPHYLAXIS_TRIGGERS: tuple[str, ...] = (
    "pcp prophylaxis",
    "pcp prevention",
    "pneumocystis prophylaxis",
    "mac prophylaxis",
    "mac prevention",
    "mycobacterium avium prophylaxis",
    "antifungal prophylaxis",
    "cmv prophylaxis",
    "cmv prevention",
    "toxoplasma prophylaxis",
    "prophylaxis dose",
    "prophylaxis dosing",
    "prophylaxis for",
    "prophylaxis in",
    "prevention dose",
    "cotrimoxazole prophylaxis",
    "tmp-smx prophylaxis",
    "bactrim prophylaxis",
    "posaconazole prophylaxis",
    "valganciclovir prophylaxis",
    "letermovir",
    "hbv prophylaxis",
    "hepatitis b reactivation",
    "immunosuppressed prophylaxis",
    "transplant prophylaxis",
    "hiv prophylaxis",
)


def _is_prophylaxis_request(text: str) -> bool:
    if _assistant_is_immunoid_intent(text):
        return False
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _PROPHYLAXIS_TRIGGERS)


_SOURCE_CONTROL_TRIGGERS: tuple[str, ...] = (
    "source control",
    "remove the line",
    "pull the line",
    "take out the line",
    "pull out the line",
    "remove the catheter",
    "line removal",
    "line out",
    "drain the abscess",
    "drain this",
    "drainage",
    "need to drain",
    "need drainage",
    "debridement",
    "dair",
    "implant retention",
    "retain the implant",
    "remove the implant",
    "joint revision",
    "two-stage",
    "one-stage",
    "surgical intervention",
    "need surgery",
    "needs surgery",
    "surgical drainage",
    "abscess drainage",
    "empyema",
    "necrotising fasciitis",
    "necrotizing fasciitis",
    "fasciitis",
    "peritonsillar",
    "parapharyngeal",
    "lead extraction",
    "device extraction",
    "cardiac device",
    "infected device",
    "infected hardware",
    "infected implant",
    "remove the port",
    "remove the picc",
)


def _is_source_control_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _SOURCE_CONTROL_TRIGGERS)


_TREATMENT_FAILURE_TRIGGERS: tuple[str, ...] = (
    "still febrile",
    "still has fever",
    "not improving",
    "not getting better",
    "not responding",
    "failing treatment",
    "treatment failure",
    "antibiotics not working",
    "antibiotics aren't working",
    "fever persists",
    "persistent fever",
    "persistent bacteraemia",
    "persistent bacteremia",
    "cultures still positive",
    "still bacteraemic",
    "still bacteremic",
    "still septic",
    "why is this patient",
    "why isn't this",
    "why is the fever",
    "day 5 still",
    "day 7 still",
    "day 3 and still",
    "drug fever",
    "could this be drug fever",
)


def _is_treatment_failure_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _TREATMENT_FAILURE_TRIGGERS)


_BIOMARKER_TRIGGERS: tuple[str, ...] = (
    "procalcitonin",
    "pct level",
    "pct is",
    "beta-d-glucan",
    "beta d glucan",
    "bdg",
    "galactomannan",
    "1,3-beta",
    "cryptococcal antigen",
    "crag",
    "cryptococcal antigen",
    "igra",
    "quantiferon",
    "t-spot",
    "histoplasma antigen",
    "blastomyces antigen",
    "aspergillus antigen",
    "biomarker",
    "what does this level mean",
    "what does this result mean",
    "interpret this result",
    "is this positive",
    "false positive",
)


def _is_biomarker_request(text: str) -> bool:
    if _assistant_detect_consult_intent(text) in {"treatment_decision", "therapy_selection"}:
        return False
    if _assistant_is_immunoid_intent(text):
        return False
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _BIOMARKER_TRIGGERS)


_FLUID_INTERPRETATION_TRIGGERS: tuple[str, ...] = (
    "csf result",
    "csf shows",
    "lumbar puncture",
    "lp result",
    "lp showed",
    "cerebrospinal fluid",
    "pleural fluid",
    "pleural tap",
    "thoracentesis",
    "ascitic fluid",
    "paracentesis",
    "ascites tap",
    "synovial fluid",
    "joint fluid",
    "joint tap",
    "arthrocentesis",
    "opening pressure",
    "wbc in the csf",
    "white cells in csf",
    "glucose ratio",
    "light's criteria",
    "saag",
    "pmn count",
    "interpret the fluid",
    "fluid interpretation",
    "peritoneal fluid",
)


def _is_fluid_interpretation_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _FLUID_INTERPRETATION_TRIGGERS)


_ALLERGY_DELABELING_TRIGGERS: tuple[str, ...] = (
    "penicillin allergy",
    "penicillin allergic",
    "allergic to penicillin",
    "beta-lactam allergy",
    "beta lactam allergy",
    "cephalosporin allergy",
    "cross-reactivity",
    "cross reactivity",
    "can i give penicillin",
    "can i use penicillin",
    "is the allergy real",
    "true allergy",
    "allergy delabel",
    "delabeling",
    "rechallenge",
    "oral challenge",
    "skin test",
    "penicillin skin test",
    "red man syndrome",
    "allergy history",
    "says they are allergic",
    "says she is allergic",
    "says he is allergic",
    "reported allergy",
    "listed as allergic",
)


def _is_allergy_delabeling_request(text: str) -> bool:
    normalized = text.lower().strip()
    explicit_triggers = (
        "is the allergy real",
        "true allergy",
        "allergy delabel",
        "delabeling",
        "rechallenge",
        "oral challenge",
        "skin test",
        "penicillin skin test",
        "red man syndrome",
    )
    if any(trigger in normalized for trigger in explicit_triggers):
        return True
    label_trigger = any(
        trigger in normalized
        for trigger in (
            "penicillin allergy",
            "penicillin allergic",
            "allergic to penicillin",
            "beta-lactam allergy",
            "beta lactam allergy",
            "cephalosporin allergy",
        )
    )
    return label_trigger and any(trigger in normalized for trigger in explicit_triggers)


_FUNGAL_MANAGEMENT_TRIGGERS: tuple[str, ...] = (
    "candidaemia",
    "candidemia",
    "candida in the blood",
    "candida fungaemia",
    "candida fungemia",
    "aspergillosis",
    "aspergillus infection",
    "invasive aspergillus",
    "cryptococcal",
    "cryptococcus",
    "mucormycosis",
    "mucor",
    "rhizopus",
    "fusarium",
    "invasive fungal",
    "fungal infection",
    "antifungal treatment",
    "antifungal therapy",
    "voriconazole treatment",
    "liposomal amphotericin",
    "amphotericin b",
    "echinocandin",
    "anidulafungin",
    "micafungin",
    "caspofungin",
    "isavuconazole",
    "candida auris",
)


def _is_fungal_management_request(text: str) -> bool:
    if _assistant_detect_consult_intent(text) in {"treatment_decision", "therapy_selection"}:
        return False
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _FUNGAL_MANAGEMENT_TRIGGERS)


_SEPSIS_TRIGGERS: tuple[str, ...] = (
    "sepsis",
    "septic shock",
    "hour-1 bundle",
    "hour 1 bundle",
    "surviving sepsis",
    "sepsis bundle",
    "qsofa",
    "sofa score",
    "lactate is",
    "lactate level",
    "lactate >",
    "vasopressor",
    "noradrenaline",
    "norepinephrine",
    "septic patient",
    "think this is sepsis",
    "looks septic",
    "blood cultures before",
    "when to de-escalate sepsis",
)


def _is_sepsis_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _SEPSIS_TRIGGERS)


_CNS_INFECTION_TRIGGERS: tuple[str, ...] = (
    "meningitis",
    "encephalitis",
    "brain abscess",
    "cerebral abscess",
    "cns infection",
    "lumbar puncture empiric",
    "suspected meningitis",
    "bacterial meningitis",
    "viral meningitis",
    "hsv encephalitis",
    "herpes encephalitis",
    "toxoplasma brain",
    "ring-enhancing",
    "ring enhancing",
    "dexamethasone meningitis",
    "ceftriaxone meningitis",
    "ampicillin listeria",
    "listeria meningitis",
    "acyclovir empiric",
    "start acyclovir",
    "treat meningitis",
)


def _is_cns_infection_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _CNS_INFECTION_TRIGGERS)


_MYCOBACTERIAL_TRIGGERS: tuple[str, ...] = (
    "tuberculosis",
    " tb ",
    "tb treatment",
    "tb therapy",
    "active tb",
    "latent tb",
    "ltbi",
    "rifampicin isoniazid",
    "hrze",
    "ripe regimen",
    "isoniazid",
    "pyrazinamide",
    "ethambutol",
    "rifabutin",
    "mac infection",
    "mycobacterium avium",
    "mycobacterium abscessus",
    "ntm infection",
    "nontuberculous",
    "non-tuberculous",
    "mdr-tb",
    "xdr-tb",
    "drug-resistant tb",
    "bedaquiline",
    "3hp regimen",
    "9hp regimen",
    "igra positive",
    "quantiferon positive",
    "treat latent",
)


def _is_mycobacterial_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _MYCOBACTERIAL_TRIGGERS)


_PREGNANCY_ANTIBIOTICS_TRIGGERS: tuple[str, ...] = (
    "pregnant",
    "pregnancy",
    "trimester",
    "antenatal",
    "prenatal",
    "in pregnancy",
    "during pregnancy",
    "safe in pregnancy",
    "antibiotics in pregnancy",
    "antibiotic safe",
    "is it safe",
    "breastfeeding antibiotic",
    "gbs prophylaxis",
    "group b strep",
    "intrapartum",
    "postpartum infection",
    "puerperal",
    "nitrofurantoin pregnancy",
    "fluoroquinolone pregnancy",
    "doxycycline pregnancy",
)


def _is_pregnancy_antibiotics_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _PREGNANCY_ANTIBIOTICS_TRIGGERS)


_TRAVEL_MEDICINE_TRIGGERS: tuple[str, ...] = (
    "returned traveller",
    "returned traveler",
    "travel history",
    "travel to",
    "travelled to",
    "traveled to",
    "returning from",
    "came back from",
    "malaria",
    "dengue",
    "typhoid",
    "enteric fever",
    "leptospirosis",
    "chikungunya",
    "zika",
    "yellow fever",
    "schistosomiasis",
    "visceral leishmaniasis",
    "kala-azar",
    "strongyloides",
    "tropical fever",
    "fever in traveller",
    "fever after travel",
    "artemether",
    "artesunate",
    "antimalarial",
    "travel medicine",
    "ebola",
    "viral haemorrhagic fever",
    "vhf",
    "eosinophilia travel",
)


def _is_travel_medicine_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _TRAVEL_MEDICINE_TRIGGERS)


_IMPRESSION_PLAN_TRIGGERS: tuple[str, ...] = (
    "impression and plan",
    "impression & plan",
    "assessment and plan",
    "assessment & plan",
    "a&p",
    "a & p",
    "write an impression",
    "write the impression",
    "write an assessment",
    "write the assessment",
    "write a plan",
    "write the plan",
    "id consult note",
    "consult note",
    "write up the consult",
    "generate impression",
    "generate assessment",
    "write the note",
    "formulate a plan",
    "what's my impression",
    "what is my impression",
    "note for the chart",
    "write a note",
    "impression_plan",
)


def _is_impression_plan_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _IMPRESSION_PLAN_TRIGGERS)


_DUKE_CRITERIA_TRIGGERS: tuple[str, ...] = (
    "duke criteria",
    "duke's criteria",
    "modified duke",
    "endocarditis criteria",
    "ie criteria",
    "definite ie",
    "possible ie",
    "rejected ie",
    "major criteria",
    "minor criteria",
    "duke classification",
    "duke_criteria",
)


def _is_duke_criteria_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _DUKE_CRITERIA_TRIGGERS)


_AST_MEANING_TRIGGERS: tuple[str, ...] = (
    "what does this susceptibility mean",
    "what does the susceptibility mean",
    "what does susceptible mean",
    "what does resistant mean",
    "what does s/i/r mean",
    "esbl",
    "extended spectrum beta",
    "amp c",
    "ampc",
    "merino trial",
    "d-zone",
    "d zone",
    "inducible clindamycin",
    "hvisa",
    "hvisa",
    "vancomycin mic",
    "daptomycin lung",
    "what does the ast mean",
    "interpret this ast",
    "explain the ast",
    "what does resistant mean",
    "what does intermediate mean",
    "ast_meaning",
)


def _is_ast_meaning_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _AST_MEANING_TRIGGERS)


_COMPLEXITY_TRIGGERS: tuple[str, ...] = (
    "is this complex",
    "is this a complex case",
    "should i escalate",
    "should this go to mdt",
    "needs senior review",
    "high risk patient",
    "complex patient",
    "unusual case",
    "difficult case",
    "should i involve",
    "do i need to escalate",
    "complexity",
    "high complexity",
    "red flags",
    "flag this case",
    "complexity_flag",
)


def _is_complexity_request(text: str) -> bool:
    if _assistant_is_immunoid_intent(text):
        return False
    normalized = _normalize_choice(text)
    return any(_assistant_text_has_phrase(normalized, trigger) for trigger in _COMPLEXITY_TRIGGERS)


_COURSE_TRACKER_TRIGGERS: tuple[str, ...] = (
    "day of therapy",
    "day of treatment",
    "how many days",
    "what day am i on",
    "where am i in the course",
    "treatment milestone",
    "antibiotic day",
    "therapy day",
    "clearance culture",
    "when can i switch",
    "when can i stop",
    "when is the course done",
    "when do i stop",
    "course complete",
    "end of course",
    "course_tracker",
)


def _is_course_tracker_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _COURSE_TRACKER_TRIGGERS)


# ---------------------------------------------------------------------------
# HIVID — HIV antiretroviral therapy (Phase 1: initial ART + monitoring)
# ---------------------------------------------------------------------------

_HIV_INITIAL_ART_TRIGGERS: tuple[str, ...] = (
    "start art",
    "initiate art",
    "initiate antiretroviral",
    "start antiretroviral",
    "what art regimen",
    "art regimen",
    "new hiv diagnosis",
    "newly diagnosed hiv",
    "hiv positive",
    "hiv-positive",
    "hiv+",
    "newly diagnosed with hiv",
    "what to start for hiv",
    "hiv treatment",
    "antiretroviral therapy",
    "biktarvy",
    "dovato",
    "dolutegravir",
    "bictegravir",
    "start hiv meds",
    "hiv medications",
    "recommend art",
    "first-line art",
    "first line art",
    "initial hiv regimen",
    "same-day art",
    "same day art",
    "rapid art",
    "rapid start",
)

_HIV_MONITORING_TRIGGERS: tuple[str, ...] = (
    "hiv monitoring",
    "hiv labs",
    "hiv lab monitoring",
    "when to check viral load",
    "viral load monitoring",
    "cd4 monitoring",
    "cd4 count monitoring",
    "hiv follow-up labs",
    "hiv follow up labs",
    "art monitoring",
    "art labs",
    "art follow-up",
    "hiv baseline labs",
    "baseline labs for hiv",
    "when to repeat viral load",
    "viral load schedule",
    "how often check viral load",
    "hiv lab schedule",
)


def _is_hiv_initial_art_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_INITIAL_ART_TRIGGERS)


def _is_hiv_monitoring_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_MONITORING_TRIGGERS)


def _extract_hiv_context_from_text(text: str, state: AssistantState) -> dict:
    """Extract HIV-specific clinical data from the message text and merge with existing hiv_context."""
    hiv_ctx: dict = dict(state.hiv_context) if state.hiv_context else {}
    normalized = text.lower()

    # Viral load
    import re as _re
    vl_match = _re.search(r"(?:viral\s*load|vl|hiv\s*rna)\s*(?:is|of|=|:)?\s*(\d[\d,]*)", normalized)
    if vl_match and "viral_load" not in hiv_ctx:
        hiv_ctx["viral_load"] = float(vl_match.group(1).replace(",", ""))
    if "undetectable" in normalized and "viral_load" not in hiv_ctx:
        hiv_ctx["viral_load"] = 0

    # CD4
    cd4_match = _re.search(r"(?:cd4|cd4\+?)\s*(?:count|cells?)?\s*(?:is|of|=|:)?\s*(\d+)", normalized)
    if cd4_match and "cd4" not in hiv_ctx:
        hiv_ctx["cd4"] = int(cd4_match.group(1))

    # HBV coinfection
    if any(k in normalized for k in ("hbv", "hepatitis b", "hep b", "hbsag positive", "hbsag+")):
        if "hbv_coinfected" not in hiv_ctx:
            hiv_ctx["hbv_coinfected"] = True

    # HCV coinfection
    if any(k in normalized for k in ("hcv", "hepatitis c", "hep c")):
        if "hcv_coinfected" not in hiv_ctx:
            hiv_ctx["hcv_coinfected"] = True

    # Pregnancy
    if any(k in normalized for k in ("pregnant", "pregnancy", "gravid", "gestation")):
        if "pregnant" not in hiv_ctx:
            hiv_ctx["pregnant"] = True
        tri_match = _re.search(r"(?:t|trimester\s*)(\d)", normalized)
        if tri_match and "trimester" not in hiv_ctx:
            hiv_ctx["trimester"] = int(tri_match.group(1))

    # Active OI keywords
    oi_map = {
        "cryptococcal meningitis": "cryptococcal meningitis",
        "crypto meningitis": "cryptococcal meningitis",
        "pcp": "PCP",
        "pneumocystis": "PCP",
        "tb meningitis": "TB meningitis",
        "tuberculous meningitis": "TB meningitis",
        "pulmonary tb": "pulmonary TB",
        "tuberculosis": "TB",
        "active tb": "active TB",
        "cmv retinitis": "CMV retinitis",
        "toxoplasmosis": "toxoplasmosis",
        "mac": "MAC",
        "mycobacterium avium": "MAC",
        "kaposi": "Kaposi sarcoma",
        "pml": "PML",
        "histoplasmosis": "histoplasmosis",
    }
    for keyword, oi_name in oi_map.items():
        if keyword in normalized and "active_oi" not in hiv_ctx:
            hiv_ctx["active_oi"] = oi_name
            break

    # Prior CAB-LA PrEP
    if any(k in normalized for k in ("cab-la", "cabotegravir prep", "injectable prep", "apretude")):
        if "prior_cab_la_prep" not in hiv_ctx:
            hiv_ctx["prior_cab_la_prep"] = True

    # Currently on ART
    art_regimens = {
        "biktarvy": ["bictegravir/emtricitabine/TAF"],
        "dovato": ["dolutegravir/lamivudine"],
        "triumeq": ["dolutegravir/abacavir/lamivudine"],
        "genvoya": ["elvitegravir/cobicistat/emtricitabine/TAF"],
        "stribild": ["elvitegravir/cobicistat/emtricitabine/TDF"],
        "odefsey": ["rilpivirine/emtricitabine/TAF"],
        "complera": ["rilpivirine/emtricitabine/TDF"],
        "symtuza": ["darunavir/cobicistat/emtricitabine/TAF"],
        "cabenuva": ["cabotegravir/rilpivirine LA"],
        "juluca": ["dolutegravir/rilpivirine"],
        "descovy": ["emtricitabine/TAF"],
        "truvada": ["emtricitabine/TDF"],
        "tivicay": ["dolutegravir"],
    }
    for brand, regimen in art_regimens.items():
        if brand in normalized and "current_regimen" not in hiv_ctx:
            hiv_ctx["on_art"] = True
            hiv_ctx["current_regimen"] = regimen
            break

    # CrCl
    crcl_match = _re.search(r"(?:crcl|creatinine clearance|gfr|egfr)\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)", normalized)
    if crcl_match and "creatinine_clearance" not in hiv_ctx:
        hiv_ctx["creatinine_clearance"] = float(crcl_match.group(1))

    # Resistance mutations
    mutation_pattern = _re.findall(r"\b([A-Z]\d{2,3}[A-Z](?:/[A-Z])?)\b", text)
    if mutation_pattern and "resistance_mutations" not in hiv_ctx:
        hiv_ctx["resistance_mutations"] = mutation_pattern[:10]

    return hiv_ctx


def _assistant_hiv_initial_art_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Recommend an initial ART regimen based on patient factors."""
    from .services.consult_narrator import narrate_hiv_initial_art

    # Extract and update HIV context from the message
    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "For most treatment-naive adults, Biktarvy (bictegravir/emtricitabine/TAF) is the recommended first-line ART regimen — "
        "one pill daily, high barrier to resistance, well tolerated. "
        "Key exceptions: HBV coinfection requires tenofovir-containing backbone (Dovato is contraindicated); "
        "CrCl <30 requires adjusted backbone; VL >500,000 requires 3-drug regimen; "
        "pregnancy prefers DTG + emtricitabine/TDF; active TB on rifampin requires DTG 50mg BID. "
        "Same-day ART start is recommended for most patients — do not wait for genotype results. "
        "Exceptions: defer 4-6 weeks for cryptococcal meningitis, 4-8 weeks for TB meningitis."
    )
    answer, narration_refined = narrate_hiv_initial_art(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        fallback_message=fallback_message,
    )
    options: list[AssistantOption] = [
        AssistantOption(value="hiv_monitoring", label="Monitoring schedule"),
    ]
    if state.hiv_context and state.hiv_context.get("active_oi"):
        options.append(AssistantOption(value="hiv_oi_art_timing", label="ART timing with this OI"))
    if state.hiv_context and state.hiv_context.get("hbv_coinfected"):
        options.append(AssistantOption(value="drug_interaction", label="HBV-ART drug interactions"))
    # Cross-module bridges (CD4-based prophylaxis, pregnancy, etc.)
    cross_opts = _hivid_cross_module_options(state, current_intent="hiv_initial_art")
    for co in cross_opts:
        if not any(o.value == co.value for o in options):
            options.append(co)
    options.extend([
        AssistantOption(value="prophylaxis", label="OI prophylaxis"),
        AssistantOption(value="drug_interaction", label="Check drug interactions"),
        AssistantOption(value="consult_summary", label="Full consult summary"),
    ])
    # Deduplicate
    seen_vals: set[str] = set()
    deduped_opts: list[AssistantOption] = []
    for o in options:
        if o.value not in seen_vals:
            seen_vals.add(o.value)
            deduped_opts.append(o)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=deduped_opts,
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_initial_art"),
        tips=[
            "Same-day ART is the standard for most new HIV diagnoses — genotype results should not delay initiation.",
            "Always check HBV serologies before ART — Dovato and other non-tenofovir regimens risk HBV flare in coinfected patients.",
        ],
    )


def _assistant_hiv_monitoring_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Provide HIV lab monitoring schedule."""
    from .services.consult_narrator import narrate_hiv_monitoring

    # Extract and update HIV context
    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Baseline labs for new HIV diagnosis: HIV VL, CD4, RT-protease genotype, HBV serologies, HCV Ab, "
        "CMP, CBC, fasting lipids, glucose/HbA1c, pregnancy test if applicable, STI screening, IGRA for TB. "
        "If CD4 <100: serum CrAg. "
        "On-treatment: VL at 4-6 weeks, then every 4-8 weeks until undetectable, then every 3-6 months. "
        "CD4 every 3-6 months for 2 years, then can stop if suppressed and >300. "
        "Renal function: annually on TAF, every 3-6 months on TDF. "
        "Note: DTG/BIC increase creatinine ~0.1 mg/dL via OCT2 — not nephrotoxicity."
    )
    answer, narration_refined = narrate_hiv_monitoring(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    monitoring_opts: list[AssistantOption] = [
        AssistantOption(value="hiv_initial_art", label="Select ART regimen"),
    ]
    cross_opts = _hivid_cross_module_options(state, current_intent="hiv_monitoring")
    for co in cross_opts:
        if not any(o.value == co.value for o in monitoring_opts):
            monitoring_opts.append(co)
    monitoring_opts.extend([
        AssistantOption(value="drug_interaction", label="Check drug interactions"),
        AssistantOption(value="prophylaxis", label="OI prophylaxis"),
        AssistantOption(value="consult_summary", label="Full consult summary"),
    ])
    seen_vals2: set[str] = set()
    deduped_mon: list[AssistantOption] = []
    for o in monitoring_opts:
        if o.value not in seen_vals2:
            seen_vals2.add(o.value)
            deduped_mon.append(o)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=deduped_mon,
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_monitoring"),
        tips=[
            "DTG and BIC inhibit OCT2 and raise serum creatinine by ~0.1 mg/dL — this is a lab artifact, not nephrotoxicity. Do not switch regimens for this.",
            "After 2+ years of viral suppression with CD4 >300, routine CD4 monitoring can be safely discontinued.",
        ],
    )


# ---------------------------------------------------------------------------
# HIVID Phase 2 — PrEP, PEP, pregnancy, OI-ART timing
# ---------------------------------------------------------------------------

_HIV_PREP_TRIGGERS: tuple[str, ...] = (
    "prep",
    "pre-exposure prophylaxis",
    "pre exposure prophylaxis",
    "hiv prevention",
    "hiv prophylaxis",
    "truvada for prevention",
    "descovy for prevention",
    "cabotegravir prep",
    "cab-la prep",
    "apretude",
    "injectable prep",
    "on-demand prep",
    "2-1-1",
    "event-driven prep",
    "doxypep",
    "doxy-pep",
    "doxy pep",
    "lenacapavir prep",
)

_HIV_PEP_TRIGGERS: tuple[str, ...] = (
    "pep",
    "post-exposure prophylaxis",
    "post exposure prophylaxis",
    "needlestick",
    "needle stick",
    "hiv exposure",
    "occupational exposure",
    "sexual exposure hiv",
    "blood exposure",
    "sharps injury",
    "start pep",
)

_HIV_PREGNANCY_TRIGGERS: tuple[str, ...] = (
    "pregnant and hiv",
    "hiv in pregnancy",
    "hiv pregnancy",
    "pregnant with hiv",
    "art in pregnancy",
    "antiretroviral pregnancy",
    "pmtct",
    "mother to child",
    "vertical transmission",
    "neonatal prophylaxis",
    "hiv breastfeeding",
    "hiv delivery",
    "hiv c-section",
    "hiv cesarean",
)

_HIV_OI_ART_TIMING_TRIGGERS: tuple[str, ...] = (
    "when to start art with",
    "art timing with",
    "art timing",
    "when to start antiretroviral",
    "start art with oi",
    "start art with opportunistic",
    "art and pcp",
    "art and tb",
    "art and tuberculosis",
    "art and crypto",
    "art and cryptococcal",
    "iris risk",
    "immune reconstitution",
    "art timing tb",
    "art timing crypto",
    "art timing pcp",
    "defer art",
    "delay art",
)


def _is_hiv_prep_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_PREP_TRIGGERS)


def _is_hiv_pep_request(text: str) -> bool:
    normalized = text.lower().strip()
    # Avoid matching "pep" inside other words like "pepper" or "peptic"
    if "pep" in normalized and not any(w in normalized for w in ("pepper", "peptic", "peptide", "pepsin")):
        return True
    return any(trigger in normalized for trigger in _HIV_PEP_TRIGGERS if trigger != "pep")


def _is_hiv_pregnancy_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_PREGNANCY_TRIGGERS)


def _is_hiv_oi_art_timing_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_OI_ART_TIMING_TRIGGERS)


def _assistant_hiv_prep_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on PrEP regimen selection and monitoring."""
    from .services.consult_narrator import narrate_hiv_prep

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "PrEP options per IAS-USA 2024: "
        "(1) Oral TDF/FTC (Truvada) daily — all populations, requires CrCl >=60; "
        "(2) On-demand 2-1-1 TDF/FTC — MSM with planned anal sex only; "
        "(3) Oral TAF/FTC (Descovy) daily — cisgender men only, preferred if CrCl 30-60; "
        "(4) Injectable cabotegravir (Apretude) 600mg IM q2 months — superior to oral, all populations; "
        "(5) Lenacapavir SC q6 months — 100% efficacy in PURPOSE 1, FDA review pending. "
        "Baseline: 4th-gen HIV test, CrCl, HBV serologies, HCV Ab, STI screening. "
        "Monitor: HIV test q3 months, CrCl q6-12 months, STI q3-6 months."
    )
    answer, narration_refined = narrate_hiv_prep(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="hiv_pep", label="PEP guidance instead"),
            AssistantOption(value="hiv_initial_art", label="Patient is HIV+ — start ART"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
            AssistantOption(value="consult_summary", label="Full consult summary"),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_prep"),
        tips=[
            "Injectable cabotegravir (Apretude) was superior to daily oral TDF/FTC in trials — consider for patients with adherence challenges.",
            "DoxyPEP (doxycycline 200mg post-exposure) reduces chlamydia and syphilis in MSM/TGW — discuss alongside PrEP.",
        ],
    )


def _assistant_hiv_pep_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on PEP regimen and follow-up."""
    from .services.consult_narrator import narrate_hiv_pep

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "PEP must start within 72 hours of exposure (ideally within 2 hours). "
        "Preferred regimen: dolutegravir 50mg OD + emtricitabine/TDF 200/300mg OD x 28 days. "
        "Alternative: Biktarvy 1 pill OD x 28 days. "
        "Baseline: 4th-gen HIV Ag/Ab, HBV serologies, HCV Ab, CMP, pregnancy test. "
        "Follow-up: HIV test at 4-6 weeks and 3 months. "
        "If ongoing risk: transition to PrEP without a gap."
    )
    answer, narration_refined = narrate_hiv_pep(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="hiv_prep", label="Transition to PrEP"),
            AssistantOption(value="hiv_initial_art", label="Patient tested HIV+ — start ART"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_pep"),
        tips=[
            "If >72 hours since exposure, PEP is not recommended — offer PrEP if ongoing risk.",
            "If the source patient is on ART with undetectable VL, transmission risk is extremely low (U=U).",
        ],
    )


def _assistant_hiv_pregnancy_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on ART in pregnancy, delivery planning, and neonatal prophylaxis."""
    from .services.consult_narrator import narrate_hiv_pregnancy

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    if "pregnant" not in (hiv_ctx or {}):
        hiv_ctx = hiv_ctx or {}
        hiv_ctx["pregnant"] = True
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Preferred ART in pregnancy: dolutegravir + emtricitabine/TAF (or TDF). "
        "DTG is safe in all trimesters including the first (NTD risk ~0.2%). "
        "Cobicistat-boosted regimens are contraindicated (low levels in T2/T3). "
        "Delivery: VL <50 at 36 weeks = vaginal; VL 50-999 = C-section + IV AZT; VL >=1000 = C-section + IV AZT + intensified neonatal Rx. "
        "Neonate: low risk = AZT x 4 weeks; high risk = AZT + 3TC + NVP x 6 weeks."
    )
    answer, narration_refined = narrate_hiv_pregnancy(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="hiv_monitoring", label="Monitoring schedule in pregnancy"),
            AssistantOption(value="hiv_initial_art", label="Select ART regimen"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
            AssistantOption(value="consult_summary", label="Full consult summary"),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_pregnancy"),
        tips=[
            "DTG is now recommended in all trimesters — the NTD risk (~0.2%) is comparable to the general population background rate.",
            "Check VL at 36 weeks to plan delivery mode. VL <50 = vaginal delivery with no IV zidovudine needed.",
        ],
    )


def _assistant_hiv_oi_art_timing_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on when to start ART relative to an active OI."""
    from .services.consult_narrator import narrate_hiv_oi_art_timing

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "ART timing with OIs per IAS-USA 2024: "
        "Most OIs (PCP, toxo, MAC, histoplasma, KS, PML): start ART within 2 weeks. "
        "Pulmonary TB: CD4 <50 within 2 weeks, CD4 >=50 within 2-8 weeks. "
        "DEFER for cryptococcal meningitis (4-6 weeks, COAT trial), "
        "TB meningitis (4-8 weeks), CMV retinitis zone 1 (~2 weeks). "
        "IRIS is NOT a reason to stop ART — manage with anti-inflammatories."
    )
    answer, narration_refined = narrate_hiv_oi_art_timing(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="hiv_initial_art", label="Select ART regimen"),
            AssistantOption(value="hiv_monitoring", label="Monitoring schedule"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
            AssistantOption(value="prophylaxis", label="OI prophylaxis"),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_oi_art_timing"),
        tips=[
            "Cryptococcal meningitis: ALWAYS defer ART 4-6 weeks. The COAT trial showed early ART increases mortality.",
            "TB + rifampin: double the dolutegravir dose to 50mg BID. Bictegravir is contraindicated with rifampin.",
        ],
    )


# ---------------------------------------------------------------------------
# HIVID Phase 3 — Treatment failure, resistance, switch/simplification
# ---------------------------------------------------------------------------

_HIV_TREATMENT_FAILURE_TRIGGERS: tuple[str, ...] = (
    "virologic failure",
    "virological failure",
    "hiv failure",
    "viral load detectable",
    "vl detectable",
    "vl not suppressed",
    "hiv not suppressed",
    "hiv rebound",
    "viral rebound",
    "hiv viremia",
    "persistent viremia",
    "low-level viremia",
    "low level viremia",
    "viral blip",
    "hiv blip",
    "art failure",
    "art not working",
    "failing art",
    "failing antiretroviral",
)

_HIV_RESISTANCE_TRIGGERS: tuple[str, ...] = (
    "hiv resistance",
    "hiv genotype",
    "hiv mutation",
    "integrase resistance",
    "insti resistance",
    "nrti resistance",
    "nnrti resistance",
    "m184v",
    "k65r",
    "q148h",
    "k103n",
    "y143r",
    "n155h",
    "hiv drug resistance",
    "resistance test result",
    "genotype result",
    "resistance interpretation",
)

_HIV_SWITCH_TRIGGERS: tuple[str, ...] = (
    "switch art",
    "switch antiretroviral",
    "simplify art",
    "simplify regimen",
    "change art",
    "change hiv regimen",
    "switch to biktarvy",
    "switch to dovato",
    "switch to cabenuva",
    "injectable art",
    "long-acting art",
    "long acting art",
    "2-drug regimen",
    "two drug regimen",
    "art simplification",
    "switch from",
    "can i simplify",
)


def _is_hiv_treatment_failure_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_TREATMENT_FAILURE_TRIGGERS)


def _is_hiv_resistance_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_RESISTANCE_TRIGGERS)


def _is_hiv_switch_request(text: str) -> bool:
    normalized = text.lower().strip()
    return any(trigger in normalized for trigger in _HIV_SWITCH_TRIGGERS)


def _assistant_hiv_treatment_failure_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Evaluate HIV virologic failure — workup and switch strategy."""
    from .services.consult_narrator import narrate_hiv_treatment_failure

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Virologic failure workup: (1) Assess adherence — most common cause, >95% needed. "
        "Check polyvalent cation supplements chelating INSTIs. "
        "(2) Drug interactions — rilpivirine+PPIs, rifampin+cobicistat. "
        "(3) Absorption issues. "
        "(4) Send RT + protease + integrase genotype WHILE ON failing regimen. "
        "New regimen must include >=2 fully active agents. "
        "For multi-class resistance: fostemsavir, lenacapavir, ibalizumab — refer to academic HIV center."
    )
    answer, narration_refined = narrate_hiv_treatment_failure(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    tf_opts: list[AssistantOption] = [
        AssistantOption(value="hiv_resistance", label="Interpret resistance results"),
        AssistantOption(value="hiv_switch", label="Switch ART regimen"),
        AssistantOption(value="hiv_monitoring", label="Monitoring schedule"),
        AssistantOption(value="drug_interaction", label="Check drug interactions"),
    ]
    cross_opts = _hivid_cross_module_options(state, current_intent="hiv_treatment_failure")
    for co in cross_opts:
        if not any(o.value == co.value for o in tf_opts):
            tf_opts.append(co)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=tf_opts,
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_treatment_failure"),
        tips=[
            "Adherence is the #1 cause of virologic failure. Ask about polyvalent cation supplements (Ca, Mg, Fe, Zn) — they chelate INSTIs.",
            "Send genotype WHILE ON the failing regimen. Stopping ART before genotyping allows wild-type reversion and masks resistance.",
        ],
    )


def _assistant_hiv_resistance_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Interpret HIV resistance mutations and recommend regimen adjustment."""
    from .services.consult_narrator import narrate_hiv_resistance

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "HIV resistance interpretation: "
        "M184V = FTC/3TC resistant, increases TDF susceptibility. "
        "K65R = TDF/TAF resistant, increases AZT susceptibility. "
        "K103N = EFV/NVP resistant, RPV and DOR usually susceptible. "
        "Q148H alone = first-gen INSTI resistant, DTG/BIC usually active. "
        "Q148H + G140S + E138K = DTG/BIC may be compromised — salvage territory. "
        "DRV/r retains activity against most PI mutations (requires >=3 DRV-associated mutations). "
        "Always interpret cumulative resistance history, not just current genotype."
    )
    answer, narration_refined = narrate_hiv_resistance(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    res_opts: list[AssistantOption] = [
        AssistantOption(value="hiv_switch", label="Recommend new regimen"),
        AssistantOption(value="hiv_treatment_failure", label="Full failure workup"),
        AssistantOption(value="hiv_initial_art", label="Select ART regimen"),
        AssistantOption(value="drug_interaction", label="Check drug interactions"),
    ]
    cross_opts = _hivid_cross_module_options(state, current_intent="hiv_resistance")
    for co in cross_opts:
        if not any(o.value == co.value for o in res_opts):
            res_opts.append(co)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=res_opts,
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_resistance"),
        tips=[
            "Resistance is cumulative — archived mutations may not appear on current genotype but still affect drug activity.",
            "DTG and BIC have a high barrier to resistance. Single INSTI mutations rarely compromise them — the Q148H + secondary accumulation pathway is the main threat.",
        ],
    )


def _assistant_hiv_switch_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on ART switch or simplification for suppressed patients."""
    from .services.consult_narrator import narrate_hiv_switch

    hiv_ctx = _extract_hiv_context_from_text(message_text, state)
    state.hiv_context = hiv_ctx if hiv_ctx else state.hiv_context

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "ART switch rules: ALWAYS review full resistance history before switching. "
        "2-drug options (Dovato, Juluca) require: no prior failure, no HBV, no resistance to components. "
        "PI-to-INSTI switch is safe even with NRTI resistance if no INSTI resistance. "
        "Cabenuva (injectable CAB+RPV q1-2mo): needs no NNRTI resistance, VL suppressed, 1-2% failure risk with BMI >30. "
        "If HBV coinfected: MUST maintain tenofovir or risk fatal flare. "
        "Monitor VL at 1 month, then q3 months for 1 year after any switch."
    )
    answer, narration_refined = narrate_hiv_switch(
        question=message_text,
        hiv_context=state.hiv_context,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    sw_opts: list[AssistantOption] = [
        AssistantOption(value="hiv_resistance", label="Review resistance history"),
        AssistantOption(value="hiv_monitoring", label="Post-switch monitoring"),
        AssistantOption(value="drug_interaction", label="Check drug interactions"),
        AssistantOption(value="consult_summary", label="Full consult summary"),
    ]
    cross_opts = _hivid_cross_module_options(state, current_intent="hiv_switch")
    for co in cross_opts:
        if not any(o.value == co.value for o in sw_opts):
            sw_opts.append(co)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=sw_opts,
        suggestedPlaceholder=_build_suggested_placeholder(state, "hiv_switch"),
        tips=[
            "Never switch to a 2-drug regimen without reviewing full resistance history — archived mutations count.",
            "If switching away from tenofovir in an HBV-coinfected patient, ensure alternative HBV coverage is in place — discontinuation can cause fatal HBV flare.",
        ],
    )


# ---------------------------------------------------------------------------
# HIVID Phase 4 — Cross-module integration helpers
# ---------------------------------------------------------------------------

_ACUTE_HIV_KEYWORDS: tuple[str, ...] = (
    "acute retroviral", "acute hiv", "primary hiv", "seroconversion",
    "hiv seroconversion", "acute antiretroviral", "fiebig",
    "mononucleosis-like", "mono-like illness",
)

_HIV_AWARE_SYNDROMES: frozenset[str] = frozenset({
    "acute retroviral syndrome", "mononucleosis", "viral syndrome",
    "pharyngitis", "meningoencephalitis", "aseptic meningitis",
    "fever and rash", "lymphadenopathy",
})


def _hivid_cd4_prophylaxis_bridge(state: AssistantState) -> list[AssistantOption]:
    """If CD4 is known and <200, offer OI prophylaxis and ImmunoID chips."""
    hiv_ctx = state.hiv_context or {}
    cd4 = hiv_ctx.get("cd4")
    if cd4 is None:
        return []
    try:
        cd4_val = int(cd4)
    except (ValueError, TypeError):
        return []
    opts: list[AssistantOption] = []
    if cd4_val < 200:
        opts.append(AssistantOption(value="prophylaxis", label="OI prophylaxis (CD4 <200)"))
    if cd4_val < 100:
        opts.append(AssistantOption(value="hiv_oi_art_timing", label="OI screening & ART timing"))
    if cd4_val < 50:
        opts.append(AssistantOption(value="immunoid", label="Full immunosuppression screen"))
    return opts


def _hivid_cross_module_options(state: AssistantState, *, current_intent: str) -> list[AssistantOption]:
    """Return cross-module bridge options based on HIV context, excluding current intent."""
    hiv_ctx = state.hiv_context or {}
    opts: list[AssistantOption] = []

    # CD4-based prophylaxis bridge
    cd4_opts = _hivid_cd4_prophylaxis_bridge(state)
    opts.extend(cd4_opts)

    # HBV coinfection → drug interaction check
    if hiv_ctx.get("hbv_coinfected") and current_intent != "drug_interaction":
        opts.append(AssistantOption(value="drug_interaction", label="HBV-ART interactions"))

    # On ART + organisms known → check ART-antimicrobial interactions
    if hiv_ctx.get("on_art") and state.consult_organisms and current_intent != "drug_interaction":
        opts.append(AssistantOption(value="drug_interaction", label="ART-antimicrobial interactions"))

    # Pregnancy → pregnancy ART
    if hiv_ctx.get("pregnant") and current_intent != "hiv_pregnancy":
        opts.append(AssistantOption(value="hiv_pregnancy", label="Pregnancy ART planning"))

    # Active OI → OI-ART timing
    if hiv_ctx.get("active_oi") and current_intent != "hiv_oi_art_timing":
        opts.append(AssistantOption(value="hiv_oi_art_timing", label="ART timing with OI"))

    # Resistance mutations known → interpret them
    if hiv_ctx.get("resistance_mutations") and current_intent not in {"hiv_resistance", "hiv_switch"}:
        opts.append(AssistantOption(value="hiv_resistance", label="Interpret resistance mutations"))

    # Deduplicate by value
    seen: set[str] = set()
    deduped: list[AssistantOption] = []
    for o in opts:
        if o.value not in seen:
            seen.add(o.value)
            deduped.append(o)
    return deduped


def _probid_hiv_bridge_option(state: AssistantState) -> AssistantOption | None:
    """If ProbID syndrome looks like acute HIV/viral syndrome, offer HIVID bridge."""
    syndrome = (state.established_syndrome or "").lower()
    if not syndrome:
        return None
    if any(kw in syndrome for kw in _HIV_AWARE_SYNDROMES):
        return AssistantOption(value="hiv_initial_art", label="Could this be acute HIV? Start ART workup")
    return None


def _is_acute_hiv_mention(text: str) -> bool:
    """Check if message text mentions acute HIV/retroviral syndrome."""
    normalized = text.lower()
    return any(kw in normalized for kw in _ACUTE_HIV_KEYWORDS)


def _immunoid_hiv_bridge_options(state: AssistantState) -> list[AssistantOption]:
    """After ImmunoID, if HIV context exists or CD4 thresholds relevant, offer HIVID chips."""
    opts: list[AssistantOption] = []
    hiv_ctx = state.hiv_context or {}
    if hiv_ctx:
        # Patient has HIV context — offer relevant HIVID intents
        if not hiv_ctx.get("on_art"):
            opts.append(AssistantOption(value="hiv_initial_art", label="Start ART regimen"))
        if hiv_ctx.get("on_art"):
            opts.append(AssistantOption(value="hiv_monitoring", label="HIV monitoring schedule"))
        opts.append(AssistantOption(value="hiv_prep", label="PrEP guidance"))
    return opts


# ---------------------------------------------------------------------------
# Clarifying question system — ask ONE focused question when critical context
# is missing that would materially change the answer.
# ---------------------------------------------------------------------------

_SYNDROME_KEYWORDS: tuple[str, ...] = (
    "pneumonia", "pna", "cap", "hap", "vap", "uti", "urosepsis", "cystitis",
    "pyelonephritis", "bacteraemia", "bacteremia", "sepsis", "septic", "ssti",
    "cellulitis", "endocarditis", "osteomyelitis", "meningitis", "intra-abdominal",
    "iai", "clabsi", "cauti", "neutropenic", "febrile neutropenia", "abscess",
)

_HA_KEYWORDS: tuple[str, ...] = (
    "hospital", "nosocomial", "ha-", "healthcare", "hcap", "hap", "vap",
    "icu", "clabsi", "cauti", "nursing home", "ltcf", "inpatient",
)

_CA_KEYWORDS: tuple[str, ...] = (
    "community", "ca-", "cap", "outpatient", "community-acquired",
)

_ANTIBIOTIC_KEYWORDS: tuple[str, ...] = (
    "vancomycin", "vanco", "pip-tazo", "tazocin", "meropenem", "ertapenem",
    "ceftriaxone", "cefazolin", "cefepime", "ceftazidime", "flucloxacillin",
    "amoxicillin", "ciprofloxacin", "levofloxacin", "daptomycin", "linezolid",
    "ampicillin", "metronidazole", "azithromycin", "doxycycline", "tazobactam",
    "piperacillin", "clindamycin", "trimethoprim", "bactrim", "nitrofurantoin",
)


def _needs_clarifying_question(
    intent: str,
    message_text: str,
    state: AssistantState,
) -> tuple[str, list[AssistantOption]] | None:
    """
    Check whether a critical piece of context is missing that would materially
    change the answer. Returns (question_text, quick_reply_options) or None.

    Rules:
    - Only fires when the missing info would give a significantly different answer.
    - Never fires mid-consult when rich context is already established.
    - Never fires when the message itself already supplies the missing info.
    - Asks at most one question — the single most important gap.
    """
    msg_lower = message_text.lower()

    if intent == "empiric_therapy":
        has_syndrome = bool(state.established_syndrome)
        msg_mentions_syndrome = any(k in msg_lower for k in _SYNDROME_KEYWORDS)

        # Gap 1: No syndrome at all — regimen completely depends on this
        if not has_syndrome and not msg_mentions_syndrome:
            return (
                "Which clinical syndrome? The empiric regimen is quite different for "
                "pneumonia, UTI, bacteraemia, or skin infection — I want to make sure "
                "I give you the right first-line choice.",
                [
                    AssistantOption(
                        value="empiric_therapy",
                        label="Community pneumonia (CAP)",
                        insertText="Empiric therapy for community-acquired pneumonia",
                    ),
                    AssistantOption(
                        value="empiric_therapy",
                        label="Hospital pneumonia (HAP/VAP)",
                        insertText="Empiric therapy for hospital-acquired pneumonia",
                    ),
                    AssistantOption(
                        value="empiric_therapy",
                        label="Urosepsis / UTI",
                        insertText="Empiric therapy for urosepsis",
                    ),
                    AssistantOption(
                        value="empiric_therapy",
                        label="Bacteraemia / unknown source",
                        insertText="Empiric therapy for bacteraemia unknown source",
                    ),
                    AssistantOption(
                        value="empiric_therapy",
                        label="Skin / soft tissue (SSTI)",
                        insertText="Empiric therapy for skin and soft tissue infection",
                    ),
                ],
            )

        # Gap 2: Nosocomial-relevant syndrome without HA/CA clarity, no cultures yet
        # Use word-boundary matching to avoid false positives like "pneumoniae" → "pneumonia"
        import re as _re_clarify
        _is_nosocomial_relevant = any(
            _re_clarify.search(rf'\b{_re_clarify.escape(k)}\b', msg_lower)
            for k in ("pneumonia", "pna", "bacteraemia", "bacteremia", "sepsis", "uti", "urosepsis")
        ) or any(
            k in (state.established_syndrome or "").lower()
            for k in ("pneumonia", "bacteraemia", "bacteremia", "sepsis", "uti")
        )
        _has_acquisition = any(k in msg_lower for k in _HA_KEYWORDS + _CA_KEYWORDS)
        # If the message already names a specific organism from cultures, HA/CA is
        # clinically irrelevant — this is directed therapy, not truly empiric.
        _has_organism_in_msg = any(
            k in msg_lower for k in (
                "klebsiella", "e. coli", "escherichia", "pseudomonas", "enterobacter",
                "serratia", "proteus", "citrobacter", "morganella", "acinetobacter",
                "stenotrophomonas", "burkholderia", "staphylococcus", "s. aureus",
                "mssa", "mrsa", "enterococcus", "streptococcus", "candida",
                "blood culture", "blood cultures", "culture grew", "cultures growing",
                "cultures grew", "culture growing", "culture positive", "cultures positive",
            )
        )
        if _is_nosocomial_relevant and not _has_acquisition and not state.consult_organisms and not _has_organism_in_msg:
            return (
                "Healthcare-associated or community-acquired? "
                "This changes my first-line choice — I'd broaden for hospital-acquired "
                "(anti-pseudomonal, MRSA coverage) and can stay narrower for community.",
                [
                    AssistantOption(
                        value="empiric_therapy",
                        label="Community-acquired",
                        insertText="Community-acquired",
                    ),
                    AssistantOption(
                        value="empiric_therapy",
                        label="Healthcare-associated / nosocomial",
                        insertText="Healthcare-associated / nosocomial",
                    ),
                ],
            )

    if intent == "treatment_failure":
        # If message is vague and we have no established context, ask what they're on
        has_antibiotic = any(k in msg_lower for k in _ANTIBIOTIC_KEYWORDS)
        has_organism = bool(state.consult_organisms) or any(
            k in msg_lower for k in ("aureus", "mssa", "mrsa", "e. coli", "klebsiella",
                                      "enterococcus", "pseudomonas", "candida", "strep")
        )
        has_syndrome = bool(state.established_syndrome)
        if not has_antibiotic and not has_organism and not has_syndrome:
            return (
                "Which antibiotic is the patient on, and roughly how long? "
                "That's the key starting point — I can then work through whether it's "
                "the wrong drug, wrong dose, undrained source, or something else entirely.",
                [],
            )

    return None


def _build_clarifying_question_response(
    question: str,
    options: list[AssistantOption],
    state: AssistantState,
    intent: str | None = None,
    original_text: str | None = None,
) -> AssistantTurnResponse:
    """Build an assistant response that asks a single targeted clarifying question.

    When *intent* and *original_text* are provided, they are saved in the state
    so that the user's answer (click or free-text) routes back to the same
    intent handler with the original context preserved.
    """
    if intent:
        state.pending_intent = intent
    if original_text:
        state.pending_intent_context = original_text
    return AssistantTurnResponse(
        assistantMessage=question,
        state=state,
        options=options,
        tips=[
            "Answer in plain language or pick one of the options above.",
            "The more context you give, the more specific I can be.",
        ],
    )


def _build_suggested_placeholder(state: AssistantState, last_intent: str = "") -> str:
    """
    Build a context-aware composer placeholder that reflects what is already known
    in this consult, so the input feels like a running conversation rather than
    a blank slate after each response.
    """
    syndrome = state.established_syndrome
    orgs = state.consult_organisms or []
    pc = state.patient_context

    # Build a compact patient label
    patient_parts: List[str] = []
    if pc:
        if pc.age_years is not None:
            patient_parts.append(f"{pc.age_years}yo")
        if pc.sex:
            patient_parts.append(pc.sex)
    patient_label = " ".join(patient_parts) if patient_parts else ""

    # Intent-specific trailing suggestions
    intent_hints: dict[str, str] = {
        "empiric_therapy": "Paste cultures when back, ask about duration, or check allergy safety...",
        "mechid": "Ask about dosing, duration, IV-to-oral, or source control...",
        "doseid": "Ask about IV-to-oral step-down, OPAT eligibility, or treatment duration...",
        "allergyid": "Ask about alternative agents, allergy delabeling, or dosing...",
        "duration": "Ask about IV-to-oral, OPAT, or source control status...",
        "treatment_failure": "Share updated cultures, labs, or imaging findings...",
        "iv_to_oral": "Ask about duration, OPAT, or discharge counselling...",
        "opat": "Ask about duration, monitoring schedule, or discharge counselling...",
        "source_control": "Ask about duration, OPAT eligibility, or empiric coverage...",
        "followup_tests": "Ask about duration, treatment plan, or consult summary...",
        "impression_plan": "Ask about any part of the plan, or paste new results...",
        "hiv_initial_art": "Ask about monitoring, OI prophylaxis, drug interactions, or ART timing...",
        "hiv_monitoring": "Ask about ART regimen, drug interactions, or OI prophylaxis...",
        "hiv_prep": "Ask about PEP, ART, drug interactions, or doxyPEP for STI prevention...",
        "hiv_pep": "Ask about PrEP transition, ART if HIV+, or drug interactions...",
        "hiv_pregnancy": "Ask about delivery planning, monitoring, or neonatal prophylaxis...",
        "hiv_oi_art_timing": "Ask about ART regimen, OI prophylaxis, or drug interactions...",
        "hiv_treatment_failure": "Ask about resistance testing, regimen switch, or adherence support...",
        "hiv_resistance": "Ask about regimen switch, salvage options, or drug interactions...",
        "hiv_switch": "Ask about monitoring after switch, drug interactions, or simplification...",
    }
    hint = intent_hints.get(last_intent, "Ask a follow-up or paste new results...")

    if syndrome and orgs:
        org_str = ", ".join(orgs[:2]) + ("..." if len(orgs) > 2 else "")
        prefix = f"{patient_label + ' · ' if patient_label else ''}{org_str} · {syndrome}"
        return f"{prefix} — {hint}"
    if syndrome:
        prefix = f"{patient_label + ' · ' if patient_label else ''}{syndrome}"
        return f"{prefix} — {hint}"
    if orgs:
        org_str = ", ".join(orgs[:2]) + ("..." if len(orgs) > 2 else "")
        prefix = f"{patient_label + ' · ' if patient_label else ''}{org_str}"
        return f"{prefix} — {hint}"
    if patient_label:
        return f"{patient_label} established — {hint}"
    return "Describe the patient or paste culture results..."


def _assistant_empiric_therapy_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer an empiric therapy question — best regimen before cultures return."""
    clarifying = _needs_clarifying_question("empiric_therapy", message_text, state)
    if clarifying is not None:
        return _build_clarifying_question_response(
            clarifying[0], clarifying[1], state,
            intent="empiric_therapy", original_text=message_text,
        )

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "For empiric therapy, the best regimen depends on the syndrome and patient factors. "
        "Start a syndrome workup and I can give a more precise recommendation based on the clinical picture."
    )
    antibiogram_block: str | None = None
    if state.institutional_antibiogram:
        antibiogram_block = antibiogram_to_prompt_block(state.institutional_antibiogram)
    answer, narration_refined = narrate_empiric_therapy_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        institutional_antibiogram_block=antibiogram_block,
        fallback_message=fallback_message,
    )
    antibiogram_tip = (
        "Your institutional antibiogram is loaded — recommendations reflect local resistance rates."
        if state.institutional_antibiogram
        else "Upload your institutional antibiogram for location-specific empiric guidance."
    )
    # Empiric → next steps: paste cultures when back, allergy check, dose calc
    empiric_options: list[AssistantOption] = [
        AssistantOption(value="mechid", label="Cultures back — paste results"),
        AssistantOption(value="allergyid", label="Allergy check"),
        AssistantOption(value="doseid", label="Calculate dosing"),
    ]
    if not state.institutional_antibiogram:
        empiric_options.insert(0, AssistantOption(value="upload_antibiogram", label="Upload institutional antibiogram"))
    if not state.established_syndrome:
        empiric_options.insert(0, AssistantOption(value="probid", label="Syndrome workup first"))
    syndrome_opts = _syndrome_next_step_options(state, exclude={"source_control", "followup_tests"})
    for opt in reversed(syndrome_opts):
        empiric_options.insert(0, opt)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=empiric_options,
        suggestedPlaceholder=_build_suggested_placeholder(state, "empiric_therapy"),
        tips=[
            antibiogram_tip,
            "Once cultures return, paste the organism and susceptibilities and I'll refine to a targeted regimen.",
        ],
    )


def _assistant_iv_to_oral_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Assess IV-to-oral step-down eligibility and name the oral agent."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "IV-to-oral step-down is appropriate when the patient is afebrile for 24-48h, "
        "WBC is trending to normal, they are tolerating oral intake, and there is no endovascular or CNS source requiring IV therapy. "
        "The specific oral agent depends on the organism and susceptibilities."
    )
    answer, narration_refined = narrate_iv_to_oral_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="mechid", label="Paste susceptibilities"),
            AssistantOption(value="doseid", label="Calculate oral dose"),
            AssistantOption(value="allergyid", label="Allergy check"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "Paste the organism and AST and I can confirm the best oral agent for this isolate.",
            "For high-bioavailability options like fluoroquinolones or TMP-SMX, serum levels are equivalent to IV.",
        ],
    )


def _assistant_duration_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a treatment duration question."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Duration depends on the syndrome, organism, and whether source control has been achieved. "
        "For bacteraemia the clock typically starts from the first negative blood culture. "
        "Provide the syndrome and organism for a specific recommendation."
    )
    answer, narration_refined = narrate_duration_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="source_control", label="Source control status"),
            AssistantOption(value="opat", label="OPAT eligibility"),
            AssistantOption(value="iv_to_oral", label="IV-to-oral step-down?"),
            AssistantOption(value="doseid", label="Calculate dosing"),
            *(
                [AssistantOption(value="consult_summary", label="Full consult summary")]
                if _is_mid_consult(state)
                else [AssistantOption(value="restart", label="Start new consult")]
            ),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "duration"),
        tips=[
            "Duration runs from first negative culture for bacteraemia, not from antibiotic start.",
            "Source control status (line removed, abscess drained) is the biggest modifier of duration.",
        ],
    )


def _assistant_followup_tests_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a follow-up test ordering question — TEE, repeat cultures, imaging, drug levels."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Follow-up testing depends on the syndrome and organism. "
        "For S. aureus bacteraemia, TEE is recommended for most patients and repeat cultures at 48-72h are essential. "
        "Provide the organism and syndrome for specific guidance."
    )
    answer, narration_refined = narrate_followup_tests_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="duration", label="How long to treat?"),
            AssistantOption(value="doseid", label="Calculate dosing"),
            *(
                [AssistantOption(value="mechid", label="Paste culture results")]
                if not state.consult_organisms
                else []
            ),
            *(
                [AssistantOption(value="consult_summary", label="Full consult summary")]
                if _is_mid_consult(state)
                else [AssistantOption(value="restart", label="Start new consult")]
            ),
        ],
        tips=[
            "For S. aureus bacteraemia, TEE should not be delayed beyond 5-7 days.",
            "Document culture clearance before counting duration for bacteraemia.",
        ],
    )


def _assistant_oral_therapy_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a question about oral antibiotic options — which syndromes can be treated orally."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Oral antibiotics are appropriate for many syndromes. Bone and joint infections can be stepped down to oral after "
        "initial IV stabilisation (OVIVA trial). UTI and cystitis should always be oral. Mild CAP and cellulitis are oral from the start. "
        "Provide the syndrome and organism for specific recommendations."
    )
    answer, narration_refined = narrate_oral_therapy_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="iv_to_oral", label="Assess step-down criteria"),
            AssistantOption(value="doseid", label="Calculate oral dose"),
            AssistantOption(value="mechid", label="Paste culture results"),
            AssistantOption(value="duration", label="Confirm duration"),
        ],
        tips=[
            "OVIVA (2019): oral step-down non-inferior to IV for bone and joint infections after clinical stabilisation.",
            "High-bioavailability agents (fluoroquinolones, TMP-SMX, linezolid) achieve serum levels equivalent to IV.",
        ],
    )


def _assistant_discharge_counselling_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Generate discharge counselling — treatment plan, monitoring, red flags."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Discharge counselling should cover: the antibiotic name, form, dose, and duration; what to monitor "
        "(labs, wound, temperature); red flag symptoms prompting return to ED; and the follow-up plan. "
        "Provide the syndrome and treatment details for a specific plan."
    )
    answer, narration_refined = narrate_discharge_counselling_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="duration", label="Confirm treatment duration"),
            AssistantOption(value="doseid", label="Confirm dose"),
            AssistantOption(value="opat", label="OPAT assessment"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "Always tell the patient not to stop antibiotics early even if feeling better.",
            "Red flags for S. aureus or endocarditis: recurrent fever, new joint pain, neurological symptoms.",
        ],
    )


def _assistant_stewardship_review_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Review a listed antibiotic regimen — stop, narrow, or continue each agent."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "List your current antibiotics and I'll advise which to stop, narrow, or continue based on the cultures "
        "and clinical picture. For example: 'Patient is on vancomycin, pip-tazo, and fluconazole — cultures grew MSSA.'"
    )
    answer, narration_refined = narrate_stewardship_review_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="mechid", label="Paste full susceptibility report"),
            AssistantOption(value="iv_to_oral", label="Consider oral step-down"),
            AssistantOption(value="duration", label="Confirm duration for narrowed agent"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "MSSA on vancomycin: always narrow to beta-lactam (oxacillin or cefazolin) — beta-lactams are superior.",
            "Empiric antifungals: can usually stop at 72-96h if cultures negative and patient defervesced.",
        ],
    )


def _assistant_stewardship_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a de-escalation or antibiotic stewardship question."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "De-escalation should happen as soon as culture results allow narrowing. "
        "Key principle: always narrow MSSA from vancomycin to a beta-lactam, and de-escalate broad gram-negative coverage "
        "once susceptibilities permit. Paste the culture results and I can give a specific recommendation."
    )
    answer, narration_refined = narrate_stewardship_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="mechid", label="Paste susceptibilities"),
            AssistantOption(value="doseid", label="Calculate narrow agent dose"),
            AssistantOption(value="allergyid", label="Allergy check"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "Paste the final susceptibility report and I'll confirm the narrowest effective agent.",
            "Always narrow MSSA from vancomycin to a beta-lactam — oxacillin or cefazolin is superior.",
        ],
    )


def _assistant_sepsis_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Sepsis Hour-1 bundle: measure lactate, draw blood cultures ×2 before antibiotics, "
        "give broad-spectrum antibiotics within 1 hour, 30 mL/kg IV crystalloid if hypotensive or lactate ≥4, "
        "start vasopressors if MAP <65. De-escalate at 48-72h when cultures return. "
        "PCT-guided stopping: drop >80% from peak or <0.5 ng/mL."
    )
    answer, narration_refined = narrate_sepsis_management_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="empiric_therapy", label="Empiric antibiotic selection"),
            AssistantOption(value="source_control", label="Source control decision"),
            AssistantOption(value="doseid", label="Calculate antibiotic dose"),
            AssistantOption(value="stewardship", label="De-escalation plan"),
        ],
        tips=[
            "Every hour of antibiotic delay in septic shock increases mortality ~7% — start broad, narrow later.",
            "PCT-guided stopping: drop >80% from peak, or <0.5 ng/mL — safe to stop antibiotics in most sepsis.",
        ],
    )


def _assistant_cns_infection_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Bacterial meningitis: ceftriaxone 2g BD + dexamethasone 0.15mg/kg QDS — give dexamethasone BEFORE or WITH first antibiotic dose. "
        "Add ampicillin if Listeria risk (age >50, immunosuppressed, pregnancy). "
        "Viral encephalitis: start acyclovir 10mg/kg TDS empirically — do not wait for HSV PCR. "
        "Brain abscess: neurosurgical drainage + ceftriaxone + metronidazole."
    )
    answer, narration_refined = narrate_cns_infection_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="fluid_interpretation", label="Interpret LP result"),
            AssistantOption(value="doseid", label="Calculate dose"),
            AssistantOption(value="duration", label="Duration of treatment"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
        ],
        tips=[
            "Dexamethasone only reduces mortality if given before or with the first antibiotic — giving it after is largely futile.",
            "Listeria is cephalosporin-resistant — always add ampicillin in patients over 50 or immunosuppressed.",
        ],
    )


def _assistant_mycobacterial_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Drug-sensitive TB: 2 months HRZE (isoniazid + rifampicin + pyrazinamide + ethambutol), then 4 months HR. "
        "Add pyridoxine with isoniazid. Monitor LFTs monthly and visual acuity (ethambutol). "
        "LTBI: 3HP (rifapentine + isoniazid weekly ×12) is preferred. "
        "MAC pulmonary: azithromycin + ethambutol + rifampicin ×12 months culture-negative."
    )
    answer, narration_refined = narrate_mycobacterial_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="drug_interaction", label="Rifampicin drug interactions"),
            AssistantOption(value="doseid", label="Calculate TB drug dose"),
            AssistantOption(value="duration", label="Confirm treatment duration"),
            AssistantOption(value="followup_tests", label="Monitoring tests"),
        ],
        tips=[
            "Rifampicin induces CYP450 — check all concurrent medications before starting (tacrolimus, warfarin, azoles, ARVs, OCPs).",
            "Never start LTBI treatment without first excluding active TB — CXR + symptom screen mandatory.",
        ],
    )


def _assistant_pregnancy_antibiotics_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Safe in pregnancy (all trimesters): beta-lactams, azithromycin, clindamycin. "
        "Avoid: fluoroquinolones (cartilage), tetracyclines after 16 weeks (bone/teeth), aminoglycosides (fetal ototoxicity). "
        "Near term: avoid nitrofurantoin (≥36 weeks — haemolytic anaemia) and TMP-SMX (kernicterus). "
        "Tell me the infection type and trimester and I'll give a specific recommendation."
    )
    answer, narration_refined = narrate_pregnancy_antibiotics_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="doseid", label="Calculate safe dose"),
            AssistantOption(value="allergyid", label="Allergy check"),
            AssistantOption(value="empiric_therapy", label="Empiric options"),
        ],
        tips=[
            "Beta-lactams are first-line for almost all infections in pregnancy — use them broadly.",
            "TMP-SMX: avoid in 1st trimester (folate antagonist) and near term (kernicterus) — safe in 2nd trimester with folic acid 5mg/day.",
        ],
    )


def _assistant_travel_medicine_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Fever in a returned traveller: exclude malaria first — thick and thin blood film + RDT, repeat ×3 if negative. "
        "P. falciparum: artemether-lumefantrine ×3 days (uncomplicated) or IV artesunate (severe). "
        "Also consider dengue (thrombocytopenia + leukopenia), typhoid (blood culture, ceftriaxone), "
        "rickettsiae (eschar + rash — treat empirically with doxycycline). "
        "Tell me the travel destination and incubation period."
    )
    answer, narration_refined = narrate_travel_medicine_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="biomarker_interpretation", label="Interpret lab results"),
            AssistantOption(value="empiric_therapy", label="Empiric treatment"),
            AssistantOption(value="probid", label="Syndrome workup"),
        ],
        tips=[
            "Malaria must be excluded in any fever within 3 months of travel to endemic areas — even if the patient took prophylaxis.",
            "Suspect VHF if travel to sub-Saharan Africa within 21 days + fever + haemorrhagic features — isolate first, call ID.",
        ],
    )


def _assistant_impression_plan_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Generate a structured ID consult impression and plan for the medical record."""
    patient_summary = _consult_prior_context_summary(state)
    syndrome = state.established_syndrome or "not yet established"
    orgs = ", ".join(state.consult_organisms) if state.consult_organisms else "not yet identified"
    fallback_message = (
        f"IMPRESSION: ID consult for {syndrome}. Causative organism(s): {orgs}. "
        "PLAN: 1. Antimicrobial therapy — [specify agent, dose, route based on AST]. "
        "2. Duration — [syndrome-specific]. "
        "3. Source control — [line removal, drainage, debridement if indicated]. "
        "4. Monitoring — drug levels, renal function, clinical response at 48-72h. "
        "5. Investigations — pending cultures, imaging follow-up, TEE if indicated. "
        "6. Stewardship — review for de-escalation at 48-72h when cultures confirmed. "
        "7. ID follow-up — [outpatient OPAT review if applicable]."
    )
    answer, narration_refined = narrate_impression_plan(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        last_probid_summary=state.last_probid_summary,
        last_mechid_summary=state.last_mechid_summary,
        last_doseid_summary=state.last_doseid_summary,
        last_allergy_summary=state.last_allergy_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="consult_summary", label="Full consult summary"),
            AssistantOption(value="discharge_counselling", label="Discharge counselling"),
            AssistantOption(value="course_tracker", label="Track treatment course"),
        ],
        tips=[
            "The impression and plan can be copied directly into the medical record — review and edit as needed before signing.",
            "ID consult notes are most useful when they include a clear diagnosis, specific antibiotic with dose, duration, and a monitoring plan.",
        ],
    )


def _assistant_duke_criteria_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Apply Modified Duke Criteria and classify IE likelihood."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Modified Duke Criteria for IE — MAJOR: (1) positive blood cultures ×2 with typical organism (S. aureus, viridans Streptococci, HACEK, Enterococcus without primary focus) "
        "or persistently positive cultures ≥12h apart; (2) endocardial involvement on echo (vegetation, abscess, new valvular regurgitation). "
        "MINOR: (1) predisposing valve disease or IVDU; (2) fever ≥38°C; (3) vascular phenomena (emboli, Janeway lesions, septic infarcts); "
        "(4) immunological phenomena (Osler nodes, Roth spots, RF); (5) positive blood cultures not meeting major criteria. "
        "DEFINITE: 2 major OR 1 major + 3 minor OR 5 minor. POSSIBLE: 1 major + 1 minor OR 3 minor. REJECTED: firm alternative diagnosis, resolution with ≤4 days antibiotics, or no pathological evidence at surgery. "
        "Tell me the clinical findings and I'll classify the case."
    )
    answer, narration_refined = narrate_duke_criteria_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="followup_tests", label="Order TEE / echo"),
            AssistantOption(value="mechid", label="Review organism + AST"),
            AssistantOption(value="duration", label="Treatment duration for IE"),
        ],
        tips=[
            "TTE sensitivity for vegetation is ~60-70%; TEE is >90% — order TEE if TTE negative and suspicion remains.",
            "S. aureus bacteraemia warrants TEE routinely regardless of Duke score — all SAB patients need echocardiography.",
        ],
    )


def _assistant_ast_meaning_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Explain the clinical meaning of AST resistance phenotypes."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Key AST clinical pitfalls: ESBL — treat with carbapenem not pip-tazo (MERINO 2018: meropenem superior for ESBL bacteraemia). "
        "MRSA — all beta-lactams unreliable even if reported 'susceptible'; use vancomycin, daptomycin, or linezolid. "
        "Vancomycin MIC ≥2 — clinical failure likely even if technically 'susceptible'; consider daptomycin or ceftaroline. "
        "AmpC (ESCPM: Enterobacter, Serratia, Citrobacter, Providencia, Morganella) — 3GC may appear susceptible in vitro but derepression occurs on therapy; use cefepime or carbapenems. "
        "D-zone test positive = inducible clindamycin resistance — do not use clindamycin. "
        "Daptomycin + pulmonary infections — daptomycin is inactivated by surfactant; never use for pneumonia. "
        "Tell me the specific result you want explained."
    )
    answer, narration_refined = narrate_ast_clinical_meaning_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="mechid", label="Full MechID review"),
            AssistantOption(value="empiric_therapy", label="Alternative therapy options"),
            AssistantOption(value="doseid", label="Dose calculation"),
        ],
        tips=[
            "The MERINO trial (NEJM 2018) definitively showed pip-tazo is inferior to meropenem for ESBL/AmpC bacteraemia — always use carbapenem.",
            "Paste the full AST panel for a complete analysis of all resistance mechanisms.",
        ],
    )


def _assistant_complexity_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Assess case complexity and escalation thresholds."""
    patient_summary = _consult_prior_context_summary(state)
    # Extract known complexity features from state context
    complexity_features: list[str] = []
    if state.consult_organisms:
        high_risk_orgs = {"mrsa", "vre", "esbl", "mdr", "pdra", "candida", "aspergillus", "mucor", "klebsiella", "acinetobacter"}
        for org in state.consult_organisms:
            if any(h in org.lower() for h in high_risk_orgs):
                complexity_features.append(f"High-risk organism: {org}")
    if state.established_syndrome:
        high_stakes = {"endocarditis", "meningitis", "osteomyelitis", "septic arthritis", "bacteraemia", "candidaemia", "brain abscess"}
        if any(s in state.established_syndrome.lower() for s in high_stakes):
            complexity_features.append(f"High-stakes syndrome: {state.established_syndrome}")
    fallback_message = (
        "Case complexity red flags requiring escalation to senior ID or MDT: "
        "(1) MDR/XDR organism with limited treatment options; "
        "(2) Immunocompromised host (transplant, haematological malignancy, biologics); "
        "(3) Prosthetic material infection (valve, joint, mesh, CIED); "
        "(4) CNS involvement; "
        "(5) Treatment failure after 72-96h of appropriate therapy; "
        "(6) Polymicrobial bacteraemia; "
        "(7) Unusual or rare pathogen. "
        "3 or more red flags = escalate to senior ID review and consider MDT (cardiac surgery, transplant team, microbiology). "
        "Tell me the specific case features."
    )
    answer, narration_refined = narrate_complexity_flag_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        complexity_features=complexity_features or None,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="consult_summary", label="Full consult summary"),
            AssistantOption(value="impression_plan", label="Generate impression + plan"),
            AssistantOption(value="followup_tests", label="Investigations to order"),
        ],
        tips=[
            "Prosthetic material infections nearly always require MDT involvement — cardiac surgery for valve, orthopaedics for joint, ID pharmacy for extended regimens.",
            "Consider FDG-PET/CT for occult embolic foci or metastatic infection when source control is unclear.",
        ],
    )


def _assistant_course_tracker_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Track day-of-therapy milestones and advise what to do next."""
    patient_summary = _consult_prior_context_summary(state)
    syndrome = state.established_syndrome or "the infection being treated"
    fallback_message = (
        f"Day-of-therapy milestones for {syndrome}: "
        "SAB — Day 0-2: remove or change IV access; Day 2-3: repeat blood cultures to confirm clearance; Day 7-14: echocardiography (TEE preferred); full 14d IV from first negative culture (uncomplicated), 28-42d (complicated/endocarditis). "
        "Candidaemia — Day 0-1: ophthalmology review (fundoscopy); remove central line; Day 14: first negative culture = clock start; stop echinocandin at Day 14 minimum. "
        "IE — Day 10-17: assess POET eligibility for oral step-down (stable, no CNS emboli, no abscess, good bioavailability organism); "
        "Bone/joint — Day 7: consider OVIVA oral step-down if stable and susceptible organism; "
        "GNR bacteraemia — Day 3-5: de-escalate on culture; total 7-14d depending on source control. "
        "Tell me the day of therapy and syndrome for specific advice."
    )
    answer, narration_refined = narrate_course_tracker_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="iv_to_oral", label="IV-to-oral step-down"),
            AssistantOption(value="stewardship", label="De-escalation review"),
            AssistantOption(value="opat", label="OPAT candidacy"),
        ],
        tips=[
            "Clock the start of treatment from the first negative blood culture, not the antibiotic start date — this matters for SAB and candidaemia duration calculations.",
            "Set a calendar reminder at day 2-3 for repeat blood cultures and at day 7 for first formal de-escalation review.",
        ],
    )


def _assistant_treatment_failure_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Structured differential for treatment failure — patient not improving on antibiotics."""
    clarifying = _needs_clarifying_question("treatment_failure", message_text, state)
    if clarifying is not None:
        return _build_clarifying_question_response(
            clarifying[0], clarifying[1], state,
            intent="treatment_failure", original_text=message_text,
        )

    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Treatment failure has a structured differential: wrong diagnosis (drug fever, non-infectious cause), "
        "wrong drug (resistance, superinfection), wrong dose (check drug levels), uncontrolled source (repeat imaging), "
        "metastatic seeding (TEE, MRI spine for S. aureus), or host immune failure. "
        "Tell me the clinical picture and current antibiotics and I'll work through it."
    )
    answer, narration_refined = narrate_treatment_failure_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="mechid", label="Paste updated cultures"),
            AssistantOption(value="followup_tests", label="What tests to order"),
            AssistantOption(value="doseid", label="Check drug levels / dosing"),
            AssistantOption(value="source_control", label="Source control decision"),
        ],
        suggestedPlaceholder=_build_suggested_placeholder(state, "treatment_failure"),
        tips=[
            "S. aureus still bacteraemic at day 3: get a TEE and MRI spine — metastatic seeding is the rule, not the exception.",
            "Drug fever classically appears day 7-10, patient looks well despite fever, eosinophilia may be present.",
        ],
    )


def _assistant_biomarker_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Interpret infectious disease biomarkers — PCT, BDG, galactomannan, CrAg, IGRA."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Biomarker interpretation depends on the clinical context. Key thresholds: "
        "procalcitonin >0.5 ng/mL suggests bacterial infection; <0.1 makes it unlikely. "
        "Beta-D-glucan >80 pg/mL is positive for invasive fungal infection (not Cryptococcus or Mucor). "
        "Galactomannan ≥0.5 (serum) suggests Aspergillus in the right host. "
        "Tell me the specific value and context."
    )
    answer, narration_refined = narrate_biomarker_interpretation_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="empiric_therapy", label="Empiric antifungal / antibiotic"),
            AssistantOption(value="followup_tests", label="What to test next"),
            AssistantOption(value="probid", label="Syndrome workup"),
        ],
        tips=[
            "PCT stopping rule: drop >80% from peak, or <0.5 ng/mL — safe to stop antibiotics in most settings.",
            "Two consecutive positive BDG results increase specificity — a single value in a low-risk patient may be a false positive.",
        ],
    )


def _assistant_fluid_interpretation_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Interpret CSF, pleural, ascitic, or synovial fluid results."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Paste the fluid results and I'll interpret them. Key rules: "
        "CSF WBC >1000 (neutrophilic) + low glucose = bacterial meningitis — treat immediately. "
        "Pleural PMN count + low pH + low glucose = empyema needing drainage. "
        "Ascitic PMN ≥250 = SBP — start cefotaxime now. "
        "Synovial WBC >50,000 strongly suggests septic arthritis."
    )
    answer, narration_refined = narrate_fluid_interpretation_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="empiric_therapy", label="Empiric therapy for this result"),
            AssistantOption(value="cns_infection", label="CNS infection guidance"),
            AssistantOption(value="source_control", label="Source control / drainage"),
            AssistantOption(value="probid", label="Syndrome workup"),
        ],
        tips=[
            "For suspected bacterial meningitis: start antibiotics immediately — do not delay for CT unless focal neurology or papilloedema.",
            "Cryptococcal meningitis: measure opening pressure at every LP — raised ICP is the main driver of early mortality.",
        ],
    )


def _assistant_allergy_delabeling_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Assess whether a reported antibiotic allergy is genuine and advise on delabeling."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Most reported penicillin allergies are not true allergies — over 90% tolerate penicillin on formal testing. "
        "Low-risk history (remote, mild rash, GI upset, childhood amoxicillin rash with viral illness) → direct oral challenge is appropriate. "
        "True IgE-mediated reactions (anaphylaxis, angioedema within 1 hour) → avoid and refer allergy. "
        "Tell me the specific reaction history and I'll stratify the risk."
    )
    answer, narration_refined = narrate_allergy_delabeling_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="allergyid", label="Find safe alternative"),
            AssistantOption(value="empiric_therapy", label="Empiric therapy options"),
            AssistantOption(value="mechid", label="Paste susceptibilities"),
        ],
        tips=[
            "Penicillin-to-cephalosporin cross-reactivity is ~1-2% (not 10%) — driven by R1 side chain, not the ring.",
            "Red man syndrome with vancomycin is NOT an allergy — it is a rate-related infusion reaction.",
        ],
    )


def _assistant_fungal_management_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Manage invasive fungal infection — candidaemia, Aspergillus, Cryptococcus, Mucor."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Invasive fungal infection management depends on the species. "
        "Candidaemia: remove lines, ophthalmology review, echinocandin first-line, 14 days from first negative culture. "
        "Aspergillosis: voriconazole or isavuconazole, minimum 6-12 weeks. "
        "Cryptococcal meningitis: amphotericin B + flucytosine induction, manage ICP with therapeutic LPs. "
        "Mucormycosis: urgent surgery + liposomal amphotericin — do NOT use voriconazole. "
        "Tell me the species and I'll give a specific plan."
    )
    answer, narration_refined = narrate_fungal_management_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="source_control", label="Source control decision"),
            AssistantOption(value="doseid", label="Calculate antifungal dose"),
            AssistantOption(value="drug_interaction", label="Check antifungal interactions"),
            AssistantOption(value="followup_tests", label="Monitoring tests"),
        ],
        tips=[
            "Candidaemia: the 14-day clock starts from the FIRST NEGATIVE blood culture — not from when you started antifungals.",
            "Mucormycosis: voriconazole is contraindicated — it has no activity against Mucorales and may promote growth.",
        ],
    )


def _assistant_drug_interaction_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a drug-drug interaction question involving antimicrobial agents."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Drug interactions with antimicrobials are common. Key ones to know: rifampin is a potent inducer that "
        "lowers levels of tacrolimus, azoles, and warfarin. Azole antifungals raise tacrolimus levels significantly. "
        "Linezolid is an MAOI — avoid with serotonergic drugs. Vancomycin with pip-tazo increases nephrotoxicity risk. "
        "Tell me the specific drugs and I'll advise on the interaction."
    )
    answer, narration_refined = narrate_drug_interaction_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        hiv_context=state.hiv_context,
        fallback_message=fallback_message,
    )
    di_opts: list[AssistantOption] = [
        AssistantOption(value="allergyid", label="Allergy check"),
        AssistantOption(value="doseid", label="Calculate adjusted dose"),
        AssistantOption(value="mechid", label="Paste susceptibilities"),
    ]
    # If HIV context exists, offer relevant HIVID chips
    if state.hiv_context:
        if state.hiv_context.get("on_art"):
            di_opts.append(AssistantOption(value="hiv_switch", label="Consider ART switch"))
        else:
            di_opts.append(AssistantOption(value="hiv_initial_art", label="Select ART regimen"))
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=di_opts,
        tips=[
            "Rifampin interactions take 2-3 days to onset and 2 weeks to offset — plan accordingly.",
            "Azole antifungals (especially fluconazole) typically require tacrolimus dose reduction of 30-50%.",
        ],
    )


def _assistant_prophylaxis_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Answer a prophylaxis dosing question for immunosuppressed patients."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Prophylaxis dosing depends on the infection risk and the degree of immunosuppression. "
        "For PCP: TMP-SMX 960mg OD or three times weekly — first choice. "
        "For antifungal prophylaxis in high-risk haematology: posaconazole 300mg OD (delayed-release). "
        "For CMV in transplant: valganciclovir 900mg OD, dose-adjusted for renal function. "
        "Tell me the patient's immunosuppressant and I'll give the specific regimen."
    )
    answer, narration_refined = narrate_prophylaxis_dose_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="immunoid", label="Full immunosuppression screen"),
            AssistantOption(value="doseid", label="Calculate prophylaxis dose"),
            AssistantOption(value="drug_interaction", label="Check drug interactions"),
        ],
        tips=[
            "PCP prophylaxis threshold: prednisolone >20mg/day for >4 weeks, or any calcineurin inhibitor + antimetabolite.",
            "Posaconazole monitoring: trough after 5-7 days; target >0.7 mg/L for prophylaxis.",
        ],
    )


def _assistant_source_control_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Advise on source control — line removal, abscess drainage, surgical debridement, implant management."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "Source control is critical in many infections. Key rules: always remove lines for S. aureus bacteraemia "
        "and Candida fungaemia. Drain any abscess >2cm. For prosthetic joint infection, discuss DAIR vs 2-stage exchange "
        "based on timing and organism. For necrotising fasciitis: immediate surgical debridement is life-saving. "
        "Tell me the specific scenario and I'll advise on the source control decision."
    )
    answer, narration_refined = narrate_source_control_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="duration", label="Duration after source control"),
            AssistantOption(value="opat", label="OPAT eligibility"),
            AssistantOption(value="doseid", label="Calculate antibiotic dose"),
            *(
                [AssistantOption(value="mechid", label="Paste susceptibilities")]
                if not state.consult_organisms
                else []
            ),
            *(
                [AssistantOption(value="consult_summary", label="Full consult summary")]
                if _is_mid_consult(state)
                else []
            ),
        ],
        tips=[
            "S. aureus bacteraemia: line removal reduces duration of bacteraemia and mortality — do not delay.",
            "Prosthetic joint DAIR: rifampicin must be added once wound is sealed — never start in bacteraemic phase.",
        ],
    )


def _opat_doseid_options(state: AssistantState) -> List[AssistantOption]:
    """Return once-daily DoseID options appropriate for OPAT based on known organisms."""
    organisms_lower = " ".join(state.consult_organisms or []).lower()
    meds_by_id = _assistant_doseid_medications_by_id()
    options: List[AssistantOption] = []
    # Suggest ertapenem for susceptible GNR (non-pseudomonal)
    if any(org in organisms_lower for org in ("coli", "klebsiella", "enterobacter", "proteus", "morganella", "serratia")):
        if "ertapenem" in meds_by_id:
            options.append(AssistantOption(value="doseid_pick:ertapenem", label="Dose ertapenem 1g OD (OPAT)"))
    # Suggest ceftriaxone for susceptible GNR or streptococcal
    if any(org in organisms_lower for org in ("coli", "klebsiella", "streptococcus", "strep", "enterobacter")):
        if "ceftriaxone" in meds_by_id:
            options.append(AssistantOption(value="doseid_pick:ceftriaxone", label="Dose ceftriaxone 2g OD (OPAT)"))
    # Generic OPAT dosing if no organism-specific match
    if not options:
        options.append(AssistantOption(value="doseid", label="Calculate OPAT dose"))
    return options[:2]  # cap at 2


def _assistant_opat_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Assess OPAT candidacy — suitability for outpatient IV antibiotic therapy at discharge."""
    patient_summary = _consult_prior_context_summary(state)
    fallback_message = (
        "OPAT is appropriate when the patient is clinically stable, the syndrome requires continued IV therapy, "
        "and there is no suitable oral alternative. Once-daily agents (ceftriaxone, ertapenem, dalbavancin) are preferred. "
        "Provide syndrome and organism details for a specific recommendation."
    )
    answer, narration_refined = narrate_opat_answer(
        question=message_text,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        patient_summary=patient_summary,
        fallback_message=fallback_message,
    )
    opat_dose_options = _opat_doseid_options(state)
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="iv_to_oral", label="Consider oral step-down instead"),
            *opat_dose_options,
            AssistantOption(value="duration", label="Confirm treatment duration"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "If a high-bioavailability oral agent covers the organism, oral step-down is preferred over OPAT.",
            "Once-daily dosing (ceftriaxone, ertapenem) makes home IV much more manageable for patients.",
        ],
    )


def _assistant_general_id_response(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse:
    """Return a grounded general ID answer for conceptual questions that don't map to a specific analysis workflow."""
    fallback_message = (
        "That's a general ID question. I can give the most precise answer when you provide patient-specific data — "
        "for example, case findings for syndrome probability, an isolate plus susceptibilities for resistance interpretation, "
        "or a drug plus renal context for dosing."
    )
    answer, narration_refined = narrate_general_id_answer(
        question=message_text,
        fallback_message=fallback_message,
    )
    return AssistantTurnResponse(
        assistantMessage=answer,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=[
            AssistantOption(value="probid", label="Syndrome workup"),
            AssistantOption(value="mechid", label="Resistance interpretation"),
            AssistantOption(value="doseid", label="Antimicrobial dosing"),
            AssistantOption(value="restart", label="Start new consult"),
        ],
        tips=[
            "If you have a specific patient, I can run a formal analysis with probability estimates and guideline-backed recommendations.",
            "Type the case details in plain language, or choose a module to start a structured workup.",
        ],
    )


def _assistant_route_by_intent(
    intent: str,
    req: AssistantTurnRequest,
    state: AssistantState,
    message_text: str,
) -> AssistantTurnResponse | None:
    """
    Route an already-classified intent to the appropriate response function.
    Shared by _assistant_llm_triage and _assistant_done_state_redirect so that
    the LLM classification step is never duplicated.
    Returns None for 'unclear' intent.
    """
    if intent == "probid":
        routed = _assistant_intake_case_from_text(req, state)
        if routed is not None:
            return routed
        return _assistant_begin_selected_workflow(
            state, PROBID_ASSISTANT_ID,
            lead_in="This looks like a syndrome workup question. ",
        )

    if intent == "mechid":
        routed = _assistant_intake_mechid_from_text(req, state)
        if routed is not None:
            return routed
        return _assistant_begin_selected_workflow(
            state, MECHID_ASSISTANT_ID,
            lead_in="This looks like a resistance or AST question. ",
        )

    if intent == "doseid":
        routed = _assistant_start_doseid_from_text(message_text, state)
        if routed is not None:
            return routed
        return _assistant_begin_selected_workflow(
            state, DOSEID_ASSISTANT_ID,
            lead_in="This looks like a dosing question. ",
        )

    if intent == "immunoid":
        routed = _assistant_start_immunoid_from_text(message_text, state)
        if routed is not None:
            return routed
        return _assistant_begin_selected_workflow(
            state, IMMUNOID_ASSISTANT_ID,
            lead_in="This looks like an immunosuppression or prophylaxis question. ",
        )

    if intent == "allergyid":
        routed = _assistant_start_allergyid_from_text(message_text, state)
        if routed is not None:
            return routed
        return _assistant_begin_selected_workflow(
            state, ALLERGYID_ASSISTANT_ID,
            lead_in="This looks like an antibiotic allergy or cross-reactivity question. ",
        )

    if intent == "empiric_therapy":
        return _assistant_empiric_therapy_response(message_text, state)
    if intent == "iv_to_oral":
        return _assistant_iv_to_oral_response(message_text, state)
    if intent == "duration":
        return _assistant_duration_response(message_text, state)
    if intent == "followup_tests":
        return _assistant_followup_tests_response(message_text, state)
    if intent == "stewardship":
        return _assistant_stewardship_response(message_text, state)
    if intent == "stewardship_review":
        return _assistant_stewardship_review_response(message_text, state)
    if intent == "opat":
        return _assistant_opat_response(message_text, state)
    if intent == "oral_therapy":
        return _assistant_oral_therapy_response(message_text, state)
    if intent == "discharge_counselling":
        return _assistant_discharge_counselling_response(message_text, state)
    if intent == "drug_interaction":
        return _assistant_drug_interaction_response(message_text, state)
    if intent == "prophylaxis":
        return _assistant_prophylaxis_response(message_text, state)
    if intent == "source_control":
        return _assistant_source_control_response(message_text, state)
    if intent == "treatment_failure":
        return _assistant_treatment_failure_response(message_text, state)
    if intent == "biomarker_interpretation":
        return _assistant_biomarker_response(message_text, state)
    if intent == "fluid_interpretation":
        return _assistant_fluid_interpretation_response(message_text, state)
    if intent == "allergy_delabeling":
        return _assistant_allergy_delabeling_response(message_text, state)
    if intent == "fungal_management":
        return _assistant_fungal_management_response(message_text, state)
    if intent == "sepsis_management":
        return _assistant_sepsis_response(message_text, state)
    if intent == "cns_infection":
        return _assistant_cns_infection_response(message_text, state)
    if intent == "mycobacterial":
        return _assistant_mycobacterial_response(message_text, state)
    if intent == "pregnancy_antibiotics":
        return _assistant_pregnancy_antibiotics_response(message_text, state)
    if intent == "travel_medicine":
        return _assistant_travel_medicine_response(message_text, state)
    if intent == "impression_plan":
        return _assistant_impression_plan_response(message_text, state)
    if intent == "duke_criteria":
        return _assistant_duke_criteria_response(message_text, state)
    if intent == "ast_meaning":
        return _assistant_ast_meaning_response(message_text, state)
    if intent == "complexity_flag":
        return _assistant_complexity_response(message_text, state)
    if intent == "course_tracker":
        return _assistant_course_tracker_response(message_text, state)
    if intent == "hiv_initial_art":
        return _assistant_hiv_initial_art_response(message_text, state)
    if intent == "hiv_monitoring":
        return _assistant_hiv_monitoring_response(message_text, state)
    if intent == "hiv_prep":
        return _assistant_hiv_prep_response(message_text, state)
    if intent == "hiv_pep":
        return _assistant_hiv_pep_response(message_text, state)
    if intent == "hiv_pregnancy":
        return _assistant_hiv_pregnancy_response(message_text, state)
    if intent == "hiv_oi_art_timing":
        return _assistant_hiv_oi_art_timing_response(message_text, state)
    if intent == "hiv_treatment_failure":
        return _assistant_hiv_treatment_failure_response(message_text, state)
    if intent == "hiv_resistance":
        return _assistant_hiv_resistance_response(message_text, state)
    if intent == "hiv_switch":
        return _assistant_hiv_switch_response(message_text, state)
    if intent == "general_id":
        return _assistant_general_id_response(message_text, state)
    if intent == "consult_summary":
        return _assistant_consult_summary_response(state)

    # intent == "unclear" — return None to fall through
    return None


def _assistant_llm_triage(
    req: AssistantTurnRequest,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    """
    LLM-backed last-resort triage called when all deterministic routing has failed.
    Context-aware: passes consult state so routing accounts for established syndrome/organisms.
    Also extracts any clinical data embedded in the message and applies it to state before routing,
    so downstream response functions receive the richest possible context.
    Returns None if the LLM is unavailable or the call fails, to allow fall-through.
    """
    message_text = (req.message or "").strip()
    if not message_text:
        return None

    triage_result = _assistant_llm_triage_intent(message_text, state)
    if triage_result is None:
        return None

    _apply_triage_extracted_context(state, triage_result.get("extracted") or {})
    intent = triage_result.get("intent", "unclear")
    return _assistant_route_by_intent(intent, req, state, message_text)


_ALLERGY_MENTION_PATTERNS = [
    r"\ballerg(?:ic|y)\s+to\s+[\w\s\-]+",
    r"[\w\s\-]+\s+allerg(?:y|ic)",
    r"\bnkda\b",
    r"\bno\s+known\s+(?:drug\s+)?allerg",
    r"\bpcn\s+allerg",
    r"\bpenicillin\s+allerg",
]


def _extract_allergy_mention(text: str) -> str | None:
    """Extract a short allergy mention from free text, if present."""
    import re as _re
    normalized = text.lower()
    for pattern in _ALLERGY_MENTION_PATTERNS:
        match = _re.search(pattern, normalized)
        if match:
            span = text[match.start():match.end()].strip()
            return span[:120]  # cap length
    return None


def _build_polymicrobial_analyses(
    original_text: str,
    primary_result: "MechIDTextAnalyzeResponse",
    *,
    parser_strategy: str = "auto",
    parser_model: str | None = None,
    allow_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """
    When multiple organisms are present but no single organism was assigned as primary,
    attempt to split the text by organism and run individual MechID analyses per organism.
    Returns a list of dicts: [{"organism": str, "analysis": dict | None, "provisionalAdvice": dict | None}]
    """
    parsed = primary_result.parsed_request
    if parsed is None:
        return []
    organisms = parsed.mentioned_organisms or []
    # Only activate when the parser found multiple organisms but no primary
    if not organisms or parsed.organism is not None:
        return []
    if len(organisms) < 2:
        return []

    per_organism_results: List[Dict[str, Any]] = []
    text_lower = original_text.lower()

    def _org_search_terms(org: str) -> List[str]:
        """Build a ranked list of search terms to locate this organism in text."""
        org_lower = org.lower()
        terms = [org_lower]
        parts = org_lower.split()
        if len(parts) >= 2:
            # "enterococcus faecalis" → "e. faecalis", "e faecalis", "faecalis"
            terms.append(f"{parts[0][0]}. {parts[-1]}")
            terms.append(f"{parts[0][0]} {parts[-1]}")
            terms.append(parts[-1])
            terms.append(parts[0][:4])  # genus prefix
        # Common abbreviations for known organisms
        abbrev_map = {
            "staphylococcus aureus": ["s. aureus", "s aureus", "staph aureus", "mssa", "mrsa"],
            "staphylococcus epidermidis": ["s. epidermidis", "mrse", "mr-cons", "mr-cons", "cons"],
            "enterococcus faecalis": ["e. faecalis", "e faecalis", "vsa", "vre"],
            "enterococcus faecium": ["e. faecium", "e faecium", "vre"],
            "streptococcus pneumoniae": ["s. pneumoniae", "pneumococcus", "pneumo"],
            "pseudomonas aeruginosa": ["p. aeruginosa", "p aeruginosa", "pseudomonas"],
            "klebsiella pneumoniae": ["k. pneumoniae", "k pneumoniae", "klebsiella"],
            "escherichia coli": ["e. coli", "e coli"],
            "candida albicans": ["c. albicans", "c albicans"],
        }
        for canonical, abbrevs in abbrev_map.items():
            if org_lower == canonical or org_lower in abbrevs:
                terms.extend(abbrevs)
                break
        return list(dict.fromkeys(terms))  # deduplicate, preserve order

    for org in organisms[:4]:  # cap at 4 organisms
        search_terms = _org_search_terms(org)

        org_pos = -1
        matched_term_len = len(org)
        for term in search_terms:
            pos = text_lower.find(term)
            if pos != -1:
                org_pos = pos
                matched_term_len = len(term)
                break

        if org_pos == -1:
            # Organism not clearly found in text; run on full text with organism hint
            try:
                org_result = _build_mechid_text_response(
                    f"{org}: {original_text}",
                    parser_strategy=parser_strategy,
                    parser_model=parser_model,
                    allow_fallback=allow_fallback,
                )
                per_organism_results.append({
                    "organism": org,
                    "analysis": org_result.analysis.model_dump(by_alias=True) if org_result.analysis else None,
                    "provisionalAdvice": org_result.provisional_advice.model_dump(by_alias=True) if org_result.provisional_advice else None,
                    "note": "Organism name not clearly located; ran on full text with organism hint.",
                })
            except Exception:
                per_organism_results.append({
                    "organism": org,
                    "analysis": None,
                    "provisionalAdvice": None,
                    "note": "Organism mentioned but not clearly associated with distinct AST data.",
                })
            continue

        # Extract the text segment from this organism mention to the next organism mention
        segment_end = len(original_text)
        for other_org in organisms:
            if other_org == org:
                continue
            for term in _org_search_terms(other_org):
                other_pos = text_lower.find(term, org_pos + matched_term_len)
                if other_pos != -1 and other_pos < segment_end:
                    segment_end = other_pos
                    break

        segment = original_text[org_pos:segment_end].strip()
        # Minimum useful segment: at least 20 chars; otherwise use full text with organism prefix
        if len(segment) < 20:
            segment = f"{org}: {original_text}"

        try:
            org_result = _build_mechid_text_response(
                segment,
                parser_strategy=parser_strategy,
                parser_model=parser_model,
                allow_fallback=allow_fallback,
            )
            per_organism_results.append({
                "organism": org,
                "analysis": org_result.analysis.model_dump(by_alias=True) if org_result.analysis else None,
                "provisionalAdvice": org_result.provisional_advice.model_dump(by_alias=True) if org_result.provisional_advice else None,
            })
        except Exception:
            per_organism_results.append({
                "organism": org,
                "analysis": None,
                "provisionalAdvice": None,
            })

    return per_organism_results


# Syndromes where microbiology is central — proactively suggest pasting culture results
_MICRO_CENTRAL_SYNDROMES: frozenset[str] = frozenset({
    "infective endocarditis",
    "bacteremia",
    "septic arthritis",
    "prosthetic joint infection (pji)",
    "bacterial meningitis",
    "brain abscess",
    "spinal epidural abscess",
    "community-acquired pneumonia (cap)",
    "ventilator-associated pneumonia (vap)",
    "urinary tract infection (uti)",
    "diabetic foot infection / osteomyelitis",
    "necrotizing soft tissue infection",
    "invasive candidiasis",
})


_HIGH_STAKES_DOSE_SYNDROMES: frozenset[str] = frozenset({
    "infective endocarditis",
    "endocarditis",
    "osteomyelitis",
    "septic arthritis",
    "prosthetic joint infection",
    "pji",
    "bacterial meningitis",
    "meningitis",
    "brain abscess",
    "spinal epidural abscess",
    "bacteremia",
    "invasive candidiasis",
    "necrotizing soft tissue infection",
})


def _mechid_doseid_bridge_option(state: AssistantState) -> AssistantOption | None:
    """Return a 'Calculate dose' option after MechID completes for high-stakes syndromes where dosing precision matters."""
    if state.doseid_text:
        return None  # dosing already underway — don't show again
    syndrome = (state.established_syndrome or "").lower()
    organisms = state.consult_organisms
    if not organisms:
        return None  # no organism yet — no drug to dose
    is_high_stakes = syndrome and any(s in syndrome for s in _HIGH_STAKES_DOSE_SYNDROMES)
    if not is_high_stakes:
        return None
    return AssistantOption(
        value="doseid",
        label="Calculate renal-adjusted dose",
        description="I'll factor in renal function for the recommended agent.",
    )


def _probid_micro_bridge_option(state: AssistantState) -> AssistantOption | None:
    """Return a 'Paste culture results' option when syndrome is known and no organisms have been identified yet."""
    syndrome = (state.established_syndrome or "").lower()
    if not syndrome:
        return None
    if state.consult_organisms:
        return None  # already have organisms — don't show this nudge
    matches = any(s in syndrome for s in _MICRO_CENTRAL_SYNDROMES)
    if not matches:
        return None
    return AssistantOption(
        value="mechid",
        label="Paste culture results",
        description="I'll interpret the organism and susceptibility pattern in the context of this syndrome.",
    )


def _consult_prior_context_summary(state: AssistantState) -> str | None:
    """
    Build a compact one-line summary of what is already known in this consult,
    suitable for passing to narrators so they can reference prior context naturally.
    e.g. "65yo male, 80kg, SCr 1.4 · Syndrome: Infective endocarditis · Organisms: MSSA, E. faecalis"
    """
    parts: List[str] = []
    pc = state.patient_context
    if pc:
        demo: List[str] = []
        if pc.age_years is not None:
            demo.append(f"{pc.age_years}yo")
        if pc.sex:
            demo.append(pc.sex)
        if pc.total_body_weight_kg is not None:
            demo.append(f"{pc.total_body_weight_kg}kg")
        if pc.serum_creatinine_mg_dl is not None:
            demo.append(f"SCr {pc.serum_creatinine_mg_dl}")
        if pc.renal_mode != "standard":
            demo.append(pc.renal_mode.upper())
        if demo:
            parts.append(", ".join(demo))
    if state.established_syndrome:
        parts.append(f"Syndrome: {state.established_syndrome}")
    if state.consult_organisms:
        parts.append(f"Organisms: {', '.join(state.consult_organisms)}")
    # HIV context summary
    hiv_ctx = state.hiv_context or {}
    if hiv_ctx:
        hiv_parts: list[str] = ["HIV+"]
        if hiv_ctx.get("cd4") is not None:
            hiv_parts.append(f"CD4 {hiv_ctx['cd4']}")
        if hiv_ctx.get("viral_load") is not None:
            hiv_parts.append(f"VL {hiv_ctx['viral_load']}")
        if hiv_ctx.get("on_art"):
            regimen = hiv_ctx.get("current_regimen", "unknown")
            hiv_parts.append(f"on {regimen}")
        if hiv_ctx.get("hbv_coinfected"):
            hiv_parts.append("HBV+")
        parts.append(" ".join(hiv_parts))
    return " · ".join(parts) if parts else None


def _syndrome_next_step_options(state: AssistantState, *, exclude: set[str] | None = None) -> list[AssistantOption]:
    """
    Return 1-2 contextually appropriate next-step chips based on established syndrome and consult context.
    Keeps the conversation clinically focused rather than showing generic module names.
    """
    exclude = exclude or set()
    options: list[AssistantOption] = []
    syndrome = (state.established_syndrome or "").lower()
    orgs_lower = " ".join(state.consult_organisms or []).lower()

    # Endocarditis / SAB / candidaemia → TEE + clearance cultures are the immediate priority
    if any(s in syndrome for s in ("endocarditis", "bacteraemia", "bacteremia", "candidaemia", "candidemia")):
        if "followup_tests" not in exclude:
            options.append(AssistantOption(value="followup_tests", label="Order TEE / clearance cultures"))

    # S. aureus anywhere → Duke criteria / TEE if not already there
    if ("aureus" in orgs_lower or "mssa" in orgs_lower or "mrsa" in orgs_lower) and "bacteraemia" in syndrome:
        if "duke_criteria" not in exclude and not any(o.value == "followup_tests" for o in options):
            options.append(AssistantOption(value="duke_criteria", label="Apply Duke criteria"))

    # Bone / joint / osteomyelitis → IV-to-oral (OVIVA) is the key next question
    if any(s in syndrome for s in ("osteomyelitis", "septic arthritis", "bone", "joint infection", "prosthetic joint", "pjı", "pji")):
        if "iv_to_oral" not in exclude:
            options.append(AssistantOption(value="iv_to_oral", label="IV-to-oral step-down (OVIVA)"))

    # Bacteraemia / sepsis / candidaemia → source control
    if any(s in syndrome for s in ("bacteraemia", "bacteremia", "sepsis", "septic shock", "candidaemia", "candidemia")):
        if "source_control" not in exclude and not any(o.value == "source_control" for o in options):
            options.append(AssistantOption(value="source_control", label="Source control decision"))

    # CNS infections → drug interaction check (dex timing, acyclovir)
    if any(s in syndrome for s in ("meningitis", "encephalitis", "brain abscess")):
        if "drug_interaction" not in exclude:
            options.append(AssistantOption(value="drug_interaction", label="Check drug interactions"))

    return options[:2]


def _is_mid_consult(state: AssistantState) -> bool:
    """True when enough context is established that 'restart' is less useful than 'summary'."""
    return bool(state.established_syndrome or state.consult_organisms or state.last_probid_summary or state.last_mechid_summary)


def _snapshot_probid_result(state: AssistantState, text_result: "TextAnalyzeResponse", module: "SyndromeModule | None") -> None:
    """Save a compact ProbID result snapshot to state for use in consult summary."""
    if text_result.analysis is None:
        return
    snap: Dict[str, Any] = {}
    if module:
        snap["syndrome"] = _assistant_module_label(module)
    analysis = text_result.analysis
    if hasattr(analysis, "posttest_probability") and analysis.posttest_probability is not None:
        snap["probability"] = round(analysis.posttest_probability, 2)
    if hasattr(analysis, "recommendation") and analysis.recommendation:
        snap["recommendation"] = analysis.recommendation
    if hasattr(analysis, "treatment_duration_guidance") and analysis.treatment_duration_guidance:
        snap["treatmentDuration"] = analysis.treatment_duration_guidance
    if hasattr(analysis, "monitoring_recommendations") and analysis.monitoring_recommendations:
        snap["monitoring"] = analysis.monitoring_recommendations
    state.last_probid_summary = snap


def _snapshot_mechid_result(state: AssistantState, result: "MechIDTextAnalyzeResponse") -> None:
    """Save a compact MechID result snapshot to state for use in consult summary."""
    parsed = result.parsed_request
    if parsed is None:
        return
    snap: Dict[str, Any] = {}
    if parsed.organism:
        snap["organism"] = parsed.organism
    if parsed.resistance_phenotypes:
        snap["resistancePhenotypes"] = parsed.resistance_phenotypes
    if parsed.susceptibility_results:
        snap["susceptibilityResults"] = {k: v for k, v in list(parsed.susceptibility_results.items())[:6]}
    tx = parsed.tx_context
    if tx.syndrome and tx.syndrome != "Not specified":
        snap["syndrome"] = tx.syndrome
    if tx.carbapenemase_result and tx.carbapenemase_result != "Not specified":
        snap["carbapenemaseResult"] = tx.carbapenemase_result
    if tx.carbapenemase_class and tx.carbapenemase_class != "Not specified":
        snap["carbapenemaseClass"] = tx.carbapenemase_class
    state.last_mechid_summary = snap


def _snapshot_doseid_result(state: AssistantState, dose_result: Dict[str, Any]) -> None:
    """Save a compact DoseID result snapshot to state for use in consult summary."""
    if not dose_result:
        return
    snap: Dict[str, Any] = {}
    for key in ("drug", "dose", "interval", "route", "renalAdjustment", "monitoringNotes"):
        if dose_result.get(key):
            snap[key] = dose_result[key]
    state.last_doseid_summary = snap if snap else None


def _snapshot_allergy_result(state: AssistantState, allergy_result: "AntibioticAllergyAnalyzeResponse") -> None:
    """Save a compact AllergyID result snapshot to state for use in consult summary."""
    snap: Dict[str, Any] = {}
    if hasattr(allergy_result, "candidate_drug") and allergy_result.candidate_drug:
        snap["candidateDrug"] = allergy_result.candidate_drug
    if hasattr(allergy_result, "verdict") and allergy_result.verdict:
        snap["verdict"] = allergy_result.verdict
    if hasattr(allergy_result, "allergen") and allergy_result.allergen:
        snap["allergen"] = allergy_result.allergen
    if hasattr(allergy_result, "reaction_phenotype") and allergy_result.reaction_phenotype:
        snap["reactionPhenotype"] = allergy_result.reaction_phenotype
    state.last_allergy_summary = snap if snap else None


def _accumulate_consult_organisms(state: AssistantState, result: "MechIDTextAnalyzeResponse") -> None:
    """Accumulate organisms identified from a MechID result into the session's consult_organisms list."""
    parsed = result.parsed_request
    if parsed is None:
        return
    candidates: List[str] = []
    if parsed.organism:
        candidates.append(parsed.organism)
    for org in (parsed.mentioned_organisms or []):
        if org and org not in candidates:
            candidates.append(org)
    for org in candidates:
        if org and org not in state.consult_organisms:
            state.consult_organisms.append(org)


def _update_session_patient_context(state: AssistantState, text: str) -> None:
    """
    Extract patient demographics from any incoming message and merge into state.patient_context.
    New values only fill gaps — already-set values are never overwritten.
    """
    if not text.strip():
        return
    extracted = _assistant_parse_doseid_patient_context(text)
    if state.patient_context is None:
        state.patient_context = SessionPatientContext()
    pc = state.patient_context
    if pc.age_years is None and extracted.age_years is not None:
        pc.age_years = extracted.age_years
    if pc.sex is None and extracted.sex is not None:
        pc.sex = extracted.sex
    if pc.total_body_weight_kg is None and extracted.total_body_weight_kg is not None:
        pc.total_body_weight_kg = extracted.total_body_weight_kg
    if pc.height_cm is None and extracted.height_cm is not None:
        pc.height_cm = extracted.height_cm
    if pc.serum_creatinine_mg_dl is None and extracted.serum_creatinine_mg_dl is not None:
        pc.serum_creatinine_mg_dl = extracted.serum_creatinine_mg_dl
    if pc.crcl_ml_min is None and extracted.crcl_ml_min is not None:
        pc.crcl_ml_min = extracted.crcl_ml_min
    if extracted.renal_mode != "standard":
        pc.renal_mode = extracted.renal_mode
    if not pc.allergy_text:
        mention = _extract_allergy_mention(text)
        if mention:
            pc.allergy_text = mention


def _session_context_to_doseid_patient_context(
    pc: SessionPatientContext,
) -> DoseIDAssistantPatientContext:
    """Convert a SessionPatientContext into a DoseIDAssistantPatientContext for use as a fallback."""
    return DoseIDAssistantPatientContext(
        ageYears=pc.age_years,
        sex=pc.sex,
        totalBodyWeightKg=pc.total_body_weight_kg,
        heightCm=pc.height_cm,
        serumCreatinineMgDl=pc.serum_creatinine_mg_dl,
        crclMlMin=pc.crcl_ml_min,
        renalMode=pc.renal_mode,
    )


def _session_patient_context_as_doseid_text(pc: SessionPatientContext) -> str:
    """Serialize a SessionPatientContext to a compact text string for DoseID analysis seeding."""
    parts: List[str] = []
    if pc.age_years is not None:
        parts.append(f"{pc.age_years}yo")
    if pc.sex is not None:
        parts.append(pc.sex)
    if pc.total_body_weight_kg is not None:
        parts.append(f"{pc.total_body_weight_kg}kg")
    if pc.height_cm is not None:
        parts.append(f"{pc.height_cm}cm")
    if pc.serum_creatinine_mg_dl is not None:
        parts.append(f"SCr {pc.serum_creatinine_mg_dl}")
    if pc.crcl_ml_min is not None:
        parts.append(f"CrCl {pc.crcl_ml_min}")
    if pc.renal_mode == "ihd":
        parts.append("on hemodialysis")
    elif pc.renal_mode == "crrt":
        parts.append("on CRRT")
    return ", ".join(parts)


def _assistant_handle_consult_intent(
    req: AssistantTurnRequest,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    message_text = (req.message or "").strip()
    intent = _assistant_detect_consult_intent(message_text)
    if intent not in {"treatment_decision", "therapy_selection"}:
        return None
    mechid_intent = _assistant_mechid_intent_profile(message_text)
    if (
        _assistant_is_doseid_intent(message_text)
        or _assistant_is_immunoid_intent(message_text)
        or _assistant_is_allergyid_intent(message_text)
    ):
        return None
    if (
        mechid_intent["has_ast"]
        or mechid_intent["has_resistance_signal"]
        or mechid_intent["has_isolate"]
        or mechid_intent["has_explicit_mechid_words"]
    ):
        return None

    module_hint = _assistant_consult_treatment_module_hint(message_text)
    normalized = _normalize_choice(message_text)
    if module_hint is None and any(token in normalized for token in CONSULT_INTENT_FUNGAL_TOKENS):
        return _assistant_consult_fungal_clarification_response(state)

    if module_hint is not None:
        response = _assistant_intake_case_from_text(req, state, module_hint=module_hint)
        module = store.get(module_hint)
        if response is not None and module is not None:
            intro = (
                _assistant_consult_treatment_intro(module)
                if intent == "treatment_decision"
                else _assistant_consult_therapy_selection_intro(module)
            )
            response.assistant_message = intro + response.assistant_message
            response.tips = [
                (
                    "I’ll keep this framed as a treatment decision and ask for the single detail most likely to change management next."
                    if intent == "treatment_decision"
                    else "I’ll keep this framed as a therapy-selection consult and ask for the single detail most likely to change management next."
                ),
                *(response.tips or []),
            ][:3]
            return response
        if module is not None:
            return _assistant_begin_selected_syndrome_module(
                state,
                module_hint,
                lead_in=(
                    _assistant_consult_treatment_intro(module)
                    if intent == "treatment_decision"
                    else _assistant_consult_therapy_selection_intro(module)
                ),
            )

    response = _assistant_begin_selected_workflow(state, PROBID_ASSISTANT_ID)
    return _assistant_consult_generic_antimicrobial_clarification_response(
        state,
        message_text=message_text,
        intent=intent,
    )


def _assistant_explicit_non_syndrome_workflow_request(message_text: str) -> str | None:
    normalized = _normalize_choice(message_text)
    if not normalized:
        return None
    if not any(_assistant_text_has_phrase(normalized, token) for token in EXPLICIT_SYNDROME_REQUEST_TOKENS):
        return None

    for workflow_id in (ALLERGYID_ASSISTANT_ID, DOSEID_ASSISTANT_ID, IMMUNOID_ASSISTANT_ID, MECHID_ASSISTANT_ID):
        aliases = EXPLICIT_NON_SYNDROME_WORKFLOW_ALIASES.get(workflow_id, ())
        if any(_assistant_text_has_phrase(normalized, alias) for alias in aliases):
            return workflow_id
    return None


def _assistant_begin_selected_workflow(
    state: AssistantState,
    workflow_id: str,
    *,
    lead_in: str | None = None,
) -> AssistantTurnResponse:
    if workflow_id == PROBID_ASSISTANT_ID:
        state.workflow = "probid"
        state.stage = "select_syndrome_module"
        state.module_id = None
        state.preset_id = None
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.doseid_text = None
        state.allergyid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        _assistant_reset_immunoid_state(state)
        intro = lead_in or ""
        return AssistantTurnResponse(
            assistantMessage=(intro + "Which clinical syndrome would you like to assess?"),
            state=state,
            options=_assistant_syndrome_module_options(),
            tips=[
                "Choose the syndrome first, then I’ll ask for the setting and case details.",
                "You can also type the syndrome name in plain language.",
            ],
        )

    if workflow_id == MECHID_ASSISTANT_ID:
        _assistant_reset_immunoid_state(state)
        state.workflow = "mechid"
        state.stage = "mechid_describe"
        state.module_id = None
        state.preset_id = None
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.doseid_text = None
        state.allergyid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        intro = lead_in or ""
        return AssistantTurnResponse(
            assistantMessage=(
                intro
                + "Paste the organism and susceptibility pattern in plain language. "
                + "For example: 'E. coli resistant to ceftriaxone and ciprofloxacin, susceptible to meropenem, bloodstream infection in septic shock.'"
            ),
            state=state,
            options=[AssistantOption(value="restart", label="Start new consult")],
            tips=[
                "I can extract the organism, AST pattern, and basic treatment context from free text.",
                "Ask for likely resistance mechanism, therapy, or both.",
            ],
        )

    if workflow_id == DOSEID_ASSISTANT_ID:
        _assistant_reset_immunoid_state(state)
        state.workflow = "doseid"
        state.stage = "doseid_describe"
        state.module_id = None
        state.preset_id = None
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.doseid_text = None
        state.allergyid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        intro = lead_in or ""
        return AssistantTurnResponse(
            assistantMessage=(
                intro
                + "Tell me the antimicrobial or regimen plus the renal context. "
                + "For example: 'cefepime dosing on hemodialysis' or 'RIPE dosing for 62 kg, CrCl 35, female, 165 cm'."
            ),
            state=state,
            options=[AssistantOption(value="restart", label="Start new consult")],
            tips=[
                "I can handle common antibacterial, TB, antifungal, and antiviral regimens.",
                "The most useful details are weight, serum creatinine or CrCl, dialysis status, age, sex, and height.",
            ],
        )

    if workflow_id == IMMUNOID_ASSISTANT_ID:
        state.workflow = "immunoid"
        state.stage = "immunoid_select_agents"
        state.module_id = None
        state.preset_id = None
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.doseid_text = None
        state.allergyid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        _assistant_reset_immunoid_state(state)
        intro = lead_in or ""
        return AssistantTurnResponse(
            assistantMessage=(
                intro
                + "Tell me which chemotherapy, steroids, biologics, or transplant agents are planned. "
                + "You can type them in plain language, for example: 'rituximab and prednisone 20 mg daily', "
                + "or click a few common agents to get started."
            ),
            state=state,
            options=_assistant_immunoid_agent_options(state),
            tips=[
                "I will build a deterministic screening and prophylaxis checklist from the selected exposures.",
                "You can keep adding agents before continuing.",
            ],
        )

    if workflow_id == ALLERGYID_ASSISTANT_ID:
        state.workflow = "allergyid"
        state.stage = "done"
        state.module_id = None
        state.preset_id = None
        state.case_section = None
        state.case_text = None
        state.mechid_text = None
        state.doseid_text = None
        state.allergyid_text = None
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []
        state.endo_blood_culture_context = None
        state.endo_score_factor_ids = []
        _assistant_reset_immunoid_state(state)
        intro = lead_in or ""
        return AssistantTurnResponse(
            assistantMessage=(
                intro
                + "Describe the allergy history and the antibiotics you are considering. "
                + "For example: 'amoxicillin anaphylaxis, can I use cefazolin for MSSA bacteremia?'"
            ),
            state=state,
            options=[AssistantOption(value="restart", label="Start new consult")],
            tips=[
                "The most useful details are the culprit antibiotic, what happened, and the candidate drug you want to use.",
                "If the reaction was severe, say so explicitly, for example SJS/TEN, DRESS, organ injury, or hemolysis.",
            ],
        )

    return _assistant_begin_selected_syndrome_module(state, workflow_id, lead_in=lead_in)


def _assistant_begin_selected_syndrome_module(
    state: AssistantState,
    module_id: str,
    *,
    lead_in: str | None = None,
) -> AssistantTurnResponse:
    state.workflow = "probid"
    _assistant_reset_immunoid_state(state)
    state.module_id = module_id
    state.preset_id = None
    state.pending_intake_text = None
    state.pending_followup_workflow = None
    state.pending_followup_text = None
    state.case_section = None
    state.case_text = None
    state.mechid_text = None
    state.doseid_text = None
    state.allergyid_text = None
    state.pretest_factor_ids = []
    state.pretest_factor_labels = []
    state.endo_blood_culture_context = None
    state.endo_score_factor_ids = []

    module = store.get(module_id)
    if module is None:
        raise HTTPException(status_code=400, detail=f"Selected module '{module_id}' not found")

    state.stage = "select_preset"
    intro = lead_in or f"Great, we’ll work on {_assistant_module_label(module)}. "
    return AssistantTurnResponse(
        assistantMessage=(intro + "Which setting/pretest context fits this case best?"),
        state=state,
        options=_assistant_preset_options(module),
        tips=[
            "You can click an option or type something like 'ED', 'ICU', or the preset name.",
            "Type 'restart' anytime to begin a new consult.",
        ],
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
    _assistant_sanitize_endo_imaging_question_parse(text_result, message_text)

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
    inferred_pretest_factor_ids = _assistant_infer_pretest_factor_ids_from_text(module, message_text, state)
    _assistant_reset_immunoid_state(state)
    explicit_preset_supported = bool(preset_hint) or _assistant_text_explicitly_supports_preset(message_text, module)
    implicit_preset_note: str | None = None
    allow_implicit_preset = False
    if module.id == "endo" and _assistant_is_endo_imaging_question(message_text):
        if not text_result.parsed_request.preset_id:
            text_result.parsed_request.preset_id = "endo_low"
        allow_implicit_preset = True
        implicit_preset_note = (
            "I defaulted the baseline setting to ED / inpatient for now because this reads like an active endocarditis imaging question. "
            "If this is really outpatient or tertiary-referral context, tell me and I’ll update the probability."
        )

    if module.pretest_presets and (
        not text_result.parsed_request.preset_id
        or (not explicit_preset_supported and not allow_implicit_preset)
    ):
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
        _assistant_merge_pretest_factor_ids(state, inferred_pretest_factor_ids)
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
    _assistant_merge_pretest_factor_ids(state, inferred_pretest_factor_ids)
    _sync_pretest_factor_labels(state, module)
    _apply_pretest_factors_to_parsed_request(module=module, state=state, parsed_request=text_result.parsed_request)
    _sync_text_result_references(
        text_result=text_result,
        module=module,
        selected_pretest_factor_ids=state.pretest_factor_ids,
    )
    _assistant_cache_probid_case_result(state, text_result)
    state.stage = "confirm_case"
    review_message, narration_refined = _assistant_probid_review_message(
        module,
        text_result,
        state,
        prefix=(
            "I parsed your case description and pre-populated the calculator inputs. "
            + (implicit_preset_note + " " if implicit_preset_note else "")
        ),
    )
    return AssistantTurnResponse(
        assistantMessage=review_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=_assistant_review_options_for_case(module, text_result, state),
        analysis=text_result,
        tips=[
            *(
                ["If the setting is not the usual inpatient baseline, tell me and I’ll update the estimate."]
                if implicit_preset_note
                else []
            ),
            "Reply with the single follow-up detail I asked for, in normal words, and I will keep the case moving.",
            "If the extraction already looks right, select the clinical syndrome to get the therapy recommendation.",
        ],
    )


def _assistant_append_case_item_from_selection(
    module: SyndromeModule,
    state: AssistantState,
    selection: str,
) -> tuple[str | None, str | None]:
    if not selection.startswith("insert_text:"):
        return None, None
    item_id = selection.split(":", 1)[1]
    item = _assistant_case_item_by_id(module, item_id)
    if item is None or not _assistant_case_item_allowed(module, item, state):
        return None, None
    present_text, _ = _assistant_case_item_text(item, module)
    existing_lines = {line.strip().lower() for line in (state.case_text or "").splitlines() if line.strip()}
    if present_text.strip().lower() not in existing_lines:
        state.case_text = _append_case_text(state.case_text, present_text)
    return item_id, item.label


def _assistant_is_mechid_intent(message: str | None) -> bool:
    return _assistant_mechid_intent_profile(message)["strong_mechid_trigger"]


def _assistant_mechid_intent_profile(message: str | None) -> Dict[str, bool]:
    text = _normalize_choice(message)
    profile = {
        "has_isolate": False,
        "has_ast": False,
        "has_resistance_signal": False,
        "has_explicit_mechid_words": False,
        "has_explicit_therapy_words": False,
        "has_treatment_question": False,
        "has_isolate_context_words": False,
        "strong_mechid_trigger": False,
        "ambiguous_isolate_only": False,
    }
    if not text:
        return profile

    try:
        parsed = parse_mechid_text(message or "")
    except MechIDEngineError:
        parsed = None

    tx_context = parsed.get("txContext", {}) if isinstance(parsed, dict) else {}
    has_isolate = bool(
        parsed
        and (
            parsed.get("organism")
            or parsed.get("mentionedOrganisms")
        )
    )
    has_ast = bool(parsed and parsed.get("susceptibilityResults"))
    has_resistance_signal = bool(
        parsed
        and (
            parsed.get("resistancePhenotypes")
            or tx_context.get("carbapenemaseResult") not in {None, "", "Not specified"}
            or tx_context.get("carbapenemaseClass") not in {None, "", "Not specified"}
        )
    )
    has_explicit_mechid_words = any(_assistant_text_has_phrase(text, token) for token in MECHID_INTENT_TOKENS)
    has_isolate_context_words = any(token in text for token in ("isolate", "culture", "cultures", "organism", "bug"))
    has_explicit_therapy_words = any(_assistant_text_has_phrase(text, token) for token in MECHID_THERAPY_INTENT_TOKENS) and (
        has_isolate
        or has_ast
        or has_resistance_signal
        or has_isolate_context_words
        or any(token in text for token in ("susceptibility", "susceptible", "resistant", "culture", "cultures", "organism", "isolate", "ast"))
    )
    has_treatment_question = (
        ("treat" in text or "therapy" in text or "antibiotic" in text or "manage" in text or "cover" in text)
        and ("how" in text or "what" in text or "which" in text or "recommend" in text or "choice" in text)
    )
    strong_mechid_trigger = (
        has_ast
        or has_resistance_signal
        or has_explicit_mechid_words
        or has_explicit_therapy_words
        or (has_isolate and has_treatment_question)
        or (has_isolate_context_words and has_treatment_question)
    )

    profile.update(
        {
            "has_isolate": has_isolate,
            "has_ast": has_ast,
            "has_resistance_signal": has_resistance_signal,
            "has_explicit_mechid_words": has_explicit_mechid_words,
            "has_explicit_therapy_words": has_explicit_therapy_words,
            "has_treatment_question": has_treatment_question,
            "has_isolate_context_words": has_isolate_context_words,
            "strong_mechid_trigger": strong_mechid_trigger,
            "ambiguous_isolate_only": has_isolate and not strong_mechid_trigger,
        }
    )
    return profile


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
    mechid_result = _assistant_effective_mechid_result(
        mechid_result,
        established_syndrome=state.established_syndrome,
    )

    _assistant_reset_immunoid_state(state)
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
    _accumulate_consult_organisms(state, mechid_result)
    _snapshot_mechid_result(state, mechid_result)
    review_message, narration_refined = _assistant_mechid_review_message(
        mechid_result,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        institutional_antibiogram=state.institutional_antibiogram or None,
    )
    return AssistantTurnResponse(
        assistantMessage=review_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=_assistant_mechid_review_options(mechid_result, established_syndrome=state.established_syndrome),
        mechidAnalysis=mechid_result,
        tips=[
            "Reply with the next susceptibility or context detail I asked for, in normal words, and I will update the case.",
            "If the extraction already looks right, select the clinical syndrome to get the therapy recommendation.",
        ],
    )


def _assistant_immunoid_normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


IMMUNOID_LOCATION_CONTEXT_MARKERS = (
    "born in",
    "from ",
    "lived in",
    "grew up in",
    "travel to",
    "traveled to",
    "travelled to",
    "resided in",
    "immigrated from",
    "visited",
    "visit to",
    "returned from",
)

IMMUNOID_TB_COUNTRY_ALIASES = (
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
    "sudan",
    "south sudan",
    "egypt",
    "yemen",
)

IMMUNOID_STRONGY_COUNTRY_ALIASES = (
    "mexico",
    "guatemala",
    "honduras",
    "el salvador",
    "nicaragua",
    "costa rica",
    "panama",
    "colombia",
    "venezuela",
    "ecuador",
    "peru",
    "bolivia",
    "brazil",
    "paraguay",
    "argentina",
    "chile",
    "uruguay",
    "haiti",
    "dominican republic",
    "jamaica",
    "cuba",
    "puerto rico",
    "egypt",
    "morocco",
    "algeria",
    "tunisia",
    "sudan",
    "south sudan",
    "ethiopia",
    "eritrea",
    "somalia",
    "kenya",
    "uganda",
    "tanzania",
    "nigeria",
    "ghana",
    "cameroon",
    "angola",
    "mozambique",
    "madagascar",
    "yemen",
    "saudi arabia",
    "iraq",
    "iran",
    "syria",
    "lebanon",
    "jordan",
    "afghanistan",
    "pakistan",
    "india",
    "bangladesh",
    "sri lanka",
    "nepal",
    "thailand",
    "laos",
    "cambodia",
    "myanmar",
    "indonesia",
    "philippines",
    "vietnam",
    "malaysia",
    "timor leste",
    "papua new guinea",
    "fiji",
)

def _assistant_immunoid_has_location_context(normalized: str) -> bool:
    return any(token in normalized for token in IMMUNOID_LOCATION_CONTEXT_MARKERS)


def _assistant_immunoid_has_country_match(normalized: str, aliases: tuple[str, ...]) -> bool:
    padded = f" {normalized} "
    for alias in aliases:
        if f" {alias} " not in padded:
            continue
        if alias == "mexico" and " new mexico " in padded:
            continue
        return True
    return False


def _assistant_immunoid_mentions_generic_steroid(message_text: str) -> bool:
    normalized = _assistant_immunoid_normalize(message_text)
    generic_tokens = (
        " steroid ",
        " steroids ",
        " corticosteroid ",
        " corticosteroids ",
        " glucocorticoid ",
        " glucocorticoids ",
    )
    padded = f" {normalized} "
    if any(token in padded for token in generic_tokens):
        return True
    # If a specific steroid was already recognized we do not need the generic fallback.
    if _assistant_detect_immunoid_agent_ids(message_text):
        return False
    return any(name in padded for name in (" prednisone ", " prednisolone ", " methylprednisolone ", " dexamethasone ", " hydrocortisone "))


def _assistant_reset_immunoid_state(state: AssistantState) -> None:
    state.immunoid_selected_regimen_ids = []
    state.immunoid_selected_agent_ids = []
    state.immunoid_planned_steroid_duration_days = None
    state.immunoid_anticipated_prolonged_profound_neutropenia = None
    state.immunoid_hbv_hbsag = "unknown"
    state.immunoid_hbv_anti_hbc = "unknown"
    state.immunoid_hbv_anti_hbs = "unknown"
    state.immunoid_tb_screen_result = "unknown"
    state.immunoid_tb_endemic_exposure = None
    state.immunoid_strongyloides_exposure = None
    state.immunoid_strongyloides_igg = "unknown"
    state.immunoid_coccidioides_exposure = None
    state.immunoid_histoplasma_exposure = None
    state.immunoid_signal_sources = {}


def _assistant_immunoid_agents_by_id() -> Dict[str, Dict[str, Any]]:
    return {entry["id"]: entry for entry in list_immunoid_agents()}


def _assistant_immunoid_regimens_by_id() -> Dict[str, Dict[str, Any]]:
    return {entry["id"]: entry for entry in list_immunoid_regimens()}


def _assistant_set_immunoid_signal_source(state: AssistantState, signal_id: str, source: str) -> None:
    state.immunoid_signal_sources[signal_id] = source


IMMUNOID_AGENT_FUZZY_MIN_RATIO = 0.9
IMMUNOID_AGENT_FUZZY_MAX_ALIAS_TOKENS = 4


def _assistant_detect_immunoid_regimen_ids(message_text: str) -> List[str]:
    normalized = f" {_assistant_immunoid_normalize(message_text)} "
    matches: List[tuple[int, int, str]] = []
    for regimen_id, entry in IMMUNOID_REGIMENS.items():
        aliases = {
            _assistant_immunoid_normalize(regimen_id.replace("_", " ")),
            _assistant_immunoid_normalize(entry["name"]),
            *(_assistant_immunoid_normalize(alias) for alias in entry.get("aliases", ())),
        }
        for alias in aliases:
            if not alias:
                continue
            needle = f" {alias} "
            index = normalized.find(needle)
            if index >= 0:
                matches.append((index, index + len(needle), regimen_id))
                break
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    ordered: List[str] = []
    seen: set[str] = set()
    occupied_spans: List[tuple[int, int]] = []
    for start, end, regimen_id in matches:
        if regimen_id in seen:
            continue
        if any(start >= existing_start and end <= existing_end for existing_start, existing_end in occupied_spans):
            continue
        seen.add(regimen_id)
        ordered.append(regimen_id)
        occupied_spans.append((start, end))
    return ordered


def _assistant_apply_immunoid_regimen_defaults(state: AssistantState, regimen_id: str) -> None:
    regimen = IMMUNOID_REGIMENS.get(regimen_id)
    if regimen is None:
        return
    defaults = regimen.get("defaults", {})
    steroid_days = defaults.get("planned_steroid_duration_days")
    if steroid_days is not None and state.immunoid_planned_steroid_duration_days is None:
        state.immunoid_planned_steroid_duration_days = steroid_days
        _assistant_set_immunoid_signal_source(state, "planned_steroid_duration_days", "regimen")
    anticipated_neutropenia = defaults.get("anticipated_prolonged_profound_neutropenia")
    if anticipated_neutropenia is not None and state.immunoid_anticipated_prolonged_profound_neutropenia is None:
        state.immunoid_anticipated_prolonged_profound_neutropenia = anticipated_neutropenia
        _assistant_set_immunoid_signal_source(state, "anticipated_prolonged_profound_neutropenia", "regimen")


def _assistant_detect_immunoid_agent_ids(message_text: str) -> List[str]:
    normalized = f" {_assistant_immunoid_normalize(message_text)} "
    matches: List[tuple[int, str]] = []
    for entry in list_immunoid_agents():
        aliases = {
            _assistant_immunoid_normalize(entry["name"]),
            _assistant_immunoid_normalize(entry["id"].replace("_", " ")),
            *(_assistant_immunoid_normalize(alias) for alias in entry.get("aliases", ())),
        }
        for alias in aliases:
            if not alias:
                continue
            needle = f" {alias} "
            index = normalized.find(needle)
            if index >= 0:
                matches.append((index, entry["id"]))
                break
    matches.sort(key=lambda item: item[0])
    ordered: List[str] = []
    seen: set[str] = set()
    for _, agent_id in matches:
        if agent_id in seen:
            continue
        seen.add(agent_id)
        ordered.append(agent_id)
    if ordered:
        return ordered

    tokens = _assistant_immunoid_normalize(message_text).split()
    if not tokens:
        return []
    fuzzy_matches: List[tuple[float, int, str]] = []
    for entry in list_immunoid_agents():
        aliases = {
            _assistant_immunoid_normalize(entry["name"]),
            _assistant_immunoid_normalize(entry["id"].replace("_", " ")),
            *(_assistant_immunoid_normalize(alias) for alias in entry.get("aliases", ())),
        }
        for alias in aliases:
            alias_tokens = alias.split()
            if not alias_tokens or len(alias_tokens) > IMMUNOID_AGENT_FUZZY_MAX_ALIAS_TOKENS:
                continue
            for start in range(0, len(tokens) - len(alias_tokens) + 1):
                phrase = " ".join(tokens[start : start + len(alias_tokens)])
                score = SequenceMatcher(None, phrase, alias).ratio()
                if score >= IMMUNOID_AGENT_FUZZY_MIN_RATIO:
                    fuzzy_matches.append((score, start, entry["id"]))
                    break
    fuzzy_matches.sort(key=lambda item: (-item[0], item[1]))
    for _, _, agent_id in fuzzy_matches:
        if agent_id in seen:
            continue
        seen.add(agent_id)
        ordered.append(agent_id)
    return ordered


def _assistant_is_immunoid_intent(message_text: str) -> bool:
    normalized = _normalize_choice(message_text)
    if any(token in normalized for token in IMMUNOID_INTENT_TOKENS):
        return True
    if _assistant_immunoid_mentions_generic_steroid(message_text):
        return True
    if _assistant_detect_immunoid_regimen_ids(message_text):
        return True
    detected_agents = _assistant_detect_immunoid_agent_ids(message_text)
    if detected_agents:
        return True
    keywords = ("prophyl", "screen", "immunosupp", "chemotherapy", "steroid", "biologic", "reactivation")
    return any(keyword in normalized for keyword in keywords)


def _assistant_is_allergyid_intent(message_text: str) -> bool:
    normalized = _normalize_choice(message_text)
    if not normalized:
        return False
    reaction_tokens = (
        "allergy",
        "allergic",
        "anaphylaxis",
        "hives",
        "urticaria",
        "angioedema",
        "rash",
        "reaction",
        "throat tightness",
        "sjs",
        "ten",
        "dress",
    )
    has_reaction_token = any(
        _assistant_text_has_phrase(normalized, token)
        for token in reaction_tokens
    )
    if any(
        _assistant_text_has_phrase(normalized, token)
        for token in (
            "bacteremia",
            "bloodstream infection",
            "endocarditis",
            "persistent bacteremia",
            "positive blood cultures",
            "osteomyelitis",
            "septic arthritis",
            "cystitis",
            "pneumonia",
        )
    ) and not has_reaction_token:
        return False
    if any(_assistant_text_has_phrase(normalized, token) for token in ALLERGYID_INTENT_TOKENS):
        return True
    medication_count = len(_assistant_detect_doseid_medication_ids(message_text))
    if medication_count >= 2 and has_reaction_token:
        return True
    return any(
        _assistant_text_has_phrase(normalized, phrase)
        for phrase in (
            "can i use",
            "can i still use",
            "can i use again",
            "can they use",
            "could i use",
            "could they use",
            "we are thinking about",
            "thinking about",
            "wondering about",
            "may need",
            "now needs",
            "what can i use",
            "what can they use",
            "best antibiotic with",
            "best antibiotic if allergic",
        )
    ) and (
        has_reaction_token
        or any(
            _assistant_text_has_phrase(normalized, token)
            for token in (
                "caused",
                "headache",
                "nausea",
                "vomiting",
                "diarrhea",
                "gi upset",
                "gastrointestinal",
                "unknown",
            )
        )
    )


def _assistant_parse_immunoid_serology_state(message_text: str, aliases: List[str]) -> str | None:
    normalized = _assistant_immunoid_normalize(message_text)
    for alias in aliases:
        alias_norm = _assistant_immunoid_normalize(alias)
        if not alias_norm:
            continue
        patterns = (
            rf"{re.escape(alias_norm)}\s+(?:is\s+)?(positive|negative|unknown|pos|neg)",
            rf"(positive|negative|unknown|pos|neg)\s+{re.escape(alias_norm)}",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            token = match.group(1)
            if token in {"positive", "pos"}:
                return "positive"
            if token in {"negative", "neg"}:
                return "negative"
            return "unknown"
    return None


def _assistant_parse_immunoid_tb_state(message_text: str) -> str | None:
    normalized = _assistant_immunoid_normalize(message_text)
    aliases = ("igra", "quantiferon", "quanti feron", "qft", "t spot", "tspot", "tst", "ppd", "tb screen")
    for alias in aliases:
        patterns = (
            rf"{re.escape(alias)}\s+(?:is\s+)?(positive|negative|indeterminate|pos|neg|indet)",
            rf"(positive|negative|indeterminate|pos|neg|indet)\s+{re.escape(alias)}",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            token = match.group(1)
            if token in {"positive", "pos"}:
                return "positive"
            if token in {"negative", "neg"}:
                return "negative"
            return "indeterminate"
    return None


def _assistant_parse_immunoid_duration_days(message_text: str) -> int | None:
    normalized = _assistant_immunoid_normalize(message_text)
    match = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", normalized)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("week"):
        return value * 7
    if unit.startswith("month"):
        return value * 30
    return value


def _assistant_parse_immunoid_yes_no_unknown(message_text: str) -> bool | None | str:
    normalized = _normalize_choice(message_text)
    if any(token in normalized for token in ("not sure", "unknown", "unclear", "unsure")):
        return "unknown"
    if any(token in normalized for token in ("no", "none", "not", "doesn't", "does not", "nope", "absent")):
        return False
    if any(token in normalized for token in ("yes", "yep", "present", "does", "has", "expected", "will")):
        return True
    return None


def _assistant_parse_immunoid_context_from_text(state: AssistantState, message_text: str) -> List[str]:
    updates: List[str] = []
    for regimen_id in _assistant_detect_immunoid_regimen_ids(message_text):
        if regimen_id not in state.immunoid_selected_regimen_ids:
            state.immunoid_selected_regimen_ids.append(regimen_id)
            updates.append(f"regimen:{regimen_id}")
        for agent_id in IMMUNOID_REGIMENS.get(regimen_id, {}).get("component_agent_ids", ()):
            if agent_id not in state.immunoid_selected_agent_ids:
                state.immunoid_selected_agent_ids.append(agent_id)
                updates.append(f"agent:{agent_id}")
        _assistant_apply_immunoid_regimen_defaults(state, regimen_id)

    for agent_id in _assistant_detect_immunoid_agent_ids(message_text):
        if agent_id not in state.immunoid_selected_agent_ids:
            state.immunoid_selected_agent_ids.append(agent_id)
            updates.append(f"agent:{agent_id}")

    hbsag = _assistant_parse_immunoid_serology_state(message_text, ["hbsag", "surface antigen"])
    if hbsag and hbsag != state.immunoid_hbv_hbsag:
        state.immunoid_hbv_hbsag = hbsag
        _assistant_set_immunoid_signal_source(state, "hbv_hbsag", "text")
        updates.append("hbv_hbsag")

    anti_hbc = _assistant_parse_immunoid_serology_state(message_text, ["anti hbc", "anti h b c", "core antibody", "hbcab"])
    if anti_hbc and anti_hbc != state.immunoid_hbv_anti_hbc:
        state.immunoid_hbv_anti_hbc = anti_hbc
        _assistant_set_immunoid_signal_source(state, "hbv_anti_hbc", "text")
        updates.append("hbv_anti_hbc")

    anti_hbs = _assistant_parse_immunoid_serology_state(message_text, ["anti hbs", "anti h b s", "surface antibody", "hbsab"])
    if anti_hbs and anti_hbs != state.immunoid_hbv_anti_hbs:
        state.immunoid_hbv_anti_hbs = anti_hbs
        _assistant_set_immunoid_signal_source(state, "hbv_anti_hbs", "text")
        updates.append("hbv_anti_hbs")

    tb_state = _assistant_parse_immunoid_tb_state(message_text)
    if tb_state and tb_state != state.immunoid_tb_screen_result:
        state.immunoid_tb_screen_result = tb_state
        _assistant_set_immunoid_signal_source(state, "tb_screen_result", "text")
        updates.append("tb_screen")

    strongy_igg = _assistant_parse_immunoid_serology_state(message_text, ["strongyloides igg", "strongy igg", "strongyloides serology"])
    if strongy_igg and strongy_igg != state.immunoid_strongyloides_igg:
        state.immunoid_strongyloides_igg = strongy_igg
        _assistant_set_immunoid_signal_source(state, "strongyloides_igg", "text")
        updates.append("strongyloides_igg")

    duration_days = _assistant_parse_immunoid_duration_days(message_text)
    if duration_days is not None and duration_days != state.immunoid_planned_steroid_duration_days:
        state.immunoid_planned_steroid_duration_days = duration_days
        _assistant_set_immunoid_signal_source(state, "planned_steroid_duration_days", "text")
        updates.append("steroid_duration")

    neutropenia_norm = _assistant_immunoid_normalize(message_text)
    if "neutropenia" in neutropenia_norm:
        yes_no = _assistant_parse_immunoid_yes_no_unknown(message_text)
        if yes_no in {True, False} and yes_no != state.immunoid_anticipated_prolonged_profound_neutropenia:
            state.immunoid_anticipated_prolonged_profound_neutropenia = yes_no
            _assistant_set_immunoid_signal_source(state, "anticipated_prolonged_profound_neutropenia", "text")
            updates.append("neutropenia")

    tb_endemic_regions = (
        "tb endemic",
        "tb high incidence",
        "asia",
        "africa",
        "latin america",
        "india",
        "philippines",
        "china",
        "vietnam",
        "peru",
        "brazil",
        "haiti",
        "sub saharan africa",
    )
    tb_exposure_markers = IMMUNOID_LOCATION_CONTEXT_MARKERS + (
        "incarceration",
        "jail",
        "prison",
        "homeless shelter",
        "nursing home",
        "tb contact",
        "close contact with tb",
    )
    tb_negative_markers = (
        "no tb endemic exposure",
        "no tb travel",
        "no tb risk factors",
        "no travel to tb endemic area",
        "no known tb exposure",
    )
    if any(token in neutropenia_norm for token in tb_negative_markers):
        if state.immunoid_tb_endemic_exposure is not False:
            state.immunoid_tb_endemic_exposure = False
            _assistant_set_immunoid_signal_source(state, "tb_endemic_exposure", "text")
            updates.append("tb_endemic_exposure")
    elif (
        any(token in neutropenia_norm for token in tb_endemic_regions)
        and any(token in neutropenia_norm for token in tb_exposure_markers)
    ) or (
        _assistant_immunoid_has_location_context(neutropenia_norm)
        and _assistant_immunoid_has_country_match(neutropenia_norm, IMMUNOID_TB_COUNTRY_ALIASES)
    ) or "tb endemic exposure" in neutropenia_norm:
        if state.immunoid_tb_endemic_exposure is not True:
            state.immunoid_tb_endemic_exposure = True
            _assistant_set_immunoid_signal_source(state, "tb_endemic_exposure", "text")
            updates.append("tb_endemic_exposure")

    cocci_locations = (
        "arizona",
        "california central valley",
        "new mexico",
        "west texas",
        "southern nevada",
        "utah",
        "washington state",
        "northern mexico",
    )
    cocci_negative_markers = ("no coccidioides exposure", "no cocci exposure", "no arizona exposure", "no arizona travel")
    if any(token in neutropenia_norm for token in cocci_negative_markers):
        if state.immunoid_coccidioides_exposure is not False:
            state.immunoid_coccidioides_exposure = False
            _assistant_set_immunoid_signal_source(state, "coccidioides_exposure", "text")
            updates.append("cocci_exposure")
    elif any(token in neutropenia_norm for token in cocci_locations):
        if state.immunoid_coccidioides_exposure is not True:
            state.immunoid_coccidioides_exposure = True
            _assistant_set_immunoid_signal_source(state, "coccidioides_exposure", "text")
            updates.append("cocci_exposure")

    strongy_regions = (
        "latin america",
        "caribbean",
        "sub saharan africa",
        "southeast asia",
        "oceania",
        "appalachia",
        "southeastern us",
        "peru",
        "brazil",
    )
    strongy_negative_markers = ("no strongyloides exposure", "no endemic exposure", "no mexico exposure")
    if any(token in neutropenia_norm for token in strongy_negative_markers):
        if state.immunoid_strongyloides_exposure is not False:
            state.immunoid_strongyloides_exposure = False
            _assistant_set_immunoid_signal_source(state, "strongyloides_exposure", "text")
            updates.append("strongyloides_exposure")
    elif any(token in neutropenia_norm for token in strongy_regions) or (
        _assistant_immunoid_has_location_context(neutropenia_norm)
        and _assistant_immunoid_has_country_match(neutropenia_norm, IMMUNOID_STRONGY_COUNTRY_ALIASES)
    ):
        if state.immunoid_strongyloides_exposure is not True:
            state.immunoid_strongyloides_exposure = True
            _assistant_set_immunoid_signal_source(state, "strongyloides_exposure", "text")
            updates.append("strongyloides_exposure")

    histo_regions = (
        "histoplasma",
        "histoplasmosis",
        "ohio river valley",
        "mississippi river valley",
        "central us",
        "eastern us",
        "central america",
        "south america",
        "missouri",
        "arkansas",
        "kentucky",
        "tennessee",
        "indiana",
        "ohio",
    )
    histo_exposure_markers = (
        "bat",
        "bird droppings",
        "bird guano",
        "cave",
        "spelunk",
        "chicken coop",
        "demolition",
        "excavation",
        "dusty soil",
        "soil exposure",
    )
    histo_negative_markers = (
        "no histoplasma exposure",
        "no histoplasmosis exposure",
        "no bat exposure",
        "no cave exposure",
        "no bird droppings exposure",
    )
    if any(token in neutropenia_norm for token in histo_negative_markers):
        if state.immunoid_histoplasma_exposure is not False:
            state.immunoid_histoplasma_exposure = False
            _assistant_set_immunoid_signal_source(state, "histoplasma_exposure", "text")
            updates.append("histoplasma_exposure")
    elif any(token in neutropenia_norm for token in histo_regions) or any(
        token in neutropenia_norm for token in histo_exposure_markers
    ):
        if state.immunoid_histoplasma_exposure is not True:
            state.immunoid_histoplasma_exposure = True
            _assistant_set_immunoid_signal_source(state, "histoplasma_exposure", "text")
            updates.append("histoplasma_exposure")
    return updates


def _assistant_immunoid_request_from_state(state: AssistantState) -> ImmunoAnalyzeRequest:
    return ImmunoAnalyzeRequest(
        selectedRegimenIds=state.immunoid_selected_regimen_ids,
        selectedAgentIds=state.immunoid_selected_agent_ids,
        plannedSteroidDurationDays=state.immunoid_planned_steroid_duration_days,
        anticipatedProlongedProfoundNeutropenia=state.immunoid_anticipated_prolonged_profound_neutropenia,
        hbvHbsAg=state.immunoid_hbv_hbsag,
        hbvAntiHbc=state.immunoid_hbv_anti_hbc,
        hbvAntiHbs=state.immunoid_hbv_anti_hbs,
        tbScreenResult=state.immunoid_tb_screen_result,
        tbEndemicExposure=state.immunoid_tb_endemic_exposure,
        strongyloidesExposure=state.immunoid_strongyloides_exposure,
        strongyloidesIgg=state.immunoid_strongyloides_igg,
        coccidioidesExposure=state.immunoid_coccidioides_exposure,
        histoplasmaExposure=state.immunoid_histoplasma_exposure,
    )


def _assistant_immunoid_analysis_from_state(state: AssistantState) -> ImmunoAnalyzeResponse:
    result = ImmunoAnalyzeResponse(**analyze_immunoid(_assistant_immunoid_request_from_state(state).model_dump()))
    for item in result.exposure_summary:
        source = state.immunoid_signal_sources.get(item.id)
        if source:
            item.source = source
    return result


def _assistant_immunoid_agent_options(state: AssistantState) -> List[AssistantOption]:
    regimens_by_id = _assistant_immunoid_regimens_by_id()
    agents_by_id = _assistant_immunoid_agents_by_id()
    options: List[AssistantOption] = []
    for regimen_id in IMMUNOID_COMMON_REGIMEN_IDS:
        entry = regimens_by_id.get(regimen_id)
        if entry is None:
            continue
        options.append(
            AssistantOption(
                value=f"immunoid_regimen:{regimen_id}",
                label=entry["name"],
                description=", ".join(entry["componentAgentNames"][:3]),
            )
        )
    for agent_id in IMMUNOID_COMMON_AGENT_IDS:
        entry = agents_by_id.get(agent_id)
        if entry is None:
            continue
        options.append(
            AssistantOption(
                value=f"immunoid_agent:{agent_id}",
                label=entry["name"],
                description=entry["drugClass"],
            )
        )
    if state.immunoid_selected_agent_ids or state.immunoid_selected_regimen_ids:
        options.append(AssistantOption(value="immunoid_continue", label="Continue"))
        options.append(AssistantOption(value="immunoid_clear_agents", label="Clear selections"))
    options.append(AssistantOption(value="restart", label="Start new consult"))
    return options


def _assistant_immunoid_steroid_options() -> List[AssistantOption]:
    return [
        AssistantOption(value="immunoid_agent:prednisone_20", label="Prednisone >= 20 mg/day"),
        AssistantOption(value="immunoid_agent:prednisolone_20", label="Prednisolone >= 20 mg/day"),
        AssistantOption(value="immunoid_agent:methylpred_16", label="Methylprednisolone >= 16 mg/day"),
        AssistantOption(value="immunoid_agent:dexamethasone_3", label="Dexamethasone >= 3 mg/day"),
        AssistantOption(value="immunoid_agent:hydrocortisone_80", label="Hydrocortisone >= 80 mg/day"),
        AssistantOption(value="restart", label="Start new consult"),
    ]


def _assistant_immunoid_followup_options(question_id: str) -> List[AssistantOption]:
    if question_id == "steroid_duration":
        return [
            AssistantOption(value="immunoid_answer:steroid_duration:28_plus", label="28 days or more"),
            AssistantOption(value="immunoid_answer:steroid_duration:lt_28", label="Less than 28 days"),
            AssistantOption(value="immunoid_answer:steroid_duration:unknown", label="Not sure"),
            AssistantOption(value="restart", label="Start new consult"),
        ]
    return [
        AssistantOption(value=f"immunoid_answer:{question_id}:yes", label="Yes"),
        AssistantOption(value=f"immunoid_answer:{question_id}:no", label="No"),
        AssistantOption(value=f"immunoid_answer:{question_id}:unknown", label="Not sure"),
        AssistantOption(value="restart", label="Start new consult"),
    ]


def _assistant_immunoid_extra_context_options(state: AssistantState) -> List[AssistantOption]:
    options: List[AssistantOption] = []
    if any(agent_id in state.immunoid_selected_agent_ids for agent_id in {"rituximab", "obinutuzumab", "ocrelizumab", "ofatumumab", "ofatumumab_kesimpta", "ublituximab"}):
        options.extend(
            [
                AssistantOption(value="immunoid_set:hbv_hbsag:positive", label="HBsAg positive"),
                AssistantOption(value="immunoid_set:hbv_hbsag:negative", label="HBsAg negative"),
                AssistantOption(value="immunoid_set:hbv_anti_hbc:positive", label="anti-HBc positive"),
                AssistantOption(value="immunoid_set:hbv_anti_hbc:negative", label="anti-HBc negative"),
            ]
        )
    if any(agent_id in state.immunoid_selected_agent_ids for agent_id in {"infliximab", "adalimumab", "etanercept", "certolizumab", "golimumab", "tofacitinib", "baricitinib", "upadacitinib", "ruxolitinib"}):
        options.extend(
            [
                AssistantOption(value="immunoid_set:tb_screen:positive", label="TB screen positive"),
                AssistantOption(value="immunoid_set:tb_screen:negative", label="TB screen negative"),
                AssistantOption(value="immunoid_set:tb_screen:indeterminate", label="TB screen indeterminate"),
            ]
        )
    return options[:6]


def _assistant_apply_immunoid_selection(state: AssistantState, selection: str) -> bool:
    if selection.startswith("immunoid_regimen:"):
        regimen_id = selection.split(":", 1)[1]
        regimen = IMMUNOID_REGIMENS.get(regimen_id)
        if regimen is None:
            return False
        if regimen_id not in state.immunoid_selected_regimen_ids:
            state.immunoid_selected_regimen_ids.append(regimen_id)
        for agent_id in regimen.get("component_agent_ids", ()):
            if agent_id not in state.immunoid_selected_agent_ids and agent_id in _assistant_immunoid_agents_by_id():
                state.immunoid_selected_agent_ids.append(agent_id)
        _assistant_apply_immunoid_regimen_defaults(state, regimen_id)
        return True
    if selection.startswith("immunoid_agent:"):
        agent_id = selection.split(":", 1)[1]
        if agent_id not in state.immunoid_selected_agent_ids and agent_id in _assistant_immunoid_agents_by_id():
            state.immunoid_selected_agent_ids.append(agent_id)
        return True
    if selection == "immunoid_clear_agents":
        state.immunoid_selected_regimen_ids = []
        state.immunoid_selected_agent_ids = []
        state.immunoid_planned_steroid_duration_days = None
        state.immunoid_anticipated_prolonged_profound_neutropenia = None
        state.immunoid_signal_sources.pop("planned_steroid_duration_days", None)
        state.immunoid_signal_sources.pop("anticipated_prolonged_profound_neutropenia", None)
        return True
    if selection.startswith("immunoid_set:"):
        _, field_id, value = selection.split(":", 2)
        if field_id == "hbv_hbsag":
            state.immunoid_hbv_hbsag = value
            _assistant_set_immunoid_signal_source(state, "hbv_hbsag", "selection")
        elif field_id == "hbv_anti_hbc":
            state.immunoid_hbv_anti_hbc = value
            _assistant_set_immunoid_signal_source(state, "hbv_anti_hbc", "selection")
        elif field_id == "tb_screen":
            state.immunoid_tb_screen_result = value
            _assistant_set_immunoid_signal_source(state, "tb_screen_result", "selection")
        return True
    if selection.startswith("immunoid_answer:"):
        _, question_id, value = selection.split(":", 2)
        if question_id == "steroid_duration":
            state.immunoid_planned_steroid_duration_days = 28 if value == "28_plus" else (14 if value == "lt_28" else None)
            _assistant_set_immunoid_signal_source(state, "planned_steroid_duration_days", "selection")
        elif question_id == "tb_endemic_exposure":
            state.immunoid_tb_endemic_exposure = True if value == "yes" else (False if value == "no" else None)
            _assistant_set_immunoid_signal_source(state, "tb_endemic_exposure", "selection")
        elif question_id == "strongyloides_exposure":
            state.immunoid_strongyloides_exposure = True if value == "yes" else (False if value == "no" else None)
            _assistant_set_immunoid_signal_source(state, "strongyloides_exposure", "selection")
        elif question_id == "coccidioides_exposure":
            state.immunoid_coccidioides_exposure = True if value == "yes" else (False if value == "no" else None)
            _assistant_set_immunoid_signal_source(state, "coccidioides_exposure", "selection")
        elif question_id == "histoplasma_exposure":
            state.immunoid_histoplasma_exposure = True if value == "yes" else (False if value == "no" else None)
            _assistant_set_immunoid_signal_source(state, "histoplasma_exposure", "selection")
        elif question_id == "prolonged_profound_neutropenia":
            state.immunoid_anticipated_prolonged_profound_neutropenia = True if value == "yes" else (False if value == "no" else None)
            _assistant_set_immunoid_signal_source(state, "anticipated_prolonged_profound_neutropenia", "selection")
        return True
    return False


def _assistant_immunoid_followup_message(result: ImmunoAnalyzeResponse) -> str:
    question = result.follow_up_questions[0]
    selected_names = ", ".join(regimen.name for regimen in result.selected_regimens) or ", ".join(
        agent.name for agent in result.selected_agents
    ) or "the selected agents"
    if result.recommendations:
        preview = ", ".join(rec.title for rec in result.recommendations[:2])
        return (
            f"I identified these exposures: {selected_names}. I already have a preliminary checklist ({preview}), "
            f"but one missing detail will change the result: {question.prompt}"
        )
    return (
        f"I identified these exposures: {selected_names}. Before I finalize the screening and prophylaxis checklist, "
        f"I need one more detail: {question.prompt}"
    )


def _assistant_immunoid_final_message(result: ImmunoAnalyzeResponse) -> str:
    if not result.recommendations:
        return (
            "I mapped the selected immunosuppressive agents, but no current deterministic screening or prophylaxis rule fired "
            "from the context you provided. Add more serologies, exposure history, or regimen details if you want me to refine it."
        )
    tests = [rec.summary for rec in result.recommendations if rec.category == "screening"]
    prophylaxis = [rec.summary for rec in result.recommendations if rec.category == "prophylaxis"]
    context_items = [rec.summary for rec in result.recommendations if rec.category in {"referral", "context", "monitoring"}]
    parts: List[str] = []
    if prophylaxis:
        parts.append("Prophylaxis or protocol review: " + " ".join(prophylaxis[:3]))
    if tests:
        parts.append("Before therapy, check: " + " ".join(tests[:3]))
    if context_items:
        parts.append("Additional context: " + " ".join(context_items[:3]))
    if not parts:
        parts.append("I generated a rule-backed checklist from the selected immunosuppression profile.")
    return "\n\n".join(parts)


def _assistant_immunoid_response(
    state: AssistantState,
    *,
    prefix: str | None = None,
) -> AssistantTurnResponse:
    result = _assistant_immunoid_analysis_from_state(state)
    if result.follow_up_questions:
        state.stage = "immunoid_collect_context"
        question = result.follow_up_questions[0]
        options = _assistant_immunoid_followup_options(question.id)
        options.extend(_assistant_immunoid_extra_context_options(state))
        fallback_message = ((prefix or "") + _assistant_immunoid_followup_message(result)).strip()
        narrated_message, narration_refined = narrate_immunoid_assistant_message(
            immunoid_result=result,
            fallback_message=fallback_message,
            follow_up_stage=True,
        )
        return AssistantTurnResponse(
            assistantMessage=narrated_message,
            assistantNarrationRefined=narration_refined,
            state=state,
            options=options,
            immunoidAnalysis=result,
            tips=[
                question.reason,
                "You can answer in plain language or click one of the choices.",
            ],
        )

    state.stage = "done"
    doseid_options = _immunoid_doseid_options(result)
    options: List[AssistantOption] = [
        *doseid_options,
        AssistantOption(value="add_more_details", label="Update this case"),
    ]
    # ImmunoID → HIVID bridge: offer HIV-specific chips when relevant
    hiv_bridge_opts = _immunoid_hiv_bridge_options(state)
    options.extend(hiv_bridge_opts)
    if _is_mid_consult(state):
        options.append(AssistantOption(value="consult_summary", label="Full consult summary"))
    else:
        options.append(AssistantOption(value="restart", label="Start new consult"))
    options.extend(_assistant_immunoid_extra_context_options(state))
    fallback_message = ((prefix or "") + _assistant_immunoid_final_message(result)).strip()
    narrated_message, narration_refined = narrate_immunoid_assistant_message(
        immunoid_result=result,
        fallback_message=fallback_message,
        follow_up_stage=False,
    )
    tips = [
        "Add another serology result, exposure history point, or regimen detail if you want me to refine the same checklist.",
        "Every recommendation shown here comes from the deterministic rule set and its attached citations.",
    ]
    if doseid_options:
        tips.insert(0, "If you want, I can calculate the renal-adjusted dose for the recommended prophylaxis agent.")
    return AssistantTurnResponse(
        assistantMessage=narrated_message,
        assistantNarrationRefined=narration_refined,
        state=state,
        options=options,
        immunoidAnalysis=result,
        tips=tips,
    )


def _assistant_start_immunoid_from_text(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    if not message_text or not _assistant_is_immunoid_intent(message_text):
        return None
    detected_regimens = _assistant_detect_immunoid_regimen_ids(message_text)
    detected_agents = _assistant_detect_immunoid_agent_ids(message_text)
    if not detected_regimens and not detected_agents and not _assistant_immunoid_mentions_generic_steroid(message_text):
        return None

    state.workflow = "immunoid"
    state.module_id = None
    state.preset_id = None
    state.pending_intake_text = None
    state.case_section = None
    state.case_text = None
    state.mechid_text = None
    state.pretest_factor_ids = []
    state.pretest_factor_labels = []
    state.endo_blood_culture_context = None
    state.endo_score_factor_ids = []
    _assistant_reset_immunoid_state(state)
    if not detected_agents and _assistant_immunoid_mentions_generic_steroid(message_text):
        state.stage = "immunoid_select_agents"
        return AssistantTurnResponse(
            assistantMessage=(
                "I picked up a steroid exposure, but I still need the specific corticosteroid and threshold. "
                "Please tell me which steroid, the approximate daily dose, and whether it will be prednisone-equivalent "
                "20 mg/day or more for at least 4 weeks."
            ),
            state=state,
            options=_assistant_immunoid_steroid_options(),
            tips=[
                "A useful reply would be: 'prednisone 20 mg daily for 6 weeks' or 'methylprednisolone 8 mg daily for 10 days'.",
                "That distinction matters because the PJP rule is tied to dose and duration.",
            ],
        )
    state.immunoid_selected_regimen_ids = detected_regimens
    state.immunoid_selected_agent_ids = detected_agents
    for regimen_id in detected_regimens:
        for agent_id in IMMUNOID_REGIMENS.get(regimen_id, {}).get("component_agent_ids", ()):
            if agent_id not in state.immunoid_selected_agent_ids:
                state.immunoid_selected_agent_ids.append(agent_id)
        _assistant_apply_immunoid_regimen_defaults(state, regimen_id)
    _assistant_parse_immunoid_context_from_text(state, message_text)
    return _assistant_immunoid_response(state)


def _assistant_doseid_normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", (value or "").lower()).strip()


def _assistant_text_mentions_doseid_age(normalized: str) -> bool:
    return bool(
        re.search(
            r"\bage\s*(?:is|=|:)?\s*\d{1,3}\b|\b\d{1,3}\s*(?:yo|y o|y/o|yr old|yrs old|year old|years old)\b",
            normalized,
        )
    )


def _assistant_text_mentions_doseid_crcl(normalized: str) -> bool:
    return bool(re.search(r"\b(?:crcl|creatinine clearance)\b", normalized))


def _assistant_text_mentions_doseid_scr(normalized: str) -> bool:
    return bool(re.search(r"\b(?:scr|s\s*cr|serum creatinine|creatinine|cr)\b", normalized))


def _assistant_text_mentions_doseid_ihd(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:hemodialysis|haemodialysis|dialysis|esrd|ihd|hd)\b",
            normalized,
        )
    )


def _assistant_text_mentions_doseid_crrt(normalized: str) -> bool:
    return bool(re.search(r"\b(?:crrt|cvvh|cvvhd|cvvhdf)\b", normalized))


def _assistant_doseid_medications_by_id() -> Dict[str, Any]:
    return {med.id: med for med in list_medications()}


def _assistant_doseid_alias_map() -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}
    for med in list_medications():
        aliases = {
            med.id.replace("_", " "),
            med.name.lower(),
            med.name.lower().replace("/", " "),
            med.name.lower().replace("/", ""),
        }
        if med.id == "piperacillin_tazobactam":
            aliases.update({"zosyn", "pip tazo", "piptazo"})
        elif med.id == "ampicillin_sulbactam":
            aliases.update({"unasyn", "unsyn"})
        elif med.id == "tmp_smx":
            aliases.update({"tmp smx", "tmp-smx", "bactrim", "septra", "trimethoprim sulfamethoxazole"})
        elif med.id == "nitrofurantoin":
            aliases.update({"macrobid", "macrodantin", "nitro"})
        elif med.id == "fosfomycin":
            aliases.update({"monurol"})
        elif med.id == "cefepime":
            aliases.update({"maxipime"})
        elif med.id == "ceftriaxone":
            aliases.update({"rocephin"})
        elif med.id == "meropenem":
            aliases.update({"merrem"})
        elif med.id == "ertapenem":
            aliases.update({"invanz"})
        elif med.id == "aztreonam":
            aliases.update({"azactam"})
        elif med.id == "cefiderocol":
            aliases.update({"fetroja"})
        elif med.id == "ceftaroline":
            aliases.update({"teflaro"})
        elif med.id == "ceftazidime_avibactam":
            aliases.update({"avycaz", "ceftaz avi", "ceftaz-avi"})
        elif med.id == "ciprofloxacin":
            aliases.update({"cipro"})
        elif med.id == "levofloxacin":
            aliases.update({"levaquin"})
        elif med.id == "linezolid":
            aliases.update({"zyvox"})
        elif med.id == "daptomycin":
            aliases.update({"cubicin"})
        elif med.id == "vancomycin_iv":
            aliases.update({"vanc", "iv vancomycin", "vancomycin", "vancocin"})
        elif med.id == "metronidazole":
            aliases.update({"flagyl"})
        elif med.id == "clindamycin":
            aliases.update({"cleocin"})
        elif med.id == "amoxicillin_clavulanate":
            aliases.update({"augmentin"})
        elif med.id == "amoxicillin":
            aliases.update({"amoxil"})
        elif med.id == "liposomal_amphotericin_b":
            aliases.update({"ambisome", "l ampho", "liposomal amphotericin"})
        elif med.id == "isavuconazole":
            aliases.update({"cresemba"})
        elif med.id == "posaconazole":
            aliases.update({"noxafil"})
        elif med.id == "fluconazole":
            aliases.update({"diflucan"})
        elif med.id == "micafungin":
            aliases.update({"mycamine"})
        elif med.id == "voriconazole":
            aliases.update({"vfend"})
        elif med.id == "caspofungin":
            aliases.update({"cancidas"})
        elif med.id == "foscarnet":
            aliases.update({"foscavir"})
        elif med.id == "famciclovir":
            aliases.update({"famvir"})
        elif med.id == "moxifloxacin_tb":
            aliases.update({"moxifloxacin", "moxi"})
        elif med.id == "acyclovir_iv":
            aliases.update({"iv acyclovir", "zovirax iv"})
        elif med.id == "acyclovir_po":
            aliases.update({"oral acyclovir", "po acyclovir", "zovirax"})
        elif med.id == "valacyclovir":
            aliases.update({"valtrex"})
        elif med.id == "ganciclovir_iv":
            aliases.update({"cytovene"})
        elif med.id == "valganciclovir":
            aliases.update({"valcyte"})
        elif med.id == "oseltamivir":
            aliases.update({"tamiflu"})
        alias_map[med.id] = sorted(_assistant_doseid_normalize(alias) for alias in aliases if alias)
    return alias_map


def _assistant_detect_doseid_medication_ids(message_text: str) -> List[str]:
    normalized = _assistant_doseid_normalize(message_text)
    medication_ids: List[str] = []
    if _assistant_text_has_phrase(normalized, "ripe") or _assistant_text_has_phrase(normalized, "rhze"):
        medication_ids.extend(["rifampin", "isoniazid", "pyrazinamide", "ethambutol"])
    for med_id, aliases in _assistant_doseid_alias_map().items():
        if med_id in medication_ids:
            continue
        for alias in aliases:
            if alias and _assistant_text_has_phrase(normalized, alias):
                medication_ids.append(med_id)
                break
    return medication_ids


def _assistant_text_mentions_doseid_schedule(normalized: str) -> bool:
    return bool(re.search(r"\bq\d{1,2}(?:-\d{1,2})?(?:h|hr|hrs)\b|\b(?:bid|tid|qid|qod)\b", normalized))


def _assistant_has_doseid_context_signal(normalized: str) -> bool:
    if _assistant_text_mentions_doseid_age(normalized):
        return True
    if _assistant_text_mentions_doseid_crcl(normalized):
        return True
    if _assistant_text_mentions_doseid_scr(normalized):
        return True
    if _assistant_text_mentions_doseid_ihd(normalized):
        return True
    if _assistant_text_mentions_doseid_crrt(normalized):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(kg|kgs|kilograms?|kilos?|lb|lbs|pounds?|cm)\b", normalized):
        return True
    return any(token in normalized for token in ("renal function", "kidneys are normal", "kidney function"))


def _assistant_is_doseid_intent(message_text: str) -> bool:
    normalized = f" {_assistant_doseid_normalize(message_text)} "
    if not normalized.strip():
        return False
    if (" susceptible dose dependent " in normalized or " sdd " in normalized):
        explicit_dose_phrases = (
            " dosing ",
            " dosage ",
            " what is the dose ",
            " what's the dose ",
            " calculate dosing ",
            " calculate the dose ",
            " dose this ",
            " renal dose ",
            " renal dosing ",
            " hemodialysis ",
            " dialysis ",
            " crcl ",
            " creatinine clearance ",
            " ripe ",
            " rhze ",
        )
        if not any(phrase in normalized for phrase in explicit_dose_phrases):
            return False
    if _assistant_detect_doseid_medication_ids(message_text):
        if any(f" {token} " in normalized for token in DOSEID_INTENT_TOKENS):
            return True
        if _assistant_text_mentions_doseid_schedule(normalized):
            return True
        return _assistant_has_doseid_context_signal(normalized)
    return any(f" {token} " in normalized for token in DOSEID_INTENT_TOKENS)


def _assistant_parse_doseid_patient_context(message_text: str) -> DoseIDAssistantPatientContext:
    normalized = _assistant_doseid_normalize(message_text)
    renal_mode = "standard"
    if _assistant_text_mentions_doseid_crrt(normalized):
        renal_mode = "crrt"
    elif _assistant_text_mentions_doseid_ihd(normalized):
        renal_mode = "ihd"

    age_years = None
    age_match = re.search(
        r"\bage\s*(?:is|=|:)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:yo|y o|y/o|yr old|yrs old|year old|years old)\b",
        normalized,
    )
    if age_match:
        age_years = int(age_match.group(1) or age_match.group(2))

    sex = None
    sex_match = re.search(
        r"\b(?:sex|gender)\s*(?:is|=|:)?\s*(male|female|man|woman|m|f)\b",
        normalized,
    )
    if sex_match:
        sex_token = str(sex_match.group(1) or "").strip()
        sex = "male" if sex_token in {"male", "man", "m"} else "female"
    elif re.search(r"\bmale\b|\bman\b", normalized):
        sex = "male"
    elif re.search(r"\bfemale\b|\bwoman\b", normalized):
        sex = "female"

    weight_kg = None
    weight_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|kilos?)\b", normalized)
    if weight_match:
        weight_kg = float(weight_match.group(1))
    else:
        pounds_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b", normalized)
        if pounds_match:
            weight_kg = round(float(pounds_match.group(1)) / 2.20462, 1)

    height_cm = None
    height_match = re.search(r"\b(\d+(?:\.\d+)?)\s*cm\b", normalized)
    if height_match:
        height_cm = float(height_match.group(1))
    else:
        feet_inches_match = re.search(r"\b(\d)\s*(?:ft|feet|foot)\s*(\d{1,2})?\s*(?:in|inch|inches)?\b", normalized)
        if feet_inches_match:
            feet = int(feet_inches_match.group(1))
            inches = int(feet_inches_match.group(2) or 0)
            height_cm = round(((feet * 12) + inches) * 2.54, 1)
        else:
            feet_inches_quote_match = re.search(r"\b(\d)\s*'\s*(\d{1,2})\s*(?:\"|in|inch|inches)?\b", message_text)
            if feet_inches_quote_match:
                feet = int(feet_inches_quote_match.group(1))
                inches = int(feet_inches_quote_match.group(2))
                height_cm = round(((feet * 12) + inches) * 2.54, 1)

    scr = None
    scr_match = re.search(
        r"\b(?:scr|s\s*cr|serum creatinine|creatinine|cr)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if scr_match:
        scr = float(scr_match.group(1))

    crcl = None
    crcl_match = re.search(
        r"\b(?:crcl|creatinine clearance)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if crcl_match:
        crcl = float(crcl_match.group(1))

    return DoseIDAssistantPatientContext(
        ageYears=age_years,
        sex=sex,
        totalBodyWeightKg=weight_kg,
        heightCm=height_cm,
        serumCreatinineMgDl=scr,
        crclMlMin=crcl,
        renalMode=renal_mode,
    )


def _assistant_doseid_has_any(normalized: str, tokens: tuple[str, ...]) -> bool:
    return any(token in normalized for token in tokens)


DOSEID_CNS_TOKENS = ("meningitis", "cns", "ventriculitis", "brain abscess", "central nervous system")
DOSEID_PSEUDOMONAL_TOKENS = ("pseudomonas", "pseudomonal")
DOSEID_SEVERE_TOKENS = ("septic shock", "shock", "critical illness", "critically ill", "severe", "deep seated", "high inoculum")
DOSEID_BACTEREMIA_TOKENS = ("bacteremia", "bloodstream", "endocarditis", "endovascular")
DOSEID_BONE_JOINT_TOKENS = ("bone and joint", "bone", "joint", "osteomyelitis", "septic arthritis", "prosthetic joint", "pji")
DOSEID_MUCOSAL_CANDIDA_TOKENS = ("mucosal", "thrush", "oropharyngeal", "vaginal candidiasis")
DOSEID_PJP_TOKENS = ("pjp", "pneumocystis")
DOSEID_STENO_TOKENS = ("steno", "stenotrophomonas")
DOSEID_SSTI_TOKENS = ("skin", "soft tissue", "ssti", "cellulitis")
DOSEID_INTRAABDOMINAL_TOKENS = ("intraabdominal", "intra abdominal", "abdominal", "polymicrobial")
DOSEID_TOXIN_SUPPRESSION_TOKENS = ("necrotizing", "toxin", "toxin suppression", "group a strep", "strep pyogenes", "tss")


def _assistant_doseid_structured_context_indication(medication_id: str, normalized: str) -> str | None:
    bloodstream = "syndrome: bloodstream infection" in normalized
    endocarditis = "focus: endocarditis" in normalized
    cystitis = "syndrome: uncomplicated cystitis" in normalized
    complicated_uti = "syndrome: complicated uti / pyelonephritis" in normalized
    bone_joint = "syndrome: bone/joint infection" in normalized
    pneumonia = "syndrome: pneumonia (hap/vap or severe cap)" in normalized
    intraabdominal = "syndrome: intra-abdominal infection" in normalized
    cns = "syndrome: cns infection" in normalized
    enterococcal = any(token in normalized for token in ("enterococcus", "enterococcal", "e faecium", "e faecalis"))
    vre = any(token in normalized for token in ("vre", "vrefm", "vancomycin resistant enterococcus"))

    if medication_id == "nitrofurantoin" and cystitis:
        return "uncomplicated_cystitis"
    if medication_id == "fosfomycin":
        if cystitis:
            return "uncomplicated_cystitis"
        if complicated_uti:
            return "complicated_cystitis"
    if medication_id == "daptomycin" and (endocarditis or bloodstream):
        if vre or "enterococcus faecium" in normalized:
            return "vre_high_burden"
        return "bacteremia_endovascular"
    if medication_id == "linezolid" and (endocarditis or bloodstream or bone_joint or pneumonia):
        return "standard_bacterial"
    if medication_id == "ampicillin" and (endocarditis or bloodstream or enterococcal or cns):
        return "high_exposure"
    if medication_id == "ceftriaxone":
        if cns:
            return "meningitis"
        if endocarditis or bloodstream:
            return "serious_infection"
    if medication_id == "cefazolin" and (endocarditis or bloodstream or bone_joint):
        return "complicated_or_deep"
    if medication_id == "nafcillin" and (endocarditis or bloodstream):
        return "mssa_high_burden"
    if medication_id == "penicillin_g" and (endocarditis or bloodstream or cns):
        return "cns_or_high_exposure"
    if medication_id == "vancomycin_iv" and (endocarditis or bloodstream):
        return "serious_mrsa_or_invasive"
    if medication_id == "levofloxacin" and pneumonia:
        return "pneumonia_or_pseudomonas"
    if medication_id == "ciprofloxacin" and pneumonia:
        return "high_exposure_pseudomonal"
    if medication_id == "clindamycin" and bone_joint:
        return "bone_joint_infection"
    if medication_id == "metronidazole" and intraabdominal:
        return "intraabdominal_coverage"
    if medication_id == "ampicillin_sulbactam" and intraabdominal:
        return "surgical_or_intraabdominal"
    return None


def _assistant_doseid_indication_for_query(medication_id: str, message_text: str) -> str:
    normalized = _assistant_doseid_normalize(message_text)
    structured_context = _assistant_doseid_structured_context_indication(medication_id, normalized)
    if structured_context is not None:
        return structured_context
    if medication_id == "tmp_smx":
        if "prophyl" in normalized and "pjp" in normalized:
            return "pjp_prophylaxis"
        if _assistant_doseid_has_any(normalized, DOSEID_PJP_TOKENS):
            return "pjp_treatment"
        if _assistant_doseid_has_any(normalized, DOSEID_STENO_TOKENS):
            return "stenotrophomonas"
        if _assistant_doseid_has_any(normalized, DOSEID_BONE_JOINT_TOKENS):
            return "staph_bone_joint"
        if _assistant_doseid_has_any(normalized, DOSEID_SSTI_TOKENS):
            return "ssti"
        if _assistant_doseid_has_any(normalized, DOSEID_BACTEREMIA_TOKENS):
            return "gnr_bacteremia"
    if medication_id == "nitrofurantoin":
        return "uncomplicated_cystitis"
    if medication_id == "fosfomycin":
        if any(token in normalized for token in ("complicated cystitis", "recurrent cystitis", "multiple dose")):
            return "complicated_cystitis"
        return "uncomplicated_cystitis"
    if medication_id == "ampicillin_sulbactam" and _assistant_doseid_has_any(normalized, DOSEID_INTRAABDOMINAL_TOKENS):
        return "surgical_or_intraabdominal"
    if medication_id in {"cefepime", "meropenem"} and _assistant_doseid_has_any(normalized, DOSEID_CNS_TOKENS):
        return "cns_meningitis"
    if medication_id == "piperacillin_tazobactam" and (
        _assistant_doseid_has_any(normalized, DOSEID_PSEUDOMONAL_TOKENS) or _assistant_doseid_has_any(normalized, DOSEID_SEVERE_TOKENS)
    ):
        return "high_inoculum_pseudomonal"
    if medication_id == "aztreonam" and any(token in normalized for token in ("uti", "urinary", "cystitis")):
        return "uncomplicated_uti"
    if medication_id == "cefazolin" and (
        _assistant_doseid_has_any(normalized, DOSEID_BACTEREMIA_TOKENS) or _assistant_doseid_has_any(normalized, DOSEID_BONE_JOINT_TOKENS)
    ):
        return "complicated_or_deep"
    if medication_id == "ceftriaxone":
        if "endocarditis" in normalized and any(token in normalized for token in ("enterococcus faecalis", "e faecalis")):
            return "enterococcal_endocarditis_synergy"
        if _assistant_doseid_has_any(normalized, DOSEID_CNS_TOKENS):
            return "meningitis"
        if _assistant_doseid_has_any(normalized, DOSEID_BACTEREMIA_TOKENS) or any(
            token in normalized for token in ("sepsis", "severe", "invasive")
        ):
            return "serious_infection"
    if medication_id == "ceftazidime" and (
        _assistant_doseid_has_any(normalized, DOSEID_PSEUDOMONAL_TOKENS) or _assistant_doseid_has_any(normalized, DOSEID_SEVERE_TOKENS)
    ):
        return "pseudomonal_or_severe"
    if medication_id == "ertapenem" and any(token in normalized for token in ("esbl", "extended spectrum beta lactamase")):
        return "esbl_targeted"
    if medication_id == "linezolid" and any(token in normalized for token in ("tb", "ntm", "mycobacterial", "prolonged")):
        return "mycobacterial_or_long_course"
    if medication_id == "levofloxacin" and (
        _assistant_doseid_has_any(normalized, DOSEID_PSEUDOMONAL_TOKENS)
        or any(token in normalized for token in ("pneumonia", "cap", "hap", "vap"))
    ):
        return "pneumonia_or_pseudomonas"
    if medication_id == "ampicillin" and (_assistant_doseid_has_any(normalized, DOSEID_CNS_TOKENS) or "enterococcus" in normalized):
        return "high_exposure"
    if medication_id == "cefiderocol" and (
        _assistant_doseid_has_any(normalized, DOSEID_SEVERE_TOKENS)
        or any(token in normalized for token in ("arc", "augmented renal clearance", "critical"))
    ):
        return "high_clearance_or_critical"
    if medication_id == "ceftaroline" and any(
        token in normalized for token in ("mrsa", "persistent bacteremia", "endocarditis", "salvage")
    ):
        return "high_exposure_mrsa"
    if medication_id == "ceftazidime_avibactam" and _assistant_doseid_has_any(normalized, DOSEID_SEVERE_TOKENS):
        return "high_exposure_critical"
    if medication_id == "ciprofloxacin" and (
        _assistant_doseid_has_any(normalized, DOSEID_PSEUDOMONAL_TOKENS)
        or any(token in normalized for token in ("hap", "vap"))
    ):
        return "high_exposure_pseudomonal"
    if medication_id == "clindamycin":
        if _assistant_doseid_has_any(normalized, DOSEID_BONE_JOINT_TOKENS):
            return "bone_joint_infection"
        if _assistant_doseid_has_any(normalized, DOSEID_TOXIN_SUPPRESSION_TOKENS):
            return "adjunctive_toxin_suppression"
    if medication_id == "daptomycin":
        if any(token in normalized for token in ("vre", "vrefm", "enterococcus faecium")):
            return "vre_high_burden"
        if _assistant_doseid_has_any(normalized, DOSEID_BACTEREMIA_TOKENS):
            return "bacteremia_endovascular"
    if medication_id == "imipenem_cilastatin" and any(
        token in normalized for token in ("resistant", "mdr", "xdr", "carbapenem resistant", "esbl")
    ):
        return "high_exposure_resistant"
    if medication_id == "nafcillin" and _assistant_doseid_has_any(normalized, DOSEID_BACTEREMIA_TOKENS):
        return "mssa_high_burden"
    if medication_id == "penicillin_g" and (
        _assistant_doseid_has_any(normalized, DOSEID_CNS_TOKENS)
        or any(token in normalized for token in ("endocarditis", "deep", "high inoculum"))
    ):
        return "cns_or_high_exposure"
    if medication_id == "vancomycin_iv" and any(
        token in normalized for token in ("mrsa", "bacteremia", "bloodstream", "endocarditis", "invasive")
    ):
        return "serious_mrsa_or_invasive"
    if medication_id in {"isoniazid", "rifampin"} and "three times weekly" in normalized:
        return "tb_intermittent" if medication_id == "isoniazid" else "tb_daily"
    if medication_id in {"ethambutol", "pyrazinamide"} and "three times weekly" in normalized:
        return "tb_high_dose_intermittent"
    if medication_id == "oseltamivir":
        return "influenza_prophylaxis" if "prophyl" in normalized else "influenza_treatment"
    if medication_id in {"valganciclovir", "ganciclovir_iv"}:
        return "cmv_prophylaxis" if "prophyl" in normalized else "cmv_treatment"
    if medication_id == "foscarnet":
        return "cmv_maintenance_or_hsv" if any(token in normalized for token in ("maintenance", "salvage hsv", "hsv")) else "cmv_induction"
    if medication_id == "valacyclovir":
        return "hsv_suppression" if "suppress" in normalized else "zoster_or_treatment"
    if medication_id == "acyclovir_po":
        return "zoster_or_severe_hsv" if any(token in normalized for token in ("zoster", "shingles", "severe hsv")) else "standard_hsv"
    if medication_id == "acyclovir_iv":
        return "hsv_encephalitis_or_disseminated" if any(token in normalized for token in ("encephalitis", "disseminated")) else "standard_hsv_systemic"
    if medication_id == "famciclovir":
        if "suppress" in normalized:
            return "suppression"
        if any(token in normalized for token in ("recurrent genital", "episodic hsv", "genital hsv")):
            return "recurrent_genital_hsv"
        return "herpes_zoster"
    if medication_id == "rifampin" and any(token in normalized for token in ("hardware", "prosthetic", "biofilm")):
        return "hardware_adjuvant"
    if medication_id == "isavuconazole":
        return "stepdown_oral" if "stepdown" in normalized or "step-down" in normalized else "invasive_mold_treatment"
    if medication_id == "posaconazole":
        return "mold_prophylaxis" if "prophyl" in normalized else "invasive_fungal_treatment"
    if medication_id == "fluconazole":
        return "mucosal_candidiasis" if _assistant_doseid_has_any(normalized, DOSEID_MUCOSAL_CANDIDA_TOKENS) else "candidemia_invasive"
    if medication_id == "micafungin":
        return "esophageal_candidiasis" if "esophag" in normalized else "candidemia_invasive"
    if medication_id == "voriconazole":
        return "mold_prophylaxis" if "prophyl" in normalized else "invasive_mold_treatment"
    if medication_id == "liposomal_amphotericin_b":
        return "cryptococcal_cns_induction" if any(token in normalized for token in ("crypto", "cryptococ", "cns")) else "invasive_mold_or_severe_yeast"
    return default_indication_id(medication_id)


DOSEID_STANDARD_NONRENAL_MEDICATION_IDS = {
    "ceftriaxone",
    "linezolid",
    "clindamycin",
    "nafcillin",
    "isoniazid",
    "rifampin",
    "moxifloxacin_tb",
    "caspofungin",
    "isavuconazole",
    "posaconazole",
    "micafungin",
    "voriconazole",
    "liposomal_amphotericin_b",
}
DOSEID_ALWAYS_WEIGHT_BASED_MEDICATION_IDS = {
    "daptomycin",
    "vancomycin_iv",
    "isoniazid",
    "ethambutol",
    "pyrazinamide",
    "voriconazole",
    "liposomal_amphotericin_b",
    "foscarnet",
    "acyclovir_iv",
    "ganciclovir_iv",
}
DOSEID_WEIGHT_BASED_TMP_SMX_INDICATION_IDS = {
    "staph_bone_joint",
    "gnr_bacteremia",
    "stenotrophomonas",
    "pjp_treatment",
}
DOSEID_HEIGHT_REQUIRED_MEDICATION_IDS = {
    "daptomycin",
    "ethambutol",
    "pyrazinamide",
    "voriconazole",
    "liposomal_amphotericin_b",
    "foscarnet",
    "acyclovir_iv",
    "ganciclovir_iv",
}
DOSEID_SEX_REQUIRED_MEDICATION_IDS = {
    "ethambutol",
    "pyrazinamide",
    "voriconazole",
    "liposomal_amphotericin_b",
    "foscarnet",
    "ganciclovir_iv",
}
DOSEID_FIELD_LABELS = {
    "medication": "medication or regimen",
    "renal_function": "serum creatinine or CrCl",
    "serum_creatinine": "serum creatinine",
    "age": "age",
    "sex": "sex",
    "weight": "weight",
    "height": "height",
}
DOSEID_FOLLOW_UP_FIELD_ORDER = (
    "medication",
    "renal_function",
    "serum_creatinine",
    "age",
    "sex",
    "weight",
    "height",
)


def _doseid_patient_context_from_partial_input(
    patient: Any,
    *,
    renal_mode: str,
) -> DoseIDAssistantPatientContext:
    return DoseIDAssistantPatientContext(
        ageYears=getattr(patient, "age_years", None),
        sex=getattr(patient, "sex", None),
        totalBodyWeightKg=getattr(patient, "total_body_weight_kg", None),
        heightCm=getattr(patient, "height_cm", None),
        serumCreatinineMgDl=getattr(patient, "serum_creatinine_mg_dl", None),
        crclMlMin=getattr(patient, "crcl_ml_min", None),
        renalMode=renal_mode,
    )


def _doseid_standard_renal_inputs_required(medication_id: str, indication_id: str | None = None) -> bool:
    if medication_id == "fosfomycin" and indication_id == "uncomplicated_cystitis":
        return False
    return medication_id not in DOSEID_STANDARD_NONRENAL_MEDICATION_IDS


def _doseid_requires_weight(medication_id: str, indication_id: str) -> bool:
    if medication_id in DOSEID_ALWAYS_WEIGHT_BASED_MEDICATION_IDS:
        return True
    if medication_id == "tmp_smx" and indication_id in DOSEID_WEIGHT_BASED_TMP_SMX_INDICATION_IDS:
        return True
    if medication_id == "ceftriaxone" and indication_id == "standard_dose":
        return True
    return False


def _doseid_requires_height(medication_id: str, indication_id: str) -> bool:
    if medication_id in DOSEID_HEIGHT_REQUIRED_MEDICATION_IDS:
        return True
    if medication_id == "tmp_smx" and indication_id in DOSEID_WEIGHT_BASED_TMP_SMX_INDICATION_IDS:
        return True
    if medication_id == "ceftriaxone" and indication_id == "standard_dose":
        return True
    return False


def _doseid_requires_sex(medication_id: str) -> bool:
    return medication_id in DOSEID_SEX_REQUIRED_MEDICATION_IDS


def _doseid_add_missing_reason(
    reasons_by_field: Dict[str, List[str]],
    field_id: str,
    reason: str,
) -> None:
    bucket = reasons_by_field.setdefault(field_id, [])
    if reason not in bucket:
        bucket.append(reason)


def _doseid_reason_text(reasons: List[str]) -> str:
    if not reasons:
        return ""
    if len(reasons) == 1:
        return reasons[0]
    return ", ".join(reasons[:-1]) + f", and {reasons[-1]}"


def _doseid_follow_up_question(field_id: str, reasons: List[str]) -> DoseIDFollowUpQuestion:
    reason_text = _doseid_reason_text(reasons)
    if field_id == "medication":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="Which medication or regimen do you want dosed?",
            reason="The dose pathway starts with the drug or regimen.",
        )
    if field_id == "renal_function":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="What is the serum creatinine, or do you already have a creatinine clearance (CrCl)?",
            reason=f"I need renal function to place the regimen in the correct renal bucket for {reason_text}.",
        )
    if field_id == "serum_creatinine":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="What is the serum creatinine?",
            reason=f"I need the serum creatinine for {reason_text}.",
        )
    if field_id == "age":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="How old is the patient? If you already have a Cockcroft-Gault CrCl, you can send that instead.",
            reason=f"I need age for {reason_text}.",
        )
    if field_id == "sex":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="What is the patient's sex? If you already have a Cockcroft-Gault CrCl, you can send that instead.",
            reason=f"I need sex for {reason_text}.",
        )
    if field_id == "weight":
        return DoseIDFollowUpQuestion(
            id=field_id,
            prompt="What is the patient's weight in kg?",
            reason=f"I need weight for {reason_text}.",
        )
    return DoseIDFollowUpQuestion(
        id=field_id,
        prompt="What is the patient's height in cm?",
        reason=f"I need height for {reason_text}.",
    )


def _doseid_missing_input_questions(
    *,
    medication_ids: List[str],
    indication_ids: Dict[str, str],
    patient_context: DoseIDAssistantPatientContext,
) -> List[DoseIDFollowUpQuestion]:
    reasons_by_field: Dict[str, List[str]] = {}
    meds_by_id = _assistant_doseid_medications_by_id()
    if not medication_ids:
        _doseid_add_missing_reason(reasons_by_field, "medication", "the requested dosing pathway")
    for medication_id in medication_ids:
        indication_id = indication_ids.get(medication_id) or default_indication_id(medication_id)
        medication_label = meds_by_id.get(medication_id).name if meds_by_id.get(medication_id) is not None else medication_id.replace("_", " ")
        if _doseid_requires_weight(medication_id, indication_id) and patient_context.total_body_weight_kg is None:
            _doseid_add_missing_reason(reasons_by_field, "weight", f"{medication_label} dosing")
        if _doseid_requires_height(medication_id, indication_id) and patient_context.height_cm is None:
            _doseid_add_missing_reason(reasons_by_field, "height", f"{medication_label} body-size adjustment")
        if _doseid_requires_sex(medication_id) and patient_context.sex is None:
            _doseid_add_missing_reason(reasons_by_field, "sex", f"{medication_label} body-size adjustment")

        if patient_context.renal_mode != "standard":
            continue

        if medication_id == "foscarnet":
            if patient_context.serum_creatinine_mg_dl is None:
                _doseid_add_missing_reason(reasons_by_field, "serum_creatinine", "foscarnet adjusted creatinine clearance")
            if patient_context.age_years is None:
                _doseid_add_missing_reason(reasons_by_field, "age", "foscarnet adjusted creatinine clearance")
            if patient_context.sex is None:
                _doseid_add_missing_reason(reasons_by_field, "sex", "foscarnet adjusted creatinine clearance")
            if patient_context.total_body_weight_kg is None:
                _doseid_add_missing_reason(reasons_by_field, "weight", "foscarnet mg/kg dosing")
            if patient_context.height_cm is None:
                _doseid_add_missing_reason(reasons_by_field, "height", "foscarnet obesity-adjusted dosing weight")
            continue

        if not _doseid_standard_renal_inputs_required(medication_id, indication_id):
            continue
        if patient_context.crcl_ml_min is not None:
            continue
        if patient_context.serum_creatinine_mg_dl is None:
            _doseid_add_missing_reason(reasons_by_field, "renal_function", f"{medication_label} renal dosing")
            continue
        if patient_context.age_years is None:
            _doseid_add_missing_reason(reasons_by_field, "age", "Cockcroft-Gault renal dosing")
        if patient_context.sex is None:
            _doseid_add_missing_reason(reasons_by_field, "sex", "Cockcroft-Gault renal dosing")
        if patient_context.total_body_weight_kg is None:
            _doseid_add_missing_reason(reasons_by_field, "weight", "Cockcroft-Gault renal dosing")

    return [
        _doseid_follow_up_question(field_id, reasons_by_field[field_id])
        for field_id in DOSEID_FOLLOW_UP_FIELD_ORDER
        if field_id in reasons_by_field
    ]


def _doseid_recommendations_ready(
    *,
    medication_ids: List[str],
    indication_ids: Dict[str, str],
    patient_context: DoseIDAssistantPatientContext,
) -> tuple[List[DoseIDDoseRecommendation], List[str], List[str]]:
    warnings: List[str] = []
    recommendations: List[DoseIDDoseRecommendation] = []
    assumptions: List[str] = []
    needs_standard_renal_inputs = patient_context.renal_mode == "standard" and any(
        medication_id == "foscarnet"
        or _doseid_standard_renal_inputs_required(medication_id, indication_ids.get(medication_id))
        for medication_id in medication_ids
    )
    if needs_standard_renal_inputs or patient_context.crcl_ml_min is not None or patient_context.serum_creatinine_mg_dl is not None:
        patient, assumptions = normalize_patient_from_available_inputs(
            total_body_weight_kg=patient_context.total_body_weight_kg,
            age_years=patient_context.age_years,
            sex=patient_context.sex,
            height_cm=patient_context.height_cm,
            serum_creatinine_mg_dl=patient_context.serum_creatinine_mg_dl,
            crcl_ml_min=patient_context.crcl_ml_min,
            renal_mode=patient_context.renal_mode,
        )
    else:
        patient, assumptions = normalize_patient_from_available_inputs(
            total_body_weight_kg=patient_context.total_body_weight_kg or 70.0,
            age_years=patient_context.age_years or 50,
            sex=patient_context.sex or "male",
            height_cm=patient_context.height_cm or 170.0,
            serum_creatinine_mg_dl=1.0,
            renal_mode="ihd",
        )
    if patient_context.crcl_ml_min is not None and patient_context.serum_creatinine_mg_dl is None:
        warnings.append("Direct CrCl was used for renal bucketing. If the underlying estimate is uncertain, confirm the final regimen manually.")
    for medication_id in medication_ids[:6]:
        payload = calculate_medication(
            medication_id=medication_id,
            patient=patient,
            renal_mode=patient_context.renal_mode,
            indication_id=indication_ids.get(medication_id),
        )
        recommendations.append(_doseid_recommendation_model(payload))
    return recommendations, assumptions, warnings


def _doseid_merge_patient_context(
    message_text: str,
    local_context: DoseIDAssistantPatientContext,
    llm_context: Dict[str, Any],
) -> DoseIDAssistantPatientContext:
    normalized = _assistant_doseid_normalize(message_text)
    mentions_age = _assistant_text_mentions_doseid_age(normalized)
    mentions_crcl = _assistant_text_mentions_doseid_crcl(normalized)
    mentions_scr = _assistant_text_mentions_doseid_scr(normalized)
    mentions_ihd = _assistant_text_mentions_doseid_ihd(normalized)
    mentions_crrt = _assistant_text_mentions_doseid_crrt(normalized)
    llm_renal_mode = str(llm_context.get("renalMode") or "standard")
    llm_age = llm_context.get("ageYears") if mentions_age else None
    llm_scr = llm_context.get("serumCreatinineMgDl") if mentions_scr else None
    llm_crcl = llm_context.get("crclMlMin") if mentions_crcl else None
    if (
        llm_crcl is not None
        and local_context.age_years is not None
        and local_context.crcl_ml_min is None
        and float(llm_crcl) == float(local_context.age_years)
    ):
        llm_crcl = None
    if mentions_crrt:
        llm_renal_mode = "crrt"
    elif mentions_ihd:
        llm_renal_mode = "ihd"
    else:
        llm_renal_mode = "standard"

    return DoseIDAssistantPatientContext(
        ageYears=local_context.age_years if local_context.age_years is not None else llm_age,
        sex=local_context.sex if local_context.sex is not None else llm_context.get("sex"),
        totalBodyWeightKg=(
            local_context.total_body_weight_kg
            if local_context.total_body_weight_kg is not None
            else llm_context.get("totalBodyWeightKg")
        ),
        heightCm=local_context.height_cm if local_context.height_cm is not None else llm_context.get("heightCm"),
        serumCreatinineMgDl=(
            local_context.serum_creatinine_mg_dl
            if local_context.serum_creatinine_mg_dl is not None
            else llm_scr
        ),
        crclMlMin=local_context.crcl_ml_min if local_context.crcl_ml_min is not None else llm_crcl,
        renalMode=local_context.renal_mode if local_context.renal_mode != "standard" else llm_renal_mode,
    )


def _doseid_rule_parse_payload(text: str) -> Dict[str, Any]:
    medication_ids = _assistant_detect_doseid_medication_ids(text)
    patient_context = _assistant_parse_doseid_patient_context(text)
    selections = [
        {
            "medicationId": medication_id,
            "indicationId": _assistant_doseid_indication_for_query(medication_id, text),
        }
        for medication_id in medication_ids
    ]
    return {
        "medications": selections,
        "patientContext": patient_context.model_dump(by_alias=True),
        "requiresConfirmation": not medication_ids,
    }


def _doseid_merge_medication_selections(
    local_selections: List[Dict[str, Any]],
    llm_selections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in local_selections:
        medication_id = str(item.get("medicationId") or "").strip()
        if not medication_id:
            continue
        merged[medication_id] = {"medicationId": medication_id, "indicationId": item.get("indicationId")}
    for item in llm_selections:
        medication_id = str(item.get("medicationId") or "").strip()
        if not medication_id:
            continue
        indication_id = item.get("indicationId")
        if medication_id not in merged:
            merged[medication_id] = {"medicationId": medication_id, "indicationId": indication_id}
            continue
        if indication_id and (
            not merged[medication_id].get("indicationId")
            or merged[medication_id].get("indicationId") == default_indication_id(medication_id)
        ):
            merged[medication_id]["indicationId"] = indication_id
    return list(merged.values())


def _build_doseid_text_response(
    text: str,
    *,
    parser_strategy: str = "auto",
    parser_model: str | None = None,
    allow_fallback: bool = True,
) -> DoseIDTextAnalyzeResponse:
    warnings: List[str] = []
    parser_fallback_used = False
    local_payload = _doseid_rule_parse_payload(text)
    selected_payload: Dict[str, Any] | None = None
    parser_name = "rule-based-v1"

    if parser_strategy == "rule":
        selected_payload = local_payload
    elif parser_strategy == "openai":
        try:
            llm_payload = parse_doseid_text_with_openai(text=text, parser_model=parser_model)
            selected_payload = {
                "medications": _doseid_merge_medication_selections(
                    local_payload.get("medications", []),
                    list(llm_payload.get("medications", [])),
                ),
                "patientContext": _doseid_merge_patient_context(
                    text,
                    DoseIDAssistantPatientContext.model_validate(local_payload.get("patientContext", {})),
                    dict(llm_payload.get("patientContext") or {}),
                ).model_dump(by_alias=True),
                "requiresConfirmation": bool(llm_payload.get("requiresConfirmation") or not llm_payload.get("medications")),
            }
            parser_name = str(llm_payload.get("parser") or "openai-doseid")
            for ambiguity in llm_payload.get("ambiguities", []):
                warnings.append(f"OpenAI DoseID parser note: {ambiguity}")
        except LLMParserError as exc:
            if not allow_fallback:
                return DoseIDTextAnalyzeResponse(
                    text=text,
                    parsedRequest=None,
                    warnings=[f"OpenAI DoseID parser failed: {exc}"],
                    requiresConfirmation=True,
                    parser="openai-doseid",
                    parserFallbackUsed=False,
                    analysis=None,
                )
            selected_payload = local_payload
            parser_fallback_used = True
            warnings.append(f"OpenAI DoseID parser unavailable/failed, used rule parser fallback: {exc}")
    else:
        openai_err: str | None = None
        if (os.getenv("OPENAI_API_KEY") or "").strip():
            try:
                llm_payload = parse_doseid_text_with_openai(text=text, parser_model=parser_model)
                selected_payload = {
                    "medications": _doseid_merge_medication_selections(
                        local_payload.get("medications", []),
                        list(llm_payload.get("medications", [])),
                    ),
                    "patientContext": _doseid_merge_patient_context(
                        DoseIDAssistantPatientContext.model_validate(local_payload.get("patientContext", {})),
                        dict(llm_payload.get("patientContext") or {}),
                    ).model_dump(by_alias=True),
                    "requiresConfirmation": bool(llm_payload.get("requiresConfirmation") or not llm_payload.get("medications")),
                }
                parser_name = str(llm_payload.get("parser") or "openai-doseid")
                for ambiguity in llm_payload.get("ambiguities", []):
                    warnings.append(f"OpenAI DoseID parser note: {ambiguity}")
            except LLMParserError as exc:
                openai_err = str(exc)
        if selected_payload is None:
            selected_payload = local_payload
            parser_name = "rule-based-v1"
            if openai_err:
                parser_fallback_used = True
                warnings.append(f"OpenAI DoseID parser unavailable/failed, used rule parser fallback: {openai_err}")

    parsed_request = DoseIDTextParsedRequest(
        medications=[
            {
                "medicationId": item.get("medicationId"),
                "indicationId": item.get("indicationId"),
            }
            for item in selected_payload.get("medications", [])
            if item.get("medicationId")
        ],
        patientContext=selected_payload.get("patientContext", {}),
    )
    analysis = _assistant_build_doseid_analysis(
        text,
        parser_strategy=parser_strategy,
        parser_model=parser_model,
        allow_fallback=allow_fallback,
    )
    requires_confirmation = bool(selected_payload.get("requiresConfirmation")) or not parsed_request.medications
    if analysis.follow_up_questions:
        requires_confirmation = True
    return DoseIDTextAnalyzeResponse(
        parser=parser_name,
        text=text,
        parsedRequest=parsed_request,
        warnings=warnings,
        requiresConfirmation=requires_confirmation,
        parserFallbackUsed=parser_fallback_used,
        analysis=analysis,
    )


def _assistant_doseid_simple_numeric_reply(reply: str) -> str | None:
    normalized = _assistant_doseid_normalize(reply)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*(mg/?dl|mg dl|ml/?min|ml min|kg|kgs|kilograms?|kilos?|cm|lb|lbs))?", normalized)
    if not match:
        return None
    number = match.group(1)
    unit = match.group(2) or ""
    if unit in {"mg/dl", "mg dl"}:
        unit = "mg/dl"
    elif unit in {"ml/min", "ml min"}:
        unit = "ml/min"
    elif unit in {"kgs", "kilogram", "kilograms", "kilo", "kilos"}:
        unit = "kg"
    return f"{number} {unit}".strip()


def _assistant_doseid_followup_field_ids(existing_text: str | None) -> List[str]:
    prior_analysis = _assistant_build_doseid_analysis(existing_text or "", parser_strategy="rule")
    return [question.id for question in prior_analysis.follow_up_questions]


def _assistant_doseid_multi_field_reply_fragments(
    reply: str,
    missing_field_ids: List[str],
) -> tuple[List[str], set[str]]:
    normalized = _assistant_doseid_normalize(reply)
    fragments: Dict[str, str] = {}
    explicit_fields: set[str] = set()

    def remember(field_id: str, fragment: str | None, *, explicit: bool = False) -> None:
        if field_id in missing_field_ids and fragment and field_id not in fragments:
            fragments[field_id] = fragment.strip()
            if explicit:
                explicit_fields.add(field_id)

    age_match = re.search(
        r"\bage\s*(?:is|=|:)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:yo|y o|y/o|yr old|yrs old|year old|years old)\b",
        normalized,
    )
    if age_match:
        remember("age", f"age {age_match.group(1) or age_match.group(2)}", explicit=True)

    sex_match = re.search(
        r"\b(?:sex|gender)\s*(?:is|=|:)?\s*(male|female|man|woman|m|f)\b",
        normalized,
    )
    if sex_match:
        sex_token = str(sex_match.group(1) or "").strip()
        remember("sex", "male" if sex_token in {"male", "man", "m"} else "female", explicit=True)
    elif re.search(r"\bmale\b|\bman\b", normalized):
        remember("sex", "male", explicit=True)
    elif re.search(r"\bfemale\b|\bwoman\b", normalized):
        remember("sex", "female", explicit=True)

    weight_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|kilos?)\b", normalized)
    if weight_match:
        remember("weight", f"{weight_match.group(1)} kg", explicit=True)
    else:
        pounds_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b", normalized)
        if pounds_match:
            remember("weight", f"{pounds_match.group(1)} lb", explicit=True)

    height_match = re.search(r"\b(\d+(?:\.\d+)?)\s*cm\b", normalized)
    if height_match:
        remember("height", f"{height_match.group(1)} cm", explicit=True)
    else:
        feet_inches_match = re.search(r"\b(\d)\s*(?:ft|feet|foot)\s*(\d{1,2})?\s*(?:in|inch|inches)?\b", normalized)
        if feet_inches_match:
            feet = feet_inches_match.group(1)
            inches = feet_inches_match.group(2) or "0"
            remember("height", f"{feet} ft {inches} in", explicit=True)
        else:
            feet_inches_quote_match = re.search(r"\b(\d)\s*'\s*(\d{1,2})\s*(?:\"|in|inch|inches)?\b", reply)
            if feet_inches_quote_match:
                remember(
                    "height",
                    f"{feet_inches_quote_match.group(1)} ft {feet_inches_quote_match.group(2)} in",
                    explicit=True,
                )

    crcl_match = re.search(
        r"\b(?:crcl|creatinine clearance)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if crcl_match:
        remember("renal_function", f"crcl {crcl_match.group(1)}", explicit=True)

    scr_match = re.search(
        r"\b(?:scr|s\s*cr|serum creatinine|creatinine|cr)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\b",
        normalized,
    )
    if scr_match:
        scr_fragment = f"serum creatinine {scr_match.group(1)}"
        remember("serum_creatinine", scr_fragment, explicit=True)
        remember("renal_function", scr_fragment, explicit=True)

    chunks = [chunk.strip() for chunk in re.split(r"[,;\n]+", reply) if chunk.strip()]
    bare_numeric_chunks: List[tuple[str, str, float, bool]] = []
    for chunk in chunks:
        chunk_norm = _assistant_doseid_normalize(chunk)
        if not chunk_norm:
            continue
        if chunk_norm in {"male", "man", "m"}:
            remember("sex", "male", explicit=True)
            continue
        if chunk_norm in {"female", "woman", "f"}:
            remember("sex", "female", explicit=True)
            continue
        simple = _assistant_doseid_simple_numeric_reply(chunk)
        if simple is None:
            continue
        simple_parts = simple.split()
        value = simple_parts[0]
        unit = simple_parts[1] if len(simple_parts) > 1 else ""
        if unit in {"kg", "lb", "lbs"}:
            remember("weight", f"{value} {unit}", explicit=True)
            continue
        if unit == "cm":
            remember("height", f"{value} cm", explicit=True)
            continue
        if unit in {"mg/dl", "mg", "dl"}:
            fragment = f"serum creatinine {value}"
            remember("serum_creatinine", fragment, explicit=True)
            remember("renal_function", fragment, explicit=True)
            continue
        if unit in {"ml/min", "ml", "min"}:
            remember("renal_function", f"crcl {value}", explicit=True)
            continue
        bare_numeric_chunks.append((chunk, value, float(value), "." in value))

    used_numeric_indexes: set[int] = set()

    def remember_bare_numeric(index: int, field_id: str, fragment: str) -> None:
        used_numeric_indexes.add(index)
        remember(field_id, fragment)

    remaining_field_ids = [field_id for field_id in missing_field_ids if field_id not in fragments]
    if bare_numeric_chunks and remaining_field_ids:
        if "age" in remaining_field_ids:
            for index, (_chunk, value, numeric_value, is_decimal) in enumerate(bare_numeric_chunks):
                if index in used_numeric_indexes or is_decimal:
                    continue
                if numeric_value.is_integer() and 0 < numeric_value <= 120:
                    remember_bare_numeric(index, "age", f"age {value}")
                    break

        remaining_field_ids = [field_id for field_id in missing_field_ids if field_id not in fragments]
        if "height" in remaining_field_ids:
            for index, (_chunk, value, numeric_value, _is_decimal) in enumerate(bare_numeric_chunks):
                if index in used_numeric_indexes:
                    continue
                if 100 <= numeric_value <= 250:
                    remember_bare_numeric(index, "height", f"{value} cm")
                    break

        remaining_field_ids = [field_id for field_id in missing_field_ids if field_id not in fragments]
        if "weight" in remaining_field_ids:
            for index, (_chunk, value, numeric_value, _is_decimal) in enumerate(bare_numeric_chunks):
                if index in used_numeric_indexes:
                    continue
                if 20 <= numeric_value <= 350:
                    remember_bare_numeric(index, "weight", f"{value} kg")
                    break

        remaining_field_ids = [field_id for field_id in missing_field_ids if field_id not in fragments]
        renal_field_id = "serum_creatinine" if "serum_creatinine" in remaining_field_ids else None
        if renal_field_id is None and "renal_function" in remaining_field_ids:
            renal_field_id = "renal_function"
        if renal_field_id is not None:
            chosen_index: int | None = None
            renal_fragment: str | None = None
            for index, (_chunk, value, numeric_value, is_decimal) in enumerate(bare_numeric_chunks):
                if index in used_numeric_indexes:
                    continue
                if is_decimal and numeric_value < 20:
                    chosen_index = index
                    renal_fragment = f"serum creatinine {value}"
                    break
            if chosen_index is None:
                for index, (_chunk, value, numeric_value, _is_decimal) in enumerate(bare_numeric_chunks):
                    if index in used_numeric_indexes:
                        continue
                    if numeric_value >= 20:
                        chosen_index = index
                        renal_fragment = f"crcl {value}"
                        break
            if chosen_index is not None and renal_fragment is not None:
                used_numeric_indexes.add(chosen_index)
                remember(renal_field_id, renal_fragment)
                if renal_field_id == "renal_function" and renal_fragment.startswith("serum creatinine"):
                    remember("serum_creatinine", renal_fragment)

    ordered_fragments = []
    for field_id in missing_field_ids:
        fragment = fragments.get(field_id)
        if fragment and fragment not in ordered_fragments:
            ordered_fragments.append(fragment)
    return ordered_fragments, explicit_fields


def _assistant_rewrite_doseid_followup_reply(existing_text: str | None, reply: str) -> str:
    extra = (reply or "").strip()
    if not extra:
        return extra
    missing_field_ids = _assistant_doseid_followup_field_ids(existing_text)
    if not missing_field_ids:
        return extra
    multi_field_fragments, explicit_fields = _assistant_doseid_multi_field_reply_fragments(extra, missing_field_ids)
    if len(multi_field_fragments) >= 2:
        return ". ".join(multi_field_fragments)
    if len(multi_field_fragments) == 1 and explicit_fields:
        return multi_field_fragments[0]

    question_id = missing_field_ids[0]
    normalized = _assistant_doseid_normalize(extra)
    numeric_value = _assistant_doseid_simple_numeric_reply(extra)
    if question_id == "serum_creatinine" and numeric_value is not None:
        return f"serum creatinine {numeric_value.split()[0]}"
    if question_id == "renal_function":
        number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", normalized)
        if number_match:
            value = number_match.group(1)
            if re.search(r"\b(crcl|creatinine clearance|ml/?min)\b", normalized):
                return f"crcl {value}"
            if re.search(r"\b(scr|serum creatinine|creatinine|mg/?dl)\b", normalized):
                return f"serum creatinine {value}"
            return f"serum creatinine {value}" if float(value) < 20 else f"crcl {value}"
    if question_id == "age" and numeric_value is not None:
        return f"age {numeric_value.split()[0]}"
    if question_id == "weight" and numeric_value is not None:
        if re.search(r"\b(lb|lbs)\b", normalized):
            return numeric_value
        return f"{numeric_value.split()[0]} kg"
    if question_id == "height" and numeric_value is not None:
        return f"{numeric_value.split()[0]} cm"
    return extra


def _assistant_build_doseid_analysis(
    message_text: str,
    *,
    parser_strategy: str = "auto",
    parser_model: str | None = None,
    allow_fallback: bool = True,
    force_best_effort: bool = False,
    session_patient_context: SessionPatientContext | None = None,
) -> DoseIDAssistantAnalysis:
    medication_ids = _assistant_detect_doseid_medication_ids(message_text)
    patient_context = _assistant_parse_doseid_patient_context(message_text)
    warnings: List[str] = []
    indication_ids: Dict[str, str] = {
        medication_id: _assistant_doseid_indication_for_query(medication_id, message_text)
        for medication_id in medication_ids
    }
    if parser_strategy in {"auto", "openai"} and (os.getenv("OPENAI_API_KEY") or "").strip():
        try:
            llm_payload = parse_doseid_text_with_openai(text=message_text, parser_model=parser_model)
            patient_context = _doseid_merge_patient_context(
                message_text,
                patient_context,
                dict(llm_payload.get("patientContext") or {}),
            )
            for item in llm_payload.get("medications", []):
                medication_id = str(item.get("medicationId") or "").strip()
                if not medication_id:
                    continue
                if medication_id not in medication_ids:
                    medication_ids.append(medication_id)
                indication_id = str(item.get("indicationId") or "").strip()
                if indication_id and (
                    medication_id not in indication_ids
                    or indication_ids[medication_id] == default_indication_id(medication_id)
                ):
                    indication_ids[medication_id] = indication_id
            for ambiguity in llm_payload.get("ambiguities", []):
                warnings.append(f"OpenAI DoseID parser note: {ambiguity}")
        except LLMParserError as exc:
            if parser_strategy == "openai" and not allow_fallback:
                warnings.append(f"OpenAI DoseID parser failed: {exc}")
            else:
                warnings.append(f"OpenAI DoseID parser unavailable/failed, used rule parser fallback: {exc}")
    # Lowest-priority fallback: fill any remaining gaps from the session patient context
    if session_patient_context is not None:
        sc = session_patient_context
        if patient_context.age_years is None and sc.age_years is not None:
            patient_context = patient_context.model_copy(update={"age_years": sc.age_years})
        if patient_context.sex is None and sc.sex is not None:
            patient_context = patient_context.model_copy(update={"sex": sc.sex})
        if patient_context.total_body_weight_kg is None and sc.total_body_weight_kg is not None:
            patient_context = patient_context.model_copy(update={"total_body_weight_kg": sc.total_body_weight_kg})
        if patient_context.height_cm is None and sc.height_cm is not None:
            patient_context = patient_context.model_copy(update={"height_cm": sc.height_cm})
        if patient_context.serum_creatinine_mg_dl is None and sc.serum_creatinine_mg_dl is not None:
            patient_context = patient_context.model_copy(update={"serum_creatinine_mg_dl": sc.serum_creatinine_mg_dl})
        if patient_context.crcl_ml_min is None and sc.crcl_ml_min is not None:
            patient_context = patient_context.model_copy(update={"crcl_ml_min": sc.crcl_ml_min})
        if patient_context.renal_mode == "standard" and sc.renal_mode != "standard":
            patient_context = patient_context.model_copy(update={"renal_mode": sc.renal_mode})
    follow_up_questions = _doseid_missing_input_questions(
        medication_ids=medication_ids,
        indication_ids=indication_ids,
        patient_context=patient_context,
    )
    missing_inputs = [DOSEID_FIELD_LABELS[item.id] for item in follow_up_questions]

    recommendations: List[DoseIDDoseRecommendation] = []
    assumptions: List[str] = []
    provisional = False
    provisional_reasons: List[str] = []
    if medication_ids and not follow_up_questions:
        recommendations, assumptions, warnings = _doseid_recommendations_ready(
            medication_ids=medication_ids,
            indication_ids=indication_ids,
            patient_context=patient_context,
        )
    elif medication_ids and force_best_effort:
        best_effort_context = patient_context.model_copy(deep=True)
        if (
            best_effort_context.renal_mode == "standard"
            and best_effort_context.crcl_ml_min is None
            and best_effort_context.serum_creatinine_mg_dl is None
        ):
            best_effort_context.serum_creatinine_mg_dl = 1.0
            assumptions.append(
                "Renal function was not provided, so serum creatinine 1.0 mg/dL was used only to generate a provisional dosing estimate."
            )
        recommendations, extra_assumptions, extra_warnings = _doseid_recommendations_ready(
            medication_ids=medication_ids,
            indication_ids=indication_ids,
            patient_context=best_effort_context,
        )
        assumptions.extend(extra_assumptions)
        warnings.extend(extra_warnings)
        provisional = True
        provisional_reasons = [question.prompt for question in follow_up_questions[:3]]
        assumptions.insert(
            0,
            "This is a provisional dosing estimate generated with missing inputs still outstanding. Confirm the final regimen before use.",
        )

    medication_names = [
        _assistant_doseid_medications_by_id().get(med_id).name
        for med_id in medication_ids
        if _assistant_doseid_medications_by_id().get(med_id) is not None
    ]
    return DoseIDAssistantAnalysis(
        medications=medication_names,
        patientContext=patient_context,
        recommendations=recommendations,
        assumptions=assumptions,
        warnings=warnings,
        missingInputs=missing_inputs,
        followUpQuestions=follow_up_questions,
        provisional=provisional,
        provisionalReasons=provisional_reasons,
    )


def _assistant_doseid_message(result: DoseIDAssistantAnalysis) -> str:
    if not result.medications:
        return (
            "I can help with antimicrobial dosing here. Tell me the medication or regimen plus the renal context, "
            "for example 'cefepime dosing on hemodialysis' or 'RIPE dosing for 62 kg, CrCl 35'."
        )
    if result.provisional and result.recommendations:
        recommendation_summary = "; ".join(
            f"{item.medication_name}: {item.regimen}" for item in result.recommendations[:4]
        )
        missing_reason = (
            f" I am still missing {', '.join(result.missing_inputs[:3])}."
            if result.missing_inputs
            else ""
        )
        return (
            "I generated a provisional dosing estimate using placeholder assumptions because key inputs are still missing. "
            + recommendation_summary
            + missing_reason
        )
    if result.follow_up_questions:
        meds = ", ".join(result.medications)
        primary_question = result.follow_up_questions[0].prompt
        if len(result.follow_up_questions) == 1:
            return (
                f"I picked up {meds}. {primary_question} "
                "If you do not have that yet, I can still give you a provisional estimate with explicit assumptions."
            )
        remaining = ", ".join(DOSEID_FIELD_LABELS[item.id] for item in result.follow_up_questions[1:])
        return (
            f"I picked up {meds}. {primary_question} After that, I’ll also need {remaining}. "
            "If you do not have those yet, I can still give you a provisional estimate with explicit assumptions."
        )
    if not result.recommendations:
        return "I picked up the dosing question, but I still need a little more clinical detail before I can calculate a regimen."
    recommendation_summary = "; ".join(
        f"{item.medication_name}: {item.regimen}" for item in result.recommendations[:4]
    )
    caveats = " A few missing values were scaffolded, so please review the assumptions panel." if result.assumptions else ""
    return f"{recommendation_summary}.{caveats}"


def _assistant_doseid_response(
    state: AssistantState,
    *,
    message_text: str,
    prefix: str = "",
    force_best_effort: bool = False,
) -> AssistantTurnResponse:
    state.workflow = "doseid"
    state.stage = "doseid_describe"
    state.module_id = None
    state.preset_id = None
    state.case_section = None
    state.case_text = None
    state.mechid_text = None
    state.doseid_text = message_text
    state.pretest_factor_ids = []
    state.pretest_factor_labels = []
    state.endo_blood_culture_context = None
    state.endo_score_factor_ids = []
    _assistant_reset_immunoid_state(state)
    result = _assistant_build_doseid_analysis(
        message_text,
        parser_strategy=state.parser_strategy,
        parser_model=state.parser_model,
        allow_fallback=state.allow_fallback,
        force_best_effort=force_best_effort,
        session_patient_context=state.patient_context,
    )
    # Snapshot for consult summary
    if result.medications:
        med = result.medications[0]
        _snapshot_doseid_result(state, {
            "drug": med.name if hasattr(med, "name") else None,
            "route": getattr(med, "route", None),
        })
    fallback_message = ((prefix or "") + _assistant_doseid_message(result)).strip()
    message, narration_refined = narrate_doseid_assistant_message(
        doseid_result=result,
        fallback_message=fallback_message,
        established_syndrome=state.established_syndrome,
        consult_organisms=state.consult_organisms or None,
        prior_context_summary=_consult_prior_context_summary(state),
    )
    # Build next-step options based on whether dosing is complete or still provisional
    if result.follow_up_questions and result.medications:
        # Provisional: still missing inputs — primary action is to complete the calculation
        doseid_options: list[AssistantOption] = [
            AssistantOption(value="run_assessment", label="Run provisional dosing"),
            AssistantOption(value="add_more_details", label="Update dosing inputs"),
        ]
        doseid_tips: list[str] = [
            "If you do not have the next missing value yet, use Run provisional dosing and I will keep the missing inputs visible.",
            "You can add age, sex, height, weight, serum creatinine, or direct CrCl to refine the regimen.",
        ]
    else:
        # Calculation complete — offer clinically relevant next steps
        exclude_next = {"source_control", "followup_tests"}  # avoid duplicates with syndrome options below
        syndrome_opts = _syndrome_next_step_options(state, exclude=exclude_next)
        doseid_options = [
            AssistantOption(value="iv_to_oral", label="IV-to-oral step-down?"),
            AssistantOption(value="opat", label="OPAT eligibility"),
            AssistantOption(value="duration", label="How long to treat?"),
            *syndrome_opts,
            AssistantOption(value="add_more_details", label="Update dosing inputs"),
        ]
        if _is_mid_consult(state):
            doseid_options.append(AssistantOption(value="consult_summary", label="Full consult summary"))
        else:
            doseid_options.append(AssistantOption(value="restart", label="Start new consult"))
        doseid_tips = [
            "A useful reply would be: 'cefepime on HD' or 'RIPE for 62 kg, CrCl 35, female, 165 cm'.",
            "Ask about IV-to-oral step-down once the patient is clinically stable and afebrile for 48h.",
        ]

    return AssistantTurnResponse(
        assistantMessage=message,
        assistantNarrationRefined=narration_refined,
        state=state,
        doseidAnalysis=result,
        options=doseid_options,
        tips=doseid_tips,
    )


def _assistant_start_doseid_from_text(
    message_text: str,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    if not message_text or not _assistant_is_doseid_intent(message_text):
        return None
    return _assistant_doseid_response(state, message_text=message_text)


def _assistant_unique_medication_ids(medication_ids: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for medication_id in medication_ids:
        if medication_id in seen:
            continue
        seen.add(medication_id)
        unique.append(medication_id)
    return unique


def _assistant_build_doseid_followup_prompt(
    medication_ids: List[str],
    *,
    case_text: str | None,
) -> str | None:
    medication_ids = _assistant_unique_medication_ids(medication_ids)
    if not medication_ids:
        return None
    meds_by_id = _assistant_doseid_medications_by_id()
    medication_names = [meds_by_id[item].name for item in medication_ids if item in meds_by_id]
    if not medication_names:
        return None
    if medication_ids[:4] == ["rifampin", "isoniazid", "pyrazinamide", "ethambutol"]:
        prompt = "Please calculate RIPE dosing."
    else:
        prompt = f"Please calculate dosing for {_join_readable(medication_names)}."
    if case_text and case_text.strip():
        prompt += f" Case context: {case_text.strip()}"
    return prompt


def _assistant_append_dosing_invitation(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return "If you want, I can also calculate the right dose for this patient next."
    normalized = _assistant_doseid_normalize(text)
    if any(token in normalized for token in ("dose", "dosing", "doseid")):
        return text
    suffix = " If you want, I can also calculate the right dose for this patient next."
    if text.endswith(("?", "!", ".")):
        return text + suffix
    return text + "." + suffix


# ── Cross-module suggestion helpers ─────────────────────────────────────────

# Map ImmunoID recommendation IDs to DoseID medication IDs for prophylaxis dosing suggestions.
_IMMUNOID_REC_TO_DOSEID: Dict[str, str] = {
    "pjp_prophylaxis_high_risk_agents": "tmp_smx",
    "pjp_prophylaxis_combination_review": "tmp_smx",
    "pjp_prophylaxis_steroids": "tmp_smx",
    "tb_positive_manage_before_immunosuppression": "isoniazid",
}


def _immunoid_doseid_options(result: ImmunoAnalyzeResponse) -> List[AssistantOption]:
    """Build DoseID suggestion options from ImmunoID prophylaxis recommendations."""
    meds_by_id = _assistant_doseid_medications_by_id()
    seen: set[str] = set()
    options: List[AssistantOption] = []
    for rec in result.recommendations:
        med_id = _IMMUNOID_REC_TO_DOSEID.get(rec.id)
        if med_id and med_id not in seen and med_id in meds_by_id:
            seen.add(med_id)
            med_name = meds_by_id[med_id].name
            options.append(AssistantOption(
                value=f"immunoid_doseid:{med_id}",
                label=f"Dose {med_name}",
                description=f"Calculate the renal-adjusted dose for {med_name} based on this patient's context.",
            ))
    return options


def _allergyid_doseid_options(result: AntibioticAllergyAnalyzeResponse) -> List[AssistantOption]:
    """Build DoseID suggestion options for preferred alternatives from AllergyID analysis."""
    meds_by_id = _assistant_doseid_medications_by_id()
    seen: set[str] = set()
    options: List[AssistantOption] = []
    for rec in result.recommendations:
        if rec.recommendation != "preferred":
            continue
        agent_text = rec.normalized_agent or rec.agent
        medication_ids = _assistant_detect_doseid_medication_ids(agent_text)
        for med_id in medication_ids[:1]:
            if med_id in seen or med_id not in meds_by_id:
                continue
            seen.add(med_id)
            med_name = meds_by_id[med_id].name
            options.append(AssistantOption(
                value=f"doseid_pick:{med_id}",
                label=f"Dose {med_name}",
                description=f"Calculate the renal-adjusted dose for {med_name}, preferred given the allergy history.",
            ))
    return options[:3]


def _assistant_allergy_check_option(
    state: AssistantState,
    *,
    candidate_drug_names: List[str],
) -> AssistantOption | None:
    """
    Return an allergy-check option when the session has an allergy note and candidate drugs are known.
    """
    if not (state.patient_context and state.patient_context.allergy_text):
        return None
    if not candidate_drug_names:
        return None
    drug_label = _join_readable(candidate_drug_names[:2])
    return AssistantOption(
        value="allergyid_check",
        label=f"Check allergy vs. {drug_label}",
        description=f"Run an allergy compatibility check for {drug_label} given the noted allergy history.",
    )


def _assistant_handle_allergyid_check(
    state: AssistantState,
    *,
    candidate_drug_names: List[str],
) -> AssistantTurnResponse | None:
    """Build and return an AllergyID response using the session allergy note and candidate drugs."""
    if not (state.patient_context and state.patient_context.allergy_text):
        return None
    allergy_text = state.patient_context.allergy_text
    drug_phrase = ", ".join(candidate_drug_names[:3])
    query = f"Allergy history: {allergy_text}. Evaluating: {drug_phrase}."
    return _assistant_allergyid_response(state, message_text=query)


def _assistant_probid_doseid_candidate_ids(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> List[str]:
    analysis = text_result.analysis
    parsed_request = text_result.parsed_request
    case_text = state.case_text or ""
    case_norm = _assistant_doseid_normalize(case_text)
    findings = parsed_request.findings if parsed_request is not None else {}

    if analysis is None or analysis.recommendation != "treat":
        return []

    def _present(item_id: str) -> bool:
        return findings.get(item_id) == "present"

    if module.id == "active_tb":
        return ["rifampin", "isoniazid", "pyrazinamide", "ethambutol"]

    if module.id == "inv_mold":
        mucor_signal = any(
            _present(item_id)
            for item_id in {"imi_mucorales_pcr_bal", "imi_mucorales_pcr_plasma"}
        ) or "mucor" in case_norm
        aspergillus_signal = any(
            _present(item_id)
            for item_id in {
                "imi_serum_gm_odi10",
                "imi_bal_gm_odi10",
                "imi_aspergillus_pcr_bal",
                "imi_aspergillus_pcr_plasma",
                "imi_aspergillus_culture_resp",
            }
        ) or "aspergillus" in case_norm
        if mucor_signal:
            return ["liposomal_amphotericin_b", "isavuconazole", "posaconazole"]
        if aspergillus_signal:
            return ["voriconazole", "isavuconazole", "posaconazole"]
        return ["voriconazole", "liposomal_amphotericin_b", "posaconazole"]

    if module.id == "endo":
        if "mrsa" in case_norm or "methicillin resistant" in case_norm:
            return ["vancomycin_iv", "daptomycin"]
        if "mssa" in case_norm or "methicillin susceptible" in case_norm:
            return ["nafcillin", "cefazolin"]
        if state.endo_blood_culture_context == "staph":
            return ["vancomycin_iv", "cefazolin"]
        if state.endo_blood_culture_context == "enterococcus":
            if "vre" in case_norm or "vancomycin resistant enterococcus" in case_norm:
                return ["daptomycin", "linezolid"]
            return ["ampicillin", "vancomycin_iv"]
        if state.endo_blood_culture_context == "strep":
            return ["penicillin_g", "ceftriaxone"]

    if module.id == "bacterial_meningitis":
        return ["ceftriaxone", "vancomycin_iv"]

    return []


def _assistant_dfi_has_osteomyelitis_signal(text_result: TextAnalyzeResponse, case_text: str | None = None) -> bool:
    parsed_request = text_result.parsed_request
    findings = parsed_request.findings if parsed_request is not None else {}
    if any(
        findings.get(item_id) == "present"
        for item_id in {
            "dfi_probe_to_bone_positive",
            "dfi_exposed_bone",
            "dfi_xray_osteomyelitis",
            "dfi_mri_osteomyelitis_or_abscess",
            "dfi_bone_biopsy_culture_pos",
            "dfi_bone_histology_pos",
            "dfi_positive_bone_margin",
        }
    ):
        return True
    return "osteomyelitis" in _normalize_choice(case_text)


def _assistant_build_probid_mechid_followup_text(
    module: SyndromeModule,
    text_result: TextAnalyzeResponse,
    state: AssistantState,
) -> str | None:
    case_text = (state.case_text or "").strip()
    if not case_text:
        return None

    mechid_result = _build_mechid_text_response(
        case_text,
        parser_strategy=state.parser_strategy,
        parser_model=state.parser_model,
        allow_fallback=state.allow_fallback,
    )
    parsed = mechid_result.parsed_request
    if parsed is None:
        return None
    has_organism_signal = bool(parsed.organism or parsed.mentioned_organisms)
    has_ast_signal = bool(parsed.susceptibility_results)
    if not (has_organism_signal and has_ast_signal):
        return None

    if module.id != "diabetic_foot_infection":
        return case_text

    findings = text_result.parsed_request.findings if text_result.parsed_request is not None else {}
    prefix_parts: List[str] = []
    if _assistant_dfi_has_osteomyelitis_signal(text_result, case_text):
        prefix_parts.append("Diabetic foot osteomyelitis.")
    else:
        prefix_parts.append("Diabetic foot infection.")

    if findings.get("dfi_bone_biopsy_culture_pos") == "present" or any(
        token in _normalize_choice(case_text) for token in ("bone culture", "bone biopsy", "bone specimen")
    ):
        prefix_parts.append("Use the bone culture isolate and susceptibility results for treatment planning.")
    elif findings.get("dfi_deep_tissue_culture_pos") == "present" or any(
        token in _normalize_choice(case_text) for token in ("deep tissue culture", "operative culture", "tissue culture")
    ):
        prefix_parts.append("Use the deep tissue culture isolate and susceptibility results for treatment planning.")

    if findings.get("dfi_surgery_debridement_done") == "present":
        prefix_parts.append("Surgical debridement or source control has already been performed.")
    if findings.get("dfi_minor_amputation_done") == "present":
        prefix_parts.append("Minor amputation or bone resection has already been performed.")
    if findings.get("dfi_positive_bone_margin") == "present":
        prefix_parts.append("There is a positive residual bone margin after resection.")

    prefix_parts.append(case_text)
    return " ".join(prefix_parts)


def _assistant_mechid_supported_oral_candidates(result: MechIDTextAnalyzeResponse) -> List[str]:
    parsed = result.parsed_request
    if parsed is None or result.analysis is None:
        return []
    syndrome = parsed.tx_context.syndrome
    if syndrome == "Uncomplicated cystitis":
        susceptible_agents = [
            antibiotic
            for antibiotic, call in parsed.susceptibility_results.items()
            if call == "Susceptible"
        ]
        return [
            agent
            for agent in (
                "Nitrofurantoin",
                "Trimethoprim/Sulfamethoxazole",
                "Fosfomycin",
                "Ciprofloxacin",
                "Levofloxacin",
            )
            if agent in susceptible_agents
        ]
    if syndrome in {"Bone/joint infection", "Other deep-seated / high-inoculum focus"}:
        return _mechid_high_bioavailability_oral_choices(result.analysis, parsed)
    return []


def _assistant_append_detected_doseid_ids(candidates: List[str], text: str | None) -> None:
    if not text:
        return
    for medication_id in _assistant_detect_doseid_medication_ids(text):
        if medication_id not in candidates:
            candidates.append(medication_id)


def _assistant_mechid_doseid_candidate_ids(result: MechIDTextAnalyzeResponse) -> List[str]:
    candidates: List[str] = []
    for item in _assistant_mechid_supported_oral_candidates(result):
        _assistant_append_detected_doseid_ids(candidates, item)
    if result.provisional_advice is not None:
        for item in result.provisional_advice.oral_options or []:
            _assistant_append_detected_doseid_ids(candidates, item)
        for item in result.provisional_advice.recommended_options or []:
            _assistant_append_detected_doseid_ids(candidates, item)
    if result.analysis is not None:
        for item in result.analysis.dosing_recommendations or []:
            _assistant_append_detected_doseid_ids(candidates, item.medication_name)
        for item in result.analysis.therapy_notes or []:
            _assistant_append_detected_doseid_ids(candidates, item)
        for item in result.analysis.favorable_signals or []:
            _assistant_append_detected_doseid_ids(candidates, item)
    if _assistant_mechid_should_pair_levo_with_rifampin(result) and "rifampin" not in candidates:
        candidates.insert(1 if candidates else 0, "rifampin")
    return _assistant_unique_medication_ids(candidates)[:4]


def _assistant_mechid_doseid_options(result: MechIDTextAnalyzeResponse) -> List[AssistantOption]:
    medication_ids = _assistant_mechid_doseid_candidate_ids(result)
    if not medication_ids:
        return []
    meds_by_id = _assistant_doseid_medications_by_id()
    options: List[AssistantOption] = []
    pair_levo_with_rifampin = _assistant_mechid_should_pair_levo_with_rifampin(result)
    for medication_id in medication_ids:
        med = meds_by_id.get(medication_id)
        if med is None:
            continue
        description = "Carry this antibiotic into DoseID with the same case context."
        if pair_levo_with_rifampin and medication_id == "levofloxacin":
            description = "Carry levofloxacin plus rifampin into DoseID with the same hardware-associated staphylococcal context."
        options.append(
            AssistantOption(
                value=f"doseid_pick:{medication_id}",
                label=f"Dose {med.name}",
                description=description,
            )
        )
    return options


def _assistant_build_mechid_doseid_prompt(
    result: MechIDTextAnalyzeResponse,
    *,
    case_text: str | None,
    medication_ids: List[str] | None = None,
) -> str | None:
    explicit_medication_ids = medication_ids is not None
    medication_ids = _assistant_unique_medication_ids(
        medication_ids if medication_ids is not None else _assistant_mechid_doseid_candidate_ids(result)
    )
    if not medication_ids:
        return None
    meds_by_id = _assistant_doseid_medications_by_id()
    medication_names = [meds_by_id[item].name for item in medication_ids if item in meds_by_id]
    if not medication_names:
        return None

    parsed = result.parsed_request
    context_bits: List[str] = []
    if parsed is not None:
        if parsed.organism:
            context_bits.append(f"Organism: {parsed.organism}")
        if parsed.tx_context.syndrome != "Not specified":
            context_bits.append(f"Syndrome: {parsed.tx_context.syndrome}")
        if parsed.tx_context.focus_detail != "Not specified":
            context_bits.append(f"Focus: {parsed.tx_context.focus_detail}")
        if parsed.tx_context.severity != "Not specified":
            context_bits.append(f"Severity: {parsed.tx_context.severity}")
        if parsed.tx_context.oral_preference:
            context_bits.append("Prefer an oral option if supported")

    prompt = f"Please calculate dosing for {_join_readable(medication_names)} using the same case context."
    if context_bits:
        prompt += " " + " ".join(context_bits) + "."
    if case_text and case_text.strip() and not (explicit_medication_ids and parsed is not None):
        prompt += f" Case context: {case_text.strip()}"
    return prompt


def _assistant_start_doseid_from_probid_state(state: AssistantState) -> AssistantTurnResponse | None:
    if not state.module_id or not (state.case_text or "").strip():
        return None
    module = store.get(state.module_id)
    if module is None:
        return None
    text_result = _assistant_parse_case_text(module, state)
    if text_result.parsed_request is None:
        return None
    try:
        text_result.analysis = _analyze_internal(text_result.parsed_request)
    except HTTPException:
        return None
    medication_ids = _assistant_probid_doseid_candidate_ids(module, text_result, state)
    prompt = _assistant_build_doseid_followup_prompt(medication_ids, case_text=state.case_text)
    if not prompt:
        return None
    response = _assistant_start_doseid_from_text(prompt, state)
    if response is not None:
        response.assistant_message = "I carried the consult forward into dosing. " + response.assistant_message
    return response


def _assistant_start_doseid_from_mechid_state(state: AssistantState) -> AssistantTurnResponse | None:
    return _assistant_start_selected_doseid_from_mechid_state(state, medication_id=None)


def _assistant_expand_mechid_selected_doseid_ids(
    result: MechIDTextAnalyzeResponse,
    medication_id: str | None,
) -> List[str] | None:
    if medication_id is None:
        return None
    medication_ids = [medication_id]
    if medication_id == "levofloxacin" and _assistant_mechid_should_pair_levo_with_rifampin(result):
        medication_ids.append("rifampin")
    return _assistant_unique_medication_ids(medication_ids)


def _assistant_start_selected_doseid_from_mechid_state(
    state: AssistantState,
    *,
    medication_id: str | None,
) -> AssistantTurnResponse | None:
    if not (state.mechid_text or "").strip():
        return None
    result = _build_mechid_text_response(
        state.mechid_text,
        parser_strategy=state.parser_strategy,
        parser_model=state.parser_model,
        allow_fallback=state.allow_fallback,
    )
    result = _assistant_effective_mechid_result(
        result,
        established_syndrome=state.established_syndrome,
    )
    selected_ids = _assistant_expand_mechid_selected_doseid_ids(result, medication_id)
    prompt = _assistant_build_mechid_doseid_prompt(
        result,
        case_text=state.mechid_text,
        medication_ids=selected_ids,
    )
    if not prompt:
        return None
    response = _assistant_start_doseid_from_text(prompt, state)
    if response is not None:
        syndrome = result.parsed_request.tx_context.syndrome if result.parsed_request is not None else "Not specified"
        focus = result.parsed_request.tx_context.focus_detail if result.parsed_request is not None else "Not specified"
        context_label = focus if focus != "Not specified" else syndrome
        med_name = None
        if medication_id:
            med = _assistant_doseid_medications_by_id().get(medication_id)
            if med is not None:
                med_name = med.name
        paired_name = None
        if selected_ids and len(selected_ids) > 1:
            meds_by_id = _assistant_doseid_medications_by_id()
            paired_names = [meds_by_id[item].name for item in selected_ids if item in meds_by_id]
            if paired_names:
                paired_name = _join_readable(paired_names)
        if context_label != "Not specified":
            if paired_name:
                response.assistant_message = (
                    f"I carried {paired_name} forward into dosing using the same {context_label.lower()} context. "
                    + response.assistant_message
                )
            elif med_name:
                response.assistant_message = (
                    f"I carried {med_name} forward into dosing using the same {context_label.lower()} context. "
                    + response.assistant_message
                )
            else:
                response.assistant_message = (
                    f"I carried the isolate consult forward into dosing using the same {context_label.lower()} context. "
                    + response.assistant_message
                )
        else:
            if paired_name:
                response.assistant_message = f"I carried {paired_name} forward into dosing. " + response.assistant_message
            elif med_name:
                response.assistant_message = f"I carried {med_name} forward into dosing. " + response.assistant_message
            else:
                response.assistant_message = "I carried the isolate consult forward into dosing. " + response.assistant_message
    return response


def _select_module_from_turn(req: AssistantTurnRequest) -> str | None:
    sel = (req.selection or "").strip()
    if sel == PROBID_ASSISTANT_ID:
        return sel
    if sel == MECHID_ASSISTANT_ID:
        return sel
    if sel == DOSEID_ASSISTANT_ID:
        return sel
    if sel == IMMUNOID_ASSISTANT_ID:
        return sel
    if sel == ALLERGYID_ASSISTANT_ID:
        return sel
    if sel and store.get(sel):
        return sel

    msg = (req.message or "").strip()
    if not msg:
        return None
    explicit_workflow_id = _assistant_explicit_non_syndrome_workflow_request(msg)
    if explicit_workflow_id is not None:
        return explicit_workflow_id
    if _assistant_is_allergyid_intent(msg):
        return ALLERGYID_ASSISTANT_ID
    if _assistant_is_mechid_intent(msg):
        return MECHID_ASSISTANT_ID
    if _assistant_is_doseid_intent(msg):
        return DOSEID_ASSISTANT_ID
    if _assistant_is_immunoid_intent(msg):
        return IMMUNOID_ASSISTANT_ID
    explicit_syndrome_module_id = _assistant_explicit_syndrome_module_request(msg)
    if explicit_syndrome_module_id is not None:
        return explicit_syndrome_module_id
    if msg in {"syndrome", "probability", "clinical syndrome probability", "probid", "syndrome probability"}:
        return PROBID_ASSISTANT_ID

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
    if pending_workflow == "doseid":
        response = _assistant_start_doseid_from_text(pending_text, state)
        if response is not None:
            response.assistant_message = (
                "I carried the same case into the dosing lane. " + response.assistant_message
            )
        return response
    if pending_workflow == "immunoid":
        response = _assistant_start_immunoid_from_text(pending_text, state)
        if response is not None:
            response.assistant_message = (
                "I carried the same case into the immunosuppression checklist lane. " + response.assistant_message
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


def _assistant_done_state_redirect(
    req: AssistantTurnRequest,
    state: AssistantState,
) -> AssistantTurnResponse | None:
    """
    When the assistant is in 'done' state and the user sends a free-form message,
    check whether it is a redirect to a different module or intent rather than
    an update to the current workflow.

    Strategy:
    1. Fast-path keyword checks cover clearly-phrased redirects (doseid, duration, iv_to_oral, etc.)
    2. For question-like messages not caught by fast-path, LLM triage runs with full consult context.
    3. Returns a redirect response if a different intent is identified; returns None if the message
       should be treated as a case update for the current workflow.
    """
    msg = (req.message or "").strip()
    if not msg:
        return None

    # Run all fast-path checks in order — same list as the select_module dispatcher.
    # These cover the vast majority of clearly-phrased mid-consult redirects.
    if _is_impression_plan_request(msg):
        return _assistant_impression_plan_response(msg, state)
    if _is_consult_summary_request(msg):
        return _assistant_consult_summary_response(state)
    if _is_empiric_therapy_request(msg):
        return _assistant_empiric_therapy_response(msg, state)
    if _is_iv_to_oral_request(msg):
        return _assistant_iv_to_oral_response(msg, state)
    if _is_duration_request(msg):
        return _assistant_duration_response(msg, state)
    if _is_followup_test_request(msg):
        return _assistant_followup_tests_response(msg, state)
    if _is_stewardship_request(msg):
        return _assistant_stewardship_response(msg, state)
    if _is_stewardship_review_request(msg):
        return _assistant_stewardship_review_response(msg, state)
    if _is_opat_request(msg):
        return _assistant_opat_response(msg, state)
    if _is_oral_therapy_request(msg):
        return _assistant_oral_therapy_response(msg, state)
    if _is_discharge_counselling_request(msg):
        return _assistant_discharge_counselling_response(msg, state)
    if _is_drug_interaction_request(msg):
        return _assistant_drug_interaction_response(msg, state)
    if _is_prophylaxis_request(msg):
        return _assistant_prophylaxis_response(msg, state)
    if _is_source_control_request(msg):
        return _assistant_source_control_response(msg, state)
    if _is_treatment_failure_request(msg):
        return _assistant_treatment_failure_response(msg, state)
    if _is_biomarker_request(msg):
        return _assistant_biomarker_response(msg, state)
    if _is_fluid_interpretation_request(msg):
        return _assistant_fluid_interpretation_response(msg, state)
    if _is_allergy_delabeling_request(msg):
        return _assistant_allergy_delabeling_response(msg, state)
    if _is_fungal_management_request(msg):
        return _assistant_fungal_management_response(msg, state)
    if _is_sepsis_request(msg):
        return _assistant_sepsis_response(msg, state)
    if _is_cns_infection_request(msg):
        return _assistant_cns_infection_response(msg, state)
    if _is_mycobacterial_request(msg):
        return _assistant_mycobacterial_response(msg, state)
    if _is_pregnancy_antibiotics_request(msg):
        return _assistant_pregnancy_antibiotics_response(msg, state)
    if _is_travel_medicine_request(msg):
        return _assistant_travel_medicine_response(msg, state)
    if _is_duke_criteria_request(msg):
        return _assistant_duke_criteria_response(msg, state)
    if _is_ast_meaning_request(msg):
        return _assistant_ast_meaning_response(msg, state)
    if _is_complexity_request(msg):
        return _assistant_complexity_response(msg, state)
    if _is_course_tracker_request(msg):
        return _assistant_course_tracker_response(msg, state)
    if _is_hiv_initial_art_request(msg):
        return _assistant_hiv_initial_art_response(msg, state)
    if _is_hiv_monitoring_request(msg):
        return _assistant_hiv_monitoring_response(msg, state)
    if _is_hiv_prep_request(msg):
        return _assistant_hiv_prep_response(msg, state)
    if _is_hiv_pep_request(msg):
        return _assistant_hiv_pep_response(msg, state)
    if _is_hiv_pregnancy_request(msg):
        return _assistant_hiv_pregnancy_response(msg, state)
    if _is_hiv_oi_art_timing_request(msg):
        return _assistant_hiv_oi_art_timing_response(msg, state)
    if _is_hiv_treatment_failure_request(msg):
        return _assistant_hiv_treatment_failure_response(msg, state)
    if _is_hiv_resistance_request(msg):
        return _assistant_hiv_resistance_response(msg, state)
    if _is_hiv_switch_request(msg):
        return _assistant_hiv_switch_response(msg, state)

    # Cross-workflow direct parsers: catch doseid or mechid phrasing when in a different workflow
    if state.workflow != "doseid":
        doseid_rsp = _assistant_start_doseid_from_text(msg, state)
        if doseid_rsp is not None:
            return doseid_rsp
    if state.workflow not in {"allergyid", "mechid"}:
        mechid_rsp = _assistant_start_mechid_from_text(msg, state)
        if mechid_rsp is not None:
            return mechid_rsp

    # LLM triage for natural-language redirects not caught by the fast-path.
    # Only triggered for question-like messages to avoid adding latency to case-update replies.
    _msg_lower = msg.lower()
    is_question_like = (
        "?" in msg
        or _msg_lower.startswith((
            "what ", "how ", "can i ", "can we ", "should i ", "should we ",
            "when ", "why ", "is it ", "is this ", "could i ", "would ",
            "do i need", "do we need", "tell me", "give me",
        ))
    )
    if is_question_like:
        triage_result = _assistant_llm_triage_intent(msg, state)
        if triage_result is not None:
            _apply_triage_extracted_context(state, triage_result.get("extracted") or {})
            intent = triage_result.get("intent", "unclear")
            # If triage gives a workflow intent matching the current one, let the update proceed
            if intent in {"unclear", state.workflow}:
                return None
            # Otherwise redirect — single LLM call, no duplication
            return _assistant_route_by_intent(intent, req, state, msg)

    return None


@app.post("/v1/assistant/turn", response_model=AssistantTurnResponse)
def assistant_turn(req: AssistantTurnRequest) -> AssistantTurnResponse:
    state = _assistant_initial_state(req)
    user_text = _normalize_choice(req.message or req.selection)
    restart_requested = user_text in {"restart", "start over", "reset", "new case"}

    if restart_requested:
        saved_patient_context = state.patient_context  # preserve across restarts — same patient, new question
        state = AssistantState(
            workflow="probid",
            caseText=None,
            mechidText=None,
            doseidText=None,
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
            patientContext=saved_patient_context,
        )

    # Extract patient demographics from every incoming message and accumulate into session context
    if (req.message or "").strip():
        _update_session_patient_context(state, req.message or "")

    # Handle follow-up to a clarifying question — route back to the saved intent
    # with original context + the user's answer combined.
    if state.pending_intent and not restart_requested:
        _pi_intent = state.pending_intent
        _pi_original = state.pending_intent_context or ""
        _pi_follow_up = (req.message or "").strip()
        if not _pi_follow_up:
            # User clicked an option — use the label / insertText (sent as selection)
            _pi_follow_up = (req.selection or "").strip()
        _pi_combined = f"{_pi_original} {_pi_follow_up}".strip() if _pi_original else _pi_follow_up
        # Clear pending state before routing to avoid infinite loops
        state.pending_intent = None
        state.pending_intent_context = None
        return _assistant_route_by_intent(_pi_intent, req, state, _pi_combined)

    # Route option clicks whose value is a known intent name.
    # When the user clicks a chip like "Calculate dosing" (value="doseid") or
    # "IV-to-oral step-down" (value="iv_to_oral"), the frontend sends
    # selection=<value> with no message.  Module IDs (probid, mechid, etc.)
    # are caught later by _select_module_from_turn, but intent names
    # (empiric_therapy, duration, opat, ...) would otherwise fall through
    # to the default welcome screen.
    _ROUTABLE_INTENT_SELECTIONS: frozenset[str] = frozenset({
        "empiric_therapy", "iv_to_oral", "duration", "followup_tests",
        "stewardship", "stewardship_review", "opat", "oral_therapy",
        "discharge_counselling", "drug_interaction", "prophylaxis",
        "source_control", "treatment_failure", "biomarker_interpretation",
        "fluid_interpretation", "allergy_delabeling", "fungal_management",
        "sepsis_management", "cns_infection", "mycobacterial",
        "pregnancy_antibiotics", "travel_medicine", "impression_plan",
        "duke_criteria", "ast_meaning", "complexity_flag", "course_tracker",
        "consult_summary", "general_id",
        "hiv_initial_art", "hiv_monitoring", "hiv_prep", "hiv_pep",
        "hiv_pregnancy", "hiv_oi_art_timing", "hiv_treatment_failure",
        "hiv_resistance", "hiv_switch",
    })
    _sel_value = (req.selection or "").strip()
    if _sel_value in _ROUTABLE_INTENT_SELECTIONS and not restart_requested:
        # Use req.message if present (user typed + clicked), otherwise
        # build a minimal message from the intent name for the handler.
        _sel_msg = (req.message or "").strip()
        if not _sel_msg:
            _sel_msg = _sel_value.replace("_", " ")
        _sel_routed = _assistant_route_by_intent(_sel_value, req, state, _sel_msg)
        if _sel_routed is not None:
            return _sel_routed

    if state.stage == "select_module":
        message_text = (req.message or "").strip()
        if message_text:
            explicit_workflow_id = _assistant_explicit_non_syndrome_workflow_request(message_text)
            if explicit_workflow_id is not None:
                workflow_label = next(
                    (
                        option.label
                        for option in _assistant_module_options()
                        if option.value == explicit_workflow_id
                    ),
                    explicit_workflow_id,
                )
                return _assistant_begin_selected_workflow(
                    state,
                    explicit_workflow_id,
                    lead_in=f"This reads like an explicit request for {workflow_label}, so I’ll start in that pathway. ",
                )
            direct_doseid_response = _assistant_start_doseid_from_text(message_text, state)
            if direct_doseid_response is not None:
                return direct_doseid_response
            direct_allergy_response = _assistant_start_allergyid_from_text(message_text, state)
            if direct_allergy_response is not None:
                return direct_allergy_response
            explicit_syndrome_module_id = _assistant_explicit_syndrome_module_request(message_text)
            if explicit_syndrome_module_id is not None:
                explicit_module = store.get(explicit_syndrome_module_id)
                explicit_case_response = _assistant_intake_case_from_text(
                    req,
                    state,
                    module_hint=explicit_syndrome_module_id,
                )
                if explicit_case_response is not None:
                    if explicit_module is not None:
                        explicit_case_response.assistant_message = (
                            f"This reads like an explicit request to assess {_assistant_module_label(explicit_module)}, so I’ll start in that syndrome pathway. "
                            + explicit_case_response.assistant_message
                        )
                    explicit_case_response.tips = [
                        "If you also want dosing, resistance, or allergy help afterward, we can carry the same case forward.",
                        *(explicit_case_response.tips or []),
                    ][:3]
                    return explicit_case_response
                if explicit_module is not None:
                    return _assistant_begin_selected_syndrome_module(
                        state,
                        explicit_syndrome_module_id,
                        lead_in=(
                            f"This reads like an explicit request to assess {_assistant_module_label(explicit_module)}, so I’ll start in that syndrome pathway. "
                        ),
                    )
            if _is_impression_plan_request(message_text):
                return _assistant_impression_plan_response(message_text, state)
            if _is_consult_summary_request(message_text):
                return _assistant_consult_summary_response(state)
            if _is_empiric_therapy_request(message_text):
                return _assistant_empiric_therapy_response(message_text, state)
            if _is_iv_to_oral_request(message_text):
                return _assistant_iv_to_oral_response(message_text, state)
            if _is_duration_request(message_text):
                return _assistant_duration_response(message_text, state)
            if _is_followup_test_request(message_text):
                return _assistant_followup_tests_response(message_text, state)
            if _is_stewardship_request(message_text):
                return _assistant_stewardship_response(message_text, state)
            if _is_stewardship_review_request(message_text):
                return _assistant_stewardship_review_response(message_text, state)
            if _is_opat_request(message_text):
                return _assistant_opat_response(message_text, state)
            if _is_oral_therapy_request(message_text):
                return _assistant_oral_therapy_response(message_text, state)
            if _is_discharge_counselling_request(message_text):
                return _assistant_discharge_counselling_response(message_text, state)
            if _is_drug_interaction_request(message_text):
                return _assistant_drug_interaction_response(message_text, state)
            if _is_prophylaxis_request(message_text):
                return _assistant_prophylaxis_response(message_text, state)
            if _is_source_control_request(message_text):
                return _assistant_source_control_response(message_text, state)
            if _is_treatment_failure_request(message_text):
                return _assistant_treatment_failure_response(message_text, state)
            if _is_biomarker_request(message_text):
                return _assistant_biomarker_response(message_text, state)
            if _is_fluid_interpretation_request(message_text):
                return _assistant_fluid_interpretation_response(message_text, state)
            if _is_allergy_delabeling_request(message_text):
                return _assistant_allergy_delabeling_response(message_text, state)
            if _is_fungal_management_request(message_text):
                return _assistant_fungal_management_response(message_text, state)
            if _is_sepsis_request(message_text):
                return _assistant_sepsis_response(message_text, state)
            if _is_cns_infection_request(message_text):
                return _assistant_cns_infection_response(message_text, state)
            if _is_mycobacterial_request(message_text):
                return _assistant_mycobacterial_response(message_text, state)
            if _is_pregnancy_antibiotics_request(message_text):
                return _assistant_pregnancy_antibiotics_response(message_text, state)
            if _is_travel_medicine_request(message_text):
                return _assistant_travel_medicine_response(message_text, state)
            if _is_impression_plan_request(message_text):
                return _assistant_impression_plan_response(message_text, state)
            if _is_duke_criteria_request(message_text):
                return _assistant_duke_criteria_response(message_text, state)
            if _is_ast_meaning_request(message_text):
                return _assistant_ast_meaning_response(message_text, state)
            if _is_complexity_request(message_text):
                return _assistant_complexity_response(message_text, state)
            if _is_course_tracker_request(message_text):
                return _assistant_course_tracker_response(message_text, state)
            if _is_hiv_initial_art_request(message_text):
                return _assistant_hiv_initial_art_response(message_text, state)
            if _is_hiv_monitoring_request(message_text):
                return _assistant_hiv_monitoring_response(message_text, state)
            if _is_hiv_prep_request(message_text):
                return _assistant_hiv_prep_response(message_text, state)
            if _is_hiv_pep_request(message_text):
                return _assistant_hiv_pep_response(message_text, state)
            if _is_hiv_pregnancy_request(message_text):
                return _assistant_hiv_pregnancy_response(message_text, state)
            if _is_hiv_oi_art_timing_request(message_text):
                return _assistant_hiv_oi_art_timing_response(message_text, state)
            if _is_hiv_treatment_failure_request(message_text):
                return _assistant_hiv_treatment_failure_response(message_text, state)
            if _is_hiv_resistance_request(message_text):
                return _assistant_hiv_resistance_response(message_text, state)
            if _is_hiv_switch_request(message_text):
                return _assistant_hiv_switch_response(message_text, state)
            consult_intent_response = _assistant_handle_consult_intent(req, state)
            if consult_intent_response is not None:
                return consult_intent_response
            direct_doseid_response = _assistant_start_doseid_from_text(message_text, state)
            if direct_doseid_response is not None:
                return direct_doseid_response
            direct_immunoid_response = _assistant_start_immunoid_from_text(message_text, state)
            if direct_immunoid_response is not None:
                return direct_immunoid_response
            probid_preview = _assistant_preview_case_from_text(message_text, state, require_high_confidence=True)
            if _assistant_is_endo_imaging_question(message_text):
                endo_preview = _assistant_preview_case_from_text(
                    message_text,
                    state,
                    module_hint="endo",
                    require_high_confidence=False,
                )
                if endo_preview is not None:
                    probid_preview = endo_preview
            mechid_intent = _assistant_mechid_intent_profile(message_text)
            mechid_preview = _assistant_preview_mechid_from_text(message_text, state)
            if (
                probid_preview is not None
                and probid_preview[1].id == "endo"
                and _assistant_is_endo_imaging_question(message_text)
            ):
                case_response = _assistant_intake_case_from_text(req, state)
                if case_response is not None:
                    case_response.assistant_message = (
                        "This reads mainly like an endocarditis imaging question, so I’ll focus on the syndrome side first. "
                        + case_response.assistant_message
                    )
                    case_response.tips = [
                        "If this is really more about antibiotic choice for an isolate, we can still move into resistance afterward.",
                        *(case_response.tips or []),
                    ][:3]
                    return case_response
            if probid_preview is not None and mechid_preview is not None and mechid_intent["strong_mechid_trigger"]:
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
            if probid_preview is not None and mechid_intent["ambiguous_isolate_only"]:
                state.pending_followup_workflow = "mechid"
                state.pending_followup_text = message_text
                case_response = _assistant_intake_case_from_text(req, state)
                if case_response is not None:
                    case_response.assistant_message = (
                        "I also noticed an isolate in your text, but this reads more like a syndrome question than a resistance question. "
                        "I’ll keep the isolate in mind and focus on the syndrome first. "
                        + case_response.assistant_message
                    )
                    case_response.tips = [
                        "If you later want resistance help, tell me the susceptibilities or ask which antibiotics you would use.",
                        *(case_response.tips or []),
                    ][:3]
                    return case_response
            if probid_preview is None and mechid_intent["ambiguous_isolate_only"]:
                isolate_preview = _build_mechid_text_response(
                    message_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                return AssistantTurnResponse(
                    assistantMessage=(
                        "I picked up an isolate, but I’m not yet sure whether you want resistance help or a syndrome workup. "
                        "If you want resistance help, add a few susceptibilities or ask which antibiotics you would use. "
                        "If you want syndrome help, describe the symptoms, source, or key test results."
                    ),
                    state=state,
                    mechidAnalysis=isolate_preview,
                    options=_assistant_module_options(),
                    tips=[
                        "For resistance help, a useful reply would be: 'E. coli resistant to ceftriaxone, susceptible to meropenem.'",
                        "For syndrome help, a useful reply would be: 'fever, flank pain, pyuria, and positive urine culture'.",
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
            return _assistant_begin_selected_workflow(state, chosen_module_id)

        _msg_text = (req.message or "").strip()
        if _is_impression_plan_request(_msg_text):
            return _assistant_impression_plan_response(_msg_text, state)
        if _is_consult_summary_request(_msg_text):
            return _assistant_consult_summary_response(state)
        if _is_empiric_therapy_request(_msg_text):
            return _assistant_empiric_therapy_response(_msg_text, state)
        if _is_iv_to_oral_request(_msg_text):
            return _assistant_iv_to_oral_response(_msg_text, state)
        if _is_duration_request(_msg_text):
            return _assistant_duration_response(_msg_text, state)
        if _is_followup_test_request(_msg_text):
            return _assistant_followup_tests_response(_msg_text, state)
        if _is_stewardship_request(_msg_text):
            return _assistant_stewardship_response(_msg_text, state)
        if _is_stewardship_review_request(_msg_text):
            return _assistant_stewardship_review_response(_msg_text, state)
        if _is_opat_request(_msg_text):
            return _assistant_opat_response(_msg_text, state)
        if _is_oral_therapy_request(_msg_text):
            return _assistant_oral_therapy_response(_msg_text, state)
        if _is_discharge_counselling_request(_msg_text):
            return _assistant_discharge_counselling_response(_msg_text, state)
        if _is_drug_interaction_request(_msg_text):
            return _assistant_drug_interaction_response(_msg_text, state)
        if _is_prophylaxis_request(_msg_text):
            return _assistant_prophylaxis_response(_msg_text, state)
        if _is_source_control_request(_msg_text):
            return _assistant_source_control_response(_msg_text, state)
        if _is_treatment_failure_request(_msg_text):
            return _assistant_treatment_failure_response(_msg_text, state)
        if _is_biomarker_request(_msg_text):
            return _assistant_biomarker_response(_msg_text, state)
        if _is_fluid_interpretation_request(_msg_text):
            return _assistant_fluid_interpretation_response(_msg_text, state)
        if _is_allergy_delabeling_request(_msg_text):
            return _assistant_allergy_delabeling_response(_msg_text, state)
        if _is_fungal_management_request(_msg_text):
            return _assistant_fungal_management_response(_msg_text, state)
        if _is_sepsis_request(_msg_text):
            return _assistant_sepsis_response(_msg_text, state)
        if _is_cns_infection_request(_msg_text):
            return _assistant_cns_infection_response(_msg_text, state)
        if _is_mycobacterial_request(_msg_text):
            return _assistant_mycobacterial_response(_msg_text, state)
        if _is_pregnancy_antibiotics_request(_msg_text):
            return _assistant_pregnancy_antibiotics_response(_msg_text, state)
        if _is_travel_medicine_request(_msg_text):
            return _assistant_travel_medicine_response(_msg_text, state)
        if _is_duke_criteria_request(_msg_text):
            return _assistant_duke_criteria_response(_msg_text, state)
        if _is_ast_meaning_request(_msg_text):
            return _assistant_ast_meaning_response(_msg_text, state)
        if _is_complexity_request(_msg_text):
            return _assistant_complexity_response(_msg_text, state)
        if _is_course_tracker_request(_msg_text):
            return _assistant_course_tracker_response(_msg_text, state)

        llm_triage_response = _assistant_llm_triage(req, state)
        if llm_triage_response is not None:
            return llm_triage_response

        return AssistantTurnResponse(
            assistantMessage=(
                "Hi! I'm your IDAssistant. Ask me an infectious diseases question, or upload an AST image so I can help with your ID question."
            ),
            state=state,
            options=_assistant_module_options(),
            tips=[
                "Choose resistance mechanism and therapy, antimicrobial dosing, clinical syndrome probability, or immunosuppression screening and prophylaxis.",
                "You can also ask consult-style questions such as 'should I start antifungal treatment?' or 'does this look like endocarditis?'.",
            ],
        )

    if state.stage == "select_syndrome_module":
        chosen_module_id = _select_module_from_turn(req)
        if chosen_module_id and chosen_module_id not in {PROBID_ASSISTANT_ID, MECHID_ASSISTANT_ID, DOSEID_ASSISTANT_ID, IMMUNOID_ASSISTANT_ID, ALLERGYID_ASSISTANT_ID}:
            return _assistant_begin_selected_syndrome_module(state, chosen_module_id)

        return AssistantTurnResponse(
            assistantMessage="Which clinical syndrome would you like to assess?",
            state=state,
            options=_assistant_syndrome_module_options(),
            tips=[
                "Choose the syndrome first, then I’ll ask for the setting and case details.",
                "You can also type the syndrome name in plain language.",
            ],
        )

    if state.stage == "doseid_describe":
        if _is_ready_to_assess(req) and (state.doseid_text or "").strip():
            return _assistant_doseid_response(
                state,
                message_text=state.doseid_text or "",
                prefix="I ran the dosing lane with the current information. ",
                force_best_effort=True,
            )
        if req.message and req.message.strip():
            rewritten_message = _assistant_rewrite_doseid_followup_reply(state.doseid_text, req.message)
            state.doseid_text = _append_case_text(state.doseid_text, rewritten_message)
            return _assistant_doseid_response(
                state,
                message_text=state.doseid_text,
                prefix="I updated the dosing consult. ",
            )
        return AssistantTurnResponse(
            assistantMessage=(
                "Tell me the antimicrobial or regimen plus the renal context, and I’ll calculate the dose. "
                "For example: 'cefepime on hemodialysis' or 'RIPE for 62 kg, CrCl 35'."
            ),
            state=state,
            options=[AssistantOption(value="restart", label="Start new consult")],
            tips=[
                "The most useful details are weight, renal function, dialysis status, age, sex, and height.",
            ],
        )

    if state.stage in {"immunoid_select_agents", "immunoid_collect_context"}:
        state.workflow = "immunoid"
        selection = (req.selection or "").strip()
        if selection and _assistant_apply_immunoid_selection(state, selection):
            if selection == "immunoid_clear_agents":
                state.stage = "immunoid_select_agents"
                return AssistantTurnResponse(
                    assistantMessage="I cleared the current immunosuppression list. Add the planned agents again.",
                    state=state,
                    options=_assistant_immunoid_agent_options(state),
                    tips=[
                        "You can type the agents in plain language or click a few common ones.",
                    ],
                )
            if state.immunoid_selected_agent_ids:
                return _assistant_immunoid_response(state, prefix="I updated the ImmunoID context. ")

        if req.message and req.message.strip():
            updates = _assistant_parse_immunoid_context_from_text(state, req.message)
            if state.immunoid_selected_agent_ids:
                lead = "I updated the ImmunoID context. " if updates else ""
                return _assistant_immunoid_response(state, prefix=lead)

        if not state.immunoid_selected_agent_ids:
            state.stage = "immunoid_select_agents"
            return AssistantTurnResponse(
                assistantMessage=(
                    "I still need the planned immunosuppressive agents before I can build the checklist. "
                    "Type them in plain language or click a few common options."
                ),
                state=state,
                options=_assistant_immunoid_agent_options(state),
                tips=[
                    "Examples: rituximab, infliximab, prednisone 20 mg/day, cyclophosphamide, tacrolimus.",
                ],
            )

        return _assistant_immunoid_response(state)

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
        mechid_result = _assistant_effective_mechid_result(
            mechid_result,
            established_syndrome=state.established_syndrome,
        )
        _accumulate_consult_organisms(state, mechid_result)
        _snapshot_mechid_result(state, mechid_result)
        review_message, narration_refined = _assistant_mechid_review_message(
            mechid_result,
            established_syndrome=state.established_syndrome,
            consult_organisms=state.consult_organisms or None,
            institutional_antibiogram=state.institutional_antibiogram or None,
        )
        # If syndrome is already established, show extraction + offer to proceed
        _desc_review_msg = review_message
        if state.established_syndrome and (mechid_result.analysis is not None or mechid_result.provisional_advice is not None):
            _desc_review_msg += f"\n\nSyndrome context: **{state.established_syndrome}**. Select 'Get therapy recommendation' when the extraction looks right, or add more details."
        return AssistantTurnResponse(
            assistantMessage=_desc_review_msg,
            assistantNarrationRefined=narration_refined,
            state=state,
            options=_assistant_mechid_review_options(mechid_result, established_syndrome=state.established_syndrome),
            mechidAnalysis=mechid_result,
            tips=[
                "Select the clinical syndrome so I can tailor the therapy recommendation.",
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
        mechid_result = _assistant_effective_mechid_result(
            mechid_result,
            established_syndrome=state.established_syndrome,
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

        # ── Syndrome selection → auto-run therapy recommendation ──
        _sel = (req.selection or "").strip()
        if _sel.startswith("mechid_set_syndrome:"):
            _syndrome_label = _sel.split(":", 1)[1].strip()
            if _syndrome_label and _syndrome_label != "Other":
                state.established_syndrome = _syndrome_label
            mechid_result = _assistant_effective_mechid_result(
                mechid_result,
                established_syndrome=state.established_syndrome,
            )
            # Auto-proceed to therapy recommendation (same as run_assessment path)
            if mechid_result.analysis is not None or mechid_result.provisional_advice is not None:
                state.stage = "done"
                _accumulate_consult_organisms(state, mechid_result)
                _snapshot_mechid_result(state, mechid_result)
                poly_analyses = _build_polymicrobial_analyses(
                    state.mechid_text or "",
                    mechid_result,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                narrated_message, narration_refined = _assistant_mechid_review_message(
                    mechid_result,
                    final=True,
                    established_syndrome=state.established_syndrome,
                    consult_organisms=state.consult_organisms or None,
                    polymicrobial_analyses=poly_analyses or None,
                    institutional_antibiogram=state.institutional_antibiogram or None,
                )
                done_options = [
                    AssistantOption(value="duration", label="How long to treat?"),
                    AssistantOption(value="add_more_details", label="Update this case"),
                ]
                if _is_mid_consult(state):
                    done_options.append(AssistantOption(value="consult_summary", label="Full consult summary"))
                else:
                    done_options.append(AssistantOption(value="restart", label="Start new consult"))
                syndrome_opts = _syndrome_next_step_options(state, exclude={"source_control"})
                for opt in reversed(syndrome_opts):
                    done_options.insert(0, opt)
                done_tips = [
                    "Add another susceptibility, test result, or clinical detail anytime and I will update the same case.",
                    "Review the mechanism, cautions, therapy notes, and references in the analysis panel.",
                ]
                doseid_options = _assistant_mechid_doseid_options(mechid_result)
                if doseid_options:
                    narrated_message = _assistant_append_dosing_invitation(narrated_message)
                    for option in reversed(doseid_options):
                        done_options.insert(0, option)
                    done_tips.insert(
                        0,
                        "If you want, pick one of the suggested antibiotics and I can carry it into DoseID using the same case context.",
                    )
                    _mechid_meds = _assistant_doseid_medications_by_id()
                    _mechid_candidate_names = [
                        _mechid_meds[opt.value.split(":", 1)[1]].name
                        for opt in doseid_options
                        if opt.value.startswith("doseid_pick:") and opt.value.split(":", 1)[1] in _mechid_meds
                    ]
                    allergy_opt = _assistant_allergy_check_option(state, candidate_drug_names=_mechid_candidate_names)
                    if allergy_opt:
                        done_options.append(allergy_opt)
                else:
                    dose_bridge = _mechid_doseid_bridge_option(state)
                    if dose_bridge:
                        done_options.insert(0, dose_bridge)
                        done_tips.insert(0, "For this syndrome, precise renal-adjusted dosing is important — I can calculate it now.")
                return AssistantTurnResponse(
                    assistantMessage=narrated_message,
                    assistantNarrationRefined=narration_refined,
                    state=state,
                    options=done_options,
                    mechidAnalysis=mechid_result,
                    tips=done_tips,
                )
            # No analysis/provisional_advice yet — ask for more data
            review_message, narration_refined = _assistant_mechid_review_message(
                mechid_result,
                established_syndrome=state.established_syndrome,
                consult_organisms=state.consult_organisms or None,
                institutional_antibiogram=state.institutional_antibiogram or None,
            )
            return AssistantTurnResponse(
                assistantMessage=review_message + "\n\nI need more susceptibility data before I can give a therapy recommendation. Paste additional AST results.",
                assistantNarrationRefined=narration_refined,
                state=state,
                options=[
                    AssistantOption(value="add_more_details", label="Add case detail"),
                    AssistantOption(value="restart", label="Start new consult"),
                ],
                mechidAnalysis=mechid_result,
                tips=["Add more susceptibility calls so I can refine the therapy recommendation."],
            )

        if _is_ready_to_assess(req):
            if mechid_result.analysis is None and mechid_result.provisional_advice is None:
                review_message, narration_refined = _assistant_mechid_review_message(
                    mechid_result,
                    established_syndrome=state.established_syndrome,
                    consult_organisms=state.consult_organisms or None,
                    institutional_antibiogram=state.institutional_antibiogram or None,
                )
                return AssistantTurnResponse(
                    assistantMessage=review_message,
                    assistantNarrationRefined=narration_refined,
                    state=state,
                    options=_assistant_mechid_review_options(mechid_result, established_syndrome=state.established_syndrome),
                    mechidAnalysis=mechid_result,
                    tips=[
                        "I still need a clearer organism and susceptibility pattern before I can finalize the interpretation.",
                    ],
                )
            state.stage = "done"
            _accumulate_consult_organisms(state, mechid_result)
            _snapshot_mechid_result(state, mechid_result)
            poly_analyses = _build_polymicrobial_analyses(
                state.mechid_text or "",
                mechid_result,
                parser_strategy=state.parser_strategy,
                parser_model=state.parser_model,
                allow_fallback=state.allow_fallback,
            )
            narrated_message, narration_refined = _assistant_mechid_review_message(
                mechid_result,
                final=True,
                established_syndrome=state.established_syndrome,
                consult_organisms=state.consult_organisms or None,
                polymicrobial_analyses=poly_analyses or None,
                institutional_antibiogram=state.institutional_antibiogram or None,
            )
            done_options = [
                AssistantOption(value="duration", label="How long to treat?"),
                AssistantOption(value="add_more_details", label="Update this case"),
            ]
            if _is_mid_consult(state):
                done_options.append(AssistantOption(value="consult_summary", label="Full consult summary"))
            else:
                done_options.append(AssistantOption(value="restart", label="Start new consult"))
            # Add syndrome-specific next steps (e.g. TEE for endocarditis, source control for bacteraemia)
            syndrome_opts = _syndrome_next_step_options(state, exclude={"source_control"})
            for opt in reversed(syndrome_opts):
                done_options.insert(0, opt)
            done_tips = [
                "Add another susceptibility, test result, or clinical detail anytime and I will update the same case.",
                "Review the mechanism, cautions, therapy notes, and references in the analysis panel.",
            ]
            if state.pending_followup_workflow == "probid" and (state.pending_followup_text or "").strip():
                done_options.insert(0, AssistantOption(value="continue_to_syndrome", label="Continue to syndrome"))
                done_tips.insert(0, "If you want, I can carry the same case into the syndrome workup next.")
            doseid_options = _assistant_mechid_doseid_options(mechid_result)
            if doseid_options:
                narrated_message = _assistant_append_dosing_invitation(narrated_message)
                insert_at = 1 if done_options and done_options[0].value == "continue_to_syndrome" else 0
                for option in reversed(doseid_options):
                    done_options.insert(insert_at, option)
                done_tips.insert(
                    1 if done_tips and "syndrome" in done_tips[0].lower() else 0,
                    "If you want, pick one of the suggested antibiotics and I can carry it into DoseID using the same case context.",
                )
                _mechid_meds = _assistant_doseid_medications_by_id()
                _mechid_candidate_names = [
                    _mechid_meds[opt.value.split(":", 1)[1]].name
                    for opt in doseid_options
                    if opt.value.startswith("doseid_pick:") and opt.value.split(":", 1)[1] in _mechid_meds
                ]
                allergy_opt = _assistant_allergy_check_option(state, candidate_drug_names=_mechid_candidate_names)
                if allergy_opt:
                    done_options.append(allergy_opt)
            else:
                # No specific doseid_options from mechid, but for high-stakes syndromes offer a generic dose bridge
                dose_bridge = _mechid_doseid_bridge_option(state)
                if dose_bridge:
                    done_options.insert(0, dose_bridge)
                    done_tips.insert(0, "For this syndrome, precise renal-adjusted dosing is important — I can calculate it now.")
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
            options=_assistant_mechid_review_options(mechid_result, established_syndrome=state.established_syndrome),
            mechidAnalysis=mechid_result,
            tips=[
                "Keep replying in normal words and I will keep the case moving one question at a time.",
                "If the extraction looks right, select the clinical syndrome to get the therapy recommendation.",
            ],
        )

    if state.stage == "select_preset":
        if not state.module_id:
            state.stage = "select_syndrome_module"
            return AssistantTurnResponse(
                assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                state=state,
                options=_assistant_syndrome_module_options(),
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
                state.stage = "select_syndrome_module"
                return AssistantTurnResponse(
                    assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                    state=state,
                    options=_assistant_syndrome_module_options(),
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
            state.stage = "select_syndrome_module"
            return AssistantTurnResponse(
                assistantMessage="I need the syndrome first. Which syndrome would you like to approach today?",
                state=state,
                options=_assistant_syndrome_module_options(),
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
                "Tap Continue consult when you’re ready to move on.",
            ],
        )

    if state.stage == "describe_case":
        module = store.get(state.module_id or "")
        if module is None:
            raise HTTPException(status_code=400, detail=f"Module '{state.module_id}' not found")

        selection = (req.selection or "").strip()
        message_text = (req.message or "").strip()
        inserted_item = _assistant_append_case_item_from_selection(module, state, selection)
        if inserted_item[0] is not None:
            text_result = _assistant_parse_case_text(module, state)
            current_label = _assistant_case_section_label(module, state.case_section)
            return AssistantTurnResponse(
                assistantMessage=(
                    f"Added {inserted_item[1]} to {current_label}. "
                    "Keep going in this section, or tap Continue consult when you’re ready to move on."
                ),
                assistantNarrationRefined=False,
                state=state,
                options=_assistant_case_prompt_options(module, state, section_override=state.case_section),
                analysis=text_result,
                tips=[
                    "You can keep tapping findings or type the detail in your own words.",
                    "When this section is done, continue to the next part of the case.",
                ],
            )
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
                    f"Describe the case in plain language. Start with {_assistant_case_section_label(module, 'exam_vitals')}, then laboratory, microbiology, and radiographic tests below. Use the Present/Absent toggle if a finding is negative."
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
                next_label = _assistant_case_section_label(module, next_section)
                current_label = _assistant_case_section_label(module, current_section)
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
                "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
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
        inserted_item = _assistant_append_case_item_from_selection(module, state, selection)
        if inserted_item[0] is not None:
            text_result = _assistant_parse_case_text(module, state)
            return AssistantTurnResponse(
                assistantMessage=_assistant_concise_probid_follow_up(
                    module,
                    text_result,
                    state,
                    lead=f"Okay, I added {inserted_item[1]}.",
                ),
                assistantNarrationRefined=False,
                state=state,
                options=_assistant_review_options_for_case(module, text_result, state),
                analysis=text_result,
                tips=[
                    "Keep replying in normal words and I will keep the case moving one question at a time.",
                    "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
                ],
            )
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
                    "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
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
                    "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
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
            text_result = _assistant_cached_probid_case_result(state)
            if text_result is None:
                text_result = _assistant_parse_case_text(module, state)
            if text_result.parsed_request is not None and text_result.analysis is None:
                try:
                    text_result.analysis = _analyze_internal(text_result.parsed_request)
                except HTTPException as exc:
                    text_result.warnings.append(f"Parsed request could not be analyzed yet: {exc.detail}")
                    text_result.requires_confirmation = True
            _assistant_cache_probid_case_result(state, text_result)
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
                case_text=state.case_text,
            )
            if _assistant_case_can_run_provisional_consult(module, text_result, state):
                final_message = (
                    "This is a best-effort consult using the currently available data, so treat it as provisional until the missing high-yield details are clarified. "
                    + final_message
                )
            narrated_message, narration_refined = narrate_probid_assistant_message(
                text_result=text_result,
                fallback_message=final_message,
                module_label=_assistant_module_label(module),
                prior_context_summary=_consult_prior_context_summary(state),
            )
            # Carry the identified syndrome forward so MechID and DoseID can use it
            if not state.established_syndrome and module is not None:
                state.established_syndrome = _assistant_module_label(module)
            _snapshot_probid_result(state, text_result, module)
            mechid_followup_text = _assistant_build_probid_mechid_followup_text(module, text_result, state)
            if mechid_followup_text:
                state.pending_followup_workflow = "mechid"
                state.pending_followup_text = mechid_followup_text
            elif state.pending_followup_workflow == "mechid":
                state.pending_followup_workflow = None
                state.pending_followup_text = None
            done_options = [
                AssistantOption(value="add_more_details", label="Update this case"),
            ]
            if _is_mid_consult(state):
                done_options.append(AssistantOption(value="consult_summary", label="Full consult summary"))
            else:
                done_options.append(AssistantOption(value="restart", label="Start new consult"))
            done_tips = [
                "Add another test result or case detail anytime and I will update the same consult.",
                "Review `understood` to confirm what I extracted from your text.",
            ]
            if state.pending_followup_workflow == "mechid" and (state.pending_followup_text or "").strip():
                done_options.insert(0, AssistantOption(value="continue_to_resistance", label="Continue to resistance"))
                done_tips.insert(0, "If you want, I can carry the same case into the isolate/resistance interpretation next.")
            micro_bridge = _probid_micro_bridge_option(state)
            if micro_bridge and not any(o.value in {"continue_to_resistance", "mechid"} for o in done_options):
                done_options.insert(0, micro_bridge)
                done_tips.insert(0, "If culture results are back, paste them and I'll interpret resistance and therapy in the context of this syndrome.")
            # No organisms yet but syndrome established → offer empiric therapy
            elif state.established_syndrome and not state.consult_organisms:
                done_options.insert(0, AssistantOption(value="empiric_therapy", label="What to start empirically?"))
                done_tips.insert(0, "Cultures pending? I can recommend empiric coverage based on the syndrome while you wait for results.")
            # Acute HIV / viral syndrome bridge → offer ART workup
            hiv_bridge = _probid_hiv_bridge_option(state)
            if hiv_bridge:
                done_options.insert(0, hiv_bridge)
                done_tips.insert(0, "Acute retroviral syndrome should be on the differential — I can start the HIV ART workup if 4th-gen Ag/Ab is positive.")
            elif _is_acute_hiv_mention(text_result.get("raw_text", "") if isinstance(text_result, dict) else ""):
                done_options.insert(0, AssistantOption(value="hiv_initial_art", label="Start ART for acute HIV"))
                done_tips.insert(0, "If HIV Ag/Ab is confirmed, same-day ART is recommended — I can help select the regimen.")
            probid_doseid_ids = _assistant_probid_doseid_candidate_ids(module, text_result, state)
            if probid_doseid_ids:
                narrated_message = _assistant_append_dosing_invitation(narrated_message)
                done_options.insert(
                    1 if done_options and done_options[0].value in {"continue_to_resistance", "mechid"} else 0,
                    AssistantOption(value="continue_to_dosing", label="Continue to dosing"),
                )
                done_tips.insert(
                    1 if done_tips and ("resistance" in done_tips[0].lower() or "culture" in done_tips[0].lower()) else 0,
                    "If you want, I can keep going into antimicrobial dosing for the treatment options suggested by this consult.",
                )
                _probid_meds_by_id = _assistant_doseid_medications_by_id()
                _probid_candidate_names = [_probid_meds_by_id[m].name for m in probid_doseid_ids if m in _probid_meds_by_id]
                allergy_opt = _assistant_allergy_check_option(state, candidate_drug_names=_probid_candidate_names)
                if allergy_opt:
                    done_options.append(allergy_opt)
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
            _assistant_merge_pretest_factor_ids(
                state,
                _assistant_infer_pretest_factor_ids_from_text(module, req.message, state),
            )
            _sync_pretest_factor_labels(state, module)
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
                    "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
                ],
            )

        text_result = _assistant_cached_probid_case_result(state)
        if text_result is None:
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
        if req.message and req.message.strip() and state.workflow != "allergyid":
            direct_allergy_response = _assistant_start_allergyid_from_text(req.message.strip(), state)
            if direct_allergy_response is not None:
                return direct_allergy_response
        if state.workflow == "immunoid" and selection and _assistant_apply_immunoid_selection(state, selection):
            if not state.immunoid_selected_agent_ids:
                state.stage = "immunoid_select_agents"
                return AssistantTurnResponse(
                    assistantMessage="I cleared the current immunosuppression list. Add the planned agents again.",
                    state=state,
                    options=_assistant_immunoid_agent_options(state),
                    tips=["You can type the agents in plain language or click a few common ones."],
                )
            return _assistant_immunoid_response(state, prefix="I updated the immunosuppression checklist. ")
        if selection in {"continue_to_syndrome", "continue_to_resistance"}:
            followup_response = _assistant_start_pending_followup(state)
            if followup_response is not None:
                return followup_response
        if state.workflow == "mechid" and selection.startswith("doseid_pick:"):
            medication_id = selection.split(":", 1)[1].strip()
            if medication_id:
                followup_response = _assistant_start_selected_doseid_from_mechid_state(
                    state,
                    medication_id=medication_id,
                )
                if followup_response is not None:
                    return followup_response
        if selection.startswith("immunoid_doseid:"):
            medication_id = selection.split(":", 1)[1].strip()
            if medication_id:
                meds_by_id = _assistant_doseid_medications_by_id()
                med = meds_by_id.get(medication_id)
                doseid_text = f"{med.name} dosing" if med else medication_id
                if state.patient_context:
                    pc_text = _session_patient_context_as_doseid_text(state.patient_context)
                    if pc_text:
                        doseid_text = f"{doseid_text}. {pc_text}"
                return _assistant_doseid_response(
                    state,
                    message_text=doseid_text,
                    prefix="I carried the prophylaxis agent into the dosing lane. ",
                )
        if selection == "allergyid_check":
            candidate_names: List[str] = []
            if state.workflow == "probid" and state.module_id:
                module = store.get(state.module_id or "")
                if module is not None:
                    _check_result = _assistant_parse_case_text(module, state)
                    _check_ids = _assistant_probid_doseid_candidate_ids(module, _check_result, state)
                    _check_meds = _assistant_doseid_medications_by_id()
                    candidate_names = [_check_meds[m].name for m in _check_ids if m in _check_meds]
            elif state.workflow == "mechid" and state.mechid_text:
                _mechid_check_result = _build_mechid_text_response(
                    state.mechid_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                _mechid_dose_opts = _assistant_mechid_doseid_options(_mechid_check_result)
                _mechid_check_meds = _assistant_doseid_medications_by_id()
                candidate_names = [
                    _mechid_check_meds[opt.value.split(":", 1)[1]].name
                    for opt in _mechid_dose_opts
                    if opt.value.startswith("doseid_pick:") and opt.value.split(":", 1)[1] in _mechid_check_meds
                ]
            allergy_response = _assistant_handle_allergyid_check(state, candidate_drug_names=candidate_names)
            if allergy_response is not None:
                return allergy_response
        if selection == "continue_to_dosing":
            if state.workflow == "probid":
                followup_response = _assistant_start_doseid_from_probid_state(state)
                if followup_response is not None:
                    return followup_response
            if state.workflow == "mechid":
                followup_response = _assistant_start_doseid_from_mechid_state(state)
                if followup_response is not None:
                    return followup_response
            return AssistantTurnResponse(
                assistantMessage=(
                    "I could not confidently carry this consult into dosing yet. Add the likely treatment agent or regimen, "
                    "and I’ll calculate the dose from there."
                ),
                state=state,
                options=[AssistantOption(value="restart", label="Start new consult")],
                tips=[
                    "For example: 'dose vancomycin for MRSA endocarditis' or 'RIPE dosing for 62 kg, CrCl 35'.",
                ],
            )
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
            if state.workflow == "immunoid":
                return AssistantTurnResponse(
                    assistantMessage=(
                        "Add the new serology result, exposure history, steroid duration, neutropenia expectation, or another immunosuppressive agent in plain language."
                    ),
                    state=state,
                    options=[AssistantOption(value="restart", label="Start new consult")],
                    tips=[
                        "For example: 'anti-HBc positive', 'IGRA negative', 'lived in Arizona', or 'prednisone for 6 weeks'.",
                    ],
                )
            if state.workflow == "allergyid":
                return AssistantTurnResponse(
                    assistantMessage=(
                        "Add the new allergy detail or candidate antibiotic in plain language, and I will update the same allergy consult."
                    ),
                    state=state,
                    options=[AssistantOption(value="restart", label="Start new consult")],
                    tips=[
                        "For example: 'the reaction was only GI upset', 'it was ceftriaxone not amoxicillin', or 'can I use cefepime instead?'",
                    ],
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
            # Before treating the message as a case update, check whether it is actually
            # a redirect to a different intent (e.g. "what's the dose?" while in mechid done).
            _done_redirect = _assistant_done_state_redirect(req, state)
            if _done_redirect is not None:
                return _done_redirect

            if state.workflow == "mechid" and state.mechid_text:
                previous_result = _build_mechid_text_response(
                    state.mechid_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                previous_result = _assistant_effective_mechid_result(
                    previous_result,
                    established_syndrome=state.established_syndrome,
                )
                state.mechid_text = _append_case_text(state.mechid_text, req.message)
                updated_result = _build_mechid_text_response(
                    state.mechid_text,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                updated_result = _assistant_effective_mechid_result(
                    updated_result,
                    established_syndrome=state.established_syndrome,
                )
                _accumulate_consult_organisms(state, updated_result)
                _snapshot_mechid_result(state, updated_result)
                upd_poly = _build_polymicrobial_analyses(
                    state.mechid_text or "",
                    updated_result,
                    parser_strategy=state.parser_strategy,
                    parser_model=state.parser_model,
                    allow_fallback=state.allow_fallback,
                )
                updated_message, narration_refined = _assistant_mechid_review_message(
                    updated_result,
                    final=True,
                    established_syndrome=state.established_syndrome,
                    consult_organisms=state.consult_organisms or None,
                    polymicrobial_analyses=upd_poly or None,
                    institutional_antibiogram=state.institutional_antibiogram or None,
                )
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
                doseid_options = _assistant_mechid_doseid_options(updated_result)
                if doseid_options:
                    updated_message = _assistant_append_dosing_invitation(updated_message)
                    insert_at = 1 if done_options and done_options[0].value == "continue_to_syndrome" else 0
                    for option in reversed(doseid_options):
                        done_options.insert(insert_at, option)
                    done_tips.insert(
                        1 if done_tips and "syndrome" in done_tips[0].lower() else 0,
                        "If you want, pick one of the suggested antibiotics and I can carry it into DoseID using the same case context.",
                    )
                    _upd_mechid_meds = _assistant_doseid_medications_by_id()
                    _upd_mechid_names = [
                        _upd_mechid_meds[opt.value.split(":", 1)[1]].name
                        for opt in doseid_options
                        if opt.value.startswith("doseid_pick:") and opt.value.split(":", 1)[1] in _upd_mechid_meds
                    ]
                    allergy_opt = _assistant_allergy_check_option(state, candidate_drug_names=_upd_mechid_names)
                    if allergy_opt:
                        done_options.append(allergy_opt)
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
                    _assistant_merge_pretest_factor_ids(
                        state,
                        _assistant_infer_pretest_factor_ids_from_text(module, req.message, state),
                    )
                    _sync_pretest_factor_labels(state, module)
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
                                "If the extraction looks right already, select the clinical syndrome to get the therapy recommendation.",
                            ],
                        )
                    missing_suggestions = _top_missing_tests(module, updated_result.parsed_request, limit=3, state=state)
                    final_message = _build_probid_consult_message(
                        module,
                        updated_result.analysis,
                        missing_suggestions=missing_suggestions,
                        include_panel_note=True,
                        case_text=state.case_text,
                    )
                    narrated_message, narration_refined = narrate_probid_assistant_message(
                        text_result=updated_result,
                        fallback_message=final_message,
                        module_label=_assistant_module_label(module),
                        prior_context_summary=_consult_prior_context_summary(state),
                    )
                    # Keep established_syndrome current with the active module
                    if module is not None:
                        state.established_syndrome = _assistant_module_label(module)
                    _snapshot_probid_result(state, updated_result, module)
                    mechid_followup_text = _assistant_build_probid_mechid_followup_text(module, updated_result, state)
                    if mechid_followup_text:
                        state.pending_followup_workflow = "mechid"
                        state.pending_followup_text = mechid_followup_text
                    elif state.pending_followup_workflow == "mechid":
                        state.pending_followup_workflow = None
                        state.pending_followup_text = None
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
                    updated_doseid_ids = _assistant_probid_doseid_candidate_ids(module, updated_result, state)
                    if updated_doseid_ids:
                        narrated_message = _assistant_append_dosing_invitation(narrated_message)
                        done_options.insert(
                            1 if done_options and done_options[0].value == "continue_to_resistance" else 0,
                            AssistantOption(value="continue_to_dosing", label="Continue to dosing"),
                        )
                        done_tips.insert(
                            1 if done_tips and "resistance" in done_tips[0].lower() else 0,
                            "If you want, I can keep going into antimicrobial dosing for the treatment options suggested by this consult.",
                        )
                        _upd_meds_by_id = _assistant_doseid_medications_by_id()
                        _upd_candidate_names = [_upd_meds_by_id[m].name for m in updated_doseid_ids if m in _upd_meds_by_id]
                        allergy_opt = _assistant_allergy_check_option(state, candidate_drug_names=_upd_candidate_names)
                        if allergy_opt:
                            done_options.append(allergy_opt)
                    # Acute HIV bridge for second ProbID done path
                    hiv_bridge2 = _probid_hiv_bridge_option(state)
                    if hiv_bridge2 and not any(o.value == hiv_bridge2.value for o in done_options):
                        done_options.insert(0, hiv_bridge2)
                        done_tips.insert(0, "Acute retroviral syndrome should be on the differential — I can start the HIV ART workup if confirmed.")
                    # No organisms + syndrome → empiric
                    if state.established_syndrome and not state.consult_organisms and not any(o.value == "empiric_therapy" for o in done_options):
                        done_options.insert(0, AssistantOption(value="empiric_therapy", label="What to start empirically?"))
                    return AssistantTurnResponse(
                        assistantMessage=f"{lead} {narrated_message}",
                        assistantNarrationRefined=narration_refined,
                        state=state,
                        options=done_options,
                        analysis=updated_result,
                        tips=done_tips,
                    )
            if state.workflow == "immunoid" and state.immunoid_selected_agent_ids:
                _assistant_parse_immunoid_context_from_text(state, req.message)
                return _assistant_immunoid_response(
                    state,
                    prefix="I updated the immunosuppression checklist with the new information. ",
                )
            if state.workflow == "allergyid":
                try:
                    state.allergyid_text = _assistant_merge_allergyid_followup_text(state.allergyid_text, req.message)
                except Exception:
                    state.allergyid_text = _append_case_text(state.allergyid_text, req.message)
                return _assistant_allergyid_response(
                    state,
                    message_text=state.allergyid_text or req.message,
                    prefix="I updated the allergy consult with the new information. ",
                )

    # LLM triage: attempt to route or answer before resetting to the generic fallback
    if not restart_requested and (req.message or "").strip():
        llm_triage_response = _assistant_llm_triage(req, state)
        if llm_triage_response is not None:
            return llm_triage_response

    # done / fallback
    if restart_requested:
        state = AssistantState(
            workflow="probid",
            caseText=None,
            mechidText=None,
            allergyidText=None,
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
        state.doseid_text = None
        state.allergyid_text = None
        _assistant_reset_immunoid_state(state)
        state.pretest_factor_ids = []
        state.pretest_factor_labels = []

    return AssistantTurnResponse(
        assistantMessage=(
            "Ready for another case. You can start a syndrome workup, resistance interpretation, dosing consult, or immunosuppression review."
        ),
        state=state,
        options=_assistant_module_options(),
        tips=["Type 'restart' anytime to reset the conversation."],
    )
