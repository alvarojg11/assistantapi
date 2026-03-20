from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.services.doseid_service import list_medications  # noqa: E402
from app.services.immunoid_regimens import IMMUNOID_REGIMENS  # noqa: E402
from app.services.immunoid_source_catalog import IMMUNOID_SOURCE_AGENTS  # noqa: E402
from app.services.mechid_engine import list_mechid_organisms  # noqa: E402
from scripts.smoke_test_new_syndromes import NEW_SYNDROME_CASES  # noqa: E402


BASE_DATASET = BACKEND_ROOT / "app" / "data" / "assistant_eval_cases.json"
OUTPUT_DATASETS = {
    "standard": BACKEND_ROOT / "app" / "data" / "assistant_eval_cases_100.json",
    "expanded": BACKEND_ROOT / "app" / "data" / "assistant_eval_cases_160.json",
    "humanized": BACKEND_ROOT / "app" / "data" / "assistant_eval_cases_humanized_260.json",
    "noisy": BACKEND_ROOT / "app" / "data" / "assistant_eval_cases_noisy_360.json",
}


DOSEID_PROMPTS = {
    "cefepime": "Can you help with cefepime dosing? 72 kg adult, CrCl 45 mL/min, severe infection.",
    "ceftriaxone": "Can you help with ceftriaxone dosing? 70 kg adult, CrCl 45 mL/min, serious infection.",
    "cefazolin": "Can you help with cefazolin dosing? 82 kg adult, CrCl 38 mL/min, complicated MSSA infection.",
    "meropenem": "Can you help with meropenem dosing? 85 kg adult, CrCl 28 mL/min, severe infection.",
    "piperacillin_tazobactam": "Can you help with piperacillin-tazobactam dosing? 88 kg adult, CrCl 32 mL/min, severe intra-abdominal infection.",
    "vancomycin_iv": "Can you help with vancomycin dosing? 90 kg adult, serum creatinine 1.4 mg/dL, serious MRSA infection.",
    "daptomycin": "Can you help with daptomycin dosing? 95 kg adult, CrCl 42 mL/min, bacteremia.",
    "linezolid": "Can you help with linezolid dosing? 76 kg adult, CrCl 40 mL/min, serious gram-positive infection.",
    "levofloxacin": "Can you help with levofloxacin dosing? 74 kg adult, CrCl 36 mL/min, pneumonia.",
    "ciprofloxacin": "Can you help with ciprofloxacin dosing? 78 kg adult, CrCl 34 mL/min, pseudomonal infection.",
    "aztreonam": "Can you help with aztreonam dosing? 81 kg adult, CrCl 30 mL/min, gram-negative infection.",
    "ertapenem": "Can you help with ertapenem dosing? 79 kg adult, CrCl 29 mL/min, ESBL urinary infection.",
    "nafcillin": "Can you help with nafcillin dosing? 84 kg adult, CrCl 50 mL/min, MSSA bacteremia.",
    "ampicillin_sulbactam": "Can you help with ampicillin-sulbactam dosing? 80 kg adult, CrCl 27 mL/min, intra-abdominal infection.",
    "amoxicillin": "Can you help with amoxicillin dosing? 75 kg adult, normal renal function, streptococcal infection.",
    "amoxicillin_clavulanate": "Can you help with amoxicillin-clavulanate dosing? 77 kg adult, CrCl 44 mL/min, respiratory infection.",
    "tmp_smx": "Can you help with trimethoprim-sulfamethoxazole dosing? 73 kg adult, CrCl 31 mL/min, skin infection.",
    "nitrofurantoin": "Can you help with nitrofurantoin dosing? 68 kg adult, normal renal function, uncomplicated cystitis.",
    "fosfomycin": "Can you help with fosfomycin dosing? 69 kg adult, normal renal function, uncomplicated cystitis.",
    "fluconazole": "Can you help with fluconazole dosing? 80 kg adult, CrCl 35 mL/min, candidemia.",
    "micafungin": "Can you help with micafungin dosing? 83 kg adult, CrCl 33 mL/min, candidemia.",
    "voriconazole": "Can you help with voriconazole dosing? 75 kg adult, CrCl 48 mL/min, invasive mold infection.",
    "oseltamivir": "Can you help with oseltamivir dosing? 71 kg adult, CrCl 37 mL/min, influenza treatment.",
    "acyclovir_iv": "Can you help with acyclovir IV dosing? 78 kg adult, CrCl 41 mL/min, HSV encephalitis.",
    "valganciclovir": "Can you help with valganciclovir dosing? 72 kg adult, CrCl 34 mL/min, CMV treatment.",
}

DOSEID_FOLLOWUP_UPDATES = {
    "ceftriaxone": "180 cm male",
    "vancomycin_iv": "63 year old male, 180 cm",
    "cefepime": "actually the patient is on intermittent hemodialysis",
}

HUMANIZED_DOSEID_FOLLOWUP_MEDICATION_IDS = [
    "cefepime",
    "ceftriaxone",
    "cefazolin",
    "meropenem",
    "vancomycin_iv",
    "levofloxacin",
    "ertapenem",
    "fluconazole",
    "voriconazole",
    "acyclovir_iv",
]

IMMUNOID_REGIMEN_IDS = [
    "r_chop",
    "da_r_epoch",
    "br",
    "fcr",
    "seven_plus_three",
    "flag_ida",
    "vrd",
    "dara_vrd",
    "cybord",
    "hypercvad_a",
]

IMMUNOID_AGENT_IDS = [
    "prednisone_20",
    "rituximab",
    "infliximab",
    "tofacitinib",
    "cyclophosphamide",
    "tacrolimus",
    "mycophenolate_mofetil",
    "eculizumab",
    "daratumumab",
    "lenalidomide",
]

PROBID_ROUTE_CASES = [
    {
        "id": "probid_route_cap_therapy_selection",
        "description": "A pneumonia treatment question should route into CAP preset selection.",
        "message": "what would you use for this pneumonia?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "cap"},
            "options_include": ["pc_adult", "ed_adult"],
        },
    },
    {
        "id": "probid_route_vancomycin_clarification",
        "description": "A vancomycin start question should ask for syndrome/source.",
        "message": "should I start vancomycin?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_syndrome_module"},
            "options_include": ["cap", "endo", "inv_mold"],
        },
    },
    {
        "id": "probid_route_hold_antibiotics_clarification",
        "description": "A hold-antibiotics question should ask for syndrome/source.",
        "message": "can I hold antibiotics for now?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_syndrome_module"},
            "options_include": ["cap", "endo", "uti"],
        },
    },
    {
        "id": "probid_route_endocarditis",
        "description": "Explicit endocarditis routing should land on the endocarditis presets.",
        "message": "please evaluate for endocarditis",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "endo"},
            "options_include": ["endo_low"],
        },
    },
    {
        "id": "probid_route_generic_fungal_consult",
        "description": "A generic antifungal decision prompt should ask whether this is Candida or mold.",
        "message": "should ID start antifungal treatment for this patient with possible fungal infection?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_syndrome_module"},
            "options_include": ["inv_candida", "inv_mold"],
        },
    },
    {
        "id": "probid_route_mold_therapy_selection",
        "description": "Specific mold therapy-selection wording should route into the invasive mold pathway.",
        "message": "what antifungal would you use for a neutropenic patient with pulmonary nodules and positive galactomannan?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "inv_mold"},
        },
    },
    {
        "id": "probid_route_septic_arthritis",
        "description": "Explicit septic arthritis wording should route into the septic arthritis presets.",
        "message": "septic arthritis",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "septic_arthritis"},
            "options_include": ["sa_low"],
        },
    },
    {
        "id": "probid_route_spinal_epidural_abscess",
        "description": "Explicit spinal epidural abscess wording should route into the SEA presets.",
        "message": "spinal epidural abscess",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "spinal_epidural_abscess"},
            "options_include": ["sea_low"],
        },
    },
    {
        "id": "probid_route_nsti",
        "description": "Explicit NSTI wording should route into the necrotizing soft tissue infection presets.",
        "message": "necrotizing soft tissue infection",
        "expect": {
            "state": {
                "workflow": "probid",
                "stage": "select_preset",
                "moduleId": "necrotizing_soft_tissue_infection",
            },
            "options_include": ["nsti_low"],
        },
    },
    {
        "id": "probid_route_tb_uveitis",
        "description": "Explicit TB uveitis wording should route into the TB uveitis presets.",
        "message": "tb uveitis",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "active_tb"},
            "options_include": ["tb_low"],
        },
    },
]

ALLERGY_CASES = [
    {
        "id": "allergy_remote_rash_cefepime",
        "description": "Remote rash-only penicillin history should still surface cefepime guidance.",
        "message": "Remote childhood amoxicillin rash only, no anaphylaxis, now needs cefepime.",
        "agent": "Cefepime",
    },
    {
        "id": "allergy_remote_rash_ceftriaxone",
        "description": "Remote rash-only penicillin history should still surface ceftriaxone guidance.",
        "message": "Remote childhood amoxicillin rash only, no anaphylaxis, now needs ceftriaxone.",
        "agent": "Ceftriaxone",
    },
    {
        "id": "allergy_remote_rash_meropenem",
        "description": "Remote rash-only penicillin history should still surface meropenem guidance.",
        "message": "Remote childhood amoxicillin rash only, no anaphylaxis, now needs meropenem.",
        "agent": "Meropenem",
    },
    {
        "id": "allergy_vancomycin_infusion_reaction_ceftriaxone",
        "description": "Vancomycin infusion-reaction phrasing should preserve ceftriaxone guidance.",
        "message": "Vancomycin infusion reaction only, can I still use ceftriaxone?",
        "agent": "Ceftriaxone",
    },
    {
        "id": "allergy_immediate_piptazo_cefepime",
        "description": "Immediate beta-lactam reaction text should still surface cefepime guidance.",
        "message": "Immediate hives and throat tightness after piperacillin-tazobactam, now needs cefepime.",
        "agent": "Cefepime",
    },
    {
        "id": "allergy_immediate_penicillin_cefepime",
        "description": "Immediate penicillin reaction text should still surface cefepime guidance.",
        "message": "Immediate hives and throat tightness after penicillin, now needs cefepime.",
        "agent": "Cefepime",
    },
    {
        "id": "allergy_immediate_amoxicillin_cefepime",
        "description": "Immediate amoxicillin reaction text should still surface cefepime guidance.",
        "message": "Immediate hives and throat tightness after amoxicillin, now needs cefepime.",
        "agent": "Cefepime",
    },
    {
        "id": "allergy_generic_entry_prompt",
        "description": "A generic allergy request should still route into the AllergyID workflow.",
        "message": "Can you help with antibiotic allergy compatibility?",
        "agent": None,
    },
]

MECHID_CASES = [
    (
        "Escherichia coli",
        "Please interpret this susceptibility pattern. Escherichia coli bloodstream isolate. Ceftriaxone resistant, cefepime susceptible dose dependent, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Klebsiella pneumoniae",
        "Please interpret this susceptibility pattern. Klebsiella pneumoniae bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Klebsiella oxytoca",
        "Please interpret this susceptibility pattern. Klebsiella oxytoca bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Citrobacter freundii complex",
        "Please interpret this susceptibility pattern. Citrobacter freundii complex bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
    ),
    (
        "Citrobacter koseri",
        "Please interpret this susceptibility pattern. Citrobacter koseri bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Morganella morganii",
        "Please interpret this susceptibility pattern. Morganella morganii bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
    ),
    (
        "Proteus mirabilis",
        "Please interpret this susceptibility pattern. Proteus mirabilis bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Proteus vulgaris group",
        "Please interpret this susceptibility pattern. Proteus vulgaris group bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Salmonella enterica",
        "Please interpret this susceptibility pattern. Salmonella enterica bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Serratia marcescens",
        "Please interpret this susceptibility pattern. Serratia marcescens bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
    ),
    (
        "Enterobacter cloacae complex",
        "Please interpret this susceptibility pattern. Enterobacter cloacae complex bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
    ),
    (
        "Klebsiella aerogenes",
        "Please interpret this susceptibility pattern. Klebsiella aerogenes bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
    ),
    (
        "Pseudomonas aeruginosa",
        "Please interpret this susceptibility pattern. Pseudomonas aeruginosa bloodstream isolate. Piperacillin-tazobactam resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
    ),
    (
        "Acinetobacter baumannii complex",
        "Please interpret this susceptibility pattern. Acinetobacter baumannii complex bloodstream isolate. Ampicillin-sulbactam susceptible, meropenem resistant, ciprofloxacin resistant, trimethoprim-sulfamethoxazole susceptible.",
    ),
    (
        "Enterococcus faecalis",
        "Please interpret this susceptibility pattern. Enterococcus faecalis bloodstream isolate. Ampicillin susceptible, vancomycin susceptible, linezolid susceptible.",
    ),
    (
        "Enterococcus faecium",
        "Please interpret this susceptibility pattern. Enterococcus faecium bloodstream isolate. Ampicillin resistant, vancomycin resistant, linezolid susceptible.",
    ),
    (
        "Enterococcus gallinarum",
        "Please interpret this susceptibility pattern. Enterococcus gallinarum bloodstream isolate. Ampicillin susceptible, vancomycin resistant, linezolid susceptible.",
    ),
    (
        "Staphylococcus aureus",
        "Please interpret this susceptibility pattern. Staphylococcus aureus bloodstream isolate. Oxacillin resistant, vancomycin susceptible, daptomycin susceptible, linezolid susceptible.",
    ),
    (
        "Coagulase-negative Staphylococcus",
        "Please interpret this susceptibility pattern. Coagulase-negative Staphylococcus bloodstream isolate. Oxacillin resistant, vancomycin susceptible, daptomycin susceptible, linezolid susceptible.",
    ),
    (
        "Staphylococcus lugdunensis",
        "Please interpret this susceptibility pattern. Staphylococcus lugdunensis bloodstream isolate. Oxacillin susceptible, cefazolin susceptible, vancomycin susceptible.",
    ),
]

MECHID_DOSEID_BRIDGE_CASES = [
    ("Citrobacter freundii complex", "cefepime"),
    ("Serratia marcescens", "cefepime"),
    ("Enterobacter cloacae complex", "cefepime"),
    ("Klebsiella aerogenes", "cefepime"),
    ("Pseudomonas aeruginosa", "cefepime"),
    ("Acinetobacter baumannii complex", "ampicillin_sulbactam"),
    ("Enterococcus faecalis", "ampicillin"),
    ("Enterococcus faecium", "daptomycin"),
    ("Staphylococcus aureus", "vancomycin_iv"),
    ("Staphylococcus lugdunensis", "cefazolin"),
]

IMMUNOID_CONTEXT_CASES = [
    {
        "id": "immunoid_context_rituximab_strongy_no",
        "selection": "immunoid_answer:strongyloides_exposure:no",
        "expected_value": "no",
    },
    {
        "id": "immunoid_context_rituximab_strongy_yes",
        "selection": "immunoid_answer:strongyloides_exposure:yes",
        "expected_value": "yes",
    },
    {
        "id": "immunoid_context_rituximab_strongy_unknown",
        "selection": "immunoid_answer:strongyloides_exposure:unknown",
        "expected_value": "unknown",
    },
]

IMMUNOID_CONTEXT_PROMPT = (
    "Can you review immunosuppression prophylaxis? "
    "Rituximab plus prednisone 20 mg daily planned for 6 weeks. "
    "HBsAg negative, anti-HBc positive, anti-HBs negative. QuantiFERON negative."
)

HUMANIZED_IMMUNOID_CONTEXT_CASES = [
    {
        "id": "humanized_immunoid_context_rituximab_strongy_no",
        "prompt": (
            "Need a prophylaxis sanity check. Planning rituximab with prednisone 20 mg daily for about "
            "6 weeks. Hep B surface antigen is negative, core antibody is positive, surface antibody is "
            "negative, and Quantiferon is negative."
        ),
        "selection": "immunoid_answer:strongyloides_exposure:no",
        "expected_value": "no",
    },
    {
        "id": "humanized_immunoid_context_rituximab_strongy_yes",
        "prompt": (
            "Trying to think ahead on infection prevention before we give rituximab plus prednisone "
            "20 mg a day for 6 weeks. HBV surface Ag negative, core Ab positive, surface Ab negative, "
            "Quant gold negative."
        ),
        "selection": "immunoid_answer:strongyloides_exposure:yes",
        "expected_value": "yes",
    },
    {
        "id": "humanized_immunoid_context_rituximab_strongy_unknown",
        "prompt": (
            "Can you help me sort prophylaxis for rituximab plus prednisone 20 mg daily for 6 weeks? Hep B "
            "surface antigen negative, core antibody positive, surface antibody negative, Quantiferon "
            "negative."
        ),
        "selection": "immunoid_answer:strongyloides_exposure:unknown",
        "expected_value": "unknown",
    },
]

HUMANIZED_IMMUNOID_AGENT_IDS = [
    "rituximab",
    "prednisone_20",
    "tofacitinib",
    "eculizumab",
]

HUMANIZED_IMMUNOID_REGIMEN_IDS = [
    "r_chop",
    "da_r_epoch",
    "cybord",
]

HUMANIZED_IMMUNOID_CHOOSER_PROMPT = (
    "I need help figuring out infection prophylaxis before we start some pretty meaningful immunosuppression."
)

ALLERGY_FOLLOWUP_CASES = [
    {
        "id": "allergy_followup_immediate_penicillin_cefepime",
        "detail": "Immediate hives and throat tightness after penicillin, now needs cefepime.",
        "agent": "Cefepime",
    },
    {
        "id": "allergy_followup_vanco_infusion_ceftriaxone",
        "detail": "Vancomycin infusion reaction only, can I still use ceftriaxone?",
        "agent": "Ceftriaxone",
    },
]

ALLERGY_FOLLOWUP_PROMPT = "Can you help with antibiotic allergy compatibility? The patient says they had a penicillin allergy."

HUMANIZED_ALLERGY_CASES = [
    {
        "id": "humanized_allergy_remote_rash_cefepime",
        "description": "Conversational remote rash wording should still support cefepime.",
        "message": (
            "Trying to sort out beta-lactam options. As a kid the patient got a blotchy rash from "
            "amoxicillin, definitely no anaphylaxis, and right now I am wondering about cefepime."
        ),
        "agent": "Cefepime",
    },
    {
        "id": "humanized_allergy_remote_rash_ceftriaxone",
        "description": "Conversational remote rash wording should still support ceftriaxone.",
        "message": (
            "This sounds low-risk but I want a second pass: remote childhood amoxicillin rash only, no "
            "throat swelling or anaphylaxis, could I use ceftriaxone?"
        ),
        "agent": "Ceftriaxone",
    },
    {
        "id": "humanized_allergy_remote_rash_meropenem",
        "description": "Conversational remote rash wording should still support meropenem.",
        "message": (
            "History is just a childhood rash with amoxicillin and no immediate features. We may need "
            "meropenem now. How does that allergy history read to you?"
        ),
        "agent": "Meropenem",
    },
    {
        "id": "humanized_allergy_vancomycin_infusion_reaction_ceftriaxone",
        "description": "Vancomycin infusion-reaction wording should still preserve ceftriaxone guidance.",
        "message": (
            "The chart says vancomycin allergy, but the story sounds like flushing and itching during the "
            "infusion rather than a real allergy. Is ceftriaxone still fine?"
        ),
        "agent": "Ceftriaxone",
    },
    {
        "id": "humanized_allergy_immediate_piptazo_cefepime",
        "description": "Immediate piperacillin-tazobactam wording should still surface cefepime guidance.",
        "message": (
            "He had hives and throat tightness right after piperacillin-tazobactam, so I am treating that "
            "like a true immediate reaction. If I need cefepime now, how risky is that?"
        ),
        "agent": "Cefepime",
    },
    {
        "id": "humanized_allergy_immediate_penicillin_cefepime",
        "description": "Immediate penicillin wording should still surface cefepime guidance.",
        "message": (
            "Real-time question from the floor: immediate hives and throat tightness after penicillin in "
            "the past, but the team wants cefepime. Can you sort that out?"
        ),
        "agent": "Cefepime",
    },
    {
        "id": "humanized_allergy_immediate_amoxicillin_cefepime",
        "description": "Immediate amoxicillin wording should still surface cefepime guidance.",
        "message": (
            "This one sounds more convincing for IgE: amoxicillin caused hives and throat tightness pretty "
            "quickly. We are thinking about cefepime now."
        ),
        "agent": "Cefepime",
    },
    {
        "id": "humanized_allergy_generic_entry_prompt",
        "description": "A messy allergy-history request should still enter AllergyID.",
        "message": (
            "Can you help me untangle an antibiotic allergy list? The chart just says penicillin allergy "
            "and I do not really trust it."
        ),
        "agent": None,
    },
]

HUMANIZED_PROBID_ROUTE_CASES = [
    {
        "id": "humanized_probid_route_cap_therapy_selection",
        "description": "A conversational pneumonia treatment question should still route into CAP presets.",
        "message": "Could you help me think through what antibiotics you would start for what looks like pneumonia?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "cap"},
            "options_include": ["pc_adult", "ed_adult"],
        },
    },
    {
        "id": "humanized_probid_route_vancomycin_clarification",
        "description": "A conversational vancomycin question should still ask for syndrome/source.",
        "message": "Team keeps asking whether we should just start vancomycin, but I feel like I need the syndrome framework first.",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_syndrome_module"},
            "options_include": ["cap", "endo", "inv_mold"],
        },
    },
    {
        "id": "humanized_probid_route_hold_antibiotics_clarification",
        "description": "A conversational hold-antibiotics question should still ask for syndrome/source.",
        "message": "Before I keep broad coverage going, can I safely hold antibiotics for a bit or is that a bad idea here?",
        "expect": {
            "any_of": [
                {
                    "state": {"workflow": "probid", "stage": "select_syndrome_module"},
                    "options_include": ["cap", "endo", "uti"],
                },
                {
                    "state": {"workflow": "probid", "stage": "select_module"},
                    "options_include": ["empiric_therapy"],
                    "message_contains": ["Which clinical syndrome?"],
                },
            ],
        },
    },
    {
        "id": "humanized_probid_route_endocarditis",
        "description": "Conversational endocarditis wording should still land on endocarditis presets.",
        "message": "I am worried this bacteremia story could be endocarditis. Can you walk me through that pathway?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "endo"},
            "options_include": ["endo_low"],
        },
    },
    {
        "id": "humanized_probid_route_generic_fungal_consult",
        "description": "A conversational antifungal decision prompt should still ask Candida versus mold.",
        "message": "Need help deciding whether ID should start antifungal therapy at all here. I am not even sure if we are in Candida or mold territory.",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_syndrome_module"},
            "options_include": ["inv_candida", "inv_mold"],
        },
    },
    {
        "id": "humanized_probid_route_mold_therapy_selection",
        "description": "Conversational mold-therapy wording should still route into the invasive mold pathway.",
        "message": "Neutropenic patient, chest CT with nodules, galactomannan came back positive. What mold-active drug would you reach for?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "inv_mold"},
        },
    },
    {
        "id": "humanized_probid_route_septic_arthritis",
        "description": "Conversational septic-arthritis wording should still route into the septic arthritis presets.",
        "message": "This might be a septic joint. Can you put me in the right consult flow for septic arthritis?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "septic_arthritis"},
            "options_include": ["sa_low"],
        },
    },
    {
        "id": "humanized_probid_route_spinal_epidural_abscess",
        "description": "Conversational SEA wording should still route into the spinal epidural abscess presets.",
        "message": "Back pain plus fever plus new weakness, and I am worried about a spinal epidural abscess.",
        "expect": {
            "any_of": [
                {
                    "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "spinal_epidural_abscess"},
                    "options_include": ["sea_low"],
                },
                {
                    "state": {"workflow": "probid", "stage": "confirm_case", "moduleId": "spinal_epidural_abscess"},
                    "options_include": ["run_assessment"],
                },
            ],
        },
    },
    {
        "id": "humanized_probid_route_nsti",
        "description": "Conversational NSTI wording should still route into the necrotizing soft tissue presets.",
        "message": "This soft tissue infection is moving way too fast and the pain seems out of proportion. I need the NSTI pathway.",
        "expect": {
            "state": {
                "workflow": "probid",
                "stage": "select_preset",
                "moduleId": "necrotizing_soft_tissue_infection",
            },
            "options_include": ["nsti_low"],
        },
    },
    {
        "id": "humanized_probid_route_tb_uveitis",
        "description": "Conversational TB-uveitis wording should still route into the TB presets.",
        "message": "Ophthalmology is asking whether this uveitis picture could be TB-related. Can you route me to the TB uveitis workflow?",
        "expect": {
            "state": {"workflow": "probid", "stage": "select_preset", "moduleId": "active_tb"},
            "options_include": ["tb_low"],
        },
    },
]

HUMANIZED_GUIDED_MESSAGE_PREFIXES = [
    "Let me give the case in chunks: ",
    "A little more color: ",
    "Sorry, another detail I should have mentioned: ",
    "Micro-wise, ",
    "Imaging-wise, ",
]

NOISY_DOSEID_ALIASES = {
    "piperacillin_tazobactam": "zosyn",
    "vancomycin_iv": "vanc",
    "ertapenem": "invanz",
    "nitrofurantoin": "macrobid",
    "fosfomycin": "monurol",
    "amoxicillin_clavulanate": "augmentin",
    "levofloxacin": "levaquin",
    "fluconazole": "diflucan",
    "micafungin": "mycamine",
    "voriconazole": "vfend",
}

NOISY_DOSEID_FOLLOWUP_CASES = [
    {
        "medication_id": "cefepime",
        "initial": "Need cefepime dose pls. 72 kg pt, severe infection.",
        "updates": ["crcl 45", "actually on intermittent hemodialysis now"],
    },
    {
        "medication_id": "ceftriaxone",
        "initial": "Need ceftriaxone dose. 70 kg pt, serious infection.",
        "updates": ["crcl 45"],
    },
    {
        "medication_id": "cefazolin",
        "initial": "Need cefazolin dose pls. 82 kg pt, complicated MSSA infection.",
        "updates": ["crcl 38"],
    },
    {
        "medication_id": "meropenem",
        "initial": "Need meropenem dose. 85 kg pt, severe infection.",
        "updates": ["crcl 28"],
    },
    {
        "medication_id": "vancomycin_iv",
        "initial": "Need vanc dose. 90 kg pt, serious MRSA infection.",
        "updates": ["scr 1.4, 63 year old male, 180 cm"],
    },
    {
        "medication_id": "levofloxacin",
        "initial": "Need levaquin dose. 74 kg pt, pneumonia.",
        "updates": ["crcl 36"],
    },
    {
        "medication_id": "ertapenem",
        "initial": "Need invanz dose. 79 kg pt, ESBL urinary infection.",
        "updates": ["crcl 29"],
    },
    {
        "medication_id": "fluconazole",
        "initial": "Need diflucan dose. 80 kg pt, candidemia.",
        "updates": ["crcl 35"],
    },
    {
        "medication_id": "voriconazole",
        "initial": "Need vfend dose. 75 kg pt, invasive mold infection.",
        "updates": ["crcl 48, female, 165 cm"],
    },
    {
        "medication_id": "acyclovir_iv",
        "initial": "Need IV acyclovir dose. 78 kg pt, HSV encephalitis.",
        "updates": ["crcl 41"],
    },
]

NOISY_GUIDED_MODULE_IDS = [
    "bacterial_meningitis",
    "encephalitis",
    "spinal_epidural_abscess",
    "necrotizing_soft_tissue_infection",
    "tb_uveitis",
]

NOISY_GUIDED_MESSAGE_PREFIXES = [
    "sorry, typing fast from the room: ",
    "adding more bc the signout was messy: ",
    "one more chunk from the chart: ",
    "copying the key detail here: ",
]

NOISY_ROUTE_MESSAGE_PREFIXES = [
    "Sorry, typing from my phone. ",
    "Need the right lane first: ",
    "This may be a clumsy way to ask it, but ",
    "Quick steer before rounds: ",
]

NOISY_ALLERGY_MESSAGE_PREFIXES = [
    "Chart allergy story is messy. ",
    "Trying to clean up a probably-wrong allergy label before rounds. ",
    "Sorry this is copied straight from signout. ",
    "Phone call summary from family was basically: ",
]

NOISY_IMMUNOID_MESSAGE_PREFIXES = [
    "Sorry, this is copied from the onc note: ",
    "Trying to sort this before chemo starts: ",
    "Messy pre-treatment note says: ",
]


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a list of cases in {path}")
    return loaded


def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("'", "")
    )


def _prompt_details_from_clean_prompt(prompt: str) -> str:
    details = prompt.split("?", 1)[-1].strip()
    details = details.rstrip(".")
    details = details.replace("CrCl", "creatinine clearance")
    details = details.replace("adult", "adult patient")
    details = details.replace("normal renal function", "kidneys are normal")
    return details


def _humanized_doseid_prompt(medication_name: str, prompt: str, variant_index: int) -> str:
    details = _prompt_details_from_clean_prompt(prompt)
    templates = [
        "Quick dosing gut-check on {med}. {details}. What would you do?",
        "Probably a basic one, but can you help me dose {med}? {details}.",
        "Need to start {med} and just want a sanity check before I click the order. {details}.",
        "Trying not to underdose {med} here. {details}.",
        "Can you walk me through {med} dosing for this patient? {details}.",
    ]
    return templates[variant_index % len(templates)].format(med=medication_name, details=details)


def _humanized_followup_text(follow_up: str, variant_index: int) -> str:
    update = follow_up.rstrip(".")
    templates = [
        "Actually, one correction: {update}.",
        "Sorry, I left out something important: {update}.",
        "Quick update before you answer: {update}.",
        "One more detail that changes things a bit: {update}.",
    ]
    return templates[variant_index % len(templates)].format(update=update)


def _humanized_mechid_prompt(organism: str, prompt: str, variant_index: int) -> str:
    details = prompt.replace("Please interpret this susceptibility pattern. ", "").strip()
    details = details.replace("bloodstream isolate", "blood isolate")
    templates = [
        "Can you sanity-check this blood culture susceptibility report for me? {details}",
        "Trying to make sense of this AST from a blood culture. {details}",
        "Mind helping me interpret this blood isolate? {details}",
        "I am getting lost in this susceptibility report. {details}",
    ]
    return templates[variant_index % len(templates)].format(details=details.replace(organism, organism, 1))


def _humanized_guided_message(message: str, variant_index: int) -> str:
    prefix = HUMANIZED_GUIDED_MESSAGE_PREFIXES[variant_index % len(HUMANIZED_GUIDED_MESSAGE_PREFIXES)]
    return f"{prefix}{message}"


def _noisy_doseid_prompt(medication_id: str, medication_name: str, prompt: str, variant_index: int) -> str:
    label = NOISY_DOSEID_ALIASES.get(medication_id, medication_name).lower()
    details = _prompt_details_from_clean_prompt(prompt)
    details = details.replace("creatinine clearance", "crcl")
    details = details.replace("serum creatinine", "scr")
    templates = [
        "pls sanity-check {label} dose. {details}. sorry for the shorthand.",
        "quick one from the floor re {label}: {details}.",
        "need help w {label} dosing. {details}. med list is messy.",
        "re: {label} - {details}. what would you do?",
        "can you gut-check {label} for me? {details}. thx.",
    ]
    return templates[variant_index % len(templates)].format(label=label, details=details)


def _noisy_followup_text(update: str, variant_index: int) -> str:
    cleaned = update.rstrip(".")
    templates = [
        "sorry, missed this earlier: {update}.",
        "one correction from the bedside: {update}.",
        "forgot to mention: {update}.",
        "chart update since my first message: {update}.",
    ]
    return templates[variant_index % len(templates)].format(update=cleaned)


def _noisy_mechid_prompt(organism: str, prompt: str, variant_index: int) -> str:
    details = prompt.replace("Please interpret this susceptibility pattern. ", "").strip()
    details = details.replace("bloodstream isolate", "blood culture isolate")
    templates = [
        "micro note from overnight: {details} can you sanity-check that pattern?",
        "blood cx update - {details} i am trying to make sense of it.",
        "messy signout text: {details} what do you think is going on?",
        "from the culture report: {details} pls help interpret.",
    ]
    return templates[variant_index % len(templates)].format(details=details.replace(organism, organism, 1))


def _noisy_route_message(message: str, variant_index: int) -> str:
    prefix = NOISY_ROUTE_MESSAGE_PREFIXES[variant_index % len(NOISY_ROUTE_MESSAGE_PREFIXES)]
    return f"{prefix}{message}"


def _noisy_allergy_message(message: str, variant_index: int) -> str:
    prefix = NOISY_ALLERGY_MESSAGE_PREFIXES[variant_index % len(NOISY_ALLERGY_MESSAGE_PREFIXES)]
    return f"{prefix}{message}"


def _noisy_immunoid_message(message: str, variant_index: int) -> str:
    prefix = NOISY_IMMUNOID_MESSAGE_PREFIXES[variant_index % len(NOISY_IMMUNOID_MESSAGE_PREFIXES)]
    return f"{prefix}{message}"


def _noisy_guided_message(message: str, variant_index: int) -> str:
    prefix = NOISY_GUIDED_MESSAGE_PREFIXES[variant_index % len(NOISY_GUIDED_MESSAGE_PREFIXES)]
    return f"{prefix}{message}"


def _nonempty_any(*paths: str) -> Dict[str, Any]:
    return {"any_of": [{"nonempty_paths": [path]} for path in paths]}


def _doseid_case(medication_id: str, prompt: str, medication_name: str) -> Dict[str, Any]:
    expect: Dict[str, Any] = {
        "state": {"workflow": "doseid", "stage": "doseid_describe"},
        "json_contains": {"doseidAnalysis.medications": medication_name},
    }
    expect.update(_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"))
    return {
        "id": f"doseid_{medication_id}",
        "description": f"DoseID should parse {medication_name} dosing text into the dosing workflow.",
        "turns": [{"request": {"message": prompt}, "expect": expect}],
    }


def _doseid_followup_case(
    medication_id: str,
    prompt: str,
    medication_name: str,
    follow_up: str,
) -> Dict[str, Any]:
    return {
        "id": f"doseid_followup_{medication_id}",
        "description": f"DoseID should keep {medication_name} in the dosing lane after a follow-up update.",
        "turns": [
            {
                "request": {"message": prompt},
                "expect": {
                    "state": {"workflow": "doseid", "stage": "doseid_describe"},
                    "json_contains": {"doseidAnalysis.medications": medication_name},
                },
            },
            {
                "request": {"message": follow_up},
                "expect": {
                    "state": {"workflow": "doseid", "stage": "doseid_describe"},
                    "json_contains": {"doseidAnalysis.medications": medication_name},
                    "options_include": ["add_more_details"],
                    **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                },
            },
        ],
    }


def _immunoid_agent_case(
    agent_id: str,
    agent_name: str,
    *,
    initial_prompt: str = "Can you help with immunosuppression prophylaxis?",
    case_id_prefix: str = "immunoid_agent",
    description: str | None = None,
) -> Dict[str, Any]:
    return {
        "id": f"{case_id_prefix}_{agent_id}",
        "description": description or f"ImmunoID should accept a direct {agent_name} selection from the chooser.",
        "turns": [
            {
                "request": {"message": initial_prompt},
                "expect": {
                    "state": {"workflow": "immunoid", "stage": "immunoid_select_agents"},
                    "options_include": ["restart"],
                },
            },
            {
                "request": {"selection": f"immunoid_agent:{agent_id}"},
                "expect": {
                    "state": {"workflow": "immunoid"},
                    "options_include": ["restart"],
                    "list_item_checks": [
                        {
                            "path": "immunoidAnalysis.selectedAgents",
                            "where": {"id": agent_id},
                            "nonempty_paths": ["name"],
                        }
                    ],
                },
            },
        ],
    }


def _immunoid_regimen_case(
    regimen_id: str,
    regimen_name: str,
    *,
    initial_prompt: str = "Can you help with immunosuppression prophylaxis?",
    case_id_prefix: str = "immunoid_regimen",
    description: str | None = None,
) -> Dict[str, Any]:
    return {
        "id": f"{case_id_prefix}_{regimen_id}",
        "description": description or f"ImmunoID should accept a direct {regimen_name} selection from the chooser.",
        "turns": [
            {
                "request": {"message": initial_prompt},
                "expect": {
                    "state": {"workflow": "immunoid", "stage": "immunoid_select_agents"},
                    "options_include": ["restart"],
                },
            },
            {
                "request": {"selection": f"immunoid_regimen:{regimen_id}"},
                "expect": {
                    "state": {"workflow": "immunoid"},
                    "options_include": ["restart"],
                    "list_item_checks": [
                        {
                            "path": "immunoidAnalysis.selectedRegimens",
                            "where": {"id": regimen_id},
                            "nonempty_paths": ["name"],
                        }
                    ],
                },
            },
        ],
    }


def _immunoid_context_case(case: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "description": "ImmunoID should keep the checklist context after a structured follow-up answer.",
        "turns": [
            {
                "request": {"message": case.get("prompt", IMMUNOID_CONTEXT_PROMPT)},
                "expect": {
                    "state": {"workflow": "immunoid", "stage": "immunoid_collect_context"},
                    "options_include": [case["selection"]],
                },
            },
            {
                "request": {"selection": case["selection"]},
                "expect": {
                    "state": {"workflow": "immunoid"},
                    "options_include": ["restart"],
                    "nonempty_paths": ["immunoidAnalysis.recommendations", "immunoidAnalysis.exposureSummary"],
                    "list_item_checks": [
                        {
                            "path": "immunoidAnalysis.exposureSummary",
                            "where": {"id": "strongyloides_exposure"},
                            "equals": {"value": case["expected_value"]},
                        }
                    ],
                    "any_of": [
                        {"state": {"stage": "immunoid_collect_context"}},
                        {"state": {"stage": "done"}},
                    ],
                },
            },
        ],
    }


def _guided_probid_case(module_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    turns: List[Dict[str, Any]] = [
        {
            "request": {},
            "expect": {"state": {"stage": "select_module", "workflow": "probid"}},
        },
        {
            "request": {"selection": "probid"},
            "expect": {
                "state": {"stage": "select_syndrome_module", "workflow": "probid"},
                "options_include": [module_id],
            },
        },
        {
            "request": {"selection": module_id},
            "expect": {
                "state": {"stage": "select_preset", "workflow": "probid", "moduleId": module_id},
                "options_include": [spec["preset"]],
            },
        },
        {
            "request": {"selection": spec["preset"]},
            "expect": {
                "state": {
                    "stage": "select_pretest_factors",
                    "workflow": "probid",
                    "moduleId": module_id,
                    "presetId": spec["preset"],
                },
                "options_include": ["continue_to_case"],
            },
        },
    ]

    for turn in spec["assistant_turns"]:
        expect = {"state": {"workflow": "probid", "moduleId": module_id}}
        if turn.get("selection") == "continue_to_case":
            expect["state"]["stage"] = "describe_case"
        turns.append({"request": dict(turn), "expect": expect})

    turns.append(
        {
            "request": {"selection": "run_assessment"},
            "expect": {
                "state": {"stage": "done", "workflow": "probid", "moduleId": module_id},
                "nonempty_paths": ["analysis.analysis.appliedFindings"],
            },
        }
    )

    return {
        "id": f"probid_guided_{module_id}",
        "description": f"Guided {module_id.replace('_', ' ')} consult reaches a final ProbID analysis.",
        "turns": turns,
    }


def _humanized_guided_probid_case(module_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    case = _guided_probid_case(module_id, spec)
    case["id"] = f"humanized_{case['id']}"
    case["description"] = f"Guided {module_id.replace('_', ' ')} consult stays stable with conversational case fragments."

    humanized_turns: List[Dict[str, Any]] = []
    message_index = 0
    for turn in case["turns"]:
        turn_copy = {
            "request": dict(turn["request"]),
            "expect": dict(turn["expect"]),
        }
        if "message" in turn_copy["request"]:
            turn_copy["request"]["message"] = _humanized_guided_message(
                str(turn_copy["request"]["message"]),
                message_index,
            )
            message_index += 1
        humanized_turns.append(turn_copy)
    case["turns"] = humanized_turns
    return case


def _noisy_guided_probid_case(module_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    case = _guided_probid_case(module_id, spec)
    case["id"] = f"noisy_{case['id']}"
    case["description"] = f"Guided {module_id.replace('_', ' ')} consult stays stable with messy real-world phrasing."

    noisy_turns: List[Dict[str, Any]] = []
    message_index = 0
    for turn in case["turns"]:
        turn_copy = {
            "request": dict(turn["request"]),
            "expect": dict(turn["expect"]),
        }
        if "message" in turn_copy["request"]:
            turn_copy["request"]["message"] = _noisy_guided_message(
                str(turn_copy["request"]["message"]),
                message_index,
            )
            message_index += 1
        noisy_turns.append(turn_copy)
    case["turns"] = noisy_turns
    return case


def _allergy_case(case: Dict[str, Any]) -> Dict[str, Any]:
    expect: Dict[str, Any] = {
        "state": {"workflow": "allergyid", "stage": "done"},
        "options_include": ["restart"],
    }
    if case["agent"]:
        expect["list_item_checks"] = [
            {
                "path": "allergyidAnalysis.recommendations",
                "where": {"agent": case["agent"]},
                "nonempty_paths": ["recommendation"],
            }
        ]
    return {
        "id": case["id"],
        "description": case["description"],
        "turns": [{"request": {"message": case["message"]}, "expect": expect}],
    }


def _allergy_followup_case(case: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "description": "AllergyID should stay stable when a vague prompt is followed by concrete allergy details.",
        "turns": [
            {
                "request": {"message": ALLERGY_FOLLOWUP_PROMPT},
                "expect": {
                    "state": {"workflow": "allergyid", "stage": "done"},
                },
            },
            {
                "request": {"message": case["detail"]},
                "expect": {
                    "state": {"workflow": "allergyid", "stage": "done"},
                    "list_item_checks": [
                        {
                            "path": "allergyidAnalysis.recommendations",
                            "where": {"agent": case["agent"]},
                            "nonempty_paths": ["recommendation"],
                        }
                    ],
                },
            },
        ],
    }


def _noisy_allergy_followup_case(case: Dict[str, str], variant_index: int) -> Dict[str, Any]:
    return {
        "id": f"noisy_{case['id']}",
        "description": "AllergyID should stay stable when a messy allergy prompt is followed by clarified details.",
        "turns": [
            {
                "request": {
                    "message": _noisy_allergy_message(
                        "Can you help me sort an antibiotic allergy list? The chart mostly just says penicillin allergy.",
                        variant_index,
                    )
                },
                "expect": {
                    "state": {"workflow": "allergyid", "stage": "done"},
                },
            },
            {
                "request": {"message": _noisy_allergy_message(case["detail"], variant_index + 1)},
                "expect": {
                    "state": {"workflow": "allergyid", "stage": "done"},
                    "list_item_checks": [
                        {
                            "path": "allergyidAnalysis.recommendations",
                            "where": {"agent": case["agent"]},
                            "nonempty_paths": ["recommendation"],
                        }
                    ],
                },
            },
        ],
    }


def _mechid_case(organism: str, prompt: str) -> Dict[str, Any]:
    return {
        "id": f"mechid_{_slugify(organism)}",
        "description": f"MechID should parse and hold state for {organism}.",
        "turns": [
            {
                "request": {"message": prompt},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                    "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                    "nonempty_paths": [
                        "mechidAnalysis.parsedRequest.susceptibilityResults",
                        "mechidAnalysis.analysis.rows",
                    ],
                    "options_include": ["add_more_details", "restart"],
                },
            }
        ],
    }


def _mechid_finalize_case(organism: str, prompt: str) -> Dict[str, Any]:
    return {
        "id": f"mechid_finalize_{_slugify(organism)}",
        "description": f"MechID should finish the {organism} consult after a syndrome is selected.",
        "turns": [
            {
                "request": {"message": prompt},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                    "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                },
            },
            {
                "request": {"selection": "mechid_set_syndrome:Bacteraemia"},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "done"},
                    "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                    "nonempty_paths": ["mechidAnalysis.analysis.rows", "mechidAnalysis.analysis.references"],
                    "options_include": ["add_more_details"],
                },
            },
        ],
    }


def _mechid_doseid_bridge_case(
    organism: str,
    prompt: str,
    medication_id: str,
    medication_name: str,
) -> Dict[str, Any]:
    pick_value = f"doseid_pick:{medication_id}"
    return {
        "id": f"mechid_to_doseid_{_slugify(organism)}_{medication_id}",
        "description": f"MechID should carry {organism} into DoseID for {medication_name}.",
        "turns": [
            {
                "request": {"message": prompt},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                    "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                },
            },
            {
                "request": {"selection": "mechid_set_syndrome:Bacteraemia"},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "done"},
                    "options_include": [pick_value],
                },
            },
            {
                "request": {"selection": pick_value},
                "expect": {
                    "state": {"workflow": "doseid", "stage": "doseid_describe"},
                    "json_contains": {"doseidAnalysis.medications": medication_name},
                    **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                },
            },
        ],
    }


def _noisy_doseid_followup_case(
    medication_name: str,
    medication_id: str,
    initial_prompt: str,
    updates: List[str],
) -> Dict[str, Any]:
    turns: List[Dict[str, Any]] = [
        {
            "request": {"message": initial_prompt},
            "expect": {
                "state": {"workflow": "doseid", "stage": "doseid_describe"},
                "json_contains": {"doseidAnalysis.medications": medication_name},
                **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
            },
        }
    ]
    for index, update in enumerate(updates):
        turns.append(
            {
                "request": {"message": _noisy_followup_text(update, index)},
                "expect": {
                    "state": {"workflow": "doseid", "stage": "doseid_describe"},
                    "json_contains": {"doseidAnalysis.medications": medication_name},
                    **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                },
            }
        )
    return {
        "id": f"noisy_doseid_followup_{medication_id}",
        "description": f"DoseID should stay stable across rough correction turns for {medication_name}.",
        "turns": turns,
    }


def _noisy_mechid_doseid_bridge_case(
    organism: str,
    prompt: str,
    medication_id: str,
    medication_name: str,
    variant_index: int,
) -> Dict[str, Any]:
    pick_value = f"doseid_pick:{medication_id}"
    return {
        "id": f"noisy_mechid_to_doseid_{_slugify(organism)}_{medication_id}",
        "description": f"Messy AST input should still carry {organism} into DoseID for {medication_name}.",
        "turns": [
            {
                "request": {"message": _noisy_mechid_prompt(organism, prompt, variant_index)},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                    "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                },
            },
            {
                "request": {"selection": "mechid_set_syndrome:Bacteraemia"},
                "expect": {
                    "state": {"workflow": "mechid", "stage": "done"},
                    "options_include": [pick_value],
                },
            },
            {
                "request": {"selection": pick_value},
                "expect": {
                    "state": {"workflow": "doseid", "stage": "doseid_describe"},
                    "json_contains": {"doseidAnalysis.medications": medication_name},
                    **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                },
            },
        ],
    }


def _build_standard_cases() -> List[Dict[str, Any]]:
    base_cases = _load_cases(BASE_DATASET)

    supported_medications = {med.id: med for med in list_medications()}
    supported_regimens = set(IMMUNOID_REGIMENS)
    supported_agents = set(IMMUNOID_SOURCE_AGENTS)
    supported_organisms = set(list_mechid_organisms())

    generated_cases: List[Dict[str, Any]] = []
    generated_cases.extend(base_cases)
    generated_cases.extend(_guided_probid_case(module_id, spec) for module_id, spec in NEW_SYNDROME_CASES.items())
    generated_cases.extend(
        {
            "id": case["id"],
            "description": case["description"],
            "turns": [{"request": {"message": case["message"]}, "expect": case["expect"]}],
        }
        for case in PROBID_ROUTE_CASES
    )

    for medication_id, prompt in DOSEID_PROMPTS.items():
        medication = supported_medications.get(medication_id)
        if medication is None:
            raise SystemExit(f"Unsupported DoseID medication in generator: {medication_id}")
        generated_cases.append(_doseid_case(medication_id, prompt, medication.name))

    for regimen_id in IMMUNOID_REGIMEN_IDS:
        regimen = IMMUNOID_REGIMENS.get(regimen_id)
        if regimen is None or regimen_id not in supported_regimens:
            raise SystemExit(f"Unsupported ImmunoID regimen in generator: {regimen_id}")
        generated_cases.append(_immunoid_regimen_case(regimen_id, regimen["name"]))

    for agent_id in IMMUNOID_AGENT_IDS:
        agent = IMMUNOID_SOURCE_AGENTS.get(agent_id)
        if agent is None or agent_id not in supported_agents:
            raise SystemExit(f"Unsupported ImmunoID agent in generator: {agent_id}")
        generated_cases.append(_immunoid_agent_case(agent_id, str(agent["name"])))

    for case in ALLERGY_CASES:
        generated_cases.append(_allergy_case(case))

    for organism, prompt in MECHID_CASES:
        if organism not in supported_organisms:
            raise SystemExit(f"Unsupported MechID organism in generator: {organism}")
        generated_cases.append(_mechid_case(organism, prompt))

    return generated_cases


def _build_expanded_cases() -> List[Dict[str, Any]]:
    generated_cases = _build_standard_cases()
    supported_medications = {med.id: med for med in list_medications()}
    mechid_prompts = {organism: prompt for organism, prompt in MECHID_CASES}

    for medication_id, prompt in DOSEID_PROMPTS.items():
        medication = supported_medications[medication_id]
        follow_up = DOSEID_FOLLOWUP_UPDATES.get(
            medication_id,
            "the patient is 180 cm tall and female",
        )
        generated_cases.append(_doseid_followup_case(medication_id, prompt, medication.name, follow_up))

    for organism, prompt in MECHID_CASES:
        generated_cases.append(_mechid_finalize_case(organism, prompt))

    for organism, medication_id in MECHID_DOSEID_BRIDGE_CASES:
        medication = supported_medications.get(medication_id)
        prompt = mechid_prompts.get(organism)
        if medication is None or prompt is None:
            raise SystemExit(f"Unsupported MechID bridge case: {organism} -> {medication_id}")
        generated_cases.append(_mechid_doseid_bridge_case(organism, prompt, medication_id, medication.name))

    for case in IMMUNOID_CONTEXT_CASES:
        generated_cases.append(_immunoid_context_case(case))

    for case in ALLERGY_FOLLOWUP_CASES:
        generated_cases.append(_allergy_followup_case(case))

    return generated_cases


def _build_humanized_cases() -> List[Dict[str, Any]]:
    generated_cases = _build_expanded_cases()
    supported_medications = {med.id: med for med in list_medications()}
    supported_regimens = set(IMMUNOID_REGIMENS)
    supported_agents = set(IMMUNOID_SOURCE_AGENTS)
    supported_organisms = set(list_mechid_organisms())
    mechid_prompts = {organism: prompt for organism, prompt in MECHID_CASES}

    for index, (medication_id, prompt) in enumerate(DOSEID_PROMPTS.items()):
        medication = supported_medications.get(medication_id)
        if medication is None:
            raise SystemExit(f"Unsupported DoseID medication in humanized generator: {medication_id}")
        generated_cases.append(
            {
                "id": f"humanized_doseid_{medication_id}",
                "description": f"Conversational dosing wording should still keep {medication.name} in DoseID.",
                "turns": [
                    {
                        "request": {"message": _humanized_doseid_prompt(medication.name, prompt, index)},
                        "expect": {
                            "state": {"workflow": "doseid", "stage": "doseid_describe"},
                            "json_contains": {"doseidAnalysis.medications": medication.name},
                            **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                        },
                    }
                ],
            }
        )

    for index, medication_id in enumerate(HUMANIZED_DOSEID_FOLLOWUP_MEDICATION_IDS):
        medication = supported_medications.get(medication_id)
        prompt = DOSEID_PROMPTS.get(medication_id)
        if medication is None or prompt is None:
            raise SystemExit(f"Unsupported humanized DoseID follow-up case: {medication_id}")
        follow_up = DOSEID_FOLLOWUP_UPDATES.get(
            medication_id,
            "the patient is 180 cm tall and female",
        )
        generated_cases.append(
            {
                "id": f"humanized_doseid_followup_{medication_id}",
                "description": f"Conversational updates should keep {medication.name} in DoseID after the first turn.",
                "turns": [
                    {
                        "request": {"message": _humanized_doseid_prompt(medication.name, prompt, index)},
                        "expect": {
                            "state": {"workflow": "doseid", "stage": "doseid_describe"},
                            "json_contains": {"doseidAnalysis.medications": medication.name},
                        },
                    },
                    {
                        "request": {"message": _humanized_followup_text(follow_up, index)},
                        "expect": {
                            "state": {"workflow": "doseid", "stage": "doseid_describe"},
                            "json_contains": {"doseidAnalysis.medications": medication.name},
                            "options_include": ["add_more_details"],
                            **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                        },
                    },
                ],
            }
        )

    for index, (organism, prompt) in enumerate(MECHID_CASES):
        if organism not in supported_organisms:
            raise SystemExit(f"Unsupported MechID organism in humanized generator: {organism}")
        generated_cases.append(
            {
                "id": f"humanized_mechid_{_slugify(organism)}",
                "description": f"Conversational AST wording should still parse {organism} in MechID.",
                "turns": [
                    {
                        "request": {"message": _humanized_mechid_prompt(organism, prompt, index)},
                        "expect": {
                            "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                            "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                            "nonempty_paths": [
                                "mechidAnalysis.parsedRequest.susceptibilityResults",
                                "mechidAnalysis.analysis.rows",
                            ],
                            "options_include": ["add_more_details", "restart"],
                        },
                    }
                ],
            }
        )

    for index, (organism, medication_id) in enumerate(MECHID_DOSEID_BRIDGE_CASES):
        medication = supported_medications.get(medication_id)
        prompt = mechid_prompts.get(organism)
        if medication is None or prompt is None:
            raise SystemExit(f"Unsupported humanized MechID bridge case: {organism} -> {medication_id}")
        generated_cases.append(
            {
                "id": f"humanized_mechid_to_doseid_{_slugify(organism)}_{medication_id}",
                "description": f"Conversational AST input should still bridge {organism} into DoseID for {medication.name}.",
                "turns": [
                    {
                        "request": {"message": _humanized_mechid_prompt(organism, prompt, index)},
                        "expect": {
                            "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                            "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                        },
                    },
                    {
                        "request": {"selection": "mechid_set_syndrome:Bacteraemia"},
                        "expect": {
                            "state": {"workflow": "mechid", "stage": "done"},
                            "options_include": [f"doseid_pick:{medication_id}"],
                        },
                    },
                    {
                        "request": {"selection": f"doseid_pick:{medication_id}"},
                        "expect": {
                            "state": {"workflow": "doseid", "stage": "doseid_describe"},
                            "json_contains": {"doseidAnalysis.medications": medication.name},
                            **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                        },
                    },
                ],
            }
        )

    for case in HUMANIZED_ALLERGY_CASES:
        generated_cases.append(_allergy_case(case))

    for case in HUMANIZED_PROBID_ROUTE_CASES:
        generated_cases.append(
            {
                "id": case["id"],
                "description": case["description"],
                "turns": [{"request": {"message": case["message"]}, "expect": case["expect"]}],
            }
        )

    for case in HUMANIZED_IMMUNOID_CONTEXT_CASES:
        generated_cases.append(_immunoid_context_case(case))

    for agent_id in HUMANIZED_IMMUNOID_AGENT_IDS:
        agent = IMMUNOID_SOURCE_AGENTS.get(agent_id)
        if agent is None or agent_id not in supported_agents:
            raise SystemExit(f"Unsupported humanized ImmunoID agent in generator: {agent_id}")
        generated_cases.append(
            _immunoid_agent_case(
                agent_id,
                str(agent["name"]),
                initial_prompt=HUMANIZED_IMMUNOID_CHOOSER_PROMPT,
                case_id_prefix="humanized_immunoid_agent",
                description=f"Conversational prophylaxis wording should still allow selecting {agent['name']}.",
            )
        )

    for regimen_id in HUMANIZED_IMMUNOID_REGIMEN_IDS:
        regimen = IMMUNOID_REGIMENS.get(regimen_id)
        if regimen is None or regimen_id not in supported_regimens:
            raise SystemExit(f"Unsupported humanized ImmunoID regimen in generator: {regimen_id}")
        generated_cases.append(
            _immunoid_regimen_case(
                regimen_id,
                str(regimen["name"]),
                initial_prompt=HUMANIZED_IMMUNOID_CHOOSER_PROMPT,
                case_id_prefix="humanized_immunoid_regimen",
                description=f"Conversational prophylaxis wording should still allow selecting {regimen['name']}.",
            )
        )

    for module_id, spec in NEW_SYNDROME_CASES.items():
        generated_cases.append(_humanized_guided_probid_case(module_id, spec))

    return generated_cases


def _build_noisy_cases() -> List[Dict[str, Any]]:
    generated_cases = _build_humanized_cases()
    supported_medications = {med.id: med for med in list_medications()}
    supported_regimens = set(IMMUNOID_REGIMENS)
    supported_agents = set(IMMUNOID_SOURCE_AGENTS)
    supported_organisms = set(list_mechid_organisms())
    mechid_prompts = {organism: prompt for organism, prompt in MECHID_CASES}

    for index, (medication_id, prompt) in enumerate(DOSEID_PROMPTS.items()):
        medication = supported_medications.get(medication_id)
        if medication is None:
            raise SystemExit(f"Unsupported DoseID medication in noisy generator: {medication_id}")
        generated_cases.append(
            {
                "id": f"noisy_doseid_{medication_id}",
                "description": f"Messy shorthand should still keep {medication.name} in DoseID.",
                "turns": [
                    {
                        "request": {"message": _noisy_doseid_prompt(medication_id, medication.name, prompt, index)},
                        "expect": {
                            "state": {"workflow": "doseid", "stage": "doseid_describe"},
                            "json_contains": {"doseidAnalysis.medications": medication.name},
                            **_nonempty_any("doseidAnalysis.recommendations", "doseidAnalysis.followUpQuestions"),
                        },
                    }
                ],
            }
        )

    for case in NOISY_DOSEID_FOLLOWUP_CASES:
        medication_id = case["medication_id"]
        medication = supported_medications.get(medication_id)
        if medication is None:
            raise SystemExit(f"Unsupported noisy DoseID follow-up case: {medication_id}")
        generated_cases.append(
            _noisy_doseid_followup_case(
                medication.name,
                medication_id,
                str(case["initial"]),
                [str(item) for item in case["updates"]],
            )
        )

    for index, (organism, prompt) in enumerate(MECHID_CASES):
        if organism not in supported_organisms:
            raise SystemExit(f"Unsupported MechID organism in noisy generator: {organism}")
        generated_cases.append(
            {
                "id": f"noisy_mechid_{_slugify(organism)}",
                "description": f"Messy AST signout wording should still parse {organism} in MechID.",
                "turns": [
                    {
                        "request": {"message": _noisy_mechid_prompt(organism, prompt, index)},
                        "expect": {
                            "state": {"workflow": "mechid", "stage": "mechid_confirm"},
                            "json_equals": {"mechidAnalysis.parsedRequest.organism": organism},
                            "nonempty_paths": [
                                "mechidAnalysis.parsedRequest.susceptibilityResults",
                                "mechidAnalysis.analysis.rows",
                            ],
                            "options_include": ["add_more_details", "restart"],
                        },
                    }
                ],
            }
        )

    for index, case in enumerate(HUMANIZED_ALLERGY_CASES):
        noisy_case = dict(case)
        noisy_case["id"] = f"noisy_{case['id']}"
        noisy_case["description"] = f"Messy chart-style allergy wording should still preserve {case['agent'] or 'allergy'} guidance."
        noisy_case["message"] = _noisy_allergy_message(str(case["message"]), index)
        generated_cases.append(_allergy_case(noisy_case))

    for index, case in enumerate(ALLERGY_FOLLOWUP_CASES):
        generated_cases.append(_noisy_allergy_followup_case(case, index))

    for index, case in enumerate(HUMANIZED_PROBID_ROUTE_CASES):
        generated_cases.append(
            {
                "id": f"noisy_{case['id']}",
                "description": f"Messy route-selection wording should still behave for {case['id']}.",
                "turns": [
                    {
                        "request": {"message": _noisy_route_message(str(case["message"]), index)},
                        "expect": case["expect"],
                    }
                ],
            }
        )

    for index, case in enumerate(HUMANIZED_IMMUNOID_CONTEXT_CASES):
        noisy_case = dict(case)
        noisy_case["id"] = f"noisy_{case['id']}"
        noisy_case["prompt"] = _noisy_immunoid_message(str(case["prompt"]), index)
        generated_cases.append(_immunoid_context_case(noisy_case))

    for index, agent_id in enumerate(HUMANIZED_IMMUNOID_AGENT_IDS):
        agent = IMMUNOID_SOURCE_AGENTS.get(agent_id)
        if agent is None or agent_id not in supported_agents:
            raise SystemExit(f"Unsupported noisy ImmunoID agent in generator: {agent_id}")
        generated_cases.append(
            _immunoid_agent_case(
                agent_id,
                str(agent["name"]),
                initial_prompt=_noisy_immunoid_message(HUMANIZED_IMMUNOID_CHOOSER_PROMPT, index),
                case_id_prefix="noisy_immunoid_agent",
                description=f"Messy chooser wording should still allow selecting {agent['name']}.",
            )
        )

    for index, regimen_id in enumerate(HUMANIZED_IMMUNOID_REGIMEN_IDS):
        regimen = IMMUNOID_REGIMENS.get(regimen_id)
        if regimen is None or regimen_id not in supported_regimens:
            raise SystemExit(f"Unsupported noisy ImmunoID regimen in generator: {regimen_id}")
        generated_cases.append(
            _immunoid_regimen_case(
                regimen_id,
                str(regimen["name"]),
                initial_prompt=_noisy_immunoid_message(HUMANIZED_IMMUNOID_CHOOSER_PROMPT, index + 1),
                case_id_prefix="noisy_immunoid_regimen",
                description=f"Messy chooser wording should still allow selecting {regimen['name']}.",
            )
        )

    for module_id in NOISY_GUIDED_MODULE_IDS:
        generated_cases.append(_noisy_guided_probid_case(module_id, NEW_SYNDROME_CASES[module_id]))

    for index, (organism, medication_id) in enumerate(MECHID_DOSEID_BRIDGE_CASES):
        medication = supported_medications.get(medication_id)
        prompt = mechid_prompts.get(organism)
        if medication is None or prompt is None:
            raise SystemExit(f"Unsupported noisy MechID bridge case: {organism} -> {medication_id}")
        generated_cases.append(
            _noisy_mechid_doseid_bridge_case(
                organism,
                prompt,
                medication_id,
                medication.name,
                index,
            )
        )

    return generated_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate assistant evaluation datasets.")
    parser.add_argument(
        "--profile",
        choices=["standard", "expanded", "humanized", "noisy"],
        default="standard",
        help="Dataset profile to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to the profile's standard dataset path.",
    )
    args = parser.parse_args()

    if args.profile == "noisy":
        generated_cases = _build_noisy_cases()
        expected_count = 360
    elif args.profile == "humanized":
        generated_cases = _build_humanized_cases()
        expected_count = 260
    elif args.profile == "expanded":
        generated_cases = _build_expanded_cases()
        expected_count = 160
    else:
        generated_cases = _build_standard_cases()
        expected_count = 100

    if len(generated_cases) != expected_count:
        raise SystemExit(f"Expected {expected_count} generated cases, found {len(generated_cases)}")

    output_path = args.output or OUTPUT_DATASETS[args.profile]
    output_path.write_text(json.dumps(generated_cases, indent=2) + "\n")
    print(f"Wrote {len(generated_cases)} cases to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
