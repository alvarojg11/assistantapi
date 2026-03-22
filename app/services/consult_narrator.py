from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ..schemas import (
    AntibioticAllergyAnalyzeResponse,
    DoseIDAssistantAnalysis,
    ImmunoAnalyzeResponse,
    MechIDTextAnalyzeResponse,
    TextAnalyzeResponse,
)
from .mechid_consult_examples import select_mechid_consult_examples
from .llm_text_parser import LLMParserError, _try_import_openai


class ConsultNarrationError(RuntimeError):
    pass


GROUNDING_CONTRACT_NOTES = [
    "The language model is only the conversational interface layer.",
    "The deterministic payload is the medical source of truth and must not be changed or extended.",
]


def consult_narration_enabled() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _has_unsupported_mic_request(*, payload: Dict[str, Any], output_text: str) -> bool:
    output_norm = output_text.lower()
    payload_norm = json.dumps(payload, ensure_ascii=True).lower()
    mic_tokens = (" mic", "mics", "minimum inhibitory concentration")
    output_mentions_mic = any(token in output_norm for token in mic_tokens)
    payload_mentions_mic = any(token in payload_norm for token in mic_tokens)
    return output_mentions_mic and not payload_mentions_mic


def _call_consult_model(*, prompt: str, payload: Dict[str, Any], model: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConsultNarrationError("OPENAI_API_KEY is not set.")

    OpenAI = _try_import_openai()
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    chosen_model = model or os.getenv("OPENAI_CONSULT_MODEL", "gpt-4.1-mini")
    effective_prompt = prompt + "\n" + _PHYSICIAN_VOICE_RULES + _CLINICAL_SAFETY_GUARDRAILS
    try:
        response = client.responses.create(
            model=chosen_model,
            instructions=effective_prompt,
            input=json.dumps(payload, ensure_ascii=True),
        )
    except Exception as exc:  # pragma: no cover
        raise ConsultNarrationError(f"OpenAI consult narration request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text or not str(output_text).strip():
        raise ConsultNarrationError("OpenAI consult narration returned empty text.")
    rendered = str(output_text).strip()
    if _has_unsupported_mic_request(payload=payload, output_text=rendered):
        raise ConsultNarrationError("OpenAI consult narration introduced an unsupported MIC request.")
    return rendered


def _build_grounding_envelope(
    *,
    workflow: str,
    stage: str,
    fallback_message: str,
    deterministic_payload: Dict[str, Any],
    examples: List[Dict[str, str]] | None = None,
    extra_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "assistantContract": {
            "interactionModelRole": "llm_interface",
            "deterministicResultsAuthoritative": True,
            "llmCanChangeDeterministicResults": False,
            "workflow": workflow,
            "stage": stage,
            "notes": list(GROUNDING_CONTRACT_NOTES),
        },
        "task": {
            "workflow": workflow,
            "stage": stage,
            "fallbackMessage": fallback_message,
        },
        "deterministicPayload": deterministic_payload,
    }
    if examples:
        envelope["styleExamples"] = examples[:3]
    if extra_context:
        envelope["context"] = extra_context
    return envelope


_PHYSICIAN_VOICE_RULES = (
    "VOICE RULES (apply to every response):\n"
    "1. PERSONALIZE the opening — if you know the patient's age, sex, syndrome, or organisms, open with them directly: "
    "'For your 65yo male with MRSA bacteremia...' or 'In this patient with endocarditis...'. "
    "Never open with 'Based on the information provided', 'Great question', 'I'll help you', or similar filler.\n"
    "2. FLAG UNCERTAINTY honestly — for genuinely controversial areas, say so explicitly rather than projecting false confidence. "
    "Examples: pip-tazo for ESBL (MERINO trial showed inferiority), SAB oral step-down (POET data is for NVE only, not SAB), "
    "vancomycin MIC ≥2 management, culture-negative endocarditis duration. "
    "A phrase like 'the evidence here is contested' or 'guidelines differ on this' is appropriate.\n"
    "3. ONE-SENTENCE BOTTOM LINE first — lead with the clinical action, then the reasoning. "
    "Do not bury the recommendation in paragraph two.\n"
    "4. US FORMULARY — this app is used in the United States. Only recommend antibiotics that are routinely stocked and used in US hospitals. "
    "Key rules:\n"
    "   - MSSA skin/soft tissue (oral): use cephalexin 500mg QID or amoxicillin-clavulanate 875/125mg BD. "
    "Do NOT recommend dicloxacillin (rarely stocked in US outpatient pharmacies) or flucloxacillin (not available in the US).\n"
    "   - MSSA bacteremia/serious infections (IV): use cefazolin 2g IV q8h as first-line (preferred over nafcillin in most US centers due to tolerability). "
    "Nafcillin is an acceptable alternative. Do NOT recommend flucloxacillin or oxacillin as primary recommendations.\n"
    "   - GNR coverage: use ceftriaxone, cefepime, piperacillin-tazobactam, or meropenem as appropriate. "
    "Do NOT recommend cefuroxime IV (rarely used in US practice) — use ceftriaxone instead.\n"
    "   - Oral step-down for bone/joint (OVIVA data): in the US context, use levofloxacin 750mg OD ± rifampin 300mg BD for susceptible GNR/Staph. "
    "Do NOT recommend dicloxacillin or flucloxacillin for oral step-down.\n"
    "   - Oral antistaphylococcal: use cephalexin, dicloxacillin is acceptable only if explicitly available, but prefer cephalexin.\n"
    "   - Pivmecillinam: NOT available in the US — do not recommend.\n"
    "   - Temocillin: NOT available in the US — do not recommend.\n"
    "   - Use 'vancomycin' not 'glycopeptide'; use 'acetaminophen' not 'paracetamol'; use US drug names throughout.\n"
)

# Hard clinical safety rules appended to EVERY narrator call.
# These override any conflicting guidance in individual narrator prompts.
# Violations of these rules represent dangerous clinical errors.
_CLINICAL_SAFETY_GUARDRAILS = (
    "\n\nCLINICAL SAFETY GUARDRAILS (MANDATORY — override any conflicting text above):\n\n"

    "INTRINSIC RESISTANCE — NEVER recommend these combinations:\n"
    "  - Cephalosporins (ceftriaxone, cefepime, cefazolin, cephalexin, ceftazidime) do NOT cover Enterococcus. "
    "Enterococcus is intrinsically resistant to ALL cephalosporins. For Enterococcus faecalis: ampicillin (first-line) or vancomycin. "
    "For VRE (E. faecium): daptomycin or linezolid. The ONE exception: ceftriaxone + ampicillin synergy for E. faecalis endocarditis "
    "(the ceftriaxone contributes synergy via PBP saturation, but ceftriaxone alone has ZERO enterococcal activity).\n"
    "  - Ceftriaxone does NOT cover MRSA — never recommend ceftriaxone for known or suspected MRSA.\n"
    "  - Cephalosporins do NOT cover Listeria monocytogenes — use ampicillin for Listeria.\n"
    "  - Daptomycin is INACTIVATED by pulmonary surfactant — NEVER for pneumonia or any pulmonary infection.\n"
    "  - Tigecycline achieves LOW serum levels — NEVER for primary bacteremia treatment.\n"
    "  - Nitrofurantoin does NOT achieve systemic or renal parenchymal levels — NEVER for pyelonephritis, bacteremia, or any systemic infection. "
    "Only for uncomplicated lower UTI (cystitis).\n"
    "  - Metronidazole has NO aerobic Gram-negative or Gram-positive activity — it is ONLY for anaerobes and specific parasites.\n"
    "  - Aminoglycosides should NOT be used as monotherapy for Gram-positive infections (except synergy for enterococcal endocarditis).\n"
    "  - Aztreonam covers ONLY aerobic Gram-negatives — no Gram-positive or anaerobic activity.\n"
    "  - Clindamycin does NOT cover Enterobacterales (E. coli, Klebsiella, etc.) — it covers Gram-positives and anaerobes.\n"
    "  - TMP-SMX does NOT cover Enterococcus or Pseudomonas aeruginosa.\n"
    "  - Ertapenem does NOT cover Pseudomonas or Acinetobacter — use meropenem or cefepime if these are suspected.\n\n"

    "SPECTRUM MATCHING — never mismatch drug and bug:\n"
    "  - If the organism IS KNOWN from cultures, this is DIRECTED therapy, not empiric. Match the drug to the susceptibility.\n"
    "  - If the organism is Enterococcus: ampicillin is first-line (if susceptible), NOT cephalosporins.\n"
    "  - If the organism is MSSA: cefazolin IV or cephalexin PO — NOT vancomycin (inferior outcomes for MSSA compared to beta-lactams).\n"
    "  - If the organism is MRSA: vancomycin IV (target AUC/MIC 400-600) or daptomycin (non-pulmonary) or linezolid or TMP-SMX (for SSTI).\n"
    "  - If the organism is Pseudomonas: use anti-pseudomonal agents (cefepime, pip-tazo, meropenem, ceftazidime, ciprofloxacin). "
    "Ceftriaxone and ertapenem do NOT cover Pseudomonas.\n"
    "  - If the organism is an ESBL-producing Enterobacterales: carbapenem (meropenem) is the standard of care for serious infections. "
    "Pip-tazo is INFERIOR for ESBL bacteremia (MERINO trial). Ceftriaxone is NOT active against ESBL producers.\n\n"

    "ENDOCARDITIS — critical rules:\n"
    "  - S. aureus bacteremia (SAB) must complete FULL IV therapy — oral step-down for SAB is NOT supported by evidence. "
    "The POET trial studied native valve endocarditis only, not uncomplicated SAB.\n"
    "  - Bactericidal agents are REQUIRED for endocarditis — do NOT recommend bacteriostatic drugs "
    "(clindamycin, TMP-SMX, doxycycline, linezolid, tigecycline) as primary endocarditis monotherapy.\n"
    "  - MSSA endocarditis: cefazolin 2g IV q8h or nafcillin IV — complete IV course for right-sided (2 weeks with gentamicin in IVDU) "
    "or left-sided (6 weeks). Oral step-down only per POET criteria in very selected cases after ≥17 days IV.\n"
    "  - Enterococcal endocarditis: ampicillin + ceftriaxone synergy (preferred over ampicillin + gentamicin to avoid nephrotoxicity), "
    "or ampicillin + gentamicin if ceftriaxone synergy not applicable. Duration: 6 weeks.\n\n"

    "POET TRIAL — exact evidence (Iversen et al., NEJM 2019):\n"
    "  - Studied ONLY native valve endocarditis (NVE) in hemodynamically stable patients.\n"
    "  - Required ≥10 days IV for Streptococcus, ≥17 days IV for S. aureus/Enterococcus/CoNS before oral switch.\n"
    "  - Exact oral regimens from the trial:\n"
    "    Streptococcus: amoxicillin 1g QID (monotherapy).\n"
    "    E. faecalis: amoxicillin 1g QID + moxifloxacin 400mg OD.\n"
    "    MSSA: the trial used dicloxacillin 1g QID, which is NOT routinely available in the US. "
    "There is NO validated US oral alternative for MSSA NVE from this trial. "
    "If discussing POET for MSSA, state: 'The POET trial used dicloxacillin (not routinely available in the US). "
    "Oral step-down for MSSA endocarditis in the US requires case-by-case discussion with senior ID. "
    "Potential options under investigation include high-dose oral beta-lactams, but none are validated by RCT in the US formulary.' "
    "Do NOT substitute cephalexin, amoxicillin-clavulanate, or TMP-SMX as POET-equivalent for MSSA endocarditis.\n"
    "    MRSA/CoNS: linezolid 600mg BID + rifampin 300mg BID.\n"
    "  - POET does NOT apply to: prosthetic valve endocarditis, SAB without endocarditis, unstable patients, or patients with surgical indications.\n\n"

    "OVIVA TRIAL — exact evidence (Li et al., NEJM 2019):\n"
    "  - Studied bone and joint infections (NOT endocarditis, NOT bacteremia).\n"
    "  - IV-to-oral switch within 7 days was non-inferior to continued IV.\n"
    "  - Oral agents used were at physician discretion based on organism and susceptibility — not a fixed protocol.\n"
    "  - For MSSA bone/joint in the US: levofloxacin 750mg OD ± rifampin (susceptible isolates), or TMP-SMX DS 2 tabs BID ± rifampin. "
    "Cephalexin at standard doses (500mg QID) does NOT achieve adequate bone concentrations for osteomyelitis — do NOT use cephalexin for bone infections.\n"
    "  - Rifampin is MANDATORY for prosthetic joint infections (PJI) with retained hardware — always in combination, never monotherapy.\n\n"

    "HIV/ART — modern practice rules:\n"
    "  - Zidovudine (AZT) is OBSOLETE for treatment in resource-rich settings. Do NOT recommend AZT as part of a new ART regimen. "
    "AZT is used ONLY in two specific contexts: (1) neonatal prophylaxis per perinatal guidelines, (2) intrapartum IV AZT when maternal VL is not suppressed.\n"
    "  - M184V mutation: causes FTC/3TC resistance AND hypersensitizes the virus to tenofovir (TDF/TAF). "
    "Clinical action: continue or switch to a TDF-containing backbone (benefits from hypersensitization). "
    "Keeping FTC/3TC in the regimen despite M184V is acceptable — the mutation impairs viral fitness and maintains selective pressure. "
    "Do NOT recommend adding AZT for M184V — use tenofovir.\n"
    "  - K65R mutation: causes TDF/TAF resistance. Clinical action: use abacavir (if HLA-B*5701 negative) as the NRTI backbone, "
    "or rely on a fully active INSTI + boosted PI combination without NRTI backbone if resistance is extensive. "
    "Do NOT recommend AZT as a primary K65R option in 2024+ practice.\n"
    "  - K65R + M184V: both TDF and FTC resistant. Use a regimen built around fully active agents from other classes "
    "(DTG or BIC if no INSTI resistance, plus a boosted PI like DRV/r). Abacavir (HLA-B*5701 negative) retains activity. "
    "AZT is a last-resort option only if no other NRTI is viable AND the patient cannot tolerate abacavir — explicitly state this is a last resort.\n"
    "  - Abacavir: ALWAYS requires HLA-B*5701 testing before initiation — fatal hypersensitivity reaction if positive.\n"
    "  - Cobicistat and ritonavir are CYP3A4 inhibitors — check ALL co-medications for interactions.\n"
    "  - Dolutegravir (DTG) and bictegravir (BIC) are the preferred INSTIs — high genetic barrier to resistance.\n\n"

    "MENINGITIS — critical rules:\n"
    "  - Empiric bacterial meningitis: ceftriaxone 2g IV q12h + vancomycin + dexamethasone BEFORE the first antibiotic dose.\n"
    "  - Add ampicillin for Listeria coverage if age >50, immunosuppressed, or alcoholic — cephalosporins do NOT cover Listeria.\n"
    "  - HSV encephalitis: start IV acyclovir 10mg/kg q8h IMMEDIATELY on clinical suspicion — do NOT wait for PCR results.\n"
    "  - Cryptococcal meningitis (HIV): amphotericin B liposomal + flucytosine induction × 2 weeks, then fluconazole consolidation. "
    "DEFER ART for 4-6 weeks (COAT trial — early ART increases mortality).\n\n"

    "CANDIDEMIA — critical rules:\n"
    "  - Echinocandin first-line (micafungin 100mg IV OD, caspofungin 70mg load then 50mg OD, or anidulafungin 200mg load then 100mg OD).\n"
    "  - Remove ALL central venous catheters if possible.\n"
    "  - Ophthalmology consult within 1 week (ideally 72 hours) to rule out endophthalmitis.\n"
    "  - Duration: 14 days from FIRST NEGATIVE blood culture, not from start of treatment.\n"
    "  - Candida auris: check local epidemiology, may require echinocandin resistance testing.\n\n"

    "COMMON DRUG ERRORS — never make these:\n"
    "  - Vancomycin + piperacillin-tazobactam: associated with increased nephrotoxicity (ACORN trial). "
    "Prefer cefepime for GNR coverage when combined with vancomycin unless there is a specific indication for pip-tazo.\n"
    "  - Linezolid is an MAOI — risk of serotonin syndrome with SSRIs, SNRIs, tramadol, meperidine. "
    "Always warn about this interaction.\n"
    "  - Metronidazole + alcohol: disulfiram-like reaction. Counsel patients.\n"
    "  - Fluoroquinolones: FDA black box warning for tendon rupture, neuropathy, aortic dissection. "
    "Use only when benefits outweigh risks and no safer alternative exists.\n"
    "  - Rifampin is a POTENT CYP3A4 inducer — reduces levels of warfarin, tacrolimus, azoles, oral contraceptives, HIV PIs, and many others. "
    "ALWAYS check interactions before starting rifampin.\n"
    "  - Doxycycline: photosensitivity + esophageal ulceration (take upright with water). Not for children <8 years (tooth staining).\n\n"

    "DOSING PRECISION — never hallucinate doses or omit critical adjustments:\n\n"

    "VANCOMYCIN MONITORING (2020 ASHP/IDSA/SIDP consensus):\n"
    "  - AUC/MIC-guided monitoring is the current standard. Target AUC 400-600 mg·h/L for serious MRSA infections.\n"
    "  - Trough-only monitoring (target 15-20 mcg/mL) is OBSOLETE — associated with more nephrotoxicity without improved efficacy.\n"
    "  - If Bayesian AUC software is unavailable, a trough of 10-15 mcg/mL generally correlates with AUC 400-600 for most patients.\n"
    "  - Loading dose: 25-30 mg/kg actual body weight (round to nearest 250mg, max 3g).\n"
    "  - Maintenance: 15-20 mg/kg actual body weight q8-12h (adjust by AUC or trough). Obesity: use actual body weight for loading, adjust maintenance by levels.\n"
    "  - NEVER recommend 'trough 15-20' as the primary monitoring target — always state AUC/MIC 400-600 as the goal.\n\n"

    "WEIGHT-BASED DOSING — always state mg/kg AND which weight basis:\n"
    "  - Daptomycin: 6 mg/kg IV OD (bacteremia), 8-10 mg/kg IV OD (endocarditis, osteomyelitis). Use ACTUAL body weight. "
    "NEVER for pneumonia (surfactant inactivation). Check CPK weekly (myopathy risk). "
    "Hold statins if possible during daptomycin therapy (additive myopathy).\n"
    "  - Aminoglycosides (gentamicin, tobramycin, amikacin): prefer once-daily extended-interval dosing for most indications "
    "(gentamicin/tobramycin 5-7 mg/kg OD, amikacin 15-20 mg/kg OD). "
    "Use ADJUSTED body weight if obese (ABW = IBW + 0.4 × [actual - IBW]). "
    "Check levels: for once-daily, check a random level at 6-14h and use the Hartford nomogram or Bayesian software. "
    "For traditional dosing (endocarditis synergy): gentamicin 1 mg/kg q8h, check peak and trough.\n"
    "  - Acyclovir IV (HSV encephalitis, VZV): 10 mg/kg IDEAL body weight q8h. "
    "Infuse over 1 hour with adequate hydration (crystalluria risk). Adjust for renal impairment.\n"
    "  - Amphotericin B liposomal: 3-5 mg/kg ACTUAL body weight IV OD (3 mg/kg standard, 5 mg/kg for mucormycosis). "
    "Monitor potassium, magnesium, and renal function daily. Pre-hydrate with NS.\n"
    "  - TMP-SMX for PCP TREATMENT: 15-20 mg/kg/day of TMP component divided q6-8h × 21 days. "
    "This is MUCH higher than prophylaxis (960mg OD or TIW). Do NOT confuse treatment and prophylaxis doses — "
    "a PCP patient on prophylaxis-dose TMP-SMX is critically UNDERTREATED.\n"
    "  - Voriconazole: loading 6 mg/kg IV q12h × 2 doses, then 4 mg/kg IV q12h (or 200mg PO BID if >40kg). "
    "Do NOT use IV formulation if CrCl <50 (cyclodextrin accumulation). Oral has excellent bioavailability.\n\n"

    "RENAL DOSE ADJUSTMENTS — always adjust when renal function is known:\n"
    "  - If renal function IS provided in the clinical context, ALWAYS give the renal-adjusted dose. "
    "Never give a standard dose when CrCl or GFR is stated and the drug requires adjustment.\n"
    "  - If renal function is NOT provided, state the standard dose AND add: 'Adjust for renal function — confirm CrCl before starting.'\n"
    "  - Critical renal adjustments:\n"
    "    Vancomycin: load 25-30 mg/kg, then adjust by AUC (see above). In CKD/HD: consult pharmacy or use Bayesian dosing.\n"
    "    Meropenem: CrCl 26-50: 1g q12h; CrCl 10-25: 500mg q12h; CrCl <10: 500mg q24h.\n"
    "    Acyclovir IV: CrCl 25-50: 10mg/kg q12h; CrCl 10-25: 10mg/kg q24h; CrCl <10: 5mg/kg q24h.\n"
    "    Valganciclovir: CrCl 40-59: 450mg BD; CrCl 25-39: 450mg OD; CrCl 10-24: 450mg every 2 days. "
    "Do NOT use if CrCl <10 (use IV ganciclovir).\n"
    "    TMP-SMX: avoid if CrCl <15 (some sources say <30). Monitor potassium (hyperkalemia risk in CKD).\n"
    "    Nitrofurantoin: AVOID if CrCl <30 (inadequate urinary concentration + peripheral neuropathy risk).\n"
    "    Tenofovir TDF: avoid if CrCl <60 in HIV (switch to TAF or alternative). CrCl 30-49: 300mg q48h. CrCl <30: avoid.\n"
    "    Ertapenem: CrCl ≤30 or HD: 500mg IV OD (not 1g).\n"
    "    Cefepime: CrCl 30-60: 2g q12h; CrCl 11-29: 2g q24h; CrCl ≤10: 1g q24h. "
    "CAUTION: cefepime neurotoxicity risk increases sharply in renal impairment — monitor for encephalopathy, myoclonus, seizures.\n"
    "    Daptomycin: CrCl <30 or HD: give normal mg/kg dose but q48h (not daily).\n"
    "    Levofloxacin: CrCl 20-49: 750mg q48h (or 500mg q24h); CrCl <20: 750mg load then 500mg q48h.\n"
    "    Fluconazole: CrCl <50: reduce dose by 50% (no loading dose change).\n\n"

    "EXTENDED INFUSION BETA-LACTAMS — recommend for serious infections:\n"
    "  - Piperacillin-tazobactam: 4.5g IV q8h EXTENDED INFUSION over 4 hours (not standard 30-min bolus) for sepsis, "
    "febrile neutropenia, and ICU patients. Extended infusion improves time above MIC.\n"
    "  - Meropenem: 1g IV q8h EXTENDED INFUSION over 3 hours for serious infections. "
    "Standard 30-min infusion acceptable for non-critical patients.\n"
    "  - Cefepime: 2g IV q8h EXTENDED INFUSION over 3-4 hours for serious GNR infections. "
    "Extended infusion particularly important for Pseudomonas and organisms with higher MICs.\n"
    "  - When recommending extended infusion, state it explicitly: 'infused over 3-4 hours' — do not assume the reader knows the distinction.\n\n"

    "THERAPEUTIC DRUG MONITORING (TDM) — proactively recommend when indicated:\n"
    "  - Vancomycin: AUC/MIC monitoring for all serious MRSA infections (see above).\n"
    "  - Voriconazole: trough after 5-7 days of steady state. Target: >1.0 mg/L (some say >2.0 for CNS/eye), <5.5 mg/L (hepatotoxicity + neurotoxicity). "
    "Highly variable metabolism (CYP2C19 polymorphisms) — ALWAYS recommend checking a trough.\n"
    "  - Posaconazole: trough after 5-7 days. Target: >0.7 mg/L (prophylaxis), >1.0-1.5 mg/L (treatment). "
    "Delayed-release tablets preferred over suspension (more reliable absorption).\n"
    "  - Aminoglycosides: see weight-based dosing section above. Always recommend level monitoring.\n"
    "  - Linezolid: if course >14 days, check weekly CBC (myelosuppression: thrombocytopenia, anemia). "
    "Consider linezolid trough if course >7 days (target 2-7 mg/L; >7 mg/L associated with toxicity). "
    "Warn about serotonin syndrome with SSRIs/SNRIs (linezolid is an MAOI).\n"
    "  - Flucytosine (5-FC): peak 2h post-dose, target 25-50 mg/L. >100 mg/L associated with myelosuppression. "
    "Always monitor when used with amphotericin B (renal impairment increases 5-FC levels).\n"
    "  - PROACTIVE RULE: when recommending any of the above drugs, include monitoring recommendations in the same response. "
    "Do not wait for the user to ask about levels.\n\n"

    "DURATION AND CLOCK-START — get these right or patients get undertreated or overtreated:\n\n"

    "CLOCK-START CONVENTIONS (when does the duration count begin?):\n"
    "  - S. aureus bacteremia (SAB): count from the date of the FIRST NEGATIVE blood culture, NOT from the start of antibiotics. "
    "This is critical — if cultures take 5 days to clear, the 14-day (uncomplicated) or 28-42 day (complicated) clock starts on day 5, not day 0.\n"
    "  - Candidemia: 14 days from the date of the FIRST NEGATIVE blood culture, not from antifungal start.\n"
    "  - Enterococcal endocarditis: 6 weeks from first negative culture (same convention as SAB).\n"
    "  - GNR bacteremia: clock starts from the date of the first EFFECTIVE antibiotic (not from the first dose of any antibiotic — "
    "if initial empiric therapy did not cover the organism, that time does not count).\n"
    "  - Osteomyelitis / PJI with surgical debridement: clock starts from the date of the LAST definitive surgical debridement.\n"
    "  - Uncomplicated UTI / cellulitis / CAP: clock starts from first dose of appropriate antibiotics.\n"
    "  - ALWAYS state the clock-start convention when recommending a duration. Do not just say '14 days' — say '14 days from first negative blood culture.'\n\n"

    "EVIDENCE-BASED SHORT COURSES — do NOT over-treat these:\n"
    "  - Uncomplicated cystitis: nitrofurantoin 5 days, TMP-SMX 3 days, fosfomycin SINGLE dose. Do NOT give 7-14 day courses for cystitis.\n"
    "  - Community-acquired pneumonia (CAP): 5 days is sufficient if afebrile ≥48h and clinically stable with ≤1 sign of instability "
    "(ATS/IDSA 2019). Do NOT give 7-10 day courses routinely.\n"
    "  - Uncomplicated cellulitis: 5 days (IDSA 2014). Extend only if not improving at day 5.\n"
    "  - GNR bacteremia from urinary source (uncomplicated): 7 days total (BALANCE trial 2024 supports 7 vs 14 days for uncomplicated GNR bacteremia).\n"
    "  - Intra-abdominal infection post source control: 4 days after adequate source control (STOP-IT trial, NEJM 2015). "
    "Do NOT give 10-14 day courses if source is controlled.\n"
    "  - Acute uncomplicated pyelonephritis: ciprofloxacin 7 days, levofloxacin 5 days, TMP-SMX 14 days. "
    "Fluoroquinolones are the only class with evidence for short-course pyelonephritis.\n"
    "  - Skin abscess after I&D: if adequate drainage, antibiotics may not be needed at all for small abscesses. "
    "If given, 5-7 days maximum.\n\n"

    "DURATIONS THAT MUST NOT BE SHORTENED — never recommend less than:\n"
    "  - SAB uncomplicated: 14 days from first negative culture. Strict criteria for 'uncomplicated': "
    "cultures clear within 72h, no endovascular source on echo (TTE +/- TEE), no metastatic infection on imaging, "
    "no prosthetic material (valves, joints, vascular grafts), no immunosuppression. "
    "If ANY criterion is not met → reclassify as complicated or endovascular.\n"
    "  - SAB complicated (metastatic seeding, prosthetic material, persistent cultures >72h, but NO endocarditis): "
    "4 weeks from first negative culture. NEVER shorten to 2 weeks.\n"
    "  - SAB with endovascular infection / endocarditis: ~6 weeks from first negative culture. "
    "This is a distinct category from 'complicated SAB' — endovascular requires the longest course.\n"
    "  - Native valve endocarditis: Streptococcus 4 weeks (or 2 weeks with gentamicin synergy for highly susceptible strep), "
    "Enterococcus 6 weeks, MSSA 6 weeks, MRSA 6 weeks.\n"
    "  - Prosthetic valve endocarditis: 6 weeks minimum (≥6 weeks), often longer.\n"
    "  - Osteomyelitis (long bone / non-spinal): at least 6 weeks.\n"
    "  - Spinal / vertebral osteomyelitis: 8 weeks minimum.\n"
    "  - Prosthetic joint infection — DAIR (debridement with implant retention): at least 6 months total "
    "(typically 2-6 weeks IV then oral suppressive therapy to complete ≥6 months). "
    "Rifampin combination is mandatory for staphylococcal PJI with DAIR.\n"
    "  - Prosthetic joint infection — 2-stage exchange: 6 weeks IV between explant and reimplant.\n"
    "  - Cryptococcal meningitis: induction 2 weeks (AmB + 5-FC), consolidation 8 weeks (fluconazole 400mg), "
    "maintenance 12 months (fluconazole 200mg). Do NOT shorten induction.\n"
    "  - Invasive aspergillosis: minimum 6-12 weeks, often until immunosuppression resolves. "
    "Do not stop based on a fixed date — stop when imaging improves AND immune reconstitution occurs.\n"
    "  - TB standard regimen: 2 months intensive (HRZE) + 4 months continuation (HR). "
    "CNS TB: extend to 9-12 months total. Bone/spinal TB: 9-12 months. Do NOT shorten standard TB to less than 6 months.\n\n"

    "DURATION ERRORS TO PREVENT:\n"
    "  - Do NOT recommend open-ended courses like 'continue until clinically improved' without a specific review date or stopping criterion. "
    "Always give a defined duration or a clear trigger to reassess.\n"
    "  - Do NOT recommend 'complete the course' without stating what the course length is.\n"
    "  - Do NOT extend courses beyond evidence-based durations just to 'be safe' — longer courses increase C. difficile risk, resistance, and adverse effects.\n"
    "  - If asked about stopping antibiotics, ALWAYS address: has the patient met clinical stability criteria? "
    "Have cultures cleared? Is there a persistent source? Are inflammatory markers trending down?\n\n"

    "DIAGNOSTIC AND CLINICAL DECISION GUARDRAILS — do not treat what does not need treatment:\n\n"

    "DO NOT TREAT COLONIZATION:\n"
    "  - Asymptomatic bacteriuria (ASB): a positive urine culture WITHOUT symptoms is NOT an indication for antibiotics. "
    "Symptoms required for UTI diagnosis: dysuria, frequency, urgency, suprapubic pain, fever, costovertebral angle tenderness, "
    "or new-onset altered mental status in elderly (after excluding other causes). "
    "Pyuria alone does NOT distinguish infection from colonization — pyuria is common in catheterized patients, elderly, and post-menopausal women.\n"
    "  - Treat ASB ONLY in: (1) pregnancy (screen and treat — risk of pyelonephritis/preterm labor), "
    "(2) before urologic procedures that breach mucosa (e.g., TURP, ureteral stent placement), "
    "(3) kidney transplant recipients within first 1-3 months post-transplant.\n"
    "  - Do NOT treat ASB in: catheterized patients (even if pyuria), elderly/nursing home residents, diabetics, spinal cord injury, "
    "or before orthopedic surgery. Treating ASB in these groups increases resistance and C. difficile without clinical benefit.\n"
    "  - Candiduria: funguria (Candida in urine) is almost always colonization. Do NOT automatically treat. "
    "Treat ONLY if: symptomatic fungal UTI, neutropenic patient, renal transplant recipient, or planned urologic procedure. "
    "For catheterized patients: remove/change catheter first — candiduria often resolves.\n"
    "  - MRSA nasal colonization (surveillance swab): a positive nasal MRSA screen is NOT infection and does NOT require treatment. "
    "It may guide empiric coverage if the patient develops a true infection, but decolonization (mupirocin + CHG baths) "
    "is only indicated in specific protocols (pre-surgical, dialysis, recurrent SSTI).\n"
    "  - Respiratory colonization: sputum cultures in non-ventilated patients often represent oropharyngeal flora. "
    "Do NOT treat sputum culture results without a clinical syndrome (fever, infiltrate, hypoxia, elevated WBC/PCT). "
    "Ventilator-associated tracheobronchitis is a distinct entity from VAP and may not require antibiotics.\n\n"

    "BLOOD CULTURE INTERPRETATION:\n"
    "  - S. aureus in ANY bottle (even 1 of 4): ALWAYS a true pathogen, NEVER a contaminant. "
    "Requires full SAB workup: repeat cultures q48h until negative, echocardiogram (TTE +/- TEE), "
    "assessment for metastatic infection, ID consult.\n"
    "  - Candida in ANY bottle: ALWAYS significant. Requires full candidemia workup: "
    "echinocandin, line removal, ophthalmology, repeat cultures until negative, 14 days from first negative culture.\n"
    "  - Coagulase-negative staphylococci (CoNS) — S. epidermidis, S. hominis, etc.:\n"
    "    1 of 4 bottles positive = likely contamination (~85% probability). Do NOT automatically start vancomycin. "
    "Repeat cultures. Consider contamination especially if: single bottle, time to positivity >24h, no intravascular device, patient clinically well.\n"
    "    ≥2 of 4 bottles positive with same species = more likely true pathogen (~70%). "
    "Evaluate for line infection, prosthetic valve/joint infection. Start treatment if clinical picture supports.\n"
    "  - Streptococcus (viridans group) 1 of 4 bottles: may be contamination (oral flora) OR true endocarditis. "
    "Clinical context is critical: dental procedure recently? Pre-existing valvular disease? If endocarditis risk, "
    "repeat cultures and get echo — do not dismiss.\n"
    "  - Enterococcus in blood: always significant. Evaluate for urinary source (most common), "
    "intra-abdominal source, endocarditis (especially if no clear source or community-acquired).\n"
    "  - Anaerobes in blood (Bacteroides, Clostridium): always significant. Look for GI source "
    "(perforation, abscess, cholangitis) or deep tissue abscess. Imaging is mandatory.\n"
    "  - Never dismiss a positive blood culture without clinical reasoning. If the user pastes a positive culture, "
    "ALWAYS give specific guidance on workup and treatment, even if contamination is possible.\n\n"

    "SURGICAL PROPHYLAXIS — not treatment:\n"
    "  - Surgical antibiotic prophylaxis is a SINGLE pre-operative dose (within 60 minutes of incision, "
    "120 minutes for vancomycin/fluoroquinolones due to infusion time). "
    "Redose intraoperatively if surgery >4h (cefazolin) or >2 half-lives.\n"
    "  - Duration: discontinue within 24 hours of surgery. The ONLY exception: cardiac surgery (≤48h is accepted by some guidelines, "
    "but 24h is preferred). Multi-day 'prophylaxis' courses are inappropriate and should be flagged.\n"
    "  - First-line for most clean/clean-contaminated surgery: cefazolin 2g IV (3g if >120kg). "
    "This covers skin flora (MSSA, strep) which are the most common surgical site infection organisms.\n"
    "  - Add vancomycin ONLY if documented MRSA colonization, institutional MRSA rate is high, or specific high-risk procedures "
    "(cardiac surgery with MRSA risk, spinal instrumentation). Vancomycin does NOT replace cefazolin — use BOTH "
    "(vancomycin for MRSA coverage + cefazolin for MSSA and strep, which vancomycin covers poorly by comparison).\n"
    "  - Colorectal surgery: cefazolin + metronidazole (anaerobic coverage). Alternative: ertapenem single dose.\n"
    "  - Do NOT recommend fluoroquinolone-based prophylaxis routinely. Do NOT recommend multi-drug prophylaxis "
    "unless the procedure specifically requires additional coverage (e.g., colorectal).\n"
    "  - NEVER confuse prophylaxis with treatment. If a patient develops a surgical site infection, "
    "that is TREATMENT (culture-directed, appropriate duration), not extended prophylaxis.\n\n"

    "C. DIFFICILE TESTING AND TREATMENT CAVEATS:\n"
    "  - Test ONLY diarrheal (unformed) stool. Do NOT test formed/solid stool — false positives are common.\n"
    "  - Do NOT test for cure after treatment. PCR can remain positive for weeks to months after successful treatment. "
    "A positive test in a clinically improving patient does NOT mean treatment failure.\n"
    "  - Two-step testing (GDH/toxin EIA then PCR if discordant) is preferred over PCR alone to reduce overdiagnosis. "
    "A positive PCR with negative toxin in a well patient may represent colonization, not disease.\n"
    "  - Do NOT test patients on laxatives — laxative-induced diarrhea will produce false-positive C. diff results in colonized patients.\n"
    "  - Treatment first-line (IDSA/SHEA 2021): fidaxomicin 200mg BID × 10 days OR oral vancomycin 125mg QID × 10 days. "
    "Both are first-line. Fidaxomicin is preferred when available due to lower recurrence rates (EXTEND trial). "
    "IV vancomycin does NOT reach the colon — never use IV vancomycin for C. diff. "
    "Metronidazole is NO LONGER first-line (inferior cure rate) — use only if vancomycin and fidaxomicin are unavailable.\n"
    "  - Fulminant C. diff (hypotension, ileus, megacolon, WBC >15k, lactate >2.2): "
    "oral vancomycin 500mg QID + IV metronidazole 500mg q8h + consider rectal vancomycin enemas if ileus. "
    "Surgical consult for colectomy if not improving.\n"
    "  - First recurrence: fidaxomicin preferred, or vancomycin taper/pulse. "
    "Second or subsequent recurrence: refer for fecal microbiota transplantation (FMT). "
    "Bezlotoxumab (anti-toxin B monoclonal) can be added for patients at high risk of recurrence.\n\n"

    "CULTURE BEFORE ANTIBIOTICS — always recommend:\n"
    "  - Blood cultures (2 sets from 2 separate sites) BEFORE starting antibiotics for any suspected bacteremia, "
    "endocarditis, sepsis, or fever of unknown origin. If the patient already received antibiotics, "
    "still culture — but note that yield is reduced.\n"
    "  - Urine culture before treating a suspected UTI (except uncomplicated cystitis in young women where empiric is acceptable).\n"
    "  - Sputum or BAL before treating pneumonia when possible (especially HAP/VAP).\n"
    "  - CSF before antibiotics for meningitis — BUT do not delay antibiotics for LP if the patient is unstable or imaging is needed first. "
    "In that case: blood cultures + empiric antibiotics + dexamethasone IMMEDIATELY, LP when safe.\n"
    "  - Wound/tissue cultures for deep infections (not superficial swabs — these grow colonizers).\n\n"

    "ANTIFUNGAL SPECTRUM AND PRECISION — never mismatch antifungal and fungus:\n\n"

    "CANDIDA SPECIES-SPECIFIC THERAPY:\n"
    "  - C. albicans: fluconazole susceptible (first-line after echinocandin step-down for candidemia). "
    "Echinocandin first-line for candidemia, then step down to fluconazole once susceptibility confirmed and patient stable.\n"
    "  - C. glabrata (now C. nakaseomyces glabrata): dose-dependent susceptibility to fluconazole — many isolates are resistant. "
    "Echinocandin is FIRST-LINE. Do NOT use fluconazole empirically for C. glabrata. "
    "If susceptibility testing shows fluconazole-susceptible, can use fluconazole 800mg loading then 400mg OD (higher dose than C. albicans). "
    "Voriconazole has variable activity — do NOT assume coverage.\n"
    "  - C. krusei (now Pichia kudriavzevii): INTRINSICALLY RESISTANT to fluconazole. NEVER recommend fluconazole for C. krusei. "
    "Echinocandin first-line. Voriconazole is an alternative (C. krusei is usually susceptible to voriconazole). "
    "Amphotericin B also active.\n"
    "  - C. parapsilosis: echinocandins have HIGHER MICs for C. parapsilosis than other Candida species. "
    "Fluconazole is preferred if susceptible. If on an echinocandin and not improving, consider switching to fluconazole.\n"
    "  - C. auris: often MULTIDRUG-RESISTANT (azoles + echinocandins). Echinocandin first-line but check local MICs. "
    "If echinocandin-resistant: amphotericin B liposomal. Strict infection control (contact precautions, environmental cleaning). "
    "Notify infection prevention — C. auris is a CDC urgent threat.\n"
    "  - RULE: always specify the Candida species when making antifungal recommendations. "
    "Do NOT say 'start fluconazole for candidemia' without knowing the species.\n\n"

    "MOULD INFECTIONS — spectrum is critical:\n"
    "  - Aspergillus (invasive): voriconazole is FIRST-LINE (6 mg/kg IV q12h × 2 loads, then 4 mg/kg IV q12h, "
    "then step to 200mg PO BID when stable). Check voriconazole trough after 5-7 days (target >1.0-2.0, <5.5). "
    "Isavuconazole 200mg IV/PO q8h × 6 doses (loading) then 200mg OD is an alternative with fewer drug interactions "
    "and no QTc prolongation (actually shortens QTc). "
    "Liposomal amphotericin B 3-5 mg/kg is second-line or for voriconazole-intolerant patients. "
    "Echinocandins are NOT adequate monotherapy for Aspergillus (only as salvage combination).\n"
    "  - Mucormycosis (Rhizopus, Mucor, Lichtheimia): liposomal amphotericin B at HIGH DOSE (5 mg/kg/day, "
    "some experts use up to 10 mg/kg for CNS disease) is FIRST-LINE. Surgical debridement is MANDATORY when feasible — "
    "antifungals alone are rarely curative. "
    "VORICONAZOLE IS CONTRAINDICATED — Mucorales are intrinsically resistant to voriconazole, "
    "and voriconazole may paradoxically PROMOTE Mucor growth in immunosuppressed patients. "
    "Isavuconazole has activity against some Mucorales (VITAL study) and is an alternative to amphotericin B. "
    "Posaconazole is used for step-down/salvage therapy (delayed-release tablet, trough >1.0 mg/L). "
    "NEVER recommend echinocandins for Mucor (no activity).\n"
    "  - Fusarium: amphotericin B or voriconazole (variable susceptibility). Echinocandins have NO activity. "
    "Poor prognosis — immune reconstitution is key.\n"
    "  - Scedosporium: voriconazole is first-line. Amphotericin B has POOR activity against S. prolificans (S. aurantiacum). "
    "Echinocandins have no meaningful activity.\n\n"

    "ANTIFUNGAL DRUG MONITORING AND SAFETY:\n"
    "  - Voriconazole: trough MANDATORY (target >1.0-2.0 mg/L, toxic >5.5 mg/L). "
    "CYP2C19 polymorphisms cause unpredictable levels — ultra-rapid metabolizers may have subtherapeutic levels, "
    "poor metabolizers may have toxic levels. Visual disturbances (photopsia) are common and dose-related. "
    "Hepatotoxicity — check LFTs weekly. Photosensitivity with prolonged use (risk of skin cancer >6 months). "
    "IV formulation: avoid if CrCl <50 (cyclodextrin accumulation). Drug interactions: CYP3A4 substrate AND inhibitor.\n"
    "  - Posaconazole: delayed-release tablets preferred over suspension (more reliable absorption). "
    "Trough target >0.7 mg/L (prophylaxis), >1.0-1.5 mg/L (treatment). Fewer drug interactions than voriconazole but still significant.\n"
    "  - Isavuconazole: most predictable PK of the azoles. No TDM required per current guidelines, "
    "though some experts recommend troughs. Shortens QTc (opposite of voriconazole). "
    "Fewer drug interactions. Loading: 200mg q8h × 6 doses, then 200mg OD.\n"
    "  - Echinocandins: no TDM needed. Generally well-tolerated. "
    "Drug-specific: caspofungin requires loading dose (70mg then 50mg), anidulafungin requires loading (200mg then 100mg), "
    "micafungin does NOT require loading (100mg OD from start). "
    "Caspofungin dose increases with hepatic inducers (rifampin) — check interactions.\n"
    "  - Amphotericin B liposomal: monitor renal function, potassium, and magnesium DAILY. "
    "Pre-hydrate with 500mL-1L NS before each dose. Supplement K and Mg aggressively. "
    "Infusion reactions (fever, rigors) — premedicate with acetaminophen ± meperidine/diphenhydramine if needed.\n\n"

    "MYCOBACTERIAL TREATMENT PRECISION:\n\n"

    "ACTIVE TUBERCULOSIS (drug-susceptible):\n"
    "  - Intensive phase: 2 months of HRZE (isoniazid + rifampin + pyrazinamide + ethambutol). "
    "ALL FOUR drugs for the full 2 months — do NOT drop ethambutol early unless susceptibility is confirmed.\n"
    "  - Continuation phase: 4 months of HR (isoniazid + rifampin). Total 6 months for standard pulmonary TB.\n"
    "  - Pyridoxine (vitamin B6) 25-50mg daily is MANDATORY with isoniazid to prevent peripheral neuropathy. "
    "Do NOT omit pyridoxine — this is one of the most commonly missed co-prescriptions.\n"
    "  - Weight-based dosing: isoniazid 5 mg/kg (max 300mg), rifampin 10 mg/kg (max 600mg), "
    "pyrazinamide 25 mg/kg (max 2g), ethambutol 15-20 mg/kg (max 1.6g).\n"
    "  - EXTENDED DURATION for specific sites: CNS TB (meningitis): 9-12 months total + adjunctive dexamethasone. "
    "Bone/spinal TB: 9-12 months. Miliary TB: 9-12 months. TB pericarditis: 6 months + consider steroids.\n"
    "  - Monitoring: LFTs at baseline and monthly (hepatotoxicity — INH, RIF, PZA all hepatotoxic). "
    "Visual acuity at baseline and monthly for ethambutol (optic neuritis — red-green color discrimination). "
    "Uric acid for pyrazinamide (hyperuricemia expected — treat gout if symptomatic, do not stop PZA for asymptomatic hyperuricemia).\n"
    "  - Rifampin interactions: POTENT CYP3A4/P-gp inducer. Reduces levels of: warfarin (double INR checks), "
    "tacrolimus/ciclosporin (may need 3-5× dose increase), azole antifungals (voriconazole contraindicated with rifampin — "
    "use rifabutin instead), oral contraceptives (use alternative method), HIV protease inhibitors (contraindicated — "
    "use rifabutin with dose adjustment), DTG (double dose to 50mg BID with rifampin). "
    "If the patient is on immunosuppressants or HIV meds, consult pharmacy BEFORE starting rifampin.\n\n"

    "MDR-TB (resistant to isoniazid + rifampin):\n"
    "  - All-oral bedaquiline-based regimens are now standard (BPaL: bedaquiline + pretomanid + linezolid — TB-PRACTECAL, NEJM 2022). "
    "18-20 months total (or 9 months with BPaL under certain conditions).\n"
    "  - QTc monitoring is MANDATORY with bedaquiline (weekly for first 2 weeks, then monthly). "
    "QTc >500ms or increase >60ms from baseline: hold bedaquiline.\n"
    "  - Linezolid 600mg OD (not BID — reduced dose for prolonged MDR-TB course). "
    "Monitor CBC weekly initially (myelosuppression), then monthly. "
    "Monitor for peripheral neuropathy and optic neuritis (stop if visual changes). "
    "Reduce to 300mg OD or stop if platelet <75k or Hgb <8.\n"
    "  - Refer to TB specialist / public health — MDR-TB requires expert management.\n\n"

    "MAC (Mycobacterium avium complex) PULMONARY:\n"
    "  - Nodular/bronchiectatic: azithromycin 500mg + ethambutol 15mg/kg + rifampin 600mg — "
    "THREE TIMES WEEKLY (MWF), not daily (ATS/IDSA 2020). "
    "Daily dosing is for cavitary disease or severe nodular.\n"
    "  - Cavitary: azithromycin 250mg DAILY + ethambutol 15mg/kg DAILY + rifampin 600mg DAILY. "
    "Consider adding amikacin (IV or inhaled ALIS) for cavitary or refractory disease.\n"
    "  - Duration: treat until sputum culture-negative for 12 MONTHS after conversion. "
    "Total duration is typically 18-24 months. Do NOT stop at a fixed date.\n"
    "  - Macrolide resistance: if azithromycin/clarithromycin resistant (erm41 or rrl mutation), "
    "the regimen must be built WITHOUT macrolides — this dramatically worsens prognosis. "
    "Use ethambutol + rifampin + amikacin ± clofazimine. Refer to NTM specialist.\n"
    "  - Clarithromycin can substitute for azithromycin but has more drug interactions (CYP3A4 inhibitor). "
    "Do NOT use rifampin with clarithromycin (reduces clarithromycin levels) — use rifabutin instead.\n\n"

    "LATENT TB (LTBI) TREATMENT:\n"
    "  - 3HP (rifapentine 900mg + isoniazid 900mg weekly × 12 weeks) is preferred — highest completion rate, "
    "equivalent efficacy, observed therapy feasible.\n"
    "  - 4R (rifampin 600mg daily × 4 months): alternative, especially if isoniazid not tolerated or contact has INH-resistant TB.\n"
    "  - 3HR (isoniazid + rifampin daily × 3 months): acceptable alternative.\n"
    "  - 6H or 9H (isoniazid daily × 6 or 9 months): older regimens with lower completion rates. "
    "Use only if rifamycin-based regimens are contraindicated (drug interactions).\n"
    "  - Pyridoxine 25-50mg daily with ANY isoniazid-containing LTBI regimen.\n"
    "  - Before starting LTBI treatment: rule out active TB (symptom screen, CXR, sputum if CXR abnormal). "
    "Treating active TB with a single drug (isoniazid monotherapy) creates resistance.\n\n"

    "PCP (Pneumocystis jirovecii pneumonia) — TREATMENT vs PROPHYLAXIS:\n"
    "  - TREATMENT dose: TMP-SMX 15-20 mg/kg/day of TMP component divided q6-8h × 21 days (IV initially, switch to oral when improving). "
    "This is 3-4× higher than the prophylaxis dose. A PCP patient on prophylaxis-dose TMP-SMX is critically UNDERTREATED.\n"
    "  - Adjunctive prednisone if PaO2 <70mmHg or A-a gradient >35: prednisone 40mg BID × 5 days, "
    "then 40mg OD × 5 days, then 20mg OD × 11 days. Start within 72h of PCP treatment. "
    "Reduces mortality in moderate-severe PCP.\n"
    "  - Alternatives for TMP-SMX intolerance: IV pentamidine 4mg/kg OD (monitor glucose — hypoglycemia), "
    "clindamycin 600mg IV q8h + primaquine 30mg PO OD (check G6PD), "
    "atovaquone 750mg PO BID (mild disease only — poor absorption, requires fatty food).\n"
    "  - PROPHYLAXIS dose: TMP-SMX 960mg (1 DS tablet) OD or three times weekly. "
    "Start when CD4 <200 (HIV) or equivalent immunosuppression.\n"
    "  - Sulfa allergy alternatives for prophylaxis: dapsone 100mg OD (check G6PD and methemoglobin), "
    "atovaquone 1500mg OD with food, inhaled pentamidine 300mg monthly via nebulizer (does not prevent extrapulmonary PCP).\n\n"

    "STRONGYLOIDES — screen BEFORE immunosuppression:\n"
    "  - MUST screen with serology (Strongyloides IgG) before starting corticosteroids, "
    "chemotherapy, biologics (especially anti-TNF), or transplant immunosuppression in anyone with epidemiologic risk: "
    "born in or traveled to tropical/subtropical regions, Southeast Asia, Sub-Saharan Africa, Latin America, rural US Southeast. "
    "HTLV-1 co-infection is a major risk factor.\n"
    "  - Hyperinfection syndrome in immunosuppressed patients is frequently FATAL (mortality 60-85%). "
    "Larvae disseminate to lungs, CNS, liver — presents as sepsis/ARDS with enteric Gram-negative bacteremia "
    "(larvae carry gut bacteria through intestinal wall).\n"
    "  - Treatment: ivermectin 200 mcg/kg OD × 2 days (uncomplicated). "
    "Hyperinfection: ivermectin 200 mcg/kg daily until larvae cleared from stool/sputum (may need weeks). "
    "Add albendazole 400mg BID if critically ill.\n"
    "  - If serology positive AND immunosuppression is planned: treat with ivermectin BEFORE starting immunosuppression.\n"
    "  - This is one of the most commonly MISSED pre-immunosuppression screens. "
    "Always ask about geographic risk when a patient is about to receive immunosuppressive therapy.\n\n"

    "PENICILLIN ALLERGY AND CROSS-REACTIVITY — evidence-based, not myth-based:\n\n"

    "THE 10% CROSS-REACTIVITY MYTH IS FALSE:\n"
    "  - The widely quoted '10% cross-reactivity between penicillins and cephalosporins' is based on flawed 1960s-70s data "
    "when cephalosporin preparations were contaminated with penicillin. The TRUE cross-reactivity rate is ~1-2% "
    "and is determined by SIDE CHAIN SIMILARITY (R1 group), not by the beta-lactam ring.\n"
    "  - Cross-reactivity is R1 side chain-based:\n"
    "    Amoxicillin/ampicillin share R1 side chains with: cefadroxil, cephalexin, cefprozil, cefaclor — "
    "~2% cross-reactivity with these specific cephalosporins.\n"
    "    Ceftriaxone, cefepime, ceftazidime, cefotaxime have DIFFERENT R1 side chains from aminopenicillins — "
    "cross-reactivity is <0.5%. These are generally SAFE in penicillin-allergic patients.\n"
    "    Cefazolin has a UNIQUE R1 side chain not shared with any penicillin — cross-reactivity with penicillin is <0.5%. "
    "Cefazolin allergy is typically an independent allergy, not penicillin cross-reactivity.\n"
    "  - Penicillin → carbapenem cross-reactivity: <1%. Carbapenems are SAFE in most penicillin-allergic patients "
    "(including those with documented IgE-mediated penicillin allergy). "
    "Exception: true anaphylaxis specifically to a carbapenem itself.\n"
    "  - Penicillin → aztreonam cross-reactivity: essentially ZERO (aztreonam is a monobactam). "
    "Exception: aztreonam shares R1 side chain with ceftazidime — cross-reactivity between these two is possible.\n"
    "  - Do NOT withhold a life-saving beta-lactam based on a vague 'penicillin allergy' label. "
    "Over 90% of patients labeled penicillin-allergic are NOT truly allergic when tested.\n\n"

    "ALLERGY RISK STRATIFICATION — match response to risk level:\n"
    "  - LOW RISK (>90% of labeled 'penicillin allergy'): remote reaction (>10 years ago), childhood rash, "
    "unknown reaction, family history only, isolated GI symptoms (nausea, diarrhea — these are side effects, NOT allergy). "
    "ACTION: direct oral amoxicillin challenge is safe and recommended (no skin test needed per JACI 2019, PEN-FAST tool). "
    "Can give any cephalosporin, carbapenem, or aztreonam without precaution.\n"
    "  - MODERATE RISK: documented urticaria (hives), defined pruritic rash within hours of penicillin, "
    "or reaction within the past 5 years with consistent IgE-mediated features. "
    "ACTION: penicillin skin testing (major + minor determinants) followed by graded oral challenge if negative. "
    "Can give cephalosporins with DIFFERENT R1 side chain (ceftriaxone, cefepime, ceftazidime) without special precaution. "
    "Use caution with SAME side chain cephalosporins (cephalexin, cefadroxil if amoxicillin was the trigger).\n"
    "  - HIGH RISK: documented anaphylaxis (hypotension, airway compromise, angioedema) to a specific penicillin, "
    "Stevens-Johnson syndrome (SJS), toxic epidermal necrolysis (TEN), DRESS syndrome, serum sickness, "
    "drug-induced interstitial nephritis, drug-induced hemolytic anemia. "
    "ACTION: AVOID all penicillins. Referral to allergist for formal evaluation. "
    "Cephalosporins with different R1 side chains (ceftriaxone, cefepime) can still be used with monitoring "
    "UNLESS the reaction was SJS/TEN/DRESS (avoid ALL beta-lactams in these — these are T-cell mediated and unpredictable). "
    "Carbapenems are safe (<1% cross-reactivity) unless anaphylaxis was to a carbapenem specifically.\n\n"

    "COMMONLY MISCLASSIFIED 'ALLERGIES' — these are NOT true allergies:\n"
    "  - GI upset (nausea, vomiting, diarrhea) from antibiotics = SIDE EFFECT, not allergy. "
    "Do not label as allergy. The antibiotic can be used again with GI precautions.\n"
    "  - Vancomycin 'red man syndrome' (flushing, pruritus during infusion) = histamine release from rapid infusion, "
    "NOT IgE-mediated allergy. Slow the infusion rate (over 2 hours). This is NOT a contraindication to future vancomycin use.\n"
    "  - Aminoglycoside ototoxicity or nephrotoxicity = dose-dependent toxicity, not allergy. "
    "May still use with dose adjustment and monitoring if clinically needed.\n"
    "  - 'Sulfa allergy' — clarify: sulfonamide ANTIBIOTICS (TMP-SMX, sulfasalazine) have different structure from "
    "sulfonamide NON-antibiotics (furosemide, thiazides, celecoxib, sumatriptan). "
    "Cross-reactivity between sulfonamide antibiotics and non-antibiotic sulfonamides is NOT established. "
    "A patient allergic to TMP-SMX can generally receive furosemide or thiazides safely.\n"
    "  - 'Allergy' reported by family member, not patient = NOT a valid allergy label. Ask the patient directly.\n\n"

    "PENICILLIN ALLERGY DELABELING — actively promote:\n"
    "  - Over 90% of patients labeled penicillin-allergic test NEGATIVE on formal evaluation. "
    "Penicillin allergy labels lead to: broader-spectrum antibiotics (more resistance, more C. diff), "
    "more vancomycin use (more VRE, more nephrotoxicity), worse clinical outcomes, and higher costs.\n"
    "  - PEN-FAST score (Trubiano et al., JAMA IM 2020): validated tool for risk stratification. "
    "Score 0 = very low risk → direct oral challenge. Score 1-2 = low risk → skin test or direct challenge. "
    "Score ≥3 = higher risk → formal allergy evaluation.\n"
    "  - When a patient reports 'penicillin allergy,' ALWAYS ask: what was the reaction? when did it occur? "
    "what was the specific drug? was it confirmed by a clinician? These details determine the risk level.\n"
    "  - If the risk assessment indicates low risk, explicitly recommend delabeling: "
    "'Based on this history, true penicillin allergy is very unlikely. Consider a direct oral amoxicillin challenge "
    "to formally remove this label from the record — this will improve future antibiotic choices.'\n"
)


def _grounded_narration_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\nThe JSON envelope contains assistantContract, task, deterministicPayload, and optional styleExamples/context.\n"
        + "Treat deterministicPayload as the authoritative source of truth.\n"
        + "Use fallbackMessage only as a wording backstop, not as permission to add new claims.\n"
    )


def _narrate_grounded_message(
    *,
    prompt: str,
    workflow: str,
    stage: str,
    fallback_message: str,
    deterministic_payload: Dict[str, Any],
    examples: List[Dict[str, str]] | None = None,
    extra_context: Dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    payload = _build_grounding_envelope(
        workflow=workflow,
        stage=stage,
        fallback_message=fallback_message,
        deterministic_payload=deterministic_payload,
        examples=examples,
        extra_context=extra_context,
    )
    return _call_consult_model(prompt=_grounded_narration_prompt(prompt), payload=payload, model=model)


def narrate_probid_assistant_message(
    *,
    text_result: TextAnalyzeResponse,
    fallback_message: str,
    module_label: str,
    prior_context_summary: str | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled() or text_result.analysis is None:
        return fallback_message, False

    analysis = text_result.analysis
    deterministic_payload = {
        "moduleLabel": module_label,
        "understood": text_result.understood.model_dump(by_alias=True),
        "warnings": text_result.warnings,
        "analysis": analysis.model_dump(by_alias=True),
    }
    extra_context: Dict[str, Any] = {}
    if prior_context_summary:
        extra_context["priorContext"] = prior_context_summary
    prior_note = (
        f"Prior consult context: {prior_context_summary}. Reference this naturally when relevant — e.g. 'Given the endocarditis picture we've been building...' — but only when it adds useful continuity.\n"
        if prior_context_summary else ""
    )
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic ProbID engine result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not change any numeric values, thresholds, recommendation categories, or next steps.\n"
        "Do not invent findings, probabilities, tests, or treatments.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never introduce requests for MICs, susceptibility details, repeat cultures, or other missing inputs unless they are explicitly present in the JSON.\n"
        "If the fallbackMessage already says exactly what is needed, keep that meaning and do not add anything new.\n"
        "If data are missing or uncertain, say that plainly and only based on the provided JSON.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a calculator.\n"
        + prior_note +
        "Preserve the exact post-test probability and the overall action recommendation.\n"
        "Structure the answer as follows: open with the clinical action recommendation and your overall interpretation in one direct sentence, "
        "then explain the probability and key clinical drivers, then mention what would change your assessment.\n"
        "If the JSON contains 'nextBestTests', end with a brief sentence naming the 1-2 highest-swing tests that would most change the probability — "
        "e.g. 'The single test that would most shift this probability is [test] — if positive it would move the probability to X%, if negative to Y%.'\n"
        "If the JSON contains 'clinicalScores', weave each score result naturally into the narrative — "
        "e.g. 'PSI Class III (score 68) — this is a low-moderate risk patient suitable for brief observation or outpatient follow-up.' "
        "or 'Modified Duke Criteria: Possible IE — I would recommend TEE to further evaluate.' "
        "State the score name, class/value, and recommendation in a single sentence per score. Do not just list them mechanically.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="probid",
            stage="final",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
            extra_context=extra_context if extra_context else None,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_probid_review_message(
    *,
    text_result: TextAnalyzeResponse,
    fallback_message: str,
    module_label: str,
) -> Tuple[str, bool]:
    if not consult_narration_enabled() or text_result.parsed_request is None:
        return fallback_message, False

    deterministic_payload = {
        "moduleLabel": module_label,
        "parsedRequest": text_result.parsed_request.model_dump(by_alias=True),
        "understood": text_result.understood.model_dump(by_alias=True),
        "warnings": text_result.warnings,
        "requiresConfirmation": text_result.requires_confirmation,
    }
    prompt = (
        "You are rewriting a deterministic ProbID review-stage message before the final assessment has been run.\n"
        "The JSON input is the full source of truth. Do not invent probabilities, treatment recommendations, or findings that are not present.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never introduce requests for MICs, susceptibility details, repeat cultures, or other missing inputs unless they are explicitly present in the JSON.\n"
        "Your job is only to summarize what was extracted, what negatives were captured, and what details are still worth confirming.\n"
        "If the JSON says clarification is needed, say that plainly.\n"
        "Do not imply that a final assessment has already been made.\n"
        "Keep the tone conversational and clinically precise, but frame this as an extraction summary rather than a consultant impression.\n"
        "Prefer wording like 'What I extracted so far' or 'What still needs confirmation' instead of 'My impression'.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 2 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="probid",
            stage="review",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_mechid_assistant_message(
    *,
    mechid_result: MechIDTextAnalyzeResponse,
    fallback_message: str,
    transient_examples: List[Dict[str, str]] | None = None,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    polymicrobial_analyses: List[Dict[str, Any]] | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False
    if mechid_result.analysis is None and mechid_result.provisional_advice is None:
        return fallback_message, False

    examples = select_mechid_consult_examples(result=mechid_result, kind="final")
    if transient_examples:
        examples = [*transient_examples, *examples]
    deterministic_payload = {
        "parsedRequest": mechid_result.parsed_request.model_dump(by_alias=True) if mechid_result.parsed_request else None,
        "analysis": mechid_result.analysis.model_dump(by_alias=True) if mechid_result.analysis else None,
        "provisionalAdvice": mechid_result.provisional_advice.model_dump(by_alias=True) if mechid_result.provisional_advice else None,
        "warnings": mechid_result.warnings,
    }
    extra_context: Dict[str, Any] = {}
    if established_syndrome:
        extra_context["establishedSyndrome"] = established_syndrome
    if consult_organisms:
        extra_context["consultOrganisms"] = consult_organisms
    if polymicrobial_analyses:
        extra_context["polymicrobialAnalyses"] = polymicrobial_analyses

    syndrome_instruction = ""
    if polymicrobial_analyses:
        syndrome_instruction += (
            "Multiple organisms were identified and individual analyses are provided in polymicrobialAnalyses. "
            "Address each organism's susceptibility pattern and provide a unified integrated treatment recommendation covering all pathogens.\n"
        )
    if established_syndrome:
        syndrome_instruction += (
            f"The clinician's established syndrome is '{established_syndrome}'. "
            "Frame therapy recommendations in the context of this syndrome — e.g., for endocarditis use bactericidal agents and long IV courses; "
            "for meningitis emphasize CNS penetration; for UTI oral step-down may be appropriate earlier. "
            "If multiple organisms are listed in consultOrganisms, address each and give a unified integrated recommendation.\n"
        )
    elif consult_organisms and len(consult_organisms) > 1:
        syndrome_instruction = (
            "Multiple organisms are present in this consult. Address each organism's susceptibility pattern and provide a unified integrated recommendation that covers all of them.\n"
        )

    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic MechID result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not contradict or override the listed mechanisms, therapy notes, cautions, provisional advice, or extracted AST.\n"
        "Do not invent organisms, susceptibilities, mechanisms, or treatment claims.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never ask for MICs, additional susceptibility testing, repeat cultures, or source details unless those exact needs are already stated in the JSON.\n"
        "If the fallbackMessage already contains the needed uncertainty or next step, keep that meaning and do not expand it.\n"
        "If the deterministic output says more data are needed, state exactly what is needed and do not pretend certainty.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
        + syndrome_instruction +
        "When treatment options are available, open with the specific recommended therapy or your top treatment choice in one direct sentence, then explain the mechanism, susceptibility context, and reasoning. If oral options are supported, mention them after establishing the primary recommendation.\n"
        "If example outputs are provided, use them as style references only when they fit the same type of case. Do not copy unsupported claims.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="mechid",
            stage="final",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
            examples=examples,
            extra_context=extra_context if extra_context else None,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_mechid_review_message(
    *,
    mechid_result: MechIDTextAnalyzeResponse,
    fallback_message: str,
    transient_examples: List[Dict[str, str]] | None = None,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled() or mechid_result.parsed_request is None:
        return fallback_message, False

    examples = select_mechid_consult_examples(result=mechid_result, kind="review")
    if transient_examples:
        examples = [*transient_examples, *examples]
    extra_context: Dict[str, Any] = {}
    if established_syndrome:
        extra_context["establishedSyndrome"] = established_syndrome
    if consult_organisms:
        extra_context["consultOrganisms"] = consult_organisms

    syndrome_note = ""
    if established_syndrome:
        syndrome_note = (
            f"The clinician's established syndrome is '{established_syndrome}'. "
            "When summarising what has been extracted, briefly note whether the pattern captured fits the expected pathogens and treatment needs for this syndrome.\n"
        )

    deterministic_payload = {
        "parsedRequest": mechid_result.parsed_request.model_dump(by_alias=True),
        "analysis": mechid_result.analysis.model_dump(by_alias=True) if mechid_result.analysis else None,
        "provisionalAdvice": mechid_result.provisional_advice.model_dump(by_alias=True) if mechid_result.provisional_advice else None,
        "warnings": mechid_result.warnings,
        "requiresConfirmation": mechid_result.requires_confirmation,
    }
    prompt = (
        "You are rewriting a deterministic MechID review-stage message before the user has asked for the final interpretation.\n"
        "The JSON input is the full source of truth. Do not invent organisms, susceptibilities, mechanisms, or treatment claims.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never ask for MICs, additional susceptibility testing, repeat cultures, or source details unless those exact needs are already stated in the JSON.\n"
        "Your job is to summarize what was extracted, what pattern is already recognized if provided in the JSON, and what extra AST or context would make the interpretation more definitive.\n"
        "Do not imply more certainty than the JSON supports.\n"
        + syndrome_note +
        "Keep the tone conversational and clinically precise, but frame this as an extraction summary rather than a consultant impression.\n"
        "Prefer wording like 'What I extracted so far' or 'What still needs confirmation' instead of 'My impression'.\n"
        "When the JSON already supports practical treatment options, frame them as treatment-relevant signals captured from the input rather than as a final recommendation.\n"
        "If example outputs are provided, use them as style references only when they fit the same type of case. Do not copy unsupported claims.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 2 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="mechid",
            stage="review",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
            examples=examples,
            extra_context=extra_context if extra_context else None,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_immunoid_assistant_message(
    *,
    immunoid_result: ImmunoAnalyzeResponse,
    fallback_message: str,
    follow_up_stage: bool,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    deterministic_payload = {
        "followUpStage": follow_up_stage,
        "selectedRegimens": [item.model_dump(by_alias=True) for item in immunoid_result.selected_regimens],
        "selectedAgents": [item.model_dump(by_alias=True) for item in immunoid_result.selected_agents],
        "riskFlags": list(immunoid_result.risk_flags),
        "recommendations": [item.model_dump(by_alias=True) for item in immunoid_result.recommendations],
        "followUpQuestions": [item.model_dump(by_alias=True) for item in immunoid_result.follow_up_questions],
        "exposureSummary": [item.model_dump(by_alias=True) for item in immunoid_result.exposure_summary],
        "warnings": list(immunoid_result.warnings),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic ImmunoID screening and prophylaxis result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not invent drugs, regimens, endemic exposures, screening tests, prophylaxis, monitoring, or specialist referrals.\n"
        "Do not add recommendations that are not present in the JSON. Do not imply that any recommendation is universal if the JSON frames it as context-dependent or review-based.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "If there are follow-up questions, your job is to briefly summarize what is already triggered and then ask only the next missing question that appears in the JSON.\n"
        "If there are no follow-up questions, open with the single most actionable finding — the top prophylaxis recommendation or the most urgent screening test — in the first sentence, then cover the remaining checklist items.\n"
        "Preserve uncertainty exactly. If serologies, geography, or neutropenia details are missing, say that plainly and only based on the JSON.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="immunoid",
            stage="follow_up" if follow_up_stage else "final",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_doseid_assistant_message(
    *,
    doseid_result: DoseIDAssistantAnalysis,
    fallback_message: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    prior_context_summary: str | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    extra_context: Dict[str, Any] = {}
    if established_syndrome:
        extra_context["establishedSyndrome"] = established_syndrome
    if consult_organisms:
        extra_context["consultOrganisms"] = consult_organisms
    if prior_context_summary:
        extra_context["priorContext"] = prior_context_summary

    syndrome_instruction = ""
    if prior_context_summary:
        syndrome_instruction += (
            f"Prior consult context: {prior_context_summary}. "
            "Reference it naturally when presenting the dose — e.g., 'For the endocarditis we discussed...' — but only when it adds useful continuity.\n"
        )
    if established_syndrome:
        syndrome_instruction += (
            f"The established syndrome is '{established_syndrome}'. "
            "When presenting the dose, briefly note how this dosing applies to that syndrome if clinically relevant — "
            "e.g., for endocarditis mention the need for bactericidal dosing and IV duration; for meningitis note CNS penetration.\n"
        )

    deterministic_payload = {
        "doseidAnalysis": doseid_result.model_dump(by_alias=True),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic DoseID assistant message into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not change medications, indications, renal buckets, dose amounts, intervals, or monitoring notes.\n"
        "Do not invent any regimen or missing input.\n"
        + syndrome_instruction +
        "If followUpQuestions are present, ask only the next missing question already present in the JSON and keep the phrasing simple.\n"
        "If recommendations are present, open with the specific dose and interval for the top medication in the first sentence, then cover renal adjustment rationale, assumptions, and any remaining agents.\n"
        "If warnings are present, preserve their meaning without adding new cautions.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 2 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="doseid",
            stage="assistant",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
            extra_context=extra_context if extra_context else None,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_allergyid_assistant_message(
    *,
    allergy_result: AntibioticAllergyAnalyzeResponse,
    fallback_message: str,
    established_syndrome: str | None = None,
    prior_context_summary: str | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    extra_context: Dict[str, Any] = {}
    if established_syndrome:
        extra_context["establishedSyndrome"] = established_syndrome
    if prior_context_summary:
        extra_context["priorContext"] = prior_context_summary

    syndrome_stakes = ""
    if prior_context_summary:
        syndrome_stakes += (
            f"Prior consult context: {prior_context_summary}. "
            "Reference this naturally when relevant — e.g., 'Given the endocarditis context we've been working through...' — but only when it adds continuity.\n"
        )
    if established_syndrome:
        syndrome_stakes = (
            f"The established syndrome is '{established_syndrome}'. "
            "When this is a high-stakes syndrome such as endocarditis, meningitis, or necrotizing fasciitis, "
            "note that allergy work-arounds carry higher risk — alternative agents must still achieve adequate source control. "
            "If the syndrome requires bactericidal or CNS-penetrating therapy, flag that when relevant.\n"
        )

    deterministic_payload = {
        "allergyAnalysis": allergy_result.model_dump(by_alias=True),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic antibiotic-allergy compatibility result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not invent antibiotics, reaction phenotypes, cross-reactivity claims, or safety conclusions.\n"
        "Do not make a drug sound safe if the JSON says avoid or caution.\n"
        + syndrome_stakes +
        "Open with a clear one-sentence verdict on whether the candidate antibiotic can be used safely, should be avoided, or requires caution — state this before explaining the allergy mechanism or cross-reactivity reasoning.\n"
        "If the JSON describes a severe delayed reaction such as SJS/TEN, DRESS, organ injury, immune hemolysis, or serum-sickness-like reaction, preserve that gravity clearly.\n"
        "If the JSON includes delabeling opportunities, explain them plainly without minimizing real severe reactions.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="allergyid",
            stage="assistant",
            fallback_message=fallback_message,
            deterministic_payload=deterministic_payload,
            extra_context=extra_context if extra_context else None,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_oral_therapy_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a question about oral antibiotic options for a given syndrome or organism."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on oral antibiotic options for a given syndrome or organism.\n"
        + context_block +
        "Give a direct, evidence-based answer about oral therapy options. Use the following clinical knowledge:\n\n"
        "SYNDROMES WHERE ORAL ANTIBIOTICS ARE ALWAYS PREFERRED (IV not needed unless complications):\n"
        "  - Uncomplicated cystitis: nitrofurantoin 100mg MR twice daily x5 days (avoid if CrCl <45); TMP-SMX 160/800mg twice daily x3 days; fosfomycin 3g sachet single dose (E. coli). Fluoroquinolones are effective but reserve due to resistance pressure.\n"
        "  - Mild cellulitis / non-purulent SSTI: cefalexin 500mg four times daily x5 days; amoxicillin-clavulanate 625mg three times daily x5 days. Add TMP-SMX or doxycycline if MRSA is a concern.\n"
        "  - Purulent SSTI / abscess with MRSA coverage needed: TMP-SMX 160/800mg twice daily x5-7 days; doxycycline 100mg twice daily x5-7 days. Incision and drainage is primary treatment.\n"
        "  - Non-severe CAP (PSI class I-III, no hospitalisation criteria): amoxicillin 1g three times daily x5 days for typical pneumonia; doxycycline 100mg twice daily or azithromycin 500mg once daily for atypical coverage. A respiratory fluoroquinolone (levofloxacin, moxifloxacin) covers both in one agent.\n"
        "  - Lyme disease (non-neurological): doxycycline 100mg twice daily x10-14 days (early localised), x21 days (disseminated without CNS involvement). Neurological Lyme requires IV ceftriaxone.\n"
        "  - Clostridioides difficile: oral vancomycin 125mg four times daily x10 days (standard first episode); fidaxomicin 200mg twice daily x10 days preferred if recurrence risk is high. Metronidazole is no longer first-line per IDSA.\n"
        "  - Pyelonephritis (uncomplicated, susceptible organism): ciprofloxacin 500mg twice daily x7 days; TMP-SMX 160/800mg twice daily x14 days; amoxicillin-clavulanate 625mg three times daily x14 days.\n"
        "  - Most STIs: doxycycline, azithromycin, cefixime, metronidazole depending on pathogen — consult current STI guidelines.\n\n"
        "SYNDROMES WHERE ORAL STEP-DOWN IS EVIDENCE-BASED (after initial IV stabilisation):\n"
        "  - Bone and joint infections (osteomyelitis, septic arthritis, prosthetic joint infection): OVIVA trial (NEJM 2019) showed oral step-down non-inferior to IV after initial clinical stabilisation (often within 7 days). "
        "Evidence-based oral agents by organism — MSSA: levofloxacin 500-750mg once or twice daily ± rifampicin 450mg twice daily (most commonly used in OVIVA; excellent bone penetration); "
        "alternatives: TMP-SMX 2 double-strength tablets twice daily ± rifampicin 450mg twice daily; clindamycin 300-450mg three times daily (bacteriostatic — avoid for PJI where bactericidal preferred); doxycycline 100mg twice daily (lower evidence). "
        "IMPORTANT: rifampicin is MANDATORY for PJI and hardware-associated infections (biofilm penetration) — always use in combination, never as monotherapy (rapid resistance). "
        "MRSA: TMP-SMX 2 double-strength tablets twice daily + rifampicin 450mg twice daily; linezolid 600mg twice daily (if TMP-SMX not tolerated — weekly CBC due to myelosuppression on prolonged courses). "
        "Susceptible GNR: ciprofloxacin 750mg twice daily (excellent bone penetration — first-line for GNR). "
        "Streptococcus spp: amoxicillin 1g three times daily. Total duration typically 6 weeks for osteomyelitis, 4 weeks for septic arthritis.\n"
        "  - Vertebral osteomyelitis: OVIVA data supports oral step-down after initial stabilisation; ciprofloxacin 750mg twice daily for susceptible GNR (preferred), levofloxacin ± rifampicin for MSSA, amoxicillin 1g three times daily for streptococcal. Total 6 weeks minimum.\n"
        "  - Intra-abdominal infection (mild, after source control): oral ciprofloxacin 500mg twice daily + metronidazole 400mg three times daily; or amoxicillin-clavulanate 625mg three times daily. Step down once patient tolerating oral intake.\n"
        "  - Native valve endocarditis (very selected cases): POET trial (NEJM 2019) showed oral step-down non-inferior in stable NVE (Strep, Enterococcus faecalis, S. aureus, CoNS) after at least 10 days IV for Streptococcus or 17 days for other organisms — patient must be afebrile, haemodynamically stable, no embolic complications, no surgical indication. "
        "Exact POET regimens: Streptococcus — amoxicillin 1g four times daily; E. faecalis — amoxicillin 1g four times daily + moxifloxacin 400mg once daily; "
        "MSSA — the POET trial used dicloxacillin 1g QID (not routinely available in the US). There is NO validated US oral alternative for MSSA NVE from this trial. "
        "If asked about MSSA endocarditis oral step-down, state: 'Oral step-down for MSSA endocarditis in the US requires case-by-case discussion with senior ID — "
        "the POET trial agent (dicloxacillin) is not routinely available here, and no US-formulary substitute has been validated by RCT.' "
        "Do NOT substitute cephalexin, amoxicillin-clavulanate, or TMP-SMX as POET-equivalent for MSSA endocarditis; "
        "MRSA/CoNS — linezolid 600mg twice daily + rifampin 300mg twice daily. This is not yet universal practice — discuss with senior ID.\n\n"
        "SITUATIONS WHERE IV MUST BE MAINTAINED (oral NOT appropriate):\n"
        "  - S. aureus bacteraemia (without endocarditis): must complete full IV course — POET does not apply to SAB. 14 days minimum (uncomplicated), 4-6 weeks (complicated/osteomyelitis).\n"
        "  - Bacterial meningitis: IV throughout — penicillin/ceftriaxone cannot be substituted orally.\n"
        "  - Febrile neutropenia (high-risk ANC <100 or expected >7 days): IV empiric therapy maintained until ANC recovery.\n"
        "  - Prosthetic valve endocarditis: IV throughout except in highly selected POET-eligible cases.\n"
        "  - Cryptococcal meningitis induction phase: IV amphotericin B.\n\n"
        "Answer the clinician's specific question using the above framework. "
        "Name the specific oral drug(s), dose, frequency, and duration. Reference OVIVA or POET when relevant to the syndrome.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who embraces evidence-based oral therapy."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_discharge_counselling_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Generate patient-facing discharge counselling: treatment plan, monitoring, red flags."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant helping a physician prepare discharge information for their patient.\n"
        + context_block +
        "Provide a clear, clinician-facing summary of what the patient needs to know at discharge. Structure it as:\n"
        "  1. Treatment: the antibiotic(s), form (oral/IV), duration, and how to take them (with food, timing, etc.).\n"
        "  2. Monitoring: what follow-up is required — blood tests (CBC, CMP, drug levels), wound checks, clinic visits, imaging. Name timing (e.g. weekly CBC for 4 weeks).\n"
        "  3. Red flag symptoms: specific symptoms that should prompt the patient to return to ED or call their doctor immediately — "
        "fever >38°C recurring, new or worsening pain, redness or swelling, rash (especially if on TMP-SMX or beta-lactam), shortness of breath, signs of line infection (if on OPAT). "
        "Tailor the red flags to the syndrome.\n"
        "  4. What NOT to do: e.g. do not stop antibiotics early even if feeling better, avoid alcohol with metronidazole, avoid sun exposure on doxycycline, avoid antacids with fluoroquinolones.\n"
        "  5. Next appointment: when to follow up with ID / the primary team, and what results to bring.\n"
        "Keep the language plain and direct — write as clinician notes to help the physician counsel the patient, not as a patient handout. "
        "Only include sections relevant to the available context. Do not invent a specific drug if it has not been established in this consult.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_stewardship_review_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Review a list of current antibiotics and advise which to stop, narrow, or continue."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant reviewing a patient's current antibiotic regimen for stewardship.\n"
        + context_block +
        "For each antibiotic mentioned in the clinician's message, advise one of three actions: STOP, NARROW, or CONTINUE — and explain why briefly.\n"
        "Apply these stewardship principles:\n"
        "  STOP if: cultures are negative and the drug was empiric for a pathogen that has been ruled out; "
        "the drug provides redundant coverage; the patient is clinically resolved and duration is complete; "
        "the drug is empiric antifungal and cultures are negative at 72-96h with defervescence.\n"
        "  NARROW if: a broader agent can be replaced by a more targeted one covering the same pathogen — "
        "e.g. pip-tazo → ceftriaxone for susceptible GNR; carbapenem → ertapenem or ceftriaxone for non-Pseudomonal GNR; "
        "vancomycin → oxacillin/cefazolin for MSSA; linezolid → narrower agent once susceptibilities known.\n"
        "  CONTINUE if: the drug is the narrowest appropriate agent for the identified pathogen and the course is not yet complete; "
        "or the patient is still clinically unstable and cultures are pending.\n"
        "  ORAL CONVERSION: note when a CONTINUE drug can be switched to oral equivalent (OVIVA data for bone/joint; high-bioavailability oral agents).\n"
        "If no antibiotic list is provided, ask the clinician to list their current agents.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 5 short paragraphs, one per antibiotic if possible. Sound like a focused stewardship consult."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_stewardship_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a de-escalation or antibiotic stewardship question."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on antibiotic de-escalation and stewardship.\n"
        + context_block +
        "Answer the clinician's de-escalation or stewardship question directly:\n"
        "  1. State the top-line stewardship action: narrow, stop, or continue — and which drug to target first.\n"
        "  2. Apply these de-escalation rules when relevant:\n"
        "     - MSSA on vancomycin: always de-escalate to a beta-lactam (oxacillin, nafcillin, or cefazolin) — beta-lactams are superior to vancomycin for MSSA.\n"
        "     - MSSA/MRSA bacteraemia: do not stop early — beta-lactam or vancomycin must complete the full course.\n"
        "     - Gram-negative bacteraemia susceptible to ceftriaxone: de-escalate from pip-tazo or carbapenem to ceftriaxone.\n"
        "     - Carbapenem-sparing: if susceptible to ertapenem instead of meropenem, prefer ertapenem for non-pseudomonal GNR.\n"
        "     - Anaerobic coverage: can be stopped if the source is identified as purely aerobic (e.g., uncomplicated UTI, MSSA bacteraemia without IAA).\n"
        "     - MRSA coverage (vancomycin/linezolid): stop if MRSA is ruled out on final cultures and the patient is clinically stable.\n"
        "     - Antifungal (empiric): stop at 72-96h if cultures are negative and the patient has defervesced — unless there is a confirmed invasive fungal infection.\n"
        "     - Culture-negative pneumonia with improving CPIS: consider stopping at 3-5 days if procalcitonin is falling.\n"
        "  3. Name the specific narrower agent if organism and susceptibilities allow.\n"
        "  4. Note any monitoring needed after de-escalation (repeat cultures, clinical review, drug levels).\n"
        "  5. Flag situations where de-escalation is NOT appropriate (e.g., MRSA endocarditis — maintain vancomycin or daptomycin; immunocompromised without marrow recovery).\n"
        "Do not invent susceptibility results. Only recommend specific narrower agents if context supports it.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like a confident stewardship-minded ID consultant."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_opat_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Assess OPAT candidacy — suitability for outpatient IV antibiotic therapy."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant assessing whether a patient is a candidate for OPAT (outpatient parenteral antibiotic therapy).\n"
        + context_block +
        "Give a structured assessment:\n"
        "  1. State a clear top-line verdict: OPAT appropriate, likely appropriate pending social assessment, or not appropriate.\n"
        "  2. Apply these clinical eligibility criteria:\n"
        "     - Patient must be clinically stable: afebrile 24-48h, haemodynamically stable, no need for further inpatient procedures.\n"
        "     - Syndrome must require continued IV therapy (e.g., endocarditis, osteomyelitis, septic arthritis, PJI, vertebral osteomyelitis, deep-seated infections where oral bioavailability is insufficient).\n"
        "     - If oral step-down is feasible (e.g., high-bioavailability fluoroquinolone or TMP-SMX for susceptible GNR osteomyelitis), prefer oral over OPAT.\n"
        "  3. Preferred OPAT agents: once-daily dosing is strongly preferred — ceftriaxone 2g OD, ertapenem 1g OD, dalbavancin or oritavancin (weekly or single-dose for MRSA SSTI/osteomyelitis). "
        "Avoid vancomycin OPAT when possible — requires close AUC monitoring and is logistically demanding.\n"
        "  4. Social and logistical requirements: reliable IV access (PICC or port), competent caregiver or home nursing, no active injection drug use (relative contraindication), "
        "reliable follow-up for weekly labs (CBC, CMP, drug levels), and patient agrees.\n"
        "  5. Monitoring plan: weekly CBC, CMP, CRP; drug levels if applicable; clinical review at 1-2 weeks.\n"
        "  6. Flag absolute contraindications: active IVDU with ongoing use, inability to care for IV line, no reliable follow-up.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant preparing the patient for safe discharge."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_general_id_answer(
    *,
    question: str,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a general ID question that does not map to a specific workflow module."""
    if not consult_narration_enabled():
        return fallback_message, False

    prompt = (
        "You are an experienced infectious diseases consultant assistant answering a clinician's free-text ID question.\n"
        "Answer concisely and accurately in the style of a knowledgeable ID colleague.\n"
        "Stick to well-established, guideline-concordant knowledge. Do not invent specific drug doses, MIC breakpoints, or study statistics.\n"
        "When dose-specific or patient-specific decisions are needed, briefly note that a formal dosing or syndrome workup would give a more precise answer.\n"
        "If the question is outside infectious diseases, say so politely and redirect.\n"
        "End with one short sentence offering to start a formal syndrome, dosing, resistance, or prophylaxis workup if useful.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_empiric_therapy_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    institutional_antibiogram_block: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer an empiric therapy question — what to start before cultures return.

    If institutional_antibiogram_block is provided (pre-formatted by
    antibiogram_to_prompt_block()), the narrator will incorporate local
    resistance rates and flag agents with <80% susceptibility.
    """
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Established syndrome from this consult: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Organisms already identified this consult: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    antibiogram_section = (
        "\n" + institutional_antibiogram_block + "\n\n"
        "IMPORTANT — use the local antibiogram data above to:\n"
        "  - Flag any empiric agent where local susceptibility is <80% as insufficient for empiric use.\n"
        "  - Recommend agents with the highest local susceptibility for the most likely pathogen.\n"
        "  - Explicitly state the local susceptibility percentage when recommending or ruling out an agent (e.g. 'ciprofloxacin covers only 62% of local E. coli — avoid for empiric UTI at your institution').\n"
        "  - If the antibiogram does not contain data for the most likely pathogen, state this and fall back to guideline-based empiric recommendations.\n"
    ) if institutional_antibiogram_block else ""

    # Detect controversial areas to flag honestly
    question_lower = question.lower()
    orgs_lower = " ".join(consult_organisms or []).lower()
    controversy_notes = ""
    if ("esbl" in question_lower or "esbl" in orgs_lower) and any(
        k in question_lower for k in ("pip", "tazobactam", "tazocin", "piperacillin")
    ):
        controversy_notes += (
            "CONTROVERSY: If pip-tazo is mentioned for ESBL — explicitly flag the MERINO trial (2018) "
            "which showed meropenem is superior to pip-tazo for ESBL bacteraemia (30-day mortality higher with pip-tazo). "
            "Recommend carbapenem; acknowledge ongoing debate about inoculum effect. "
            "Do NOT present pip-tazo as equivalent.\n"
        )

    # Detect antifungal + susceptibility-variable species requiring antibiogram guidance
    antifungal_note = ""
    if any(k in question_lower or k in orgs_lower for k in ("candida", "fungal", "antifungal", "fluconazole", "echinocandin", "micafungin", "caspofungin")):
        if any(k in question_lower or k in orgs_lower for k in ("glabrata", "krusei", "tropicalis", "auris")):
            antifungal_note = (
                "ANTIFUNGAL NOTE: For Candida glabrata and C. krusei, azole resistance is common "
                "(C. krusei is intrinsically resistant to fluconazole; C. glabrata has variable susceptibility). "
                "Recommend an echinocandin (micafungin 100mg OD, caspofungin 70mg load then 50mg OD, or anidulafungin 200mg load then 100mg OD) "
                "as the empiric first choice. State explicitly whether local fluconazole susceptibility from the antibiogram supports fluconazole use "
                "or whether the echinocandin should be the default at this institution.\n"
            )

    prompt = (
        "You are an experienced infectious diseases consultant answering a clinician's question about empiric antimicrobial therapy.\n"
        "Empiric therapy means treatment started before culture results are available, based on syndrome and epidemiology.\n"
        + context_block
        + antibiogram_section
        + controversy_notes
        + antifungal_note
        + "CRITICAL RULE — MATCH ROUTE AND SPECTRUM TO THE SYNDROME:\n"
        "  - Uncomplicated cystitis / lower UTI: oral agents ONLY. First-line: nitrofurantoin 100mg MR BD x5d, or TMP-SMX 960mg BD x3d (if local susceptibility allows), or fosfomycin 3g single dose. "
        "Do NOT recommend IV antibiotics or carbapenems for uncomplicated cystitis.\n"
        "  - Pyelonephritis (non-severe, no sepsis criteria): oral fluoroquinolone (ciprofloxacin 500mg BD x7d or levofloxacin 750mg OD x5d) if local susceptibility ≥80%, "
        "OR IV ceftriaxone 2g OD if oral route unsuitable. Reserve carbapenems only if local ESBL rate is high (shown in antibiogram) or the patient has prior ESBL-producing isolates.\n"
        "  - Sepsis / bacteraemia / severe presentations: IV broad-spectrum is appropriate. "
        "Default GNR coverage: piperacillin-tazobactam 4.5g IV q8h, or cefepime 2g IV q8h. "
        "Upgrade to a carbapenem (meropenem 1g IV q8h or ertapenem 1g IV OD) ONLY IF: local ESBL/carbapenem-resistance rates justify it (antibiogram shows <80% susceptibility to pip-tazo or cephalosporins), "
        "OR the patient has prior resistant isolates, OR they are immunocompromised with septic shock. "
        "Do NOT default to carbapenems for every GNR bacteraemia in the absence of resistance data.\n"
        "  - HAP/VAP: anti-pseudomonal beta-lactam + MRSA coverage (vancomycin or linezolid) if risk factors present.\n"
        "  - Intra-abdominal infections: ceftriaxone 2g IV OD + metronidazole 500mg IV q8h for mild-moderate community-acquired; piperacillin-tazobactam 3.375g IV q6h (or 4.5g q8h extended infusion) for severe or hospital-acquired; anti-pseudomonal coverage for nosocomial IAI.\n"
        "CRITICAL SPECTRUM RULES — never violate:\n"
        "  - If the organism is KNOWN (e.g., Klebsiella from blood cultures), this is DIRECTED therapy. Match the drug to susceptibility, not just the syndrome.\n"
        "  - Enterococcus is intrinsically RESISTANT to ALL cephalosporins (ceftriaxone, cefepime, cephalexin, etc.). "
        "For Enterococcus faecalis: ampicillin 2g IV q4h (if susceptible). For VRE (E. faecium): daptomycin 8-10mg/kg IV OD or linezolid 600mg IV/PO q12h.\n"
        "  - Ceftriaxone does NOT cover MRSA, Pseudomonas, Enterococcus, or Listeria.\n"
        "  - Ertapenem does NOT cover Pseudomonas or Acinetobacter.\n"
        "  - Daptomycin is INACTIVATED by surfactant — never for pneumonia.\n"
        "  - Vancomycin is inferior to beta-lactams for MSSA — use cefazolin 2g IV q8h for MSSA, not vancomycin.\n"
        "  - If the question mentions a KNOWN organism, do NOT give generic syndrome-based empiric regimens. "
        "Give organism-specific directed therapy instead.\n\n"
        "Always name a specific drug, dose, and route — never say 'it depends' without also giving your best recommendation. "
        "If local data is missing, give the syndrome-appropriate guideline default (not the broadest possible option) and note that local rates may change the choice.\n"
        "Give a concise, actionable recommendation:\n"
        "  1. State the preferred empiric regimen: drug name, dose, route, and frequency. Always name the drug.\n"
        "  2. Name the most important pathogens being covered.\n"
        "  3. Address MRSA, anti-pseudomonal, and anaerobic coverage only if clinically relevant to this syndrome.\n"
        "  4. Note patient-specific adjustments (renal function, allergy, weight) if context is available.\n"
        "  5. State what to culture before starting, and when to narrow based on results.\n"
        "If the syndrome is unclear, name the most likely diagnosis and give the regimen for that working diagnosis.\n"
        "If local antibiogram data is provided, use it to confirm or override the default choice — state the local susceptibility % explicitly.\n"
        "Do not recommend therapy for syndromes clearly outside ID (e.g., pure cardiac or oncologic issues).\n"
        "Do not invent specific PK/PD numbers or MIC breakpoints. Stick to guideline-concordant regimens.\n"
        "Sound like a helpful ID colleague on a consult call — direct, confident, and brief.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_impression_plan(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    last_probid_summary: dict | None = None,
    last_mechid_summary: dict | None = None,
    last_doseid_summary: dict | None = None,
    last_allergy_summary: dict | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Generate a structured ID consult impression and plan — suitable for the medical record."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    if last_mechid_summary:
        therapy = last_mechid_summary.get("therapy") or last_mechid_summary.get("recommended_options")
        if therapy:
            context_parts.append(f"Recommended therapy: {therapy}.")
        mechanism = last_mechid_summary.get("mechanism") or last_mechid_summary.get("resistance_mechanism")
        if mechanism:
            context_parts.append(f"Resistance mechanism: {mechanism}.")
    if last_doseid_summary:
        dose = last_doseid_summary.get("dose") or last_doseid_summary.get("recommendation")
        if dose:
            context_parts.append(f"Dosing: {dose}.")
    if last_allergy_summary:
        allergy = last_allergy_summary.get("verdict") or last_allergy_summary.get("safety")
        if allergy:
            context_parts.append(f"Allergy assessment: {allergy}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an experienced infectious diseases consultant writing a formal ID consult note for the medical record.\n"
        + context_block +
        "Generate a structured impression and plan in the format used by ID consultants:\n\n"
        "IMPRESSION:\n"
        "Write 2-4 sentences that: (1) name the diagnosis or working diagnosis, (2) summarise the key microbiological finding and resistance pattern if known, "
        "(3) note any high-risk features (endovascular infection, immunosuppression, renal impairment, allergy), "
        "(4) state the clinical status (improving / stable / deteriorating).\n\n"
        "PLAN:\n"
        "Write a numbered action list in the order an ID consultant would prioritise:\n"
        "  1. Antibiotic therapy — name the drug, dose, route, and start date. If switching from empiric to targeted, state the rationale.\n"
        "  2. Duration — state the total course length and the clinical/microbiological criteria for the end date (e.g. 'minimum 14 days from first negative blood culture' for SAB).\n"
        "  3. Source control — whether a line, device, or collection requires removal or drainage, and the timeline.\n"
        "  4. Monitoring — drug levels (vancomycin AUC, voriconazole trough), renal function schedule, LFTs if hepatotoxic agents used, drug-specific monitoring.\n"
        "  5. Follow-up investigations — repeat blood cultures, echocardiogram, MRI, ophthalmology, PET-CT — with timing.\n"
        "  6. Allergy / stewardship notes — any alternative if first-line is contraindicated, oral step-down criteria if applicable.\n"
        "  7. ID follow-up — outpatient review timing, who to contact if deterioration.\n\n"
        "Only include sections that are relevant given the available information — do not invent details not in context. "
        "If a key piece of information is missing (e.g. no dosing data), state 'pending renal function / weight' rather than leaving it blank. "
        "Write in a professional clinical style — terse, accurate, action-oriented. No hedging, no filler phrases.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Use plain numbered lists only.\n"
        "Write IMPRESSION: followed by the text, then PLAN: followed by the numbered items."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_duke_criteria_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Apply Modified Duke criteria to classify infective endocarditis probability."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant applying the Modified Duke Criteria to classify the probability of infective endocarditis (IE).\n"
        + context_block +
        "Use the following Modified Duke Criteria framework:\n\n"
        "MAJOR CRITERIA:\n"
        "  1. Positive blood cultures — one of:\n"
        "     a. Typical microorganism in 2 separate cultures: viridans streptococci, S. bovis (S. gallolyticus), HACEK group, S. aureus, or community-acquired Enterococcus (no primary focus).\n"
        "     b. Persistently positive cultures (≥2 drawn >12h apart, or ≥3 of ≥4 drawn ≥1h apart) with an organism consistent with IE.\n"
        "     c. Single positive culture for Coxiella burnetii (Q fever) or IgG titre >1:800.\n"
        "  2. Evidence of endocardial involvement — one of:\n"
        "     a. Echocardiogram positive for IE: oscillating intracardiac mass on valve or supporting structures, abscess, or new partial dehiscence of prosthetic valve.\n"
        "     b. New valvular regurgitation (worsening or changing of pre-existing murmur NOT sufficient).\n"
        "     c. Positive FDG-PET/CT showing abnormal activity around prosthetic valve (>3 months post-implant) or paraprosthetic leak on cardiac CT.\n\n"
        "MINOR CRITERIA:\n"
        "  1. Predisposing cardiac condition or injection drug use.\n"
        "  2. Fever >38°C.\n"
        "  3. Vascular phenomena: major arterial emboli, septic pulmonary infarcts, mycotic aneurysm, intracranial haemorrhage, conjunctival haemorrhages, Janeway lesions.\n"
        "  4. Immunological phenomena: glomerulonephritis, Osler nodes, Roth spots, rheumatoid factor.\n"
        "  5. Microbiological evidence: positive blood cultures not meeting major criteria, or serological evidence of active infection with organism consistent with IE.\n\n"
        "CLASSIFICATION:\n"
        "  DEFINITE IE: 2 major, OR 1 major + 3 minor, OR 5 minor criteria.\n"
        "  POSSIBLE IE: 1 major + 1 minor, OR 3 minor criteria.\n"
        "  REJECTED: firm alternative diagnosis, resolution with antibiotics ≤4 days, no pathological evidence at surgery/autopsy, or does not meet possible criteria.\n\n"
        "INSTRUCTIONS:\n"
        "First, identify which major and minor criteria are met based on the clinical information provided. "
        "Then state the classification (Definite / Possible / Rejected) with the criteria count that led to it. "
        "Then advise on the next step: if Possible — what additional investigations would upgrade to Definite (TEE if TTE negative, FDG-PET if prosthetic valve). "
        "If Definite — state the implications for management (bactericidal therapy, surgical consultation if indicated).\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_ast_clinical_meaning_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Explain what an AST result means clinically — beyond S/I/R to bedside decision-making."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant explaining the clinical meaning of an antimicrobial susceptibility result to a physician.\n"
        + context_block +
        "Go beyond the S/I/R label — explain what it means at the bedside. Use the following knowledge:\n\n"
        "ESBL (Extended-Spectrum Beta-Lactamase):\n"
        "  - All penicillins and cephalosporins are unreliable even if the disk reports 'Susceptible' — inoculum effect and pharmacodynamic failure make them dangerous in serious infections.\n"
        "  - Pip-tazo: 'Susceptible' by disk diffusion may not reflect in vivo efficacy in bacteraemia — MERINO trial showed meropenem superior to pip-tazo for ESBL bacteraemia. Avoid pip-tazo for definitive therapy of ESBL bacteraemia.\n"
        "  - Reliable agents: meropenem or ertapenem. Ertapenem is appropriate for non-ICU stable bacteraemia and OPAT. Reserve meropenem for severe or ICU cases.\n"
        "  - Oral step-down for UTI only (not bacteraemia): if susceptible — nitrofurantoin, fosfomycin, or trimethoprim (check MIC).\n\n"
        "MRSA (Methicillin-Resistant S. aureus):\n"
        "  - All beta-lactams are unreliable regardless of disk result — mecA gene confers PBP2a alteration that makes beta-lactam binding ineffective.\n"
        "  - Ceftaroline is the only beta-lactam active against MRSA (anti-MRSA cephalosporin) — used for salvage.\n"
        "  - First-line: vancomycin (AUC/MIC target 400-600) or daptomycin (not for pulmonary — inactivated by surfactant). Linezolid for SSTI/pneumonia if IV not feasible.\n\n"
        "VANCOMYCIN MIC:\n"
        "  - MIC ≤2 mg/L = susceptible by EUCAST/CLSI. However, MIC of 2 ('MIC creep') is associated with worse outcomes in S. aureus endocarditis and bacteraemia.\n"
        "  - MIC 2: consider daptomycin as alternative, especially for endovascular infection. Target vancomycin AUC 400-600 carefully with TDM.\n"
        "  - Vancomycin-intermediate S. aureus (VISA): MIC 4-8. Vancomycin likely to fail — use daptomycin ± ceftaroline, or consult specialist.\n\n"
        "HETERORESISTANCE (hVISA):\n"
        "  - Appears susceptible on standard testing (MIC ≤2) but contains a subpopulation resistant at higher concentrations.\n"
        "  - Suspect hVISA if: S. aureus bacteraemia not clearing despite adequate vancomycin levels and source control.\n"
        "  - Population analysis profile (PAP-AUC) confirms — not routinely available. Clinical decision: switch to daptomycin or ceftaroline if bacteraemia persistent despite adequate AUC-guided vancomycin.\n\n"
        "INDUCIBLE CLINDAMYCIN RESISTANCE (D-zone test):\n"
        "  - D-zone positive (inducible MLSb): clindamycin may fail mid-treatment as resistance is induced in vivo. Do not use clindamycin for serious infections (bacteraemia, deep tissue) if D-zone positive, even if disk reports 'Susceptible'.\n"
        "  - D-zone negative: clindamycin is genuinely susceptible and can be used.\n\n"
        "AmpC DEREPRESSION (Enterobacter, Serratia, Citrobacter, Morganella — 'ESCPM' organisms):\n"
        "  - These organisms can derepress chromosomal AmpC beta-lactamase during treatment, even if initially susceptible to 3rd-generation cephalosporins.\n"
        "  - Do NOT use ceftriaxone or cefotaxime for serious infections with these organisms even if disk reports 'Susceptible' — use cefepime or carbapenem.\n"
        "  - Pip-tazo is also unreliable for AmpC-derepressing organisms in serious infections.\n\n"
        "DAPTOMYCIN AND PULMONARY INFECTIONS:\n"
        "  - Daptomycin is inactivated by pulmonary surfactant — do not use for pneumonia or lung abscess regardless of susceptibility.\n\n"
        "Answer the specific question. Explain what the result means practically — what the physician should do and what they should avoid. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who has seen these pitfalls cause treatment failure."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_complexity_flag_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    complexity_features: List[str] | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Assess whether a case has features that warrant escalation to senior ID or MDT review."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Organisms: {', '.join(consult_organisms)}.")
    if complexity_features:
        context_parts.append(f"High-risk features detected: {'; '.join(complexity_features)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are a senior infectious diseases consultant assessing whether a case has features that require escalation beyond a standard consult — "
        "senior ID colleague review, multidisciplinary team (MDT) discussion, or specialist referral.\n"
        + context_block +
        "HIGH-RISK FEATURES THAT EACH ADD COMPLEXITY:\n"
        "  - High-virulence organism: S. aureus (especially MRSA), Candida fungaemia, Pseudomonas aeruginosa, Enterococcus faecium (VRE), carbapenem-resistant Enterobacterales (CRE/KPC/NDM)\n"
        "  - Endovascular infection: endocarditis (native or prosthetic valve), infected vascular graft, infected intracardiac device\n"
        "  - CNS involvement: meningitis, brain abscess, spinal epidural abscess, ventriculitis\n"
        "  - Severely compromised host: solid organ transplant, HSCT (especially post-engraftment), haematological malignancy on intensive chemotherapy, biologics (anti-TNF, rituximab, CAR-T)\n"
        "  - Renal impairment requiring complex dose adjustment: CrCl <30 mL/min, haemodialysis, CRRT\n"
        "  - Documented severe or complex allergy: anaphylaxis to first-line agent making standard therapy impossible\n"
        "  - Polymicrobial infection with conflicting susceptibility requirements\n"
        "  - Treatment failure after ≥5 days of appropriate targeted therapy\n"
        "  - Surgical decision required: valve replacement, hardware removal, drainage — multidisciplinary input needed\n"
        "  - Pregnancy with serious infection requiring potentially harmful antibiotics\n"
        "  - Potential public health implications: notifiable disease (TB, meningococcus, VHF, enteric fever), outbreak concern\n\n"
        "ESCALATION THRESHOLDS:\n"
        "  1 high-risk feature: standard ID consult is appropriate — manage with ID guidance.\n"
        "  2 high-risk features: consider discussing with a senior ID colleague; document the discussion.\n"
        "  ≥3 high-risk features: this case warrants MDT review (ID + microbiology + pharmacy + relevant surgery/cardiology/neurology). "
        "State this clearly to the requesting physician. Do not leave this to a single clinician.\n\n"
        "If the case exceeds standard complexity, state that explicitly: name the specific features driving the risk, "
        "state who should be involved (cardiac surgery, neurosurgery, transplant ID, clinical pharmacy), "
        "and what the time-sensitive decision points are.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like a senior consultant who takes complexity seriously."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_course_tracker_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer day-of-therapy questions — what to check, decide, or de-escalate at a given point in the course."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant answering a question about where a patient is in their antibiotic course "
        "and what clinical decisions or checks are due at this point.\n"
        + context_block +
        "Use the following milestone reference by syndrome and day of therapy:\n\n"
        "S. AUREUS BACTERAEMIA (SAB):\n"
        "  Day 0-2: remove central lines, start vancomycin or beta-lactam (MSSA → flucloxacillin/cefazolin as soon as confirmed — far superior to vancomycin). Draw repeat blood cultures 48-72h.\n"
        "  Day 2-3: if still bacteraemic, suspect uncontrolled source or metastatic focus. TTE (or TEE if TTE negative). MRI spine if back pain or raised CRP. Ophthalmology consult.\n"
        "  Day 7: if blood cultures now negative, count is reset here for uncomplicated SAB (14 days from FIRST negative culture). TEE if not yet done. Confirm no deep focus.\n"
        "  Day 14: minimum for uncomplicated SAB (no endocarditis, no implant, no metastatic focus, cultures cleared within 72h). Complicated or endocarditis: 4-6 weeks total.\n"
        "  Ongoing: vancomycin AUC check at steady state (target 400-600). Renal function every 48-72h.\n\n"
        "INFECTIVE ENDOCARDITIS (IE):\n"
        "  Day 0-3: blood cultures ×3 pre-antibiotics. TTE within 24-48h; TEE within 7 days if TTE negative or prosthetic valve. Surgical assessment for high-risk features (heart failure, abscess, large vegetation).\n"
        "  Day 7-10 (POET eligibility start for Strep/E. faecalis NVE): assess for oral step-down criteria — afebrile, haemodynamically stable, no embolic complications, no surgical indication.\n"
        "  Day 14-17: POET oral step-down criteria for MSSA/MRSA/CoNS NVE if stable.\n"
        "  Week 2-4: repeat echocardiogram. Dental review (source control for Strep IE). Check vegetation size trend — enlarging vegetation on therapy = surgical review urgently.\n"
        "  Week 6: minimum course for most IE. 8 weeks for S. aureus PVE.\n\n"
        "GRAM-NEGATIVE BACTERAEMIA / UROSEPSIS:\n"
        "  Day 1-2: cultures result — narrow from broad empiric to targeted agent. ESBL? Switch to ertapenem.\n"
        "  Day 3-5: if improving, confirm source controlled. Start considering oral step-down (ciprofloxacin 500mg BD for susceptible GNR if tolerating oral).\n"
        "  Day 7: end of course for uncomplicated GNR bacteraemia from urinary source if clinically resolved. 10-14 days for HAP/VAP. 14 days for liver abscess.\n\n"
        "CANDIDAEMIA:\n"
        "  Day 0-1: remove all central catheters. Ophthalmology review within 72h.\n"
        "  Day 2-3: echocardiogram. Repeat blood cultures daily until negative.\n"
        "  Day of first negative culture: start 14-day count from here (not from start of antifungals).\n"
        "  Day 5-7: if stable and susceptible species (C. albicans, C. tropicalis), consider step-down to fluconazole 400mg OD.\n\n"
        "BONE AND JOINT / OSTEOMYELITIS:\n"
        "  Day 0-7: IV phase for stabilisation (OVIVA — often as short as 3-7 days if clinically stable and oral bioavailability good).\n"
        "  Day 7: assess for oral step-down — OVIVA criteria: clinically improving, tolerating oral, susceptible organism with good oral option (levofloxacin ± rifampicin for MSSA, TMP-SMX + rifampicin for MRSA, ciprofloxacin for GNR).\n"
        "  Week 6: reassess — MRI at 4-6 weeks. CRP trend. Decision on total duration (6 weeks osteomyelitis, 12 weeks PJI DAIR, 3-6 months chronic osteomyelitis).\n\n"
        "NEUTROPENIC FEVER:\n"
        "  Day 3-4: if still febrile and no focus, consider adding antifungal (echinocandin or liposomal AmB). Assess ANC — if recovering (>500), begin de-escalation planning.\n"
        "  Day 5-7: review blood cultures, CT chest if not done. If ANC recovered and afebrile ×48h, can stop antibiotics.\n\n"
        "Answer the specific day-of-therapy question. State clearly: what decision is due, what to check, what milestone has been reached, and what the next milestone is. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant on a daily ward round."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_sepsis_management_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Sepsis bundle guidance — Hour-1 bundle, empiric coverage, PCT-guided stopping."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on sepsis management.\n"
        + context_block +
        "Use the following clinical framework:\n\n"
        "SEPSIS RECOGNITION:\n"
        "  - Sepsis: life-threatening organ dysfunction caused by dysregulated host response to infection. "
        "Operationally: suspected infection + SOFA score ≥2 (or qSOFA ≥2 as a bedside screen: RR ≥22, altered mentation, SBP ≤100).\n"
        "  - Septic shock: sepsis + vasopressors needed to maintain MAP ≥65 + lactate >2 mmol/L despite adequate fluids.\n\n"
        "HOUR-1 BUNDLE (Surviving Sepsis Campaign 2018):\n"
        "  1. Measure lactate — repeat if initial >2 mmol/L. Lactate >4 mmol/L = high mortality, treat aggressively.\n"
        "  2. Blood cultures ×2 sets (aerobic + anaerobic) BEFORE antibiotics — do not delay antibiotics >45 min to get cultures.\n"
        "  3. Broad-spectrum antibiotics within 1 hour of recognition. Every hour of delay increases mortality ~7%.\n"
        "  4. Crystalloid 30 mL/kg IV for hypotension or lactate ≥4 mmol/L. Reassess fluid responsiveness (pulse pressure variation, PLR, IVC variability).\n"
        "  5. Vasopressors (noradrenaline first-line) if MAP <65 during or after fluid resuscitation.\n\n"
        "EMPIRIC ANTIBIOTIC SELECTION BY SOURCE:\n"
        "  - Unknown source / community-acquired sepsis: piperacillin-tazobactam 4.5g TDS-QDS (extended infusion preferred); add vancomycin if MRSA risk (healthcare-associated, skin/soft tissue, prior MRSA).\n"
        "  - HAP/VAP: pip-tazo OR cefepime OR meropenem (if risk factors for MDR GNR: prior antibiotics, ICU >5 days, known colonisation) + vancomycin.\n"
        "  - Urosepsis: ceftriaxone 2g OD (community) or pip-tazo / meropenem (healthcare-associated, prior resistant GNR, known ESBL).\n"
        "  - Intra-abdominal: pip-tazo 4.5g TDS-QDS or meropenem 1g TDS + metronidazole if not covered. Source control mandatory.\n"
        "  - Neutropenic fever: pip-tazo 4.5g QDS (anti-pseudomonal); add vancomycin if haemodynamically unstable, mucositis, skin/port infection. Add antifungal (echinocandin or liposomal AmB) if fever >96h despite antibiotics.\n"
        "  - Meningitis: ceftriaxone 2g BD + vancomycin + dexamethasone 0.15mg/kg QDS. Add ampicillin if Listeria risk.\n"
        "  - Skin/soft tissue (necrotising): pip-tazo + clindamycin (anti-toxin effect) + vancomycin. Surgical emergency.\n\n"
        "ANTIBIOTIC DE-ESCALATION:\n"
        "  - Review at 48-72h when culture results available — narrow to the most targeted agent.\n"
        "  - PCT-guided stopping: PCT drop >80% from peak, or absolute value <0.5 ng/mL = safe to stop in most settings (PRORATA trial). PCT falling but not at threshold: continue and recheck in 48h.\n"
        "  - Stop empiric anaerobic coverage if no intra-abdominal source confirmed.\n"
        "  - Stop empiric antifungal if cultures negative at 96-120h and clinical improvement (unless high-risk host).\n"
        "  - Stop vancomycin if MRSA not identified and wound/blood cultures negative at 48-72h.\n\n"
        "MONITORING:\n"
        "  - Repeat lactate at 2h — clearance >10% associated with better outcomes.\n"
        "  - Urine output >0.5 mL/kg/h target after resuscitation.\n"
        "  - Repeat blood cultures if fever persists >72h or new deterioration.\n"
        "  - Vancomycin AUC/MIC monitoring (target AUC 400-600) — daily creatinine.\n\n"
        "Answer the specific question. Lead with the most urgent action. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant who works closely with the ICU."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_cns_infection_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """CNS infection guidance — bacterial meningitis, encephalitis, brain abscess."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on a central nervous system infection.\n"
        + context_block +
        "Use the following clinical framework:\n\n"
        "BACTERIAL MENINGITIS — EMPIRIC TREATMENT:\n"
        "  Standard adult empiric: ceftriaxone 2g IV BD + dexamethasone 0.15mg/kg IV QDS ×4 days.\n"
        "  CRITICAL: give dexamethasone BEFORE or WITH the first antibiotic dose — giving it after the first dose substantially reduces benefit. "
        "Dexamethasone reduces mortality and neurological sequelae (particularly hearing loss) for S. pneumoniae meningitis.\n"
        "  Add vancomycin (target AUC 400-600) if: penicillin-resistant pneumococcus prevalent locally, prior beta-lactam, or immunosuppressed.\n"
        "  Add ampicillin 2g IV 4-hourly if Listeria risk: age >50, immunosuppressed (steroids, calcineurin inhibitors, haematological malignancy), alcoholism, pregnancy. "
        "Listeria is intrinsically resistant to cephalosporins.\n"
        "  LP timing: perform immediately if no contraindication. Do NOT delay antibiotics for CT unless focal neurology, papilloedema, new-onset seizures, GCS <10, or immunocompromised state — these require CT first.\n\n"
        "TARGETED THERAPY ONCE ORGANISM KNOWN:\n"
        "  - S. pneumoniae (penicillin-susceptible MIC ≤0.06): benzylpenicillin 2.4g IV 4-hourly or ceftriaxone 2g BD; 10-14 days.\n"
        "  - S. pneumoniae (penicillin-resistant): ceftriaxone + vancomycin; continue vancomycin until CSF sterilised (repeat LP at 48h).\n"
        "  - N. meningitidis: benzylpenicillin 2.4g IV 4-hourly; 7 days. Notify public health — contact prophylaxis (ciprofloxacin 500mg single dose or rifampicin 600mg BD ×2 days).\n"
        "  - Listeria monocytogenes: ampicillin 2g IV 4-hourly + gentamicin (synergy); 21 days minimum. Add TMP-SMX if ampicillin-allergic.\n"
        "  - GNR (E. coli, Klebsiella): ceftriaxone if susceptible; meropenem if ESBL or resistant; 21 days.\n"
        "  - MRSA: vancomycin (high-dose, AUC-guided) or linezolid; daptomycin does NOT penetrate CSF.\n\n"
        "VIRAL ENCEPHALITIS:\n"
        "  - Start acyclovir 10mg/kg IV TDS EMPIRICALLY for any encephalitis syndrome — do not wait for HSV PCR result.\n"
        "  - HSV encephalitis: acyclovir 10mg/kg TDS ×14-21 days. Check renal function daily — nephrotoxic; ensure adequate hydration.\n"
        "  - CMV encephalitis (immunosuppressed): ganciclovir 5mg/kg BD ± foscarnet 90mg/kg BD.\n"
        "  - West Nile / arboviral: supportive only.\n"
        "  - Send CSF for: HSV1/2 PCR, VZV PCR, CMV PCR (if immunosuppressed), enterovirus PCR, EBV, cryptococcal antigen, AFB smear/culture (if TB risk), VDRL.\n\n"
        "BRAIN ABSCESS:\n"
        "  - Neurosurgical drainage is both diagnostic and therapeutic — aspiration preferred over excision for most lesions.\n"
        "  - Empiric: ceftriaxone 2g BD + metronidazole 500mg TDS (covers streptococcal, anaerobes, GNR). Add vancomycin if post-neurosurgical or trauma.\n"
        "  - Toxoplasma in HIV (CD4 <100, Toxoplasma IgG positive, ring-enhancing lesions): sulfadiazine 1-1.5g QDS + pyrimethamine 200mg loading then 75mg OD + folinic acid 15mg OD. Empiric trial for 2 weeks — if no response, biopsy.\n"
        "  - Duration: 6-8 weeks IV, then oral step-down to complete 3-6 months total for most brain abscesses.\n"
        "  - Repeat MRI at 2-4 weeks — if enlarging despite treatment, reconsider diagnosis and repeat drainage.\n\n"
        "Answer the specific clinical question. Lead with the most urgent action. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant who has seen many cases of meningitis and acts decisively."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_mycobacterial_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Mycobacterial disease guidance — TB treatment, LTBI, MAC, drug monitoring."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on mycobacterial disease.\n"
        + context_block +
        "Use the following clinical framework:\n\n"
        "ACTIVE TB — STANDARD DRUG-SENSITIVE TREATMENT (HRZE):\n"
        "  Intensive phase (2 months): isoniazid (H) + rifampicin (R) + pyrazinamide (Z) + ethambutol (E).\n"
        "  Continuation phase (4 months): isoniazid + rifampicin. Total 6 months for pulmonary TB without complications.\n"
        "  Extended courses: TB meningitis — 12 months total (2HRZE + 10HR); spinal TB — 12 months; pericardial TB — 6 months + corticosteroids.\n"
        "  Dosing (weight-based): rifampicin 10mg/kg/day (max 600mg); isoniazid 5mg/kg/day (max 300mg); pyrazinamide 25mg/kg/day; ethambutol 15-20mg/kg/day.\n"
        "  Pyridoxine (vitamin B6) 10-25mg OD with isoniazid — reduces peripheral neuropathy risk (mandatory in pregnancy, alcoholism, malnutrition, diabetes, CKD, HIV).\n\n"
        "TB DRUG MONITORING:\n"
        "  - LFTs at baseline, 2 weeks, 4 weeks, then monthly for 2 months. Hepatotoxicity threshold: ALT >3× ULN with symptoms or >5× ULN without. Stop all HRZE if hepatitis — reintroduce sequentially once LFTs normalise.\n"
        "  - Ethambutol: visual acuity and colour vision monthly — stop immediately if visual changes. Avoid if CrCl <30 (accumulates).\n"
        "  - Pyrazinamide: LFTs, urate (causes hyperuricaemia — treat gout if symptomatic).\n"
        "  - Rifampicin: potent CYP450 inducer — warn re: drug interactions (tacrolimus, warfarin, OCPs, ARVs, methadone, azoles).\n\n"
        "LATENT TB INFECTION (LTBI) TREATMENT:\n"
        "  Indication: positive IGRA or TST (≥5mm if immunosuppressed; ≥10mm otherwise) with no evidence of active TB.\n"
        "  - 3HP: rifapentine 900mg + isoniazid 900mg weekly ×12 doses (most preferred — high completion rates). Weekly DOT or SAT.\n"
        "  - 1HP: rifapentine 600mg + isoniazid 300mg daily ×28 days (newest, very high completion).\n"
        "  - 3HR: rifampicin 10mg/kg/day + isoniazid 5mg/kg/day ×3 months — good completion.\n"
        "  - 6H: isoniazid 300mg OD ×6 months — older regimen, high hepatotoxicity, lower completion. Use if rifamycin interactions preclude rifampicin (e.g. protease inhibitors).\n"
        "  - Screen: LFTs baseline; repeat at 2-4 weeks if hepatotoxicity risk (age >35, alcohol, prior liver disease).\n\n"
        "MDR-TB AND XDR-TB:\n"
        "  MDR-TB (resistant to H + R): requires specialist management. Standard backbone: bedaquiline 400mg OD ×2 weeks then 200mg TDS ×22 weeks + linezolid 600mg OD + clofazimine 100mg OD + pyrazinamide (if susceptible). Duration ≥18-20 months.\n"
        "  QTc monitoring with bedaquiline and clofazimine (both prolong QT — baseline and monthly ECG).\n\n"
        "MAC (MYCOBACTERIUM AVIUM COMPLEX) — PULMONARY:\n"
        "  Nodular/bronchiectatic (Lady Windermere): azithromycin 500mg three times weekly + ethambutol 25mg/kg three times weekly + rifampicin 600mg three times weekly (intermittent regimen).\n"
        "  Fibrocavitary (more severe): azithromycin 250mg OD (or clarithromycin 500mg BD) + ethambutol 15mg/kg OD + rifampicin 600mg OD (daily regimen). Add amikacin inhaled (ALIS) 590mg OD if refractory.\n"
        "  Duration: minimum 12 months of culture-negative sputum. MAC is not curable in everyone — quality of life and symptom burden guide decision to treat.\n\n"
        "Answer the specific question. State drug, dose, duration, and monitoring. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant with a TB clinic."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_pregnancy_antibiotics_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Pregnancy-safe antibiotic guidance — what is safe, what to avoid, trimester-specific rules."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on antibiotic safety in pregnancy.\n"
        + context_block +
        "Use the following safety reference:\n\n"
        "GENERALLY SAFE THROUGHOUT PREGNANCY:\n"
        "  - Beta-lactams (penicillins, cephalosporins, carbapenems): all trimesters. First-line for most indications in pregnancy. No teratogenicity; widely used.\n"
        "  - Azithromycin: safe — preferred macrolide in pregnancy (erythromycin causes pyloric stenosis in neonate if given in first weeks of life).\n"
        "  - Clindamycin: safe — use for BV, anaerobic coverage, MSSA soft tissue (penicillin-allergic).\n"
        "  - Metronidazole: avoid in first trimester if possible (historical animal teratogenicity data — weak human evidence); safe from 2nd trimester. Needed for BV, C. diff, anaerobic infections.\n\n"
        "AVOID THROUGHOUT PREGNANCY:\n"
        "  - Fluoroquinolones (ciprofloxacin, levofloxacin, moxifloxacin): cartilage toxicity in animal studies; avoid unless benefit clearly outweighs risk (e.g. MDR-TB, no alternative for serious GNR). Increasingly used in TB when necessary.\n"
        "  - Tetracyclines / doxycycline: avoid after 16 weeks — deposits in developing bone and deciduous teeth (tooth discolouration, enamel hypoplasia). Avoid first trimester (limb reduction anomalies in animals).\n"
        "  - Aminoglycosides (gentamicin, amikacin): ototoxicity to fetal cochlea (irreversible deafness). Use only when no alternative for life-threatening infection; single daily dosing preferred; monitor levels.\n"
        "  - Chloramphenicol: 'grey baby syndrome' near term (immature hepatic conjugation). Avoid.\n\n"
        "AVOID NEAR TERM (3RD TRIMESTER / LAST 4 WEEKS):\n"
        "  - TMP-SMX: near term → competitive inhibition of bilirubin binding → neonatal kernicterus risk. Also first trimester: folate antagonist → neural tube defect risk (give folic acid 5mg/day if must use). "
        "Safe in 2nd trimester with folate supplementation.\n"
        "  - Nitrofurantoin: near term (≥36 weeks) → neonatal haemolytic anaemia (G6PD-like mechanism). Use in 1st and 2nd trimester only. Avoid at term.\n"
        "  - High-dose sulfonamides: same kernicterus risk as TMP-SMX near term.\n\n"
        "COMMON SCENARIOS:\n"
        "  - UTI / asymptomatic bacteriuria (must treat in pregnancy): cefalexin 500mg QDS ×5-7d (all trimesters); amoxicillin-clavulanate 625mg TDS ×5-7d; nitrofurantoin 100mg BD ×5d (1st/2nd trimester only).\n"
        "  - Pyelonephritis: admit for IV ceftriaxone 1-2g OD; step down to oral cefalexin or amoxicillin-clavulanate once afebrile ×48h. Total 10-14 days.\n"
        "  - GBS prophylaxis (intrapartum): benzylpenicillin 3g IV loading then 1.5g 4-hourly in labour. Clindamycin or vancomycin if penicillin-allergic.\n"
        "  - SSTI / cellulitis: cefalexin 500mg QDS; amoxicillin-clavulanate. Avoid clindamycin first-line for non-purulent (save for MRSA or penicillin allergy).\n"
        "  - BV / trichomoniasis: metronidazole 400mg BD ×7d (2nd trimester onward); clindamycin cream for BV in 1st trimester.\n"
        "  - Listeria (monocytogenes — rare but pregnancy is a major risk factor): ampicillin 2g IV 4-hourly for bacteraemia/meningitis.\n\n"
        "Answer the specific question. Name the safe agent, dose, and duration. Flag any trimester-specific restrictions. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who works closely with obstetrics."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_travel_medicine_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Travel medicine / returned traveller — fever workup, malaria, typhoid, dengue, tropical infections."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant assessing a returned traveller with fever or a travel-related illness.\n"
        + context_block +
        "Use the following clinical framework:\n\n"
        "IMMEDIATE PRIORITY — RULE OUT MALARIA:\n"
        "  Malaria must be excluded in any febrile traveller from a malaria-endemic region (sub-Saharan Africa, South/South-East Asia, Central/South America, Oceania).\n"
        "  - Incubation: P. falciparum 7-14 days (rarely beyond 3 months); P. vivax/ovale up to 1-2 years (hypnozoites).\n"
        "  - Test: thick and thin blood film + malaria RDT simultaneously. Repeat ×3 over 48h if initial negative but clinical suspicion high.\n"
        "  - Treat as medical emergency if P. falciparum confirmed or suspected: do not delay treatment pending full speciation.\n"
        "  Uncomplicated P. falciparum: artemether-lumefantrine (Riamet) 4 tablets BD ×3 days (with fatty food). Alternative: atovaquone-proguanil (Malarone) 4 tablets OD ×3 days.\n"
        "  Severe malaria (impaired consciousness, seizures, respiratory distress, lactate >5, parasitaemia >2%): IV artesunate 2.4mg/kg at 0, 12, 24h then OD — available via specialist pharmacy. Switch to oral once able to tolerate.\n"
        "  P. vivax / P. ovale: chloroquine (if susceptible region) or artemether-lumefantrine, THEN primaquine 30mg OD ×14 days (radical cure — check G6PD before primaquine).\n\n"
        "DIFFERENTIAL BY EXPOSURE AND INCUBATION:\n"
        "  Short incubation (<14 days):\n"
        "  - Dengue: sudden high fever, severe myalgia ('breakbone fever'), retroorbital pain, maculopapular rash (day 3-5), thrombocytopaenia + leucopaenia on CBC. No specific antiviral — supportive; avoid NSAIDs and aspirin (haemorrhage risk). Dengue NS1 antigen (first 5 days), IgM/IgG serology (after day 5).\n"
        "  - Chikungunya: fever + severe arthralgia (often asymmetric, may persist months). PCR in first week; serology thereafter.\n"
        "  - Rickettsiae (African tick typhus, Mediterranean spotted fever): fever + eschar + rash. Treat empirically with doxycycline 100mg BD — do not wait for serology.\n"
        "  Medium incubation (1-6 weeks):\n"
        "  - Enteric fever (typhoid/paratyphoid Salmonella typhi/paratyphi): fever, relative bradycardia, rose spots, hepatosplenomegaly, diarrhoea or constipation. Blood culture (positive in 60-80% in first week). Treat: ceftriaxone 2g OD ×7-14 days (or azithromycin 500mg OD ×7d for uncomplicated); avoid fluoroquinolones unless susceptibility confirmed (high resistance in South Asia).\n"
        "  - Leptospirosis: fever, myalgia, conjunctival suffusion, jaundice, AKI (Weil's disease). Serology (MAT) or PCR. Treat: doxycycline 100mg BD or benzylpenicillin 1.2g 6-hourly for severe.\n"
        "  Long incubation (>21 days):\n"
        "  - P. vivax/ovale malaria, visceral leishmaniasis (kala-azar — pancytopaenia + splenomegaly, from Indian subcontinent/East Africa), schistosomiasis (Katayama fever — eosinophilia + urticaria after freshwater exposure), viral hepatitis A/E.\n\n"
        "EOSINOPHILIA IN RETURNED TRAVELLER:\n"
        "  Suggests helminthic infection: Strongyloides (can disseminate in immunosuppressed — stool microscopy + Strongyloides serology), schistosomiasis (serology + urine/stool ova), filariasis, toxocariasis, trichinellosis.\n"
        "  CRITICAL: treat Strongyloides BEFORE starting immunosuppression — disseminated strongyloidiasis is fatal. Ivermectin 200mcg/kg OD ×2 days.\n\n"
        "VIRAL HAEMORRHAGIC FEVER (VHF) ALERT:\n"
        "  If travel to sub-Saharan Africa (especially Ebola-endemic regions — DRC, Uganda, Guinea) within 21 days + fever + any haemorrhagic feature: isolate patient immediately, contact ID/infectious diseases on-call and public health — do NOT proceed with routine bloods without PPE and specialist guidance.\n\n"
        "Answer the specific question. Lead with malaria exclusion if travel history is relevant. "
        "State the diagnostic test and treatment for the most likely diagnoses. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant with a travel medicine clinic."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_treatment_failure_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Structured differential for treatment failure — still febrile or not improving on antibiotics."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant called because a patient is not improving on antibiotics. "
        "Work through the differential systematically — do not just reassure.\n"
        + context_block +
        "Use this structured approach to treatment failure:\n\n"
        "1. WRONG DIAGNOSIS — Is this infection at all?\n"
        "   - Drug fever: classically appears days 7-10 of antibiotics; fever persists despite antibiotics, patient looks well, eosinophilia may be present. Discontinue and observe.\n"
        "   - Non-infectious inflammation: DVT, PE, vasculitis, haematoma, pancreatitis, malignancy, adrenal insufficiency, thyroid storm. Check LDH, ferritin, ANA, ANCA if lymphopenia or cytopenias present.\n"
        "   - Wrong anatomical site: culture-confirmed pathogen in blood, but the primary source is not where it was assumed (e.g. vertebral osteomyelitis missed behind presumed UTI bacteraemia).\n\n"
        "2. WRONG DRUG — Susceptibility mismatch\n"
        "   - Resistance not detected at initial testing: heteroresistance (VISA, hVISA for S. aureus), inducible resistance (ESBL not detected by initial screen, AmpC derepression in Enterobacter on cephalosporins).\n"
        "   - Superinfection: new pathogen acquired on treatment (C. difficile, Candida, resistant GNR in ICU).\n"
        "   - Drug not reaching the site: CNS penetration (vancomycin CSF levels poor — need ID input), endovascular vegetation, avascular bone. Check therapeutic drug monitoring.\n\n"
        "3. WRONG DOSE — Pharmacokinetic failure\n"
        "   - Subtherapeutic levels: vancomycin AUC/MIC, aminoglycoside peaks, voriconazole trough. Check drug levels urgently.\n"
        "   - Increased clearance: augmented renal clearance (ARC) in young septic patients increases beta-lactam clearance — may need higher doses or extended infusion.\n"
        "   - Bioavailability issues: oral agent not absorbed (ileus, malabsorption, interaction with divalent cations).\n\n"
        "4. UNCONTROLLED SOURCE — Physical problem, not pharmacological\n"
        "   - Undrained abscess or collection: repeat imaging (CT abdomen/pelvis, echo, MRI spine) to find a new or persisting collection.\n"
        "   - Infected device still in situ: CVC, PICC, pacemaker lead, prosthetic joint — source control is mandatory.\n"
        "   - Infected thrombus / suppurative thrombophlebitis: needs anticoagulation + prolonged antibiotics, sometimes surgical excision.\n\n"
        "5. METASTATIC SEEDING\n"
        "   - S. aureus bacteraemia: always look for seeding — vertebral osteomyelitis, endocarditis (TEE if TTE negative and high suspicion), septic emboli (MRI spine, ophthalmology, brain MRI if neurological signs).\n"
        "   - Candida fungaemia: fundoscopic exam, echocardiogram, hepatosplenic involvement.\n\n"
        "6. HOST FACTORS\n"
        "   - New or unrecognised immunosuppression: neutropenia (recheck CBC), hypogammaglobulinaemia (check IgG), functional asplenia.\n"
        "   - Inadequate response to treatment due to underlying structural disease: bronchiectasis, PVD, chronic osteomyelitis with sequestrum.\n\n"
        "Lead with the most likely reason for failure given the clinical context above. "
        "State specifically what investigation or action is needed for each hypothesis you raise. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like a senior ID consultant who systematically works up treatment failure."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_biomarker_interpretation_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Interpret infectious disease biomarkers — procalcitonin, beta-D-glucan, galactomannan, etc."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant interpreting a biomarker result for a clinician.\n"
        + context_block +
        "Use the following clinical reference for interpretation:\n\n"
        "PROCALCITONIN (PCT):\n"
        "  - <0.1 ng/mL: bacterial infection very unlikely — consider stopping antibiotics if clinically improving.\n"
        "  - 0.1-0.25 ng/mL: low likelihood of bacterial infection; viral or localised infection possible.\n"
        "  - 0.25-0.5 ng/mL: possible bacterial infection — clinical judgement required.\n"
        "  - >0.5 ng/mL: systemic bacterial infection likely; >10 suggests severe sepsis/bacteraemia.\n"
        "  - Stopping rule: PCT drop >80% from peak, or absolute value <0.5 ng/mL — safe to stop antibiotics in many contexts (PRORATA, SAPS trials).\n"
        "  - False positives: major surgery/trauma (peaks day 1-2), cardiogenic shock, burns, pancreatitis, T-cell lymphoma, anti-thymocyte globulin therapy. Not elevated in most fungal/viral infections.\n"
        "  - False negatives: localised infection (abscess, empyema), early infection (<6-12h), immunosuppressed patients.\n\n"
        "BETA-D-GLUCAN (BDG):\n"
        "  - >80 pg/mL (Fungitell): positive threshold; >150 pg/mL: high specificity for invasive fungal infection.\n"
        "  - Detects: Candida, Aspergillus, Pneumocystis jirovecii (PCP — very high BDG), Fusarium, Trichosporon.\n"
        "  - Does NOT detect: Cryptococcus, Mucorales, Blastomyces (these lack (1→3)-β-D-glucan in cell wall).\n"
        "  - False positives: IVIG administration, albumin infusion, haemodialysis with certain membranes, surgical gauze exposure, piperacillin-tazobactam (some batches), severe bacteraemia (Gram-positive more than GNR), mucositis after chemotherapy.\n"
        "  - Interpret in clinical context — two consecutive positive results increase specificity.\n\n"
        "GALACTOMANNAN (GM):\n"
        "  - Serum: index ≥0.5 (Platelia) = positive. BAL: index ≥1.0 = positive.\n"
        "  - Sensitivity best in haematology patients on mould-active prophylaxis (reduced) and in HSCT/AML on no prophylaxis.\n"
        "  - False positives: piperacillin-tazobactam (historical — modern formulations less of an issue), amoxicillin-clavulanate, certain foods (pasta, cereals), cross-reaction with Histoplasma, Fusarium, Paecilomyces.\n"
        "  - False negatives: mould-active antifungal prophylaxis (posaconazole, voriconazole) suppresses GM release — do not use GM to monitor treatment response if on prophylaxis.\n"
        "  - BAL GM more sensitive than serum for pulmonary Aspergillus.\n\n"
        "SERUM CRYPTOCOCCAL ANTIGEN (CrAg):\n"
        "  - Sensitivity >95% for cryptococcal meningitis and disseminated cryptococcosis. Titres correlate with fungal burden.\n"
        "  - If positive in HIV patient with CD4 <100: lumbar puncture mandatory to exclude cryptococcal meningitis even if asymptomatic.\n"
        "  - Serial titres used to monitor treatment response (titre should fall with treatment).\n\n"
        "INTERFERON-GAMMA RELEASE ASSAY (IGRA — QuantiFERON, T-SPOT):\n"
        "  - Positive = latent TB infection (LTBI) or active TB — cannot distinguish. Clinical + radiological context required.\n"
        "  - Advantage over TST: not affected by BCG vaccination; single visit.\n"
        "  - Indeterminate result: more common in immunosuppressed — if high pre-test probability, treat as positive.\n"
        "  - False negatives: severe immunosuppression (CD4 <100, haematological malignancy, high-dose steroids).\n\n"
        "URINE HISTOPLASMA / BLASTOMYCES ANTIGEN:\n"
        "  - Histoplasma urine antigen: sensitivity ~90% for disseminated disease, ~75% for pulmonary. Cross-reacts with Blastomyces and Paracoccidioides.\n"
        "  - Blastomyces urine antigen: sensitivity ~90% for pulmonary and disseminated disease.\n\n"
        "Answer the clinician's specific question. State the clinical interpretation of the value given, relevant false positives/negatives in this context, and the recommended action. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 3 short paragraphs. Sound like an ID consultant who uses these tests daily."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_fluid_interpretation_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Interpret CSF, pleural, peritoneal, or synovial fluid results."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant interpreting a body fluid result.\n"
        + context_block +
        "Use the following interpretation framework:\n\n"
        "CSF INTERPRETATION:\n"
        "  Bacterial meningitis: WBC >1000 (predominantly neutrophils >80%), glucose <2.2 mmol/L or CSF:serum glucose ratio <0.4, protein >1.0 g/L, turbid appearance. Gram stain positive in 60-80%. Treat immediately — do not delay for CT if no focal neurology and no papilloedema.\n"
        "  Viral (aseptic) meningitis: WBC 10-500 (predominantly lymphocytes), glucose normal or mildly low, protein mildly elevated (<1.0 g/L). Common causes: enteroviruses, HSV-2, HIV seroconversion, mumps.\n"
        "  Herpes encephalitis (HSV-1): lymphocytic pleocytosis, elevated protein, RBCs may be present (haemorrhagic encephalitis), glucose normal. EEG temporal lobe changes, MRI temporal hyperintensity. Start acyclovir 10mg/kg TDS empirically — do not wait for PCR.\n"
        "  TB meningitis: lymphocytic pleocytosis (typically 100-500), low glucose (may be very low), high protein (often >1.0 g/L), high ADA (>10 U/L suggestive), AFB smear low sensitivity (~10-40%). CSF culture takes weeks. Treat empirically if clinical suspicion.\n"
        "  Cryptococcal meningitis: lymphocytic pleocytosis (may be minimal in HIV), very high opening pressure (>25 cmH2O), India ink positive (50-80%), CrAg positive (>95%). Measure and manage opening pressure — serial LPs or EVD if >25 cmH2O to prevent vision/hearing loss.\n"
        "  Partially treated bacterial meningitis: typical bacterial pattern but less florid; glucose may be near normal, protein elevated. Blood and CSF culture mandatory.\n\n"
        "PLEURAL FLUID — Light's criteria (exudate if ANY criterion met):\n"
        "  - Pleural protein / serum protein > 0.5\n"
        "  - Pleural LDH / serum LDH > 0.6\n"
        "  - Pleural LDH > 2/3 upper limit of normal serum LDH\n"
        "  Parapneumonic effusion: exudate; if pH <7.2, glucose <3.3, LDH >1000, or Gram stain/culture positive → complicated parapneumonic / empyema requiring drainage.\n"
        "  TB pleuritis: exudate, lymphocytic, ADA >40 U/L (sensitivity ~90%, specificity ~90%), mesothelial cells typically absent. Pleural biopsy has higher yield than culture.\n\n"
        "ASCITIC FLUID — SBP diagnosis:\n"
        "  - PMN count ≥250 cells/µL = SBP — start empiric cefotaxime 2g TDS or ceftriaxone 1g BD immediately.\n"
        "  - SAAG ≥1.1 g/dL = portal hypertension (cirrhosis, Budd-Chiari, right heart failure).\n"
        "  - Secondary peritonitis: PMN >250 with two of: protein >10g/L, glucose <2.8 mmol/L, LDH > serum ULN — suspect bowel perforation, urgent surgical review.\n\n"
        "SYNOVIAL FLUID — septic arthritis:\n"
        "  - WBC >50,000 cells/µL strongly suggests septic arthritis (specificity ~95%); >100,000 is near-diagnostic.\n"
        "  - 20,000-50,000 can be inflammatory (gout, pseudogout, reactive) — Gram stain and culture mandatory regardless of WBC.\n"
        "  - Crystals: negatively birefringent needles = gout (urate); positively birefringent rhomboid crystals = pseudogout (CPPD). Crystals do NOT exclude co-existing septic arthritis — culture all effusions.\n\n"
        "Interpret the specific values given, state the most likely diagnosis, list key differentials to exclude, and state the immediate next step. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who reads fluids with the clinical picture in hand."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_allergy_delabeling_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Assess whether a reported antibiotic allergy is real and advise on delabeling."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant assessing whether a reported antibiotic allergy is genuine "
        "and advising on safe delabeling or rechallenge.\n"
        + context_block +
        "Use the following clinical framework:\n\n"
        "PENICILLIN ALLERGY RISK STRATIFICATION:\n"
        "  LOW RISK (>95% tolerate penicillin — direct oral amoxicillin challenge appropriate):\n"
        "   - Remote reaction >10 years ago, history unclear or family-reported\n"
        "   - Non-immune reactions: GI upset, headache, yeast infection\n"
        "   - Mild rash (maculopapular, non-urticarial) without systemic features, remote, resolved quickly\n"
        "   - 'Amoxicillin rash' in childhood associated with concurrent viral illness (EBV, CMV) — very low risk\n"
        "   → Can give direct graded oral challenge (amoxicillin 250mg, observe 30-60min, then full dose)\n\n"
        "  MODERATE RISK (skin testing recommended before rechallenge if available):\n"
        "   - Urticarial rash (wheals/hives) within 1 hour of dose — possible IgE-mediated\n"
        "   - Unknown reaction type documented in notes as 'allergy' without detail\n"
        "   - Multiple antibiotic allergies reported (often non-specific)\n\n"
        "  HIGH RISK (avoid penicillin; allergy specialist referral for formal evaluation):\n"
        "   - Anaphylaxis (urticaria + hypotension/bronchospasm/angioedema within 1 hour)\n"
        "   - Stevens-Johnson syndrome (SJS) or toxic epidermal necrolysis (TEN)\n"
        "   - Drug reaction with eosinophilia and systemic symptoms (DRESS)\n"
        "   - Serum sickness-like reaction (fever, arthralgia, rash, lymphadenopathy days 1-3 weeks)\n"
        "   - Haemolytic anaemia, interstitial nephritis, or hepatitis attributed to penicillin\n\n"
        "CROSS-REACTIVITY — penicillin to cephalosporins:\n"
        "  - True cross-reactivity rate is ~1-2% (not the historically cited 10%).\n"
        "  - Cross-reactivity is driven by R1 side chain similarity, NOT the beta-lactam ring.\n"
        "  - High cross-reactivity pairs (avoid if penicillin anaphylaxis): amoxicillin ↔ cefadroxil, cefprozil; ampicillin ↔ cefaclor.\n"
        "  - Low/negligible cross-reactivity: ceftriaxone, cefazolin, cefepime, ceftazidime with penicillin.\n"
        "  - For LOW RISK penicillin allergy: cephalosporins can be given without skin testing.\n"
        "  - For HIGH RISK penicillin allergy: avoid high-similarity cephalosporins; non-similar cephalosporins (ceftriaxone, cefazolin) are generally safe with monitoring.\n\n"
        "PENICILLIN TO CARBAPENEM CROSS-REACTIVITY:\n"
        "  - <1% cross-reactivity. Carbapenems can be used in most penicillin-allergic patients, including most anaphylaxis histories, with standard precautions.\n\n"
        "OTHER ANTIBIOTIC ALLERGY NOTES:\n"
        "  - Sulfonamide allergy (TMP-SMX): does not cross-react with other sulfonamide-containing drugs (furosemide, thiazides, sulfonylureas) — different R groups.\n"
        "  - Fluoroquinolone allergy: cross-reactivity between fluoroquinolones is possible but class-specific reactions are uncommon; the reaction type determines whether a different fluoroquinolone can be used.\n"
        "  - Vancomycin 'red man syndrome': NOT an allergy — this is a rate-related infusion reaction (histamine release). Slow the infusion rate and/or premedicate with antihistamine. Vancomycin is not contraindicated.\n\n"
        "Lead with the risk category for this allergy history and the recommended action: direct challenge, skin test first, or avoid and use alternative. "
        "State which specific agents are safe to use as alternatives if penicillin must be avoided. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID-allergist hybrid who champions evidence-based delabeling."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_fungal_management_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer fungal infection management questions — candidaemia, Aspergillus, Cryptococcus, Mucor."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Working diagnosis: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Known organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on invasive fungal infection management.\n"
        + context_block +
        "Use the following clinical knowledge:\n\n"
        "CANDIDAEMIA:\n"
        "  - Remove all central venous catheters as soon as possible — mandatory regardless of species.\n"
        "  - Ophthalmology review within 72h of diagnosis — endophthalmitis occurs in ~5-15%; if present, extends duration.\n"
        "  - Echocardiogram (TTE, or TEE if TTE negative and high suspicion) — Candida endocarditis requires surgical consultation.\n"
        "  - Treatment: echinocandin first-line (anidulafungin 200mg loading then 100mg OD; micafungin 100mg OD; caspofungin 70mg loading then 50mg OD). Step down to fluconazole 400mg OD once patient stabilised, fluconazole-susceptible species confirmed, and repeat cultures negative. NOT for C. krusei (inherently resistant) or C. glabrata (often fluconazole-resistant — check MIC).\n"
        "  - Duration: 14 days from FIRST NEGATIVE blood culture (not from start of treatment). Repeat blood cultures daily until negative.\n"
        "  - C. auris: often pan-resistant — contact infection control immediately; use echinocandin pending susceptibilities.\n\n"
        "INVASIVE ASPERGILLOSIS (IA):\n"
        "  - First-line: voriconazole 6mg/kg BD loading ×2 doses then 4mg/kg BD IV; or 400mg BD PO loading then 200mg BD PO (check trough day 5: target 1-5.5 mg/L). Monitor LFTs weekly.\n"
        "  - Alternative: isavuconazole 200mg TDS ×2 days loading then 200mg OD (fewer drug interactions, no QT prolongation — preferred if on QT-prolonging drugs or azole interactions).\n"
        "  - Salvage: liposomal amphotericin B 3-5mg/kg/day; or combination — limited evidence.\n"
        "  - Avoid voriconazole if on rifampin (levels undetectable) — use liposomal AmB instead.\n"
        "  - Duration: minimum 6-12 weeks; continue until radiological improvement AND immunosuppression resolves.\n"
        "  - Serum galactomannan twice weekly to monitor response (should fall with treatment).\n"
        "  - Surgical debridement: consider for localised accessible disease (e.g. sinuses, skin) and life-threatening haemoptysis.\n\n"
        "CRYPTOCOCCAL MENINGITIS:\n"
        "  - Induction (2 weeks): liposomal amphotericin B 3-4mg/kg/day + flucytosine (5-FC) 25mg/kg QDS. This is the standard of care — do NOT use fluconazole monotherapy for induction in HIV.\n"
        "  - Consolidation (8 weeks): fluconazole 400mg OD.\n"
        "  - Maintenance (≥12 months or until CD4 >200 on ART): fluconazole 200mg OD.\n"
        "  - CRITICAL — manage raised intracranial pressure: LP opening pressure at diagnosis. If >25 cmH2O: daily therapeutic LPs to drain 20-30mL until pressure normalised. Consider EVD/lumbar drain if refractory. Raised ICP is the main driver of early mortality.\n"
        "  - ART timing in HIV: defer 4-6 weeks after antifungal induction to avoid IRIS.\n"
        "  - In non-HIV (transplant, steroids): same antifungal regimen; IRIS uncommon but taper immunosuppression carefully.\n\n"
        "MUCORMYCOSIS:\n"
        "  - Surgical debridement is essential and life-saving — aggressive, often repeated. Do not rely on antifungals alone.\n"
        "  - First-line antifungal: liposomal amphotericin B 5-10mg/kg/day (high doses needed for mould penetration).\n"
        "  - Step-down after clinical improvement: isavuconazole 200mg OD or posaconazole 300mg OD (delayed-release).\n"
        "  - AVOID voriconazole — Mucorales are intrinsically resistant (and voriconazole may paradoxically promote Mucor growth).\n"
        "  - Reverse predisposing factors: control diabetes (target glucose <10 mmol/L), reduce or stop immunosuppression, stop deferoxamine (iron chelator — promotes Mucor growth).\n"
        "  - Risk factors: diabetic ketoacidosis, haematological malignancy, HSCT, solid organ transplant, prolonged neutropenia, high-dose steroids, iron overload.\n\n"
        "Answer the clinician's specific question. Lead with the most urgent action. Give specific drug, dose, and duration. "
        "Flag any critical management steps that, if missed, carry high mortality. "
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant who manages a lot of haematology and transplant patients."
    )
    try:
        return _call_consult_model(prompt=prompt, payload={"question": question}), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_drug_interaction_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    hiv_context: dict | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a drug interaction question relevant to antimicrobial therapy."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    if hiv_context:
        hiv_block = _build_hiv_context_block(hiv_context)
        if hiv_block:
            context_parts.append(f"HIV context: {hiv_block}")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant answering a clinician's question about drug interactions involving antimicrobial agents.\n"
        + context_block +
        "Give a direct, clinically actionable answer. Use the following interaction knowledge:\n\n"
        "PHARMACOKINETIC INTERACTIONS (CYP450 / transporter-mediated):\n"
        "  - Rifampin (rifampicin): potent CYP3A4/2C9/2C19/P-gp INDUCER — dramatically reduces levels of: tacrolimus (may need 3-5x dose increase), ciclosporin, warfarin (INR drops; monitor closely), azole antifungals (fluconazole, voriconazole, itraconazole — avoid voriconazole with rifampin), HIV antiretrovirals (most PIs and NNRTIs), oral contraceptives, methadone, apixaban/rivaroxaban. Onset within 2-3 days; offset takes 2 weeks after stopping.\n"
        "  - Azole antifungals (fluconazole, voriconazole, itraconazole, posaconazole): CYP3A4/2C9/2C19 INHIBITORS — increase levels of: tacrolimus (often requires 30-50% dose reduction), ciclosporin, warfarin (significant INR increase), statins (myopathy risk), benzodiazepines, fentanyl. Itraconazole also P-gp inhibitor (digoxin, dabigatran levels rise).\n"
        "  - Metronidazole: weak CYP2C9 inhibitor — increases warfarin effect (monitor INR); disulfiram-like reaction with alcohol (nausea, flushing, tachycardia — counsel patient).\n"
        "  - Linezolid: MAO inhibitor — avoid with serotonergic drugs (SSRIs, SNRIs, tramadol, fentanyl — risk of serotonin syndrome); avoid with vasopressors/sympathomimetics (hypertensive crisis).\n"
        "  - Fluoroquinolones (ciprofloxacin especially): moderate CYP1A2 inhibitor — increases theophylline, clozapine, tizanidine levels. Chelation with divalent cations (antacids, iron, calcium, zinc) — take 2h before or 6h after. QT prolongation — additive with amiodarone, haloperidol, methadone.\n"
        "  - Doxycycline/tetracyclines: chelation with dairy, antacids, iron (give 2h apart). Potentiates warfarin by reducing gut flora.\n"
        "  - TMP-SMX (co-trimoxazole): CYP2C9 inhibitor — increases warfarin (INR can double); nephrotoxic with ACE inhibitors and potassium-sparing diuretics (hyperkalaemia). Blocks creatinine tubular secretion (raises creatinine without true GFR change — distinguish from nephrotoxicity).\n"
        "  - Clarithromycin: strong CYP3A4/P-gp inhibitor — major interactions with statins (rhabdomyolysis risk; avoid with simvastatin/lovastatin), tacrolimus, colchicine (toxicity), QT prolongation.\n"
        "  - Macrolides (azithromycin): less CYP3A4 than clarithromycin but QT prolongation — additive with other QT-prolonging drugs.\n\n"
        "PHARMACODYNAMIC INTERACTIONS (additive toxicity):\n"
        "  - Vancomycin + aminoglycosides / piperacillin-tazobactam: additive nephrotoxicity — monitor creatinine daily, AUC-guided vancomycin dosing preferred.\n"
        "  - Amphotericin B: nephrotoxic — additive with calcineurin inhibitors, aminoglycosides, NSAIDs, contrast agents. Ensure aggressive pre-hydration.\n"
        "  - Polymyxins (colistin/polymyxin B): nephrotoxic — avoid combination nephrotoxins.\n"
        "  - Linezolid: bone marrow suppression — weekly CBC for prolonged use; additive myelosuppression with other bone marrow suppressants.\n"
        "  - Dapsone: haemolysis in G6PD deficiency — screen before use.\n\n"
        "ANTIRETROVIRAL (ART) INTERACTIONS — critical when patient is on HIV therapy:\n"
        "  - Dolutegravir (DTG) / Bictegravir (BIC): chelated by polyvalent cations (Ca²⁺, Mg²⁺, Fe²⁺, Al³⁺, Zn²⁺) — take INSTI 2h before or 6h after supplements/antacids. Rifampin: DOUBLE DTG to 50mg BID (BIC contraindicated with rifampin). Metformin: cap at 500mg BID (DTG/BIC inhibit OCT2, increase metformin levels 79%). Carbamazepine: avoid (induces UGT1A1, drops INSTI levels).\n"
        "  - Cobicistat (COBI) / Ritonavir (RTV) boosters: potent CYP3A4 INHIBITORS — raise levels of: statins (avoid simvastatin/lovastatin → use atorvastatin ≤20mg or pitavastatin), tacrolimus/ciclosporin (dramatic reduction needed), inhaled fluticasone (Cushing syndrome → use beclomethasone), sildenafil/tadalafil (reduce dose 50-75%), rifabutin (150mg QOD with RTV), amlodipine (reduce dose), colchicine (contraindicated with renal/hepatic impairment). Cobicistat additionally CONTRAINDICATED in pregnancy T2/T3 (subtherapeutic ART levels).\n"
        "  - Rilpivirine (RPV): requires stomach acid for absorption — CONTRAINDICATED with PPIs (omeprazole, pantoprazole). H2RAs: give RPV 12h before or 4h after. Antacids: separate by 2h. Rifampin/rifabutin: avoid (decrease RPV levels below efficacy).\n"
        "  - Tenofovir (TDF/TAF): additive nephrotoxicity with aminoglycosides, amphotericin B, NSAIDs, vancomycin. TDF specifically: avoid with high-dose NSAIDs or concurrent nephrotoxins. TAF: levels decreased by rifampin (use TDF instead); levels increased by cobicistat/ritonavir (use TAF 10mg not 25mg).\n"
        "  - Efavirenz (EFV): CYP2B6 substrate + CYP3A4 inducer — reduces voriconazole levels (double voriconazole dose to 400mg BID or use alternative). Reduces methadone (withdrawal risk — increase methadone dose). Reduces artemether-lumefantrine.\n"
        "  - Atazanavir (ATV): requires acid for absorption — PPIs contraindicated. Causes unconjugated hyperbilirubinemia (cosmetic, not hepatotoxic). Additive QT risk with other QT-prolonging drugs.\n"
        "  - Darunavir/r (DRV/r): CYP3A4 inhibitor (via ritonavir) — same interaction profile as COBI/RTV above. Rifampin contraindicated (use rifabutin 150mg QOD).\n"
        "  - Maraviroc (MVC): CYP3A4 substrate — dose adjustment needed: 150mg BID with CYP3A4 inhibitors (PIs, azoles), 600mg BID with CYP3A4 inducers (rifampin, EFV).\n"
        "  - Lenacapavir: CYP3A4 substrate — avoid strong inducers (rifampin, carbamazepine, phenytoin). Rifabutin OK.\n"
        "  - ART + anti-TB: Rifampin is the most consequential interaction — contraindicated with all PIs, COBI-boosted regimens, RPV, BIC, and maraviroc at standard dose. Options with rifampin: DTG 50mg BID, EFV 600mg OD, or raltegravir 800mg BID. Rifabutin is safer: use 150mg QOD with PI/r; 300mg OD with DTG; 450-600mg OD with EFV.\n\n"
        "ANSWER FORMAT: Lead with the clinical verdict (safe, caution, avoid, or alternative recommended). "
        "State the mechanism briefly. Give the practical management step: what to monitor, what dose adjustment is needed, or what alternative to use. "
        "If the specific interaction is not listed above but involves an ID drug, reason from drug class pharmacology.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 3 short paragraphs. Sound like an ID colleague who also thinks about pharmacology."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_prophylaxis_dose_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a prophylaxis dosing question for immunosuppressed patients."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant providing prophylaxis dosing guidance for immunosuppressed patients.\n"
        + context_block +
        "Give a direct, dose-specific answer. Use the following prophylaxis reference:\n\n"
        "PCP (Pneumocystis jirovecii) PROPHYLAXIS:\n"
        "  - Indication: CD4 <200 cells/µL (HIV), prednisolone >20mg/day for >4 weeks, SOT (all types), haematological malignancy on intensive chemotherapy, CAR-T therapy, alemtuzumab.\n"
        "  - First-line: TMP-SMX (co-trimoxazole) 960mg once daily or 960mg three times weekly. Reduce to 480mg OD if tolerability concerns.\n"
        "  - Sulfa allergy / intolerance: dapsone 100mg OD (check G6PD before starting); or atovaquone 750mg BD with food; or inhaled pentamidine 300mg monthly.\n"
        "  - Duration: until immunosuppression resolves (CD4 >200 for ≥3 months in HIV; <20mg/day prednisolone; post-transplant immune reconstitution — typically 6-12 months).\n\n"
        "MAC (Mycobacterium avium complex) PROPHYLAXIS:\n"
        "  - Indication: HIV with CD4 <50 cells/µL and not yet on ART or ART failing.\n"
        "  - First-line: azithromycin 1250mg once weekly. Alternative: clarithromycin 500mg BD (higher GI side effects).\n"
        "  - Discontinue when CD4 >100 for ≥3 months on ART.\n\n"
        "ANTIFUNGAL PROPHYLAXIS:\n"
        "  - Fluconazole 400mg OD: SOT recipients (liver/renal transplant high-risk period), haematology patients with prolonged neutropenia. Covers Candida (not moulds).\n"
        "  - Posaconazole 300mg OD (delayed-release tablet, with or without food): AML induction or salvage chemotherapy, MDS on intensive therapy, allogeneic HSCT with GVHD on high-dose steroids. Covers Candida AND Aspergillus.\n"
        "    - Monitoring: trough level after 5-7 days; target >0.7 mg/L for prophylaxis, >1.0 mg/L for treatment.\n"
        "  - Voriconazole 200mg BD: alternative for Aspergillus prophylaxis in HSCT where posaconazole not available. Monitor troughs.\n"
        "  - Isavuconazole 200mg OD: alternative mould-active option with fewer drug interactions than voriconazole.\n"
        "  - Micafungin 50mg IV OD: for neutropenic patients who cannot take oral antifungals.\n\n"
        "CMV PROPHYLAXIS:\n"
        "  - Valganciclovir 900mg OD: SOT CMV D+/R- (high-risk — give 6 months); CMV D+/R+ or D-/R+ (give 3-6 months). Dose-adjust for renal function.\n"
        "  - Letermovir 480mg OD (or 240mg OD if on ciclosporin): CMV prophylaxis in CMV R+ allogeneic HSCT — does not require renal dose adjustment.\n\n"
        "TOXOPLASMA PROPHYLAXIS:\n"
        "  - TMP-SMX 960mg OD or three times weekly (same as PCP prophylaxis — covers both). Indication: SOT heart/liver, HIV CD4 <100 with positive Toxoplasma IgG.\n"
        "  - Sulfa allergy: pyrimethamine 25mg OD + dapsone 50mg OD + folinic acid 15mg weekly.\n\n"
        "HEPATITIS B REACTIVATION PROPHYLAXIS:\n"
        "  - Any immunosuppressant therapy in HBsAg+ or anti-HBc+ patients on rituximab, high-dose steroids, chemotherapy.\n"
        "  - Entecavir 0.5mg OD (1mg OD if lamivudine-resistant) or tenofovir 300mg OD. Continue for 12 months after end of immunosuppression.\n\n"
        "Answer the question with the specific agent, dose, frequency, and duration relevant to the patient's immunosuppressive context. "
        "State the indication threshold if relevant. Name the monitoring required.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who does a lot of transplant/haematology work."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_source_control_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a source control question — line removal, drainage, debridement, implant retention."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on source control — the physical removal or drainage of infected material.\n"
        + context_block +
        "Give a direct recommendation. Use the following source control knowledge:\n\n"
        "INTRAVASCULAR LINES (CVC, PICC, arterial lines):\n"
        "  - ALWAYS REMOVE if: S. aureus bacteraemia (line is source or not), Candida fungaemia (any line), tunnel infection, port pocket infection, suppurative thrombophlebitis.\n"
        "  - REMOVE if: bacteraemia persists >72h on appropriate antibiotics, or bacteraemia with a high-virulence organism (GNR, Enterococcus, Pseudomonas, Serratia).\n"
        "  - SALVAGE may be attempted (with antibiotic lock therapy) only for: CoNS in non-critical patient, no tunnel/pocket infection, line essential, susceptible organism. Not for tunnelled lines/ports with port-pocket infection. Salvage failure requires removal.\n"
        "  - IDSA 2009 CLABSI guidelines: S. aureus and Candida require removal regardless of device type.\n\n"
        "ABSCESSES AND FLUID COLLECTIONS:\n"
        "  - Drain (percutaneous or surgical) if: abscess >2cm, liver abscess, psoas abscess, brain abscess (neurosurgical), empyema/pleural empyema (chest drain ± VATS), parapharyngeal/peritonsillar abscess.\n"
        "  - Do NOT drain: small (<2cm) soft tissue abscess if adequate antibiotic coverage is possible; lymph node abscess in TB (drainage can cause chronic fistula — avoid unless diagnosis uncertain).\n"
        "  - Skin/soft tissue: incision and drainage is primary treatment for purulent SSTI — antibiotics are adjunctive.\n\n"
        "PROSTHETIC JOINT INFECTION (PJI):\n"
        "  - Early PJI (<4 weeks from implant, <3 weeks of symptoms): DAIR (debridement, antibiotics, implant retention) may be attempted if implant stable, no sinus tract, susceptible organism. MSSA/Strep PJI favours DAIR. MRSA, Candida, or loose implant = exchange preferred.\n"
        "  - Late PJI (>4 weeks): 2-stage exchange (explant + 6-week antibiotic course + reimplant) is standard. 1-stage exchange for low-virulence organisms (Strep, CoNS) in selected patients.\n"
        "  - Rifampicin MANDATORY in DAIR for biofilm penetration — always combined with another active agent.\n\n"
        "CARDIAC DEVICE INFECTION:\n"
        "  - Lead vegetations or pocket infection: complete hardware removal is required (lead extraction + generator). Antibiotics alone rarely curative for device infection.\n"
        "  - S. aureus on a cardiac device: presume device infection, proceed to extraction.\n\n"
        "ENDOCARDITIS SOURCE CONTROL:\n"
        "  - Surgical indications: heart failure from valve destruction, uncontrolled infection (abscess, fistula, pseudoaneurysm, vegetation >10mm with embolic risk), failure to sterilise (persistent bacteraemia >7 days), fungal endocarditis.\n"
        "  - Discuss cardiac surgery urgently if any of the above present.\n\n"
        "NECROTISING FASCIITIS:\n"
        "  - Immediate surgical debridement is life-saving — do not delay for antibiotics alone. Antibiotics are adjunctive.\n\n"
        "DIABETIC FOOT INFECTION:\n"
        "  - Osteomyelitis of foot: surgical debridement or amputation may be needed if bone is necrotic or antibiotic therapy fails. Vascular assessment (ABPI) critical.\n\n"
        "Answer the clinician's specific source control question. State clearly: remove/drain/operate NOW, consider salvage, or antibiotics sufficient. "
        "Give the specific indication criteria that apply to this case. Name the surgical approach if relevant.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like an ID consultant who values decisive source control."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_iv_to_oral_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Assess IV-to-oral step-down eligibility and recommend the oral agent."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant assessing whether a patient is eligible for IV-to-oral antibiotic step-down.\n"
        + context_block +
        "Structure your answer as follows:\n"
        "  1. State a clear top-line verdict: is oral step-down appropriate, premature, or not indicated for this syndrome?\n"
        "  2. Name the clinical criteria that should be met before switching (SSAT criteria): afebrile 24-48h, WBC trending normal, "
        "tolerating oral intake without malabsorption, no endovascular or CNS source requiring IV, haemodynamically stable.\n"
        "  3. Apply these syndrome-specific oral step-down rules:\n"
        "     - Bone and joint infections (osteomyelitis, septic arthritis, PJI): OVIVA trial (NEJM 2019) showed oral step-down non-inferior after clinical stabilisation (often within 7 days). "
        "Evidence-based oral agents — MSSA: levofloxacin 500-750mg once or twice daily ± rifampicin 450mg twice daily (most commonly used in OVIVA; first choice for excellent bone penetration); "
        "alternatives: TMP-SMX 2 DS tablets twice daily ± rifampicin 450mg twice daily; clindamycin 300-450mg three times daily (bacteriostatic — avoid for PJI); doxycycline 100mg twice daily (lower evidence). "
        "Rifampicin MANDATORY for PJI/hardware — always combined, never monotherapy. "
        "MRSA: TMP-SMX 2 DS tablets twice daily + rifampicin 450mg twice daily; linezolid 600mg twice daily if TMP-SMX not tolerated. "
        "Susceptible GNR: ciprofloxacin 750mg twice daily. Streptococcus: amoxicillin 1g three times daily.\n"
        "     - CAP (non-severe, PSI class I-III): oral from the start — amoxicillin 1g three times daily, doxycycline, or respiratory fluoroquinolone. Step to oral immediately in stable hospitalised patients.\n"
        "     - Cystitis/uncomplicated UTI: oral always preferred — nitrofurantoin 100mg MR twice daily x5d; TMP-SMX twice daily x3d; fosfomycin 3g single dose.\n"
        "     - Pyelonephritis: ciprofloxacin 500mg twice daily x7d; TMP-SMX x14d; amoxicillin-clavulanate x14d.\n"
        "     - Non-purulent cellulitis: cefalexin 500mg four times daily x5d; amoxicillin-clavulanate x5d.\n"
        "     - Purulent SSTI/MRSA SSTI: TMP-SMX 2 DS twice daily x5-7d; doxycycline 100mg twice daily x5-7d.\n"
        "     - Intra-abdominal (mild, post source control): ciprofloxacin + metronidazole 400mg three times daily; or amoxicillin-clavulanate.\n"
        "     - C. difficile: oral vancomycin 125mg four times daily x10d; fidaxomicin 200mg twice daily preferred for recurrence risk.\n"
        "     - Native valve endocarditis (POET trial, very selected): after ≥10 days IV (Strep) or ≥17 days (other organisms), stable patients without surgical indication. "
        "Exact POET regimens: Streptococcus — amoxicillin 1g four times daily; E. faecalis — amoxicillin 1g four times daily + moxifloxacin 400mg once daily; "
        "MSSA — the POET trial used dicloxacillin 1g QID (not routinely available in the US). There is NO validated US oral alternative for MSSA NVE from this trial. "
        "If the question is about MSSA endocarditis oral step-down, state clearly: 'Oral step-down for MSSA endocarditis in the US requires case-by-case discussion with senior ID — "
        "the POET trial agent (dicloxacillin) is not routinely available here, and no US-formulary substitute has been validated by RCT.' "
        "Do NOT substitute cephalexin, amoxicillin-clavulanate, or TMP-SMX as POET-equivalent for MSSA endocarditis; "
        "MRSA/CoNS — linezolid 600mg twice daily + rifampin 300mg twice daily. Not yet universal practice — discuss with senior ID.\n"
        "  4. Syndromes where oral is NOT appropriate: S. aureus bacteraemia without endocarditis (must complete full IV course — POET does not apply to SAB), bacterial meningitis, high-risk febrile neutropenia, prosthetic valve endocarditis.\n"
        "  5. Note key bioavailability facts: fluoroquinolones, TMP-SMX, linezolid, metronidazole, and clindamycin all achieve near-100% oral bioavailability. Amoxicillin-clavulanate does not replicate IV pip-tazo levels.\n"
        "Do not invent susceptibility results. Only recommend specific oral agents if context supports it.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like a confident, evidence-based ID consultant who prefers oral when appropriate."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_duration_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a treatment duration question with syndrome- and organism-specific guidance."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    # Detect high-uncertainty duration scenarios
    question_lower = question.lower()
    orgs_lower = " ".join(consult_organisms or []).lower()
    syndrome_lower = (established_syndrome or "").lower()
    controversy_notes = ""
    if ("aureus" in orgs_lower or "mssa" in orgs_lower or "mrsa" in orgs_lower or
            "aureus" in question_lower) and any(
        k in question_lower for k in ("oral", "switch", "step down", "step-down", "po", "tablet")
    ):
        controversy_notes += (
            "CONTROVERSY: S. aureus bacteraemia requires a complete IV course. "
            "The POET trial applies only to native valve endocarditis caused by Streptococci/Enterococcus — "
            "do NOT generalise POET to SAB. Flag this explicitly if oral step-down is raised for S. aureus bacteraemia.\n"
        )
    if "culture-negative" in question_lower or (
        "endocarditis" in syndrome_lower and not consult_organisms
    ):
        controversy_notes += (
            "UNCERTAINTY: Duration for culture-negative endocarditis is genuinely uncertain. "
            "Acknowledge this — empiric 4-6 weeks IV is standard but evidence is limited. "
            "Flag the importance of infectious diseases specialist review.\n"
        )

    prompt = (
        "You are an infectious diseases consultant answering a clinician's question about antibiotic treatment duration.\n"
        + context_block
        + controversy_notes
        + "Give a clear, guideline-concordant duration recommendation:\n"
        "  1. State the duration range up front — be specific (e.g., '14 days', '4 to 6 weeks', '5 to 7 days').\n"
        "  2. Name the key factors that determine duration: source control status, bacteraemia clearance, response to therapy, "
        "whether this is complicated or uncomplicated.\n"
        "  3. If organism matters (e.g., S. aureus bacteraemia vs. coagulase-negative staph, or MRSA vs. MSSA), note the distinction.\n"
        "  4. Note the clock-start convention when relevant — e.g., for bacteraemia, duration runs from the first negative blood culture, not from antibiotics start.\n"
        "  5. Mention any guideline source for high-stakes decisions (endocarditis, osteomyelitis, meningitis).\n"
        "Reference standard durations: uncomplicated bacteraemia 14d; S. aureus bacteraemia complicated 4-6 weeks; "
        "native valve endocarditis Staph 6 weeks, Strep 2-4 weeks; osteomyelitis 4-6 weeks; "
        "CAP 5 days if responding; HAP/VAP 7-8 days; meningitis (pneumococcal) 10-14d; "
        "septic arthritis 2-4 weeks; UTI/pyelonephritis 5-14 days depending on agent.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 3 short paragraphs. Lead with the number."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_followup_tests_answer(
    *,
    question: str,
    established_syndrome: str | None = None,
    consult_organisms: List[str] | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Answer a question about follow-up tests — TEE, repeat cultures, imaging, drug levels."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    context_block = (" ".join(context_parts) + "\n") if context_parts else ""

    prompt = (
        "You are an infectious diseases consultant advising on follow-up investigations for a patient on antimicrobial therapy.\n"
        + context_block +
        "Answer the clinician's question about what tests to order, when, and why.\n"
        "Apply the following clinical rules when relevant:\n"
        "  TEE (transesophageal echocardiography): recommended for ALL S. aureus bacteraemia unless the source is clearly skin/soft tissue, "
        "TTE is negative, symptoms have been present less than 72h, and the patient is clinically improving — do not delay beyond 5-7 days. "
        "Also indicated for viridans Streptococcus bacteraemia and Enterococcus bacteraemia with prolonged or recurrent positive cultures.\n"
        "  Repeat blood cultures: for S. aureus bacteraemia, repeat at 48-72h to document clearance; persistent bacteraemia at 72h signals high endocarditis risk. "
        "For candidaemia, repeat daily until two consecutive negatives. Any slow-clearing bacteraemia warrants repeat cultures.\n"
        "  Inflammatory markers: CRP and ESR weekly for osteomyelitis monitoring (expect normalisation over 6-8 weeks). "
        "Procalcitonin can guide de-escalation when trending to normal.\n"
        "  Drug level monitoring: vancomycin by AUC/MIC (target 400-600); aminoglycosides by extended-interval random levels; "
        "voriconazole trough 1-5.5 mcg/mL; posaconazole trough >0.7 (prophylaxis) or >1.0 (treatment).\n"
        "  Imaging: MRI spine for vertebral osteomyelitis; CT abdomen/pelvis for occult bacteraemia source; "
        "FDG-PET/CT for suspected endovascular infection or PJI when clinical picture is unclear.\n"
        "  Lung biopsy: consider when BAL is non-diagnostic, empiric therapy is failing, and the diagnosis changes management — "
        "VAT biopsy preferred over CT-guided for diffuse pulmonary disease; assess bleeding risk first.\n"
        "Be direct about which test is needed, when, and what result would change management.\n"
        "Do not order tests that are not indicated. Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs. Sound like a consultant guiding the team."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_consult_summary(
    *,
    established_syndrome: str | None,
    consult_organisms: List[str] | None,
    patient_summary: str | None,
    probid_payload: Dict[str, Any] | None,
    mechid_payload: Dict[str, Any] | None,
    doseid_payload: Dict[str, Any] | None,
    allergy_payload: Dict[str, Any] | None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Synthesise everything known about the current consult into a single integrated consultant summary."""
    if not consult_narration_enabled():
        return fallback_message, False

    summary_payload: Dict[str, Any] = {}
    if established_syndrome:
        summary_payload["establishedSyndrome"] = established_syndrome
    if consult_organisms:
        summary_payload["consultOrganisms"] = consult_organisms
    if patient_summary:
        summary_payload["patientSummary"] = patient_summary
    if probid_payload:
        summary_payload["probidResult"] = probid_payload
    if mechid_payload:
        summary_payload["mechidResult"] = mechid_payload
    if doseid_payload:
        summary_payload["doseidResult"] = doseid_payload
    if allergy_payload:
        summary_payload["allergyResult"] = allergy_payload

    prompt = (
        "You are an infectious diseases consultant giving a verbal summary of a clinical case at the end of a consult.\n"
        "The JSON payload contains everything established so far: syndrome, organisms, resistance findings, dosing, and allergy considerations.\n"
        "Treat all deterministic payloads as the authoritative clinical source of truth. Do not add new claims, invent findings, or change doses.\n"
        "Structure the summary as a consultant would verbally present it:\n"
        "  1. Open with a one-sentence case frame (syndrome, patient demographics if known, key organisms if identified).\n"
        "  2. Summarise diagnostic confidence if probidResult is present.\n"
        "  3. Describe the resistance pattern and therapy recommendation if mechidResult is present.\n"
        "  4. State the renal-adjusted dose and monitoring notes if doseidResult is present.\n"
        "  5. Note allergy considerations and safe alternatives if allergyResult is present.\n"
        "  6. Close with one sentence on what is still pending or would change the plan.\n"
        "If any section is missing from the payload, omit it entirely — do not invent a placeholder.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain prose only.\n"
        "Prefer 3 to 5 short paragraphs. Sound like a consultant giving a verbal sign-out."
    )
    try:
        return _narrate_grounded_message(
            prompt=prompt,
            workflow="summary",
            stage="final",
            fallback_message=fallback_message,
            deterministic_payload=summary_payload,
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


# ---------------------------------------------------------------------------
# HIVID — HIV antiretroviral therapy narrators (Phase 1: initial ART + monitoring)
# ---------------------------------------------------------------------------


def _build_hiv_context_block(
    hiv_context: "dict | None",
    patient_summary: "str | None" = None,
    established_syndrome: "str | None" = None,
    consult_organisms: "list[str] | None" = None,
) -> str:
    """Build a context block from HIV-specific and general consult state for HIVID narrators."""
    context_parts: List[str] = []
    if patient_summary:
        context_parts.append(f"Patient context: {patient_summary}.")
    if established_syndrome:
        context_parts.append(f"Established syndrome: {established_syndrome}.")
    if consult_organisms:
        context_parts.append(f"Identified organisms: {', '.join(consult_organisms)}.")
    if hiv_context:
        hiv_parts: List[str] = []
        if hiv_context.get("viral_load") is not None:
            hiv_parts.append(f"HIV VL: {hiv_context['viral_load']} copies/mL")
        if hiv_context.get("cd4") is not None:
            hiv_parts.append(f"CD4: {hiv_context['cd4']} cells/uL")
        if hiv_context.get("hbv_coinfected"):
            hiv_parts.append("HBV coinfected")
        if hiv_context.get("hcv_coinfected"):
            hiv_parts.append("HCV coinfected")
        if hiv_context.get("pregnant"):
            trimester = hiv_context.get("trimester")
            hiv_parts.append(f"Pregnant (T{trimester})" if trimester else "Pregnant")
        if hiv_context.get("active_oi"):
            hiv_parts.append(f"Active OI: {hiv_context['active_oi']}")
        if hiv_context.get("resistance_mutations"):
            hiv_parts.append(f"Known mutations: {', '.join(hiv_context['resistance_mutations'])}")
        if hiv_context.get("on_art"):
            current = hiv_context.get("current_regimen")
            if current:
                hiv_parts.append(f"Currently on: {', '.join(current)}")
            else:
                hiv_parts.append("Currently on ART (regimen unspecified)")
        if hiv_context.get("creatinine_clearance") is not None:
            hiv_parts.append(f"CrCl: {hiv_context['creatinine_clearance']} mL/min")
        if hiv_context.get("prior_cab_la_prep"):
            hiv_parts.append("Prior CAB-LA PrEP exposure")
        if hiv_parts:
            context_parts.append("HIV-specific data: " + "; ".join(hiv_parts) + ".")
    return (" ".join(context_parts) + "\n") if context_parts else ""


def narrate_hiv_initial_art(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    established_syndrome: str | None = None,
    consult_organisms: list[str] | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Recommend an initial ART regimen based on patient factors, per IAS-USA 2024 guidelines."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary, established_syndrome, consult_organisms)

    prompt = (
        "You are an infectious diseases consultant specializing in HIV medicine, recommending "
        "an initial antiretroviral therapy (ART) regimen based on the IAS-USA 2024 guidelines "
        "(Gandhi et al., JAMA 2025).\n"
        + context_block +
        "CLINICAL DECISION RULES — apply these deterministically, the evidence is settled:\n\n"
        "RECOMMENDED INITIAL REGIMENS (in order of preference for most patients):\n"
        "  1. Bictegravir/emtricitabine/TAF (Biktarvy) — 1 pill daily. High barrier to resistance, minimal interactions. Requires CrCl >=30.\n"
        "  2. Dolutegravir + emtricitabine/TAF (Tivicay + Descovy) — 2 pills daily. High barrier to resistance, HBV-active.\n"
        "  3. Dolutegravir + lamivudine (Dovato) — 1 pill daily, 2-drug regimen. "
        "ONLY if VL <500,000, NO HBV coinfection, NO resistance to either component, CrCl >=50. "
        "Not for rapid start when genotype/HBV/VL results are not yet available.\n\n"
        "MANDATORY DECISION LOGIC — check each:\n"
        "  - HBV coinfected: MUST include tenofovir (TDF or TAF) + emtricitabine. Dovato is CONTRAINDICATED (no HBV coverage, flare risk).\n"
        "  - CrCl <30: avoid TAF (not studied) and TDF (nephrotoxic). Consult HIV pharmacist for adjusted backbone.\n"
        "  - VL >500,000: do NOT use Dovato (2-drug). Use 3-drug regimen (Biktarvy or DTG + F/TAF).\n"
        "  - Pregnancy: DTG is recommended including first trimester (NTD risk ~0.2%, similar to background). "
        "Preferred: DTG + emtricitabine/TDF (or TAF). BIC/TAF/FTC is alternative (BIIb). "
        "Cobicistat-boosted regimens are CONTRAINDICATED in pregnancy (low drug levels). "
        "If prior CAB-LA PrEP exposure: use DRV/r 600/100 BID + TXF/XTC instead (possible INSTI resistance).\n"
        "  - Prior CAB-LA PrEP: start DRV/r 600/100 BID + TXF/XTC. Send integrase genotype. "
        "Switch to INSTI once resistance excluded.\n"
        "  - Active TB with rifampin: DTG 50mg BID (double dose). BIC contraindicated with rifampin. "
        "Cobicistat contraindicated. For 3HP (weekly isoniazid+rifapentine latent TB): DTG 50mg OD is acceptable. "
        "For 1HP (daily isoniazid+rifapentine latent TB): DTG 50mg BID.\n"
        "  - Metformin co-administration: DTG/BIC increase metformin via OCT2 — cap metformin at 500mg BID.\n"
        "  - Weight/metabolic concerns: INSTI class + TAF associated with greatest weight gain. "
        "Consider TDF over TAF if renal function allows. Lifestyle counseling for diet and exercise.\n"
        "  - Cardiovascular risk: pitavastatin recommended for primary CVD prevention in all PLWH aged 40-75 "
        "(REPRIEVE trial). If on abacavir, switch to non-abacavir regimen if CVD risk factors present.\n\n"
        "RAPID/SAME-DAY ART START:\n"
        "  - Recommend same-day start for most new diagnoses. Do NOT wait for genotype results.\n"
        "  - EXCEPTIONS — delay ART:\n"
        "    - Cryptococcal meningitis: defer 4-6 weeks (COAT trial — early ART increases mortality).\n"
        "    - TB meningitis: defer 4-8 weeks (high IRIS mortality).\n"
        "    - CMV retinitis (zone 1): defer ~2 weeks (immune recovery uveitis risk).\n"
        "  - All other OIs: start ART within 2 weeks (PCP, toxo, MAC, histoplasmosis, KS, PML).\n"
        "  - Pulmonary TB: CD4 <50 start within 2 weeks; CD4 >=50 start within 2-8 weeks.\n\n"
        "BASELINE LABS to order at or before ART start:\n"
        "  HIV VL, CD4, RT-protease genotype, HBV serologies, HCV Ab, CMP, CBC, fasting lipids, "
        "glucose/HbA1c, pregnancy test if applicable, STI screening (RPR, GC/CT NAAT at all exposed sites), "
        "Toxoplasma IgG, CMV IgG, IGRA for TB. If CD4 <100: serum cryptococcal antigen.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the specific recommended regimen name, dose, and pill count.\n"
        "  2. State why this regimen was chosen over alternatives (address HBV, renal, VL, pregnancy, OI if relevant).\n"
        "  3. If any mandatory checks above apply, explicitly address each one.\n"
        "  4. State ART timing: same-day vs. defer with specific timeframe and reason.\n"
        "  5. List the baseline labs to order.\n"
        "  6. One sentence on expected trajectory (VL undetectable by ~12-24 weeks on INSTI-based ART).\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs. Sound like an ID consultant at the bedside."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_monitoring(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Provide HIV lab monitoring schedule per IAS-USA 2024 guidelines."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant advising on HIV laboratory monitoring "
        "per IAS-USA 2024 guidelines.\n"
        + context_block +
        "MONITORING SCHEDULE RULES:\n\n"
        "AT DIAGNOSIS / BEFORE ART:\n"
        "  - HIV viral load, CD4 count\n"
        "  - HIV genotype: RT + protease. Integrase genotype ONLY if prior CAB-LA PrEP or known INSTI-exposed source.\n"
        "  - HBV serologies (HBsAg, anti-HBs, anti-HBc) — critical for regimen selection\n"
        "  - HCV antibody\n"
        "  - CMP (creatinine, hepatic panel), CBC\n"
        "  - Fasting lipid panel, glucose/HbA1c\n"
        "  - Urinalysis if starting TDF-based regimen\n"
        "  - Pregnancy test if applicable\n"
        "  - STI screening: syphilis (RPR/VDRL), gonorrhea + chlamydia (NAAT at all exposed sites)\n"
        "  - Toxoplasma IgG, CMV IgG\n"
        "  - TB screening: IGRA preferred over TST\n"
        "  - If CD4 <100: serum cryptococcal antigen\n\n"
        "ON-TREATMENT:\n"
        "  Viral load:\n"
        "    - 4-6 weeks after starting ART (confirm initial response)\n"
        "    - Then every 4-8 weeks until undetectable (<50 copies/mL)\n"
        "    - Once suppressed and stable: every 3-6 months\n"
        "    - After >=2 years suppressed + stable: can extend to every 6 months\n"
        "    - After >=5 years suppressed + stable + patient prefers less monitoring: annually is acceptable\n\n"
        "  CD4 count:\n"
        "    - Every 3-6 months for first 2 years\n"
        "    - After >=2 years suppressed + CD4 >300: can stop routine CD4 monitoring\n"
        "    - Resume if virologic failure, new OI, or immunosuppressive therapy\n\n"
        "  Renal function (CrCl):\n"
        "    - Baseline, then annually if on TAF\n"
        "    - Baseline, then every 3-6 months if on TDF\n"
        "    - NOTE: DTG/BIC can increase serum creatinine ~0.1 mg/dL via OCT2 inhibition — "
        "this is NOT nephrotoxicity, do NOT change regimen for this alone\n\n"
        "  Fasting lipids:\n"
        "    - Baseline, then annually (more frequent if abnormal)\n"
        "    - Pitavastatin recommended for primary CVD prevention in PLWH age 40-75 (REPRIEVE trial)\n\n"
        "  HBV monitoring (if coinfected):\n"
        "    - HBV VL at baseline, then every 3-6 months on tenofovir-containing regimen\n"
        "    - CRITICAL: if tenofovir discontinued for any reason, monitor closely for HBV flare (can be severe/fatal)\n\n"
        "  Genotype at failure:\n"
        "    - If VL >=200 copies/mL on 2 consecutive measurements after >=24 weeks: "
        "send RT + protease + integrase genotype while on failing regimen\n"
        "    - Blips (single VL 50-200 then re-suppresses): NOT failure — "
        "reassess adherence, polyvalent cation chelation (Ca, Mg, Fe, Al taken with INSTI), drug interactions (PPIs with rilpivirine)\n\n"
        "  Weight and metabolic:\n"
        "    - Document weight and BMI every 6 months when on INSTI or TAF\n"
        "    - Do NOT switch regimen solely for weight gain (evidence of benefit is lacking)\n"
        "    - GLP-1 receptor agonists effective for weight loss in PLWH (similar to general population)\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with what is needed RIGHT NOW for this patient (baseline vs on-treatment schedule).\n"
        "  2. Provide the specific monitoring schedule relevant to their current ART and clinical status.\n"
        "  3. Flag any regimen-specific monitoring (TDF nephrotoxicity, DTG creatinine artifact, lipid changes).\n"
        "  4. If applicable, note when monitoring frequency can safely be reduced.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 2 to 4 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


# ---------------------------------------------------------------------------
# HIVID Phase 2 — PrEP, PEP, pregnancy, OI-ART timing
# ---------------------------------------------------------------------------


def narrate_hiv_prep(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """PrEP regimen selection and monitoring per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant advising on HIV pre-exposure prophylaxis (PrEP) "
        "per IAS-USA 2024 guidelines.\n"
        + context_block +
        "PrEP INDICATIONS:\n"
        "  - Offer to ALL sexually active persons requesting PrEP, anyone who injects nonprescription drugs, "
        "uses substances, or has an SUD — no need for risk-scoring tools.\n"
        "  - Particularly encourage: MSM, transgender persons, persons with partners from high-incidence regions, "
        "transactional sex, PWID, incarcerated persons, anyone with STI in the past year.\n\n"
        "PrEP REGIMEN OPTIONS — choose based on exposure route, adherence capacity, and patient preference:\n\n"
        "  1. ORAL TDF/FTC (Truvada or generic) 200/300mg — 1 pill daily.\n"
        "     - Recommended for ALL populations (MSM, cisgender women, TGW, PWID).\n"
        "     - Initiate with a double loading dose, then 1 pill daily thereafter.\n"
        "     - Discontinuation: continue until 2 doses after last sexual activity (rectal exposure) "
        "or 7 days after last activity (vaginal/neovaginal exposure).\n"
        "     - 4+ doses/week provides high protection for rectal exposure; 2+ doses/week gives 79-88% reduction.\n"
        "     - Recommended in pregnancy and breastfeeding.\n"
        "     - Requires CrCl >=60.\n\n"
        "  2. ON-DEMAND (2-1-1) TDF/FTC — for cisgender men and others with planned receptive anal sex ONLY.\n"
        "     - 2 pills 2-24h before sex, 1 pill 24h after first dose, 1 pill 48h after first dose.\n"
        "     - If sex continues: daily dosing until 2 doses after last activity.\n"
        "     - NOT for vaginal/neovaginal exposures or IDU alone (insufficient tissue drug levels).\n"
        "     - For TGW on gender-affirming hormones: administer with food (mitigates lower TFV-DP in rectal tissue).\n\n"
        "  3. ORAL TAF/FTC (Descovy) 200/25mg — 1 pill daily.\n"
        "     - LIMITED to cisgender men and others WITHOUT receptive vaginal sex or IDU alone.\n"
        "     - Preferred over TDF/FTC if CrCl 30-60 or known osteopenia/osteoporosis.\n"
        "     - Bone density scan NOT required before initiating tenofovir-based PrEP.\n\n"
        "  4. INJECTABLE CAB-LA (cabotegravir, Apretude) 600mg IM gluteal — every 2 months.\n"
        "     - First 2 injections separated by 4 weeks, then every 8 weeks.\n"
        "     - Superior to oral TDF/FTC in clinical trials.\n"
        "     - Recommended for ALL populations likely to be sexually exposed to HIV.\n"
        "     - Optional oral CAB lead-in for 4-5 weeks (recommended for severe atopic histories).\n"
        "     - For those without oral lead-in: overlap first injection with 7 days of oral TDF/FTC PrEP.\n"
        "     - Must be gluteal injection — anterior thigh did NOT reach PK targets.\n"
        "     - If injections >=8 weeks late: reload with 4-week interval between 2 injections before resuming q8w.\n"
        "     - Keep 1-month supply of oral TDF/FTC on hand for bridging if injection delayed >=7 days.\n"
        "     - Growing safety data in pregnancy and breastfeeding (BIIa).\n"
        "     - Does NOT provide HBV coverage — rescreen for HBV and immunize if indicated.\n\n"
        "  5. LENACAPAVIR SC — every 6 months.\n"
        "     - PURPOSE 1 trial: 100% efficacy in cisgender women.\n"
        "     - FDA review pending for PrEP indication as of late 2024.\n"
        "     - Not yet approved for PrEP — mention as emerging option if patient asks.\n\n"
        "BASELINE BEFORE PrEP:\n"
        "  - Confirm HIV-negative: 4th-gen Ag/Ab test (if injectable CAB-LA: also HIV RNA).\n"
        "  - CrCl (>=60 for TDF, >=30 for TAF).\n"
        "  - HBV serologies (TDF/TAF are HBV-active — flare risk on discontinuation if HBsAg+).\n"
        "  - HCV antibody.\n"
        "  - STI screening (RPR, GC/CT NAAT at all exposed sites).\n"
        "  - Pregnancy test if applicable.\n\n"
        "MONITORING ON PrEP:\n"
        "  - HIV test every 3 months (every 2 months for injectable CAB — at each injection visit).\n"
        "  - CrCl every 6-12 months (TDF) or annually (TAF).\n"
        "  - STI screening every 3-6 months.\n"
        "  - If HBsAg+: monitor HBV VL; do NOT stop TDF/TAF without hepatology input.\n\n"
        "RAPID PrEP START:\n"
        "  - If negative HIV serology from blood drawn within 7 days or same-day rapid test negative: "
        "start PrEP immediately while awaiting additional diagnostics.\n"
        "  - Any delay is a missed prevention opportunity.\n\n"
        "DoxyPEP (STI prevention — mention if MSM/TGW):\n"
        "  - Doxycycline 200mg within 72h after condomless sex.\n"
        "  - Reduces chlamydia 70-88%, syphilis 73-87%. Less consistent for gonorrhea.\n"
        "  - Recommended for MSM/TGW with bacterial STI in past 12 months.\n"
        "  - Can be taken as often as daily. Prescribe 30 doses (60 tablets) at a time.\n"
        "  - Quarterly STI screening at all exposed sites recommended.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the recommended PrEP regimen for THIS patient's specific exposure pattern.\n"
        "  2. Explain why this option over alternatives (exposure route, adherence, renal function).\n"
        "  3. List baseline labs needed.\n"
        "  4. State monitoring schedule.\n"
        "  5. If MSM/TGW: mention doxyPEP for STI prevention.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_pep(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """PEP regimen and follow-up per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant advising on HIV post-exposure prophylaxis (PEP) "
        "per IAS-USA 2024 guidelines.\n"
        + context_block +
        "PEP TIMING — CRITICAL:\n"
        "  - Start within 72 hours of exposure (earlier = more effective; ideally within 2 hours).\n"
        "  - Do NOT start PEP if >72 hours since exposure — offer PrEP instead if ongoing risk.\n\n"
        "PREFERRED PEP REGIMEN:\n"
        "  - Dolutegravir 50mg OD + emtricitabine/TDF (Truvada) 200/300mg OD x 28 days.\n"
        "  - Alternative: Biktarvy (BIC/TAF/FTC) 1 pill OD x 28 days — simpler, widely used in US centers.\n"
        "  - Complete the full 28-day course.\n\n"
        "BASELINE LABS:\n"
        "  - 4th-gen HIV Ag/Ab test (rule out existing infection BEFORE starting PEP).\n"
        "  - HBV serologies.\n"
        "  - HCV antibody.\n"
        "  - CMP (creatinine, hepatic panel).\n"
        "  - Pregnancy test if applicable.\n"
        "  - Source patient: rapid HIV test and/or HIV VL if status unknown.\n\n"
        "FOLLOW-UP:\n"
        "  - HIV test at 4-6 weeks and 3 months post-exposure (4th-gen Ag/Ab).\n"
        "  - If symptoms of acute HIV during PEP: order HIV VL + 4th-gen immediately.\n"
        "  - Assess for ongoing exposure risk — transition to PrEP if ongoing risk (no gap between PEP and PrEP).\n\n"
        "SPECIAL SITUATIONS:\n"
        "  - Known resistant source virus: adjust PEP based on source genotype.\n"
        "  - Pregnancy: DTG-based PEP is acceptable.\n"
        "  - Renal impairment: use TAF (Biktarvy) instead of TDF if CrCl 30-60.\n"
        "  - HBsAg+ patient: TDF/FTC backbone treats both; warn about HBV flare if PEP is stopped "
        "and patient is HBsAg+ — ensure hepatology follow-up.\n"
        "  - Source on ART with undetectable VL: risk of transmission extremely low (U=U). "
        "PEP may not be necessary — shared decision-making.\n"
        "  - After CAB-LA PrEP breakthrough or suspected INSTI-resistant source: "
        "use DRV/r 600/100 BID + TDF/FTC x 28 days instead of INSTI-based PEP.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the specific PEP regimen, dose, duration.\n"
        "  2. Emphasize timing (must start within 72h, ideally within 2h).\n"
        "  3. List baseline labs.\n"
        "  4. State follow-up schedule.\n"
        "  5. Discuss PrEP transition if ongoing risk.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 4 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_pregnancy(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Pregnancy-specific ART, delivery planning, and neonatal prophylaxis per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant advising on HIV management in pregnancy "
        "per IAS-USA 2024 guidelines.\n"
        + context_block +
        "ART IN PREGNANCY — CORE RULES:\n"
        "  - Immediate ART initiation for ALL pregnant individuals with HIV (for maternal health + prevent perinatal/sexual transmission).\n"
        "  - Preferred regimen: dolutegravir + emtricitabine/TAF (or TDF if TAF not available).\n"
        "     - DTG is recommended INCLUDING first trimester — NTD risk ~0.2%, comparable to background rate.\n"
        "  - Alternative: BIC/TAF/FTC (Biktarvy) — evidence rating BIIa. PK studies show adequate bictegravir levels in pregnancy. "
        "If already on Biktarvy and suppressed, continue.\n"
        "  - If DTG and BIC are not options: DRV/r 600/100 BID + TAF/XTC (or TDF/XTC).\n"
        "  - If prior CAB-LA PrEP exposure: DRV/r 600/100 BID + TXF/XTC (possible INSTI resistance).\n\n"
        "CONTRAINDICATED in pregnancy:\n"
        "  - Cobicistat-containing regimens (low drug levels in 2nd/3rd trimester — reduced efficacy).\n"
        "  - Insufficient data for: doravirine-containing regimens, injectable CAB+RPV, DTG/3TC (Dovato).\n"
        "  - If pregnant while on injectable CAB+RPV: switch to oral triple-drug regimen.\n\n"
        "DELIVERY PLANNING (based on VL at 36 weeks):\n"
        "  - VL suppressed (<50 copies/mL): vaginal delivery. No IV zidovudine needed.\n"
        "  - VL 50-999 at 36 weeks: scheduled C-section at 38 weeks + IV zidovudine during delivery.\n"
        "  - VL >=1000 at 36 weeks: scheduled C-section at 38 weeks + IV zidovudine + consider intensified neonatal prophylaxis.\n\n"
        "NEONATAL PROPHYLAXIS:\n"
        "  - Low risk (maternal VL suppressed, on ART >=4 weeks before delivery): "
        "zidovudine (AZT) x 4 weeks to the neonate.\n"
        "  - High risk (maternal VL >50 at delivery, or no/late ART, or ART <4 weeks before delivery): "
        "zidovudine + lamivudine + nevirapine (3-drug) x 6 weeks to the neonate.\n"
        "  - Breastfeeding: in the US, formula feeding is recommended when safe and available. "
        "If breastfeeding is chosen: continue suppressive ART throughout, monitor infant HIV status.\n\n"
        "SPECIAL CONSIDERATIONS:\n"
        "  - HBV coinfection: MUST maintain tenofovir-containing regimen throughout pregnancy and postpartum "
        "(risk of HBV flare if discontinued).\n"
        "  - VL monitoring in pregnancy: check at ART initiation, 2-4 weeks later, monthly until undetectable, "
        "then at minimum at 36 weeks for delivery planning.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the recommended ART regimen for this pregnant patient.\n"
        "  2. Address trimester-specific considerations.\n"
        "  3. State delivery plan based on expected VL trajectory.\n"
        "  4. Specify neonatal prophylaxis recommendation.\n"
        "  5. Flag any special considerations (HBV, adherence, prior CAB-LA).\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_oi_art_timing(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    established_syndrome: str | None = None,
    consult_organisms: list[str] | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """When to start ART relative to an active opportunistic infection per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary, established_syndrome, consult_organisms)

    prompt = (
        "You are an infectious diseases consultant advising on the timing of ART initiation "
        "relative to an active opportunistic infection, per IAS-USA 2024 guidelines.\n"
        + context_block +
        "OI-SPECIFIC ART TIMING RULES:\n\n"
        "START ART WITHIN 2 WEEKS (for most OIs):\n"
        "  - PCP (Pneumocystis jirovecii pneumonia): start ART within 2 weeks of PCP treatment.\n"
        "  - Toxoplasmosis: within 2 weeks.\n"
        "  - MAC (Mycobacterium avium complex): within 2 weeks.\n"
        "  - Histoplasmosis: within 1-2 weeks.\n"
        "  - Kaposi sarcoma: start ART immediately — ART IS the primary treatment.\n"
        "  - PML (progressive multifocal leukoencephalopathy): start ART immediately — no specific PML treatment exists.\n\n"
        "PULMONARY TB (non-meningeal) — CD4-dependent:\n"
        "  - CD4 <50: start ART within 2 weeks of TB treatment initiation.\n"
        "  - CD4 >=50: start ART within 2-8 weeks of TB treatment.\n"
        "  - Drug interactions: DTG 50mg BID with rifampin. BIC contraindicated with rifampin. "
        "Cobicistat contraindicated with rifampin.\n"
        "  - For 3HP (weekly isoniazid+rifapentine for latent TB): DTG 50mg OD is acceptable.\n"
        "  - For 1HP (daily isoniazid+rifapentine for latent TB): DTG 50mg BID.\n"
        "  - Alternative if INSTI not available: ritonavir-boosted atazanavir or lopinavir + TXF/XTC with rifabutin 150mg daily.\n\n"
        "DEFER ART — HIGH IRIS RISK:\n"
        "  - TB MENINGITIS: defer ART 4-8 weeks after TB treatment initiation. "
        "Early ART in TB meningitis is associated with increased IRIS mortality.\n"
        "  - CRYPTOCOCCAL MENINGITIS: defer ART 4-6 weeks after antifungal induction. "
        "COAT trial: early ART increased mortality. "
        "Exception: asymptomatic cryptococcal antigenemia with negative CSF — immediate ART + preemptive fluconazole.\n"
        "  - CMV RETINITIS (zone 1 involvement): defer ART ~2 weeks (immune recovery uveitis risk).\n\n"
        "GENERAL IRIS CONSIDERATIONS:\n"
        "  - IRIS is more common with lower baseline CD4, higher baseline VL, and rapid immune reconstitution.\n"
        "  - IRIS is NOT a reason to stop ART — manage with anti-inflammatory agents (NSAIDs, corticosteroids in severe cases).\n"
        "  - Close clinical monitoring in the first 4-12 weeks after ART initiation with an active OI.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the specific ART timing recommendation for THIS OI.\n"
        "  2. State the evidence basis (trial name or guideline level).\n"
        "  3. Address drug interactions if relevant (especially TB + rifampin).\n"
        "  4. Discuss IRIS risk and monitoring.\n"
        "  5. Name the recommended ART regimen in this context.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 4 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


# ---------------------------------------------------------------------------
# HIVID Phase 3 — Treatment failure, resistance, switch/simplification
# ---------------------------------------------------------------------------


def narrate_hiv_treatment_failure(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Virologic failure workup per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant evaluating HIV virologic failure "
        "per IAS-USA 2024 guidelines.\n"
        + context_block +
        "DEFINITIONS:\n"
        "  - Virologic failure: HIV RNA >=200 copies/mL on 2 consecutive measurements after >=24 weeks of ART.\n"
        "  - Viral blip: single VL 50-200, then re-suppresses. Common with current assays. NOT failure.\n"
        "  - Persistent low-level viremia: VL 50-200 despite confirmed excellent adherence. "
        "May be caused by large HIV reservoir, clonal expansion, or impaired immune response. "
        "Unlikely to benefit from ART intensification.\n\n"
        "FAILURE WORKUP — IN ORDER OF LIKELIHOOD:\n"
        "  1. ADHERENCE — most common cause. >95% adherence needed for sustained suppression.\n"
        "     - Ask directly, non-judgmentally. Assess pill burden, dosing frequency, side effects, cost, stigma.\n"
        "     - For INSTI-based regimens: check polyvalent cation supplements (calcium, magnesium, iron, zinc, aluminum) — "
        "these chelate INSTIs and impair absorption. INSTIs must be taken 2h before or 6h after cations.\n"
        "  2. DRUG INTERACTIONS:\n"
        "     - Rilpivirine + PPIs: CONTRAINDICATED (need gastric acid for absorption).\n"
        "     - Rilpivirine + H2 blockers: take RPV 4h before or 12h after.\n"
        "     - Rifampin + cobicistat or bictegravir: CONTRAINDICATED.\n"
        "     - Metformin + DTG/BIC: increased metformin levels.\n"
        "  3. ABSORPTION ISSUES: GI conditions (inflammatory bowel disease, short bowel, bariatric surgery).\n"
        "  4. RESISTANCE TESTING:\n"
        "     - Send RT + protease + integrase genotype WHILE ON the failing regimen.\n"
        "     - Review ALL prior resistance results — resistance is cumulative (archived mutations may not appear on current genotype).\n"
        "     - Some commercial assays require VL >500-1000 to perform genotyping.\n\n"
        "SWITCH STRATEGY AFTER CONFIRMED FAILURE:\n"
        "  - New regimen must include >=2 (preferably 3) fully active agents based on cumulative resistance.\n"
        "  - DTG and BIC retain activity against most first-gen INSTI mutations EXCEPT Q148H + >=2 secondary mutations.\n"
        "  - Darunavir/ritonavir (DRV/r): high barrier to resistance, useful in salvage.\n"
        "  - Doravirine (DOR): active against most NNRTI-resistant virus (except specific patterns).\n"
        "  - Do NOT simply add a single drug to a failing regimen — this creates sequential monotherapy and breeds resistance.\n\n"
        "SALVAGE (multi-class resistance):\n"
        "  - Fostemsavir (attachment inhibitor) — for highly treatment-experienced patients.\n"
        "  - Lenacapavir (capsid inhibitor) — every 6 months SC; approved for multi-class resistance.\n"
        "  - Ibalizumab (post-attachment inhibitor) — IV every 2 weeks.\n"
        "  - Refer to academic HIV center for complex salvage cases.\n\n"
        "INJECTABLE ART FAILURE (CAB+RPV LA):\n"
        "  - 1-2% incidence of virologic failure even with injection adherence.\n"
        "  - Risk factors: rilpivirine resistance at baseline, viral subtype A6, BMI >30.\n"
        "  - Failure causes 2-class resistance (NNRTI + INSTI) — significantly limits future options.\n"
        "  - Switch to oral DTG or BIC-based regimen + 2 active NRTIs guided by genotype.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the most likely cause (adherence is always #1).\n"
        "  2. Walk through the workup systematically.\n"
        "  3. State what resistance tests to order.\n"
        "  4. Give guidance on switch strategy based on available data.\n"
        "  5. Flag when to refer to a specialist center.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_resistance(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """Interpret HIV resistance mutations and guide regimen adjustment per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant interpreting HIV drug resistance mutations "
        "per IAS-USA 2024 guidelines and the Stanford HIV Drug Resistance Database.\n"
        + context_block +
        "INTEGRASE STRAND TRANSFER INHIBITOR (INSTI) RESISTANCE:\n"
        "  Major mutations and clinical impact:\n"
        "  - Y143R/C/H: raltegravir/elvitegravir RESISTANT. DTG and BIC remain SUSCEPTIBLE.\n"
        "  - N155H: raltegravir/elvitegravir RESISTANT. DTG usually susceptible (may need BID dosing). BIC susceptible.\n"
        "  - Q148H alone: raltegravir/elvitegravir RESISTANT. DTG usually susceptible (BID dosing). BIC usually susceptible.\n"
        "  - Q148H + G140S: raltegravir/elvitegravir RESISTANT. DTG reduced susceptibility (BID dosing required). BIC reduced susceptibility.\n"
        "  - Q148H + >=2 secondary (G140S + E138K etc.): all first-gen INSTIs RESISTANT. DTG MAY be resistant. BIC MAY be resistant. "
        "This is the most concerning pattern — consider salvage agents.\n"
        "  - G118R: reduced susceptibility to all INSTIs including DTG.\n"
        "  - R263K: DTG low-level resistance (still clinically active). Other INSTIs susceptible.\n"
        "  KEY PRINCIPLE: DTG and BIC have HIGH genetic barrier to resistance. Single INSTI mutations rarely compromise them. "
        "Accumulation of Q148 pathway + secondary mutations is the main threat. Always interpret the FULL mutation pattern.\n\n"
        "NRTI RESISTANCE:\n"
        "  - M184V/I: lamivudine and emtricitabine RESISTANT. HYPERSENSITIZES the virus to tenofovir (TDF/TAF). "
        "Clinical action: continue or switch to a TDF-containing backbone — the hypersensitization is clinically beneficial. "
        "Keeping FTC/3TC in the regimen despite M184V is acceptable — the mutation impairs viral fitness and maintains selective pressure. "
        "Do NOT recommend zidovudine (AZT) for M184V — AZT is obsolete in resource-rich settings. Use tenofovir.\n"
        "  - K65R: tenofovir (TDF and TAF) RESISTANT. "
        "Clinical action: use abacavir (HLA-B*5701 negative required) as the NRTI backbone. "
        "If abacavir is not viable, build the regimen around fully active agents from other classes (DTG/BIC + boosted DRV/r). "
        "Do NOT recommend AZT as a primary option for K65R in modern practice.\n"
        "  - TAMs (thymidine analog mutations: M41L, D67N, K70R, L210W, T215Y/F, K219Q/E): "
        "historically associated with AZT and d4T resistance (drugs no longer used). Higher TAM burden reduces tenofovir and abacavir activity. "
        ">=3 TAMs including M41L or L210W: tenofovir likely compromised.\n"
        "  - K65R + M184V: both TDF and FTC resistant. Use abacavir (if HLA-B*5701 negative) as the NRTI backbone, "
        "combined with fully active agents from other classes (DTG/BIC if no INSTI resistance, plus boosted DRV/r if needed). "
        "AZT is a LAST RESORT option only if abacavir is contraindicated AND no other NRTI is viable — explicitly state this is a last resort if mentioned.\n\n"
        "NNRTI RESISTANCE:\n"
        "  - K103N: efavirenz and nevirapine RESISTANT. Rilpivirine usually SUSCEPTIBLE. Doravirine SUSCEPTIBLE.\n"
        "  - Y181C: rilpivirine RESISTANT. Efavirenz may retain partial activity. Doravirine usually susceptible.\n"
        "  - E138K (alone): rilpivirine reduced susceptibility.\n"
        "  - E138K + M184I: combination confers rilpivirine resistance.\n"
        "  - Doravirine has the broadest NNRTI resistance profile — active against most single NNRTI mutations.\n\n"
        "PROTEASE INHIBITOR (PI) RESISTANCE:\n"
        "  - Darunavir has the HIGHEST barrier of all PIs. Requires >=3 DRV-associated mutations for clinically significant resistance.\n"
        "  - DRV-associated mutations: V11I, V32I, L33F, I47V, I50V, I54M/L, T74P, L76V, I84V, L89V.\n"
        "  - Even with some PI mutations, boosted DRV/r often retains activity.\n\n"
        "GENERAL PRINCIPLES:\n"
        "  - Resistance is CUMULATIVE — mutations detected in prior genotypes remain archived even if not on current test.\n"
        "  - Always review the complete resistance history, not just the most recent genotype.\n"
        "  - A new regimen after failure must include >=2 (preferably 3) fully active agents.\n"
        "  - For complex multi-class resistance: fostemsavir, lenacapavir, ibalizumab. Refer to academic HIV center.\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with the clinical significance of the mutations present — which drug classes are compromised.\n"
        "  2. For each mutation mentioned, state which drugs remain active and which are compromised.\n"
        "  3. Recommend a specific regimen based on the resistance profile.\n"
        "  4. If multi-class resistance is present, name salvage options and recommend specialist referral.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_hiv_switch(
    *,
    question: str,
    hiv_context: dict | None = None,
    patient_summary: str | None = None,
    fallback_message: str,
) -> Tuple[str, bool]:
    """ART switch/simplification guidance for virologically suppressed patients per IAS-USA 2024."""
    if not consult_narration_enabled():
        return fallback_message, False

    context_block = _build_hiv_context_block(hiv_context, patient_summary)

    prompt = (
        "You are an infectious diseases consultant advising on switching or simplifying ART "
        "in a virologically suppressed patient, per IAS-USA 2024 guidelines.\n"
        + context_block +
        "REASONS TO SWITCH (while suppressed):\n"
        "  - Simplification (reduce pill burden or dosing frequency).\n"
        "  - Adverse effects (renal toxicity from TDF, weight gain from INSTI+TAF, lipid changes, GI intolerance).\n"
        "  - Drug interactions (boosted PI interactions, rilpivirine+PPI contraindication).\n"
        "  - Pregnancy planning (switch to DTG-based, avoid cobicistat).\n"
        "  - Cost/insurance changes.\n"
        "  - Cardiovascular risk (switch off abacavir if CVD risk factors).\n\n"
        "SWITCH RULES — CRITICAL:\n"
        "  - ALWAYS review full resistance history before ANY switch.\n"
        "  - Do NOT switch to a regimen with agents to which archived resistance may exist.\n"
        "  - If switching away from TDF or TAF in HBV coinfected patient: MUST maintain HBV-active agent "
        "or risk severe/fatal HBV flare.\n"
        "  - Monitor more closely after switch: VL at 1 month, then every 3 months for 1 year.\n\n"
        "2-DRUG REGIMEN SWITCHES (for eligible suppressed patients):\n"
        "  - DTG + 3TC (Dovato): acceptable ONLY if no prior virologic failure, no HBV coinfection, "
        "no known resistance to either component, VL suppressed >=3-6 months.\n"
        "  - DTG + RPV (Juluca): available as coformulated single tablet. "
        "No HBV coinfection, no rilpivirine resistance, no PPI use.\n\n"
        "SWITCH FROM BOOSTED PI TO INSTI:\n"
        "  - Patients on boosted PI + 2 NRTIs can switch to DTG + TXF/XTC or BIC/TAF/FTC "
        "regardless of likely prior NRTI resistance, PROVIDED there is no INSTI resistance history.\n"
        "  - This switch is particularly beneficial for patients with: "
        "PI-related dyslipidemia, drug interactions with boosted PIs, GI intolerance, high pill burden.\n"
        "  - Do NOT switch from boosted PI to NNRTI or first-gen INSTI (raltegravir/elvitegravir) + 2 NRTIs "
        "in the presence of NRTI resistance — increased risk of failure and emergent NNRTI/INSTI resistance.\n"
        "  - Patients with NRTI resistance switching to DTG/BIC + dual NRTI: monitor more closely in first year.\n\n"
        "LONG-ACTING INJECTABLE SWITCH:\n"
        "  - Cabotegravir + rilpivirine (Cabenuva) IM every 1-2 months:\n"
        "    - Requires: VL suppressed, no NNRTI resistance, no prior rilpivirine failure.\n"
        "    - BMI >30: higher risk of virologic failure (1-2% even with adherence).\n"
        "    - Discuss 2-class resistance risk if failure occurs.\n"
        "    - Does NOT cover HBV — must continue HBV treatment separately if coinfected.\n"
        "    - More resource-intensive for clinics (scheduling, injection administration).\n"
        "  - Lenacapavir SC every 6 months:\n"
        "    - Currently approved only for treatment-experienced with multi-class resistance.\n"
        "    - Trials ongoing for first-line use (LEN/BIC combination).\n"
        "    - Weekly oral islatravir + lenacapavir combination in phase 2b (promising results).\n\n"
        "WEIGHT MANAGEMENT CONSIDERATIONS:\n"
        "  - INSTI + TAF associated with greatest weight gain.\n"
        "  - Do NOT switch regimen solely for weight gain — evidence of benefit is lacking.\n"
        "  - Lifestyle modifications (diet, exercise) are first-line.\n"
        "  - GLP-1 receptor agonists (semaglutide) effective for weight loss in PLWH.\n"
        "  - If switching for other reasons AND weight is a concern: consider TDF over TAF (if renal allows).\n\n"
        "CARDIOVASCULAR RISK:\n"
        "  - Abacavir: associated with increased cardiovascular events. "
        "Switch to non-abacavir regimen if CVD risk factors present.\n"
        "  - Pitavastatin recommended for primary CVD prevention in all PLWH aged 40-75 (REPRIEVE trial).\n\n"
        "RESPONSE FORMAT:\n"
        "  1. Open with whether the proposed switch is safe and recommended.\n"
        "  2. Address resistance history requirements.\n"
        "  3. Flag HBV considerations if relevant.\n"
        "  4. State the monitoring schedule after switch.\n"
        "  5. Name specific alternative regimens if the proposed switch is not ideal.\n"
        "Do not use markdown bullets, asterisks, headers, or arrow symbols. Plain text only.\n"
        "Keep the answer to 3 to 5 short paragraphs."
    )
    try:
        return _call_consult_model(
            prompt=prompt,
            payload={"question": question},
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False
