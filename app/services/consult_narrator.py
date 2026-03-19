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
    try:
        response = client.responses.create(
            model=chosen_model,
            instructions=prompt,
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
        "Structure the answer as follows: open with the clinical action recommendation and your overall interpretation in one direct sentence, then explain the probability and key clinical drivers, then mention what would change your assessment.\n"
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
        "Exact POET regimens: Streptococcus / Enterococcus faecalis — amoxicillin 2g four times daily + moxifloxacin 400mg once daily (combination); MSSA — dicloxacillin 1g four times daily (or flucloxacillin 1g four times daily); MRSA/CoNS — linezolid 600mg twice daily + rifampin 300mg twice daily. This is not yet universal practice — discuss with senior ID.\n\n"
        "SITUATIONS WHERE IV MUST BE MAINTAINED (oral NOT appropriate):\n"
        "  - S. aureus bacteraemia: must complete full IV course — do not switch to oral. 14 days minimum (uncomplicated), 4-6 weeks (complicated/endocarditis/osteomyelitis).\n"
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

    prompt = (
        "You are an experienced infectious diseases consultant answering a clinician's question about empiric antimicrobial therapy.\n"
        "Empiric therapy means treatment started before culture results are available, based on syndrome and epidemiology.\n"
        + context_block
        + antibiogram_section +
        "Give a concise, actionable recommendation:\n"
        "  1. State the preferred empiric regimen for the syndrome, including drug name and key dose range.\n"
        "  2. Name the most important pathogens being covered.\n"
        "  3. Address when to add MRSA coverage, when to consider anti-pseudomonal coverage, and when to add anaerobic coverage — only if relevant.\n"
        "  4. Note any patient-specific adjustments (renal, allergy) if context is available.\n"
        "  5. Specify what to culture before starting and when to narrow based on results.\n"
        "If the syndrome is unclear, name the most likely diagnosis and give the regimen for that working diagnosis.\n"
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
        "Exact POET regimens: Strep/E. faecalis — amoxicillin 2g four times daily + moxifloxacin 400mg once daily; MSSA — dicloxacillin 1g four times daily (or flucloxacillin 1g four times daily); MRSA/CoNS — linezolid 600mg twice daily + rifampin 300mg twice daily. Not yet universal practice — discuss with senior ID.\n"
        "  4. Syndromes where oral is NOT appropriate: S. aureus bacteraemia (must complete IV), bacterial meningitis, high-risk febrile neutropenia, prosthetic valve endocarditis.\n"
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

    prompt = (
        "You are an infectious diseases consultant answering a clinician's question about antibiotic treatment duration.\n"
        + context_block +
        "Give a clear, guideline-concordant duration recommendation:\n"
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
