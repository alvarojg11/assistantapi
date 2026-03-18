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
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic ProbID engine result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not change any numeric values, thresholds, recommendation categories, or next steps.\n"
        "Do not invent findings, probabilities, tests, or treatments.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never introduce requests for MICs, susceptibility details, repeat cultures, or other missing inputs unless they are explicitly present in the JSON.\n"
        "If the fallbackMessage already says exactly what is needed, keep that meaning and do not add anything new.\n"
        "If data are missing or uncertain, say that plainly and only based on the provided JSON.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a calculator.\n"
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
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic MechID result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not contradict or override the listed mechanisms, therapy notes, cautions, provisional advice, or extracted AST.\n"
        "Do not invent organisms, susceptibilities, mechanisms, or treatment claims.\n"
        "Do not ask for additional data unless that request already exists in the JSON input.\n"
        "Never ask for MICs, additional susceptibility testing, repeat cultures, or source details unless those exact needs are already stated in the JSON.\n"
        "If the fallbackMessage already contains the needed uncertainty or next step, keep that meaning and do not expand it.\n"
        "If the deterministic output says more data are needed, state exactly what is needed and do not pretend certainty.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
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
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_mechid_review_message(
    *,
    mechid_result: MechIDTextAnalyzeResponse,
    fallback_message: str,
    transient_examples: List[Dict[str, str]] | None = None,
) -> Tuple[str, bool]:
    if not consult_narration_enabled() or mechid_result.parsed_request is None:
        return fallback_message, False

    examples = select_mechid_consult_examples(result=mechid_result, kind="review")
    if transient_examples:
        examples = [*transient_examples, *examples]
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
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    deterministic_payload = {
        "doseidAnalysis": doseid_result.model_dump(by_alias=True),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic DoseID assistant message into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not change medications, indications, renal buckets, dose amounts, intervals, or monitoring notes.\n"
        "Do not invent any regimen or missing input.\n"
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
        ), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_allergyid_assistant_message(
    *,
    allergy_result: AntibioticAllergyAnalyzeResponse,
    fallback_message: str,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    deterministic_payload = {
        "allergyAnalysis": allergy_result.model_dump(by_alias=True),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic antibiotic-allergy compatibility result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not invent antibiotics, reaction phenotypes, cross-reactivity claims, or safety conclusions.\n"
        "Do not make a drug sound safe if the JSON says avoid or caution.\n"
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
