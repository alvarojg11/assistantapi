from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ..schemas import DoseIDAssistantAnalysis, ImmunoAnalyzeResponse, MechIDTextAnalyzeResponse, TextAnalyzeResponse
from .mechid_consult_examples import select_mechid_consult_examples
from .llm_text_parser import LLMParserError, _try_import_openai


class ConsultNarrationError(RuntimeError):
    pass


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


def narrate_probid_assistant_message(
    *,
    text_result: TextAnalyzeResponse,
    fallback_message: str,
    module_label: str,
) -> Tuple[str, bool]:
    if not consult_narration_enabled() or text_result.analysis is None:
        return fallback_message, False

    analysis = text_result.analysis
    payload = {
        "moduleLabel": module_label,
        "fallbackMessage": fallback_message,
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
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _call_consult_model(prompt=prompt, payload=payload), True
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

    payload = {
        "moduleLabel": module_label,
        "fallbackMessage": fallback_message,
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
        return _call_consult_model(prompt=prompt, payload=payload), True
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
    payload = {
        "fallbackMessage": fallback_message,
        "parsedRequest": mechid_result.parsed_request.model_dump(by_alias=True) if mechid_result.parsed_request else None,
        "analysis": mechid_result.analysis.model_dump(by_alias=True) if mechid_result.analysis else None,
        "provisionalAdvice": mechid_result.provisional_advice.model_dump(by_alias=True) if mechid_result.provisional_advice else None,
        "warnings": mechid_result.warnings,
        "examples": examples[:3],
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
        "When treatment options are available, lead with practical syndrome-specific options first, then say which option you would lean toward based on the submitted susceptibilities.\n"
        "If oral options are supported in the JSON, mention them in a clinically appropriate way. If the JSON implies IV-first treatment, keep that framing.\n"
        "If example outputs are provided, use them as style references only when they fit the same type of case. Do not copy unsupported claims.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _call_consult_model(prompt=prompt, payload=payload), True
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
    payload = {
        "fallbackMessage": fallback_message,
        "parsedRequest": mechid_result.parsed_request.model_dump(by_alias=True),
        "analysis": mechid_result.analysis.model_dump(by_alias=True) if mechid_result.analysis else None,
        "provisionalAdvice": mechid_result.provisional_advice.model_dump(by_alias=True) if mechid_result.provisional_advice else None,
        "warnings": mechid_result.warnings,
        "requiresConfirmation": mechid_result.requires_confirmation,
        "examples": examples[:3],
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
        return _call_consult_model(prompt=prompt, payload=payload), True
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

    payload = {
        "fallbackMessage": fallback_message,
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
        "If there are no follow-up questions, summarize the current rule-backed checklist in a practical consultant tone.\n"
        "Preserve uncertainty exactly. If serologies, geography, or neutropenia details are missing, say that plainly and only based on the JSON.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _call_consult_model(prompt=prompt, payload=payload), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False


def narrate_doseid_assistant_message(
    *,
    doseid_result: DoseIDAssistantAnalysis,
    fallback_message: str,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False

    payload = {
        "fallbackMessage": fallback_message,
        "doseidAnalysis": doseid_result.model_dump(by_alias=True),
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic DoseID assistant message into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not change medications, indications, renal buckets, dose amounts, intervals, or monitoring notes.\n"
        "Do not invent any regimen or missing input.\n"
        "If followUpQuestions are present, ask only the next missing question already present in the JSON and keep the phrasing simple.\n"
        "If recommendations are present, summarize them clearly without changing the numbers.\n"
        "If warnings are present, preserve their meaning without adding new cautions.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 2 short paragraphs. Plain text only."
    )
    try:
        return _call_consult_model(prompt=prompt, payload=payload), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False
