from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from ..schemas import MechIDTextAnalyzeResponse, TextAnalyzeResponse
from .llm_text_parser import LLMParserError, _try_import_openai


class ConsultNarrationError(RuntimeError):
    pass


def consult_narration_enabled() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


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
    return str(output_text).strip()


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


def narrate_mechid_assistant_message(
    *,
    mechid_result: MechIDTextAnalyzeResponse,
    fallback_message: str,
) -> Tuple[str, bool]:
    if not consult_narration_enabled():
        return fallback_message, False
    if mechid_result.analysis is None and mechid_result.provisional_advice is None:
        return fallback_message, False

    payload = {
        "fallbackMessage": fallback_message,
        "parsedRequest": mechid_result.parsed_request.model_dump(by_alias=True) if mechid_result.parsed_request else None,
        "analysis": mechid_result.analysis.model_dump(by_alias=True) if mechid_result.analysis else None,
        "provisionalAdvice": mechid_result.provisional_advice.model_dump(by_alias=True) if mechid_result.provisional_advice else None,
        "warnings": mechid_result.warnings,
    }
    prompt = (
        "You are an infectious diseases consultant rewriting a deterministic MechID result into a concise clinician-facing answer.\n"
        "The JSON input is the full source of truth. Do not contradict or override the listed mechanisms, therapy notes, cautions, provisional advice, or extracted AST.\n"
        "Do not invent organisms, susceptibilities, mechanisms, or treatment claims.\n"
        "If the deterministic output says more data are needed, state exactly what is needed and do not pretend certainty.\n"
        "Keep the tone conversational but clinical. Sound like an ID consultant, not a rules engine.\n"
        "If oral options are supported in the JSON, mention them in a clinically appropriate way. If the JSON implies IV-first treatment, keep that framing.\n"
        "Do not use markdown bullets, asterisks, or arrow symbols.\n"
        "Prefer 1 to 3 short paragraphs. Plain text only."
    )
    try:
        return _call_consult_model(prompt=prompt, payload=payload), True
    except (ConsultNarrationError, LLMParserError):
        return fallback_message, False
