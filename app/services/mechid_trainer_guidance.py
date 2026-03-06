from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from ..schemas import MechIDTextAnalyzeResponse, MechIDTrainerEvalPatch
from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai


class MechIDTrainerGuidanceError(RuntimeError):
    pass


def mechid_trainer_guidance_enabled() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def generate_mechid_trainer_targets(
    *,
    raw_text: str,
    recommendation_text: str,
    mechid_result: MechIDTextAnalyzeResponse,
    parser_model: str | None = None,
) -> Tuple[MechIDTrainerEvalPatch | None, str | None]:
    recommendation_text = (recommendation_text or "").strip()
    if not recommendation_text:
        return None, None

    if not mechid_trainer_guidance_enabled():
        return None, "OPENAI_API_KEY is not set, so recommendation guidance could not be translated automatically."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not set, so recommendation guidance could not be translated automatically."

    try:
        OpenAI = _try_import_openai()
    except LLMParserError as exc:
        return None, str(exc)

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    schema = MechIDTrainerEvalPatch.model_json_schema(by_alias=True)
    schema.pop("title", None)
    prompt = (
        "You convert a user's plain-English recommendation note into assistant targets for a MechID training example.\n"
        "Return JSON only. No markdown, no prose.\n"
        "Only include assistantGuidance, assistantReviewTarget, and assistantFinalTarget unless the user clearly asks to change something else.\n"
        "assistantReviewTarget should be extraction-focused and should not read like a final consultant impression.\n"
        "assistantFinalTarget should be clinician-facing and can sound like the desired final assistant answer.\n"
        "Do not invent organisms, susceptibilities, mechanisms, or treatment claims beyond the provided JSON.\n"
        "Output must validate against this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=True)
    )
    payload = {
        "rawText": raw_text,
        "recommendationText": recommendation_text,
        "mechidResult": mechid_result.model_dump(by_alias=True),
    }
    try:
        response = client.responses.create(
            model=parser_model or os.getenv("OPENAI_CONSULT_MODEL", "gpt-4.1-mini"),
            instructions=prompt,
            input=json.dumps(payload, ensure_ascii=True),
        )
    except Exception as exc:  # pragma: no cover
        raise MechIDTrainerGuidanceError(f"Trainer recommendation request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text or not str(output_text).strip():
        raise MechIDTrainerGuidanceError("Trainer recommendation response was empty.")

    try:
        raw_patch = _extract_json(str(output_text))
        patch = MechIDTrainerEvalPatch.model_validate(raw_patch)
    except (LLMParserError, Exception) as exc:
        raise MechIDTrainerGuidanceError(f"Trainer recommendation response was invalid JSON: {exc}") from exc
    return patch, None
