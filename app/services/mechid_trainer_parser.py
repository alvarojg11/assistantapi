from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from ..schemas import MechIDTextAnalyzeResponse, MechIDTrainerEvalCase, MechIDTrainerEvalPatch
from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai


class MechIDTrainerParseError(RuntimeError):
    pass


def mechid_trainer_parser_enabled() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def parse_mechid_trainer_correction(
    *,
    raw_text: str,
    correction_text: str,
    mechid_result: MechIDTextAnalyzeResponse,
    base_draft: MechIDTrainerEvalCase,
    parser_model: str | None = None,
) -> Tuple[MechIDTrainerEvalPatch | None, str | None]:
    correction_text = (correction_text or "").strip()
    if not correction_text:
        return None, None

    if not mechid_trainer_parser_enabled():
        return None, "OPENAI_API_KEY is not set, so plain-English corrections could not be translated automatically."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not set, so plain-English corrections could not be translated automatically."

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
        "You convert a user's plain-English correction note into a sparse JSON patch for a MechID evaluation case.\n"
        "Return JSON only. No markdown, no prose.\n"
        "Only include fields that need to change from the base draft.\n"
        "Use the exact schema provided.\n"
        "If the user corrects organism, syndrome, severity, focus detail, oral preference, susceptibility calls, mechanisms, therapy notes, or provisional behavior, encode that.\n"
        "When the user says the assistant should mention certain ideas, use assistantReviewContains or assistantFinalContains with short substring expectations.\n"
        "Do not invent clinical details that are not stated in the correction note or clearly supported by the raw case text.\n"
        "Prefer sparse, minimal patches over restating the full case.\n"
        "Output must validate against this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=True)
    )
    payload = {
        "rawText": raw_text,
        "correctionText": correction_text,
        "baseDraft": base_draft.model_dump(by_alias=True),
        "mechidResult": mechid_result.model_dump(by_alias=True),
    }
    try:
        response = client.responses.create(
            model=parser_model or os.getenv("OPENAI_CONSULT_MODEL", "gpt-4.1-mini"),
            instructions=prompt,
            input=json.dumps(payload, ensure_ascii=True),
        )
    except Exception as exc:  # pragma: no cover
        raise MechIDTrainerParseError(f"Trainer correction request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text or not str(output_text).strip():
        raise MechIDTrainerParseError("Trainer correction response was empty.")

    try:
        raw_patch = _extract_json(str(output_text))
        patch = MechIDTrainerEvalPatch.model_validate(raw_patch)
    except (LLMParserError, Exception) as exc:
        raise MechIDTrainerParseError(f"Trainer correction response was invalid JSON: {exc}") from exc
    return patch, None
