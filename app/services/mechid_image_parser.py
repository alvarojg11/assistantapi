from __future__ import annotations

import base64
import json
import os
import re
from typing import Dict, Tuple

from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai
from .mechid_llm_parser import MechIDLLMExtractionPayload, _build_mechid_catalog, _canonicalize_extraction


def _parse_data_url(data_url: str) -> Tuple[str, bytes]:
    match = re.match(r"^data:(?P<media>[-\w.+/]+);base64,(?P<data>.+)$", data_url.strip(), flags=re.DOTALL)
    if not match:
        raise LLMParserError("Image payload must be a valid base64 data URL.")
    media_type = match.group("media")
    raw_data = match.group("data")
    try:
        decoded = base64.b64decode(raw_data, validate=True)
    except Exception as exc:
        raise LLMParserError(f"Could not decode image data: {exc}") from exc
    if not decoded:
        raise LLMParserError("Uploaded image was empty.")
    return media_type, decoded


def _build_image_instructions(catalog: Dict[str, object]) -> str:
    schema = MechIDLLMExtractionPayload.model_json_schema(by_alias=True)
    schema.pop("title", None)
    return (
        "You extract a microbiology isolate susceptibility report image into a MechID input JSON.\n"
        "Return JSON only.\n"
        "This is usually a lab screenshot, report image, or scanned antibiogram for a single patient isolate.\n"
        "Your job is to read the visible organism name, susceptibility results, and any clearly stated treatment context.\n"
        "If the image appears to be a cumulative hospital antibiogram or a table not tied to one isolate, set organism to null, "
        "leave susceptibilityResults empty, set confidence to low, and explain that ambiguity in ambiguities.\n"
        "If multiple organisms are explicitly listed for the same patient, set organism to null and list them in mentionedOrganisms.\n"
        "Use resistancePhenotypes for explicit labels like MRSA, MSSA, VRE, ESBL, CRE, KPC, NDM, VIM, IMP, or OXA-48-like.\n"
        "Normalize susceptibility values to exactly one of: Susceptible, Intermediate, Resistant.\n"
        "Map shorthand such as S, I, R only when the image clearly supports that reading.\n"
        "If a value is blurry or ambiguous, omit it and mention the ambiguity instead of guessing.\n"
        "Use only organism names and antibiotics present in the provided catalog.\n"
        "If syndrome, site detail, or severity appears in the image, capture it. Otherwise use 'Not specified'.\n"
        "Set oralPreference to true only if the image or surrounding context explicitly asks for oral therapy.\n"
        "Use carbapenemaseResult and carbapenemaseClass only when clearly shown in the image.\n"
        "Recognize common carbapenemase spellings and labels such as KPC, KPC producer, blaKPC, NDM, NDM positive, blaNDM, VIM, blaVIM, IMP, IMP type, blaIMP, OXA48, OXA-48, OXA-48-like, OXA48-like, CP-CRE, MBL, and metallo-beta-lactamase.\n"
        "Set confidence to low when the image is blurry, partial, or difficult to interpret.\n"
        "Use ignoredClues for useful visible details that do not map cleanly into this schema.\n"
        "Output JSON must validate against this schema:\n"
        + json.dumps(schema, ensure_ascii=True)
        + "\n\nCATALOG:\n"
        + json.dumps(catalog, ensure_ascii=True)
    )


def parse_mechid_image_with_openai(
    *,
    image_data_url: str,
    filename: str | None = None,
    parser_model: str | None = None,
) -> Dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMParserError("OPENAI_API_KEY is not set.")

    media_type, _ = _parse_data_url(image_data_url)

    OpenAI = _try_import_openai()
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    model = parser_model or os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    catalog = _build_mechid_catalog()
    instructions = _build_image_instructions(catalog)

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Extract the organism, AST calls, carbapenemase testing, and any visible context from this image. "
                                f"Filename: {filename or 'unknown'}; media type: {media_type}."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # pragma: no cover
        raise LLMParserError(f"OpenAI image extraction request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMParserError("OpenAI image extraction response did not include text output.")

    payload = _extract_json(output_text)
    try:
        extracted = MechIDLLMExtractionPayload.model_validate(payload)
    except Exception as exc:
        raise LLMParserError(f"LLM image JSON did not match MechID extraction schema: {exc}") from exc

    normalized = _canonicalize_extraction(extracted)
    normalized["parser"] = f"openai-{model}-mechid-image-v1"
    return normalized
