from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai
from .mechid_image_parser import _parse_data_url


# ---------------------------------------------------------------------------
# Output schema (as typed dict — validated manually, no Pydantic to stay lean)
# ---------------------------------------------------------------------------
# {
#   "institution": str | null,
#   "year": str | null,
#   "organisms": {
#       "E. coli": {"ciprofloxacin": 65, "amoxicillin-clavulanate": 82, ...},
#       ...
#   },
#   "total_isolates": {"E. coli": 245, ...},   # optional — present if shown
#   "notes": ["..."],
#   "confidence": "low" | "medium" | "high",
#   "ambiguities": ["..."]
# }


_ANTIBIOGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "institution": {"type": ["string", "null"]},
        "year": {"type": ["string", "null"]},
        "organisms": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
        "total_isolates": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "notes": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["institution", "year", "organisms", "confidence"],
}


def _build_antibiogram_instructions() -> str:
    return (
        "You extract a cumulative institutional antibiogram table into structured JSON.\n"
        "Return JSON only — no markdown, no explanation.\n\n"
        "AN ANTIBIOGRAM is a summary table showing the percentage of isolates susceptible to each antibiotic "
        "across multiple organisms, typically compiled annually by a hospital microbiology laboratory.\n\n"
        "EXTRACTION RULES:\n"
        "1. The table has organisms as rows and antibiotics as columns (or vice versa). "
        "Extract each organism × antibiotic combination as a percentage susceptible (0-100 integer or float).\n"
        "2. If the table shows % resistant or % intermediate instead of % susceptible, convert: "
        "% susceptible = 100 - % resistant (ignoring intermediate unless it is the only value). "
        "If the table clearly labels values as % susceptible already, use them directly.\n"
        "3. Organism names: use standard clinical names, e.g. 'E. coli', 'K. pneumoniae', "
        "'S. aureus (MSSA)', 'S. aureus (MRSA)', 'P. aeruginosa', 'Enterococcus faecalis', "
        "'Enterococcus faecium', 'S. pneumoniae', 'K. oxytoca', 'Proteus mirabilis', "
        "'Acinetobacter baumannii', 'Serratia marcescens', 'Enterobacter cloacae'. "
        "If the table distinguishes MSSA and MRSA separately, treat them as separate entries.\n"
        "4. Antibiotic names: use full standard names, e.g. 'ciprofloxacin', 'levofloxacin', "
        "'piperacillin-tazobactam', 'ceftriaxone', 'cefazolin', 'ceftazidime', 'cefepime', "
        "'meropenem', 'ertapenem', 'imipenem', 'ampicillin', 'amoxicillin-clavulanate', "
        "'trimethoprim-sulfamethoxazole', 'doxycycline', 'vancomycin', 'linezolid', "
        "'daptomycin', 'clindamycin', 'azithromycin', 'nitrofurantoin', 'fosfomycin'.\n"
        "5. If a cell shows 'NA', '-', or is blank, omit that organism × antibiotic combination.\n"
        "6. If a cell shows a range (e.g. '78-85'), use the midpoint.\n"
        "7. Extract 'institution' (hospital or lab name) and 'year' if visible anywhere in the image. "
        "Set to null if not visible.\n"
        "8. Extract 'total_isolates' per organism if the table includes an isolate count (N=). "
        "Leave as empty object {} if not shown.\n"
        "9. Set confidence to 'high' if the table is clearly readable; 'medium' if some cells are blurry or ambiguous; "
        "'low' if less than half the table is legible or the image does not appear to be a standard antibiogram.\n"
        "10. List any parsing difficulties or ambiguous cells in 'ambiguities'.\n"
        "11. Put any useful footnote text (e.g. 'results based on urinary isolates only') in 'notes'.\n\n"
        "Output schema:\n"
        + json.dumps(_ANTIBIOGRAM_SCHEMA, ensure_ascii=True)
    )


def _validate_antibiogram_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise the raw LLM output. Returns cleaned dict or raises LLMParserError."""
    if not isinstance(payload, dict):
        raise LLMParserError("Antibiogram extraction returned non-dict JSON.")
    organisms = payload.get("organisms")
    if not isinstance(organisms, dict):
        raise LLMParserError("Antibiogram extraction missing 'organisms' object.")
    if not organisms:
        raise LLMParserError(
            "No organism data could be extracted from the image. "
            "Please upload a clearer image of the antibiogram table."
        )
    # Normalise percentages — clamp to 0-100, round to 1 decimal
    normalised_organisms: Dict[str, Dict[str, float]] = {}
    for org, abx_map in organisms.items():
        if not isinstance(abx_map, dict):
            continue
        clean_abx: Dict[str, float] = {}
        for abx, pct in abx_map.items():
            try:
                val = float(pct)
                clean_abx[abx] = round(max(0.0, min(100.0, val)), 1)
            except (TypeError, ValueError):
                pass
        if clean_abx:
            normalised_organisms[str(org)] = clean_abx
    payload["organisms"] = normalised_organisms
    payload.setdefault("institution", None)
    payload.setdefault("year", None)
    payload.setdefault("total_isolates", {})
    payload.setdefault("notes", [])
    payload.setdefault("confidence", "medium")
    payload.setdefault("ambiguities", [])
    return payload


def parse_antibiogram_image_with_openai(
    *,
    image_data_url: str,
    filename: str | None = None,
    parser_model: str | None = None,
) -> Dict[str, Any]:
    """
    Extract a cumulative institutional antibiogram from an image.

    Returns a dict with keys: institution, year, organisms, total_isolates, notes,
    confidence, ambiguities, parser.
    Raises LLMParserError on any failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMParserError("OPENAI_API_KEY is not set.")

    media_type, _ = _parse_data_url(image_data_url)

    OpenAI = _try_import_openai()
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    model = parser_model or os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    instructions = _build_antibiogram_instructions()

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
                                "Extract the full antibiogram table from this image into JSON. "
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
        raise LLMParserError(f"OpenAI antibiogram extraction request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMParserError("OpenAI antibiogram extraction response did not include text output.")

    payload = _extract_json(output_text)
    validated = _validate_antibiogram_payload(payload)
    validated["parser"] = f"openai-{model}-antibiogram-v1"
    return validated


def antibiogram_to_prompt_block(antibiogram: Dict[str, Any]) -> str:
    """
    Format a parsed antibiogram dict into a compact plain-text block
    suitable for injection into a narrator prompt.
    """
    lines: List[str] = []
    institution = antibiogram.get("institution") or "institutional"
    year = antibiogram.get("year")
    header = f"LOCAL ANTIBIOGRAM ({institution}{', ' + year if year else ''}):"
    lines.append(header)
    lines.append(
        "Values are % susceptible. Flag any agent <80% as suboptimal for empiric use "
        "and recommend alternatives with higher local susceptibility."
    )
    organisms: Dict[str, Dict[str, float]] = antibiogram.get("organisms", {})
    total_isolates: Dict[str, Any] = antibiogram.get("total_isolates", {})
    for org, abx_map in organisms.items():
        n = total_isolates.get(org)
        n_str = f" (n={int(n)})" if n else ""
        pairs = ", ".join(f"{abx} {int(pct)}%" for abx, pct in sorted(abx_map.items()))
        lines.append(f"  {org}{n_str}: {pairs}")
    notes = antibiogram.get("notes", [])
    if notes:
        lines.append("Notes: " + "; ".join(notes))
    return "\n".join(lines)
