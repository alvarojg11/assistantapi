from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai
from .mechid_engine import canonical_antibiotic_aliases, list_mechid_organisms, normalize_organism
from .mechid_text_parser import parse_mechid_text


ASTResult = Literal["Susceptible", "Intermediate", "Resistant"]


class MechIDLLMExtractionPayload(BaseModel):
    organism: str | None = None
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: Dict[str, Any] = Field(
        default_factory=lambda: {"syndrome": "Not specified", "severity": "Not specified", "oralPreference": False},
        alias="txContext",
    )
    confidence: Literal["low", "medium", "high"] = "medium"
    ambiguities: List[str] = Field(default_factory=list)
    ignored_clues: List[str] = Field(default_factory=list, alias="ignoredClues")

    model_config = {"populate_by_name": True}


def _build_mechid_catalog() -> Dict[str, Any]:
    organisms: List[Dict[str, Any]] = []
    for organism in list_mechid_organisms():
        panel_aliases = canonical_antibiotic_aliases(organism)
        organisms.append(
            {
                "organism": organism,
                "panel": sorted(set(panel_aliases.values())),
            }
        )
    return {"organisms": organisms}


def _build_output_schema() -> Dict[str, Any]:
    schema = MechIDLLMExtractionPayload.model_json_schema(by_alias=True)
    schema.pop("title", None)
    return schema


def _build_instructions(catalog: Dict[str, Any]) -> str:
    output_schema = _build_output_schema()
    return (
        "You extract lay or clinical microbiology questions into a MechID input JSON.\n"
        "Return JSON only.\n"
        "Your job is to identify the organism, named susceptibility results, and basic treatment context.\n"
        "Use only organism names and antibiotics present in the provided catalog.\n"
        "Do not infer susceptibility results that were not actually stated.\n"
        "Normalize susceptibility values to exactly one of: Susceptible, Intermediate, Resistant.\n"
        "If the organism is unclear, set organism to null instead of guessing.\n"
        "If syndrome or severity is not clearly stated, use 'Not specified'.\n"
        "Set oralPreference to true only if the user is explicitly asking for oral therapy, oral step-down, or PO options.\n"
        "Preserve as many explicitly stated AST calls as possible.\n"
        "Set confidence to low when the organism or the AST pattern is ambiguous.\n"
        "Use ambiguities for details that could support multiple interpretations.\n"
        "Use ignoredClues for clinically meaningful facts that do not map into this schema.\n"
        "Output JSON must validate against this schema:\n"
        + json.dumps(output_schema, ensure_ascii=True)
        + "\n\n"
        "CATALOG:\n"
        + json.dumps(catalog, ensure_ascii=True)
    )


def _canonicalize_extraction(payload: MechIDLLMExtractionPayload) -> Dict[str, object]:
    warnings: List[str] = []
    organism = payload.organism
    susceptibility_results: Dict[str, ASTResult] = {}
    tx_context = {
        "syndrome": payload.tx_context.get("syndrome", "Not specified") or "Not specified",
        "severity": payload.tx_context.get("severity", "Not specified") or "Not specified",
        "oralPreference": bool(payload.tx_context.get("oralPreference", False)),
    }

    if organism:
        try:
            organism = normalize_organism(organism)
        except Exception as exc:
            warnings.append(str(exc))
            organism = None

    if organism:
        aliases = canonical_antibiotic_aliases(organism)
        for raw_name, raw_state in payload.susceptibility_results.items():
            key = raw_name.strip().lower()
            antibiotic = aliases.get(key)
            if antibiotic is None:
                for alias, canonical in aliases.items():
                    if key == alias or key in alias or alias in key:
                        antibiotic = canonical
                        break
            if antibiotic is None:
                warnings.append(f"Ignored unsupported antibiotic for {organism}: {raw_name}")
                continue
            susceptibility_results[antibiotic] = raw_state

    requires_confirmation = payload.confidence == "low" or organism is None or not susceptibility_results
    if payload.ambiguities:
        warnings.append("LLM ambiguities: " + "; ".join(payload.ambiguities))
        requires_confirmation = True
    if payload.ignored_clues:
        warnings.append("LLM ignored clues: " + "; ".join(payload.ignored_clues))

    return {
        "organism": organism,
        "susceptibilityResults": susceptibility_results,
        "txContext": tx_context,
        "warnings": warnings,
        "requiresConfirmation": requires_confirmation,
    }


def parse_mechid_text_with_openai(
    *,
    text: str,
    parser_model: str | None = None,
) -> Dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMParserError("OPENAI_API_KEY is not set.")

    OpenAI = _try_import_openai()
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    model = parser_model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    catalog = _build_mechid_catalog()
    instructions = _build_instructions(catalog)

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps({"text": text}, ensure_ascii=True),
        )
    except Exception as exc:  # pragma: no cover
        raise LLMParserError(f"OpenAI request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMParserError("OpenAI response did not include text output.")

    payload = _extract_json(output_text)
    try:
        extracted = MechIDLLMExtractionPayload.model_validate(payload)
    except Exception as exc:
        raise LLMParserError(f"LLM JSON did not match MechID extraction schema: {exc}") from exc

    normalized = _canonicalize_extraction(extracted)
    rule_fallback = parse_mechid_text(text)

    if normalized["organism"] is None and rule_fallback["organism"] is not None:
        normalized["organism"] = rule_fallback["organism"]
        normalized["warnings"].append("Filled organism from rule parser.")
    if not normalized["susceptibilityResults"] and rule_fallback["susceptibilityResults"]:
        normalized["susceptibilityResults"] = rule_fallback["susceptibilityResults"]
        normalized["warnings"].append("Filled susceptibility results from rule parser.")
    if normalized["txContext"].get("syndrome") == "Not specified" and rule_fallback["txContext"].get("syndrome") != "Not specified":
        normalized["txContext"]["syndrome"] = rule_fallback["txContext"]["syndrome"]
    if normalized["txContext"].get("severity") == "Not specified" and rule_fallback["txContext"].get("severity") != "Not specified":
        normalized["txContext"]["severity"] = rule_fallback["txContext"]["severity"]
    if not normalized["txContext"].get("oralPreference") and rule_fallback["txContext"].get("oralPreference"):
        normalized["txContext"]["oralPreference"] = True

    normalized["requiresConfirmation"] = bool(
        normalized["requiresConfirmation"]
        or rule_fallback["requiresConfirmation"]
        or normalized["organism"] is None
        or not normalized["susceptibilityResults"]
    )
    normalized["parser"] = f"openai-{model}-mechid-v1"
    return normalized
