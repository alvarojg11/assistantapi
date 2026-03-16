from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai
from .mechid_engine import canonical_antibiotic_aliases, list_mechid_organisms, normalize_organism, resolve_antibiotic_name
from .mechid_text_parser import infer_phenotype_defaults, parse_mechid_text


ASTResult = Literal["Susceptible", "Intermediate", "Resistant"]


class MechIDLLMExtractionPayload(BaseModel):
    organism: str | None = None
    mentioned_organisms: List[str] = Field(default_factory=list, alias="mentionedOrganisms")
    resistance_phenotypes: List[str] = Field(default_factory=list, alias="resistancePhenotypes")
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: Dict[str, Any] = Field(
        default_factory=lambda: {
            "syndrome": "Not specified",
            "severity": "Not specified",
            "focusDetail": "Not specified",
            "oralPreference": False,
            "carbapenemaseResult": "Not specified",
            "carbapenemaseClass": "Not specified",
        },
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
        "The case may involve gram-negatives, staphylococci, enterococci, streptococci, anaerobes, or mycobacteria.\n"
        "If multiple organisms are explicitly mentioned, set organism to null and list them in mentionedOrganisms.\n"
        "Use resistancePhenotypes for explicit labels such as MRSA, MSSA, MR-CoNS, VRE, VRSA, penicillin-resistant pneumococcus, ESBL, CRE, or named carbapenemases.\n"
        "Use only organism names and antibiotics present in the provided catalog.\n"
        "Do not infer susceptibility results that were not actually stated.\n"
        "Normalize susceptibility values to exactly one of: Susceptible, Intermediate, Resistant.\n"
        "If the organism is unclear, set organism to null instead of guessing.\n"
        "If syndrome, site detail, or severity is not clearly stated, use 'Not specified'.\n"
        "Set oralPreference to true only if the user is explicitly asking for oral therapy, oral step-down, or PO options.\n"
        "Use carbapenemaseResult when the text explicitly says carbapenemase testing is positive, negative, pending, or not tested.\n"
        "Use carbapenemaseClass only when the text explicitly names a class such as KPC, OXA-48-like, NDM, VIM, or IMP.\n"
        "Normalize common variants such as OXA48, OXA 48, OXA-48, OXA-48-like, KPC producer, NDM positive, VIM positive, IMP positive, blaKPC, blaNDM, blaVIM, blaIMP, and MBL/metallo-beta-lactamase.\n"
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


def _canonicalize_carbapenemase_result(raw_value: object) -> str:
    text = str(raw_value or "").strip().lower()
    if not text or text == "not specified":
        return "Not specified"
    if any(token in text for token in ("positive", "detected", "present", "producer", "producing")):
        return "Positive"
    if any(token in text for token in ("negative", "not detected")):
        return "Negative"
    if any(token in text for token in ("pending", "not tested", "not done")):
        return "Not tested / pending"
    return "Not specified"


def _canonicalize_carbapenemase_class(raw_value: object) -> str:
    text = str(raw_value or "").strip().lower()
    if not text or text == "not specified":
        return "Not specified"
    normalized = text.replace("_", " ").replace("/", " ").replace(".", " ")
    normalized = " ".join(normalized.split())
    if any(token in normalized for token in ("blakpc", "bla kpc")):
        return "KPC"
    if "kpc" in normalized:
        return "KPC"
    if any(token in normalized for token in ("bla ndm", "blandm")):
        return "NDM"
    if "ndm" in normalized:
        return "NDM"
    if any(token in normalized for token in ("bla vim", "blavim")):
        return "VIM"
    if "vim" in normalized:
        return "VIM"
    if any(token in normalized for token in ("bla imp", "blaimp")):
        return "IMP"
    if re.search(r"\bimp(?:[- ]type)?\b", normalized):
        return "IMP"
    if re.search(r"\boxa(?:[- ]?48(?:[- ]?like)?)\b", normalized) or "oxa48" in normalized:
        return "OXA-48-like"
    if any(token in normalized for token in ("mbl", "metallo beta lactamase", "metallo-beta-lactamase")):
        return "Other / Unknown"
    return "Not specified"


def _embedded_carbapenemase_row(raw_name: str, raw_state: ASTResult) -> tuple[str, str] | None:
    row = str(raw_name or "").strip()
    if not row:
        return None
    inferred_class = _canonicalize_carbapenemase_class(row)
    if inferred_class != "Not specified":
        return "Positive", inferred_class
    lowered = row.lower()
    if any(token in lowered for token in ("carbapenemase", "cp-cre", "cp cre")):
        return "Positive", "Not specified"
    return None


def _canonicalize_extraction(payload: MechIDLLMExtractionPayload) -> Dict[str, object]:
    warnings: List[str] = []
    organism = payload.organism
    mentioned_organisms = [item for item in payload.mentioned_organisms if item]
    resistance_phenotypes: List[str] = []
    for item in payload.resistance_phenotypes:
        if not item:
            continue
        rule_labels = parse_mechid_text(str(item)).get("resistancePhenotypes", [])
        if rule_labels:
            resistance_phenotypes.extend(rule_labels)
        else:
            resistance_phenotypes.append(str(item).strip())
    resistance_phenotypes = list(dict.fromkeys(item for item in resistance_phenotypes if item))
    susceptibility_results: Dict[str, ASTResult] = {}
    tx_context = {
        "syndrome": payload.tx_context.get("syndrome", "Not specified") or "Not specified",
        "severity": payload.tx_context.get("severity", "Not specified") or "Not specified",
        "focusDetail": payload.tx_context.get("focusDetail", "Not specified") or "Not specified",
        "oralPreference": bool(payload.tx_context.get("oralPreference", False)),
        "carbapenemaseResult": _canonicalize_carbapenemase_result(payload.tx_context.get("carbapenemaseResult", "Not specified")),
        "carbapenemaseClass": _canonicalize_carbapenemase_class(payload.tx_context.get("carbapenemaseClass", "Not specified")),
    }
    if tx_context["carbapenemaseClass"] != "Not specified" and tx_context["carbapenemaseResult"] == "Not specified":
        tx_context["carbapenemaseResult"] = "Positive"

    if organism:
        try:
            organism = normalize_organism(organism)
        except Exception as exc:
            warnings.append(str(exc))
            organism = None

    normalized_mentions: List[str] = []
    for entry in mentioned_organisms:
        try:
            normalized_mentions.append(normalize_organism(entry))
        except Exception:
            normalized_mentions.append(entry)
    mentioned_organisms = list(dict.fromkeys(normalized_mentions))

    organism, phenotype_defaults, phenotype_warnings = infer_phenotype_defaults(organism, resistance_phenotypes)
    warnings.extend(phenotype_warnings)

    if organism:
        for raw_name, raw_state in payload.susceptibility_results.items():
            embedded_carb = _embedded_carbapenemase_row(raw_name, raw_state)
            if embedded_carb is not None:
                inferred_result, inferred_class = embedded_carb
                if tx_context["carbapenemaseResult"] == "Not specified":
                    tx_context["carbapenemaseResult"] = inferred_result
                if tx_context["carbapenemaseClass"] == "Not specified" and inferred_class != "Not specified":
                    tx_context["carbapenemaseClass"] = inferred_class
                warnings.append(
                    f"Interpreted '{raw_name}' as a carbapenemase result row rather than an antibiotic AST entry."
                )
                continue
            antibiotic = resolve_antibiotic_name(organism, raw_name)
            if antibiotic is None:
                warnings.append(f"Ignored unsupported antibiotic for {organism}: {raw_name}")
                continue
            susceptibility_results[antibiotic] = raw_state
        for antibiotic, state in phenotype_defaults.items():
            susceptibility_results.setdefault(antibiotic, state)

    requires_confirmation = payload.confidence == "low" or organism is None or not susceptibility_results
    if payload.ambiguities:
        warnings.append("LLM ambiguities: " + "; ".join(payload.ambiguities))
        requires_confirmation = True
    if payload.ignored_clues:
        warnings.append("LLM ignored clues: " + "; ".join(payload.ignored_clues))
    if len(mentioned_organisms) > 1:
        warnings.append("I detected more than one organism, so I cannot run single-isolate MechID inference yet.")
        requires_confirmation = True

    return {
        "organism": organism,
        "mentionedOrganisms": mentioned_organisms,
        "resistancePhenotypes": resistance_phenotypes,
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
    if not normalized.get("mentionedOrganisms") and rule_fallback.get("mentionedOrganisms"):
        normalized["mentionedOrganisms"] = rule_fallback["mentionedOrganisms"]
    if not normalized.get("resistancePhenotypes") and rule_fallback.get("resistancePhenotypes"):
        normalized["resistancePhenotypes"] = rule_fallback["resistancePhenotypes"]
    if not normalized["susceptibilityResults"] and rule_fallback["susceptibilityResults"]:
        normalized["susceptibilityResults"] = rule_fallback["susceptibilityResults"]
        normalized["warnings"].append("Filled susceptibility results from rule parser.")
    if normalized["txContext"].get("syndrome") == "Not specified" and rule_fallback["txContext"].get("syndrome") != "Not specified":
        normalized["txContext"]["syndrome"] = rule_fallback["txContext"]["syndrome"]
    if normalized["txContext"].get("severity") == "Not specified" and rule_fallback["txContext"].get("severity") != "Not specified":
        normalized["txContext"]["severity"] = rule_fallback["txContext"]["severity"]
    if normalized["txContext"].get("focusDetail") == "Not specified" and rule_fallback["txContext"].get("focusDetail") != "Not specified":
        normalized["txContext"]["focusDetail"] = rule_fallback["txContext"]["focusDetail"]
    if not normalized["txContext"].get("oralPreference") and rule_fallback["txContext"].get("oralPreference"):
        normalized["txContext"]["oralPreference"] = True
    if (
        normalized["txContext"].get("carbapenemaseResult") == "Not specified"
        and rule_fallback["txContext"].get("carbapenemaseResult") != "Not specified"
    ):
        normalized["txContext"]["carbapenemaseResult"] = rule_fallback["txContext"]["carbapenemaseResult"]
    if (
        normalized["txContext"].get("carbapenemaseClass") == "Not specified"
        and rule_fallback["txContext"].get("carbapenemaseClass") != "Not specified"
    ):
        normalized["txContext"]["carbapenemaseClass"] = rule_fallback["txContext"]["carbapenemaseClass"]

    normalized["requiresConfirmation"] = bool(
        normalized["requiresConfirmation"]
        or rule_fallback["requiresConfirmation"]
        or normalized["organism"] is None
        or not normalized["susceptibilityResults"]
    )
    normalized["parser"] = f"openai-{model}-mechid-v1"
    return normalized
