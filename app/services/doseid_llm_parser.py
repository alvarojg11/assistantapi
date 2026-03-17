from __future__ import annotations

import json
import os
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from .doseid_service import list_medications
from .llm_text_parser import LLMParserError, _extract_json, _try_import_openai


class DoseIDLLMSelection(BaseModel):
    medication_id: str = Field(alias="medicationId")
    indication_id: str | None = Field(default=None, alias="indicationId")

    model_config = {"populate_by_name": True}


class DoseIDLLMPatientContext(BaseModel):
    age_years: int | None = Field(default=None, alias="ageYears")
    sex: Literal["male", "female"] | None = None
    total_body_weight_kg: float | None = Field(default=None, alias="totalBodyWeightKg")
    height_cm: float | None = Field(default=None, alias="heightCm")
    serum_creatinine_mg_dl: float | None = Field(default=None, alias="serumCreatinineMgDl")
    crcl_ml_min: float | None = Field(default=None, alias="crclMlMin")
    renal_mode: Literal["standard", "ihd", "crrt"] = Field(default="standard", alias="renalMode")

    model_config = {"populate_by_name": True}


class DoseIDLLMExtractionPayload(BaseModel):
    medications: List[DoseIDLLMSelection] = Field(default_factory=list)
    patient_context: DoseIDLLMPatientContext = Field(alias="patientContext")
    ambiguities: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")

    model_config = {"populate_by_name": True}


def _build_doseid_catalog() -> Dict[str, object]:
    return {
        "medications": [
            {
                "id": med.id,
                "name": med.name,
                "category": med.category,
                "indications": [{"id": item.id, "label": item.label} for item in med.indications],
            }
            for med in list_medications()
        ]
    }


def _build_instructions(catalog: Dict[str, object]) -> str:
    schema = DoseIDLLMExtractionPayload.model_json_schema(by_alias=True)
    schema.pop("title", None)
    return (
        "You extract antimicrobial dosing free text into a DoseID JSON payload.\n"
        "Return JSON only. No markdown and no prose.\n"
        "Do not calculate the regimen. Only extract supported medication ids, indication ids, and patient context values.\n"
        "Use only medicationId and indicationId values that appear in the provided catalog.\n"
        "If a detail is not stated, leave it null or omit it.\n"
        "Convert pounds to kilograms and feet/inches to centimeters when clearly stated.\n"
        "If the text says hemodialysis or HD, set renalMode to ihd. If it says CRRT/CVVH/CVVHD/CVVHDF, set renalMode to crrt. Otherwise use standard.\n"
        "Phrases like '46 year old', '46-year-old', '46 yo', or 'age 46' are age and must go to ageYears.\n"
        "Creatinine values in mg/dL should go to serumCreatinineMgDl. Creatinine clearance values in mL/min should go to crclMlMin.\n"
        "Do not infer crclMlMin from age, weight, serum creatinine, or dialysis status. Only fill crclMlMin when the text explicitly gives CrCl or creatinine clearance.\n"
        "Brand or commercial names may refer to the same generic medication and should be normalized to the catalog medicationId.\n"
        "If the user clearly describes prophylaxis, meningitis, encephalitis, intermittent TB dosing, hardware rifampin use, or other supported indication-specific context, select the matching indicationId.\n"
        "If the indication is not clear, leave indicationId null.\n"
        "Set requiresConfirmation to true when the text is materially ambiguous.\n"
        "Output must validate against this schema:\n"
        + json.dumps(schema, ensure_ascii=True)
        + "\n\nCATALOG:\n"
        + json.dumps(catalog, ensure_ascii=True)
    )


def parse_doseid_text_with_openai(
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

    model = parser_model or os.getenv("OPENAI_DOSEID_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    instructions = _build_instructions(_build_doseid_catalog())

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
        extracted = DoseIDLLMExtractionPayload.model_validate(payload)
    except Exception as exc:
        raise LLMParserError(f"LLM JSON did not match DoseID extraction schema: {exc}") from exc

    return {
        "medications": [item.model_dump(by_alias=True) for item in extracted.medications],
        "patientContext": extracted.patient_context.model_dump(by_alias=True),
        "ambiguities": list(extracted.ambiguities),
        "requiresConfirmation": bool(extracted.requires_confirmation),
        "parser": f"openai-{model}-doseid-v1",
    }
