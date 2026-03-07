from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ..schemas import AnalyzeRequest, ProbIDControlsInput
from .module_store import InMemoryModuleStore
from .text_parser import ParseTextResult, empty_understanding, normalize, parse_text_to_request, summarize_parsed_request


class LLMParserError(RuntimeError):
    pass


class LLMExtractionPayload(BaseModel):
    module_id: str | None = Field(default=None, alias="moduleId")
    preset_id: str | None = Field(default=None, alias="presetId")
    findings: Dict[str, Literal["present", "absent", "unknown"]] = Field(default_factory=dict)
    ordered_finding_ids: List[str] = Field(default_factory=list, alias="orderedFindingIds")
    probid_controls: ProbIDControlsInput | None = Field(default=None, alias="probidControls")
    include_explanation: bool = Field(default=True, alias="includeExplanation")
    confidence: Literal["low", "medium", "high"] = "medium"
    ambiguities: List[str] = Field(default_factory=list)
    missing_finding_ids: List[str] = Field(default_factory=list, alias="missingFindingIds")
    ignored_clues: List[str] = Field(default_factory=list, alias="ignoredClues")

    model_config = {"populate_by_name": True}


STATUS_LABEL_TOKENS = ("not done", "unknown", "not used", "not applied", "n/a")
STATUS_ID_TOKENS = ("_na", "not_done", "unknown", "not_used")
SETTING_SIGNAL_TOKENS = (
    "outpatient",
    "clinic",
    "office",
    "ed",
    "er",
    "emergency department",
    "emergency room",
    "hospital",
    "inpatient",
    "admitted",
    "admission",
    "icu",
    "ward",
    "ventilated",
    "after day 3",
    "day 3",
    "first 72 hours",
    "first 48 hours",
)
EXCLUSIVE_RESULT_GROUPS: Dict[str, set[str]] = {
    "cap": {"cap_cxr", "cap_rvp"},
    "vap": {"vap_cxr", "vap_cpis", "vap_pct"},
    "cdi": {"cdi_test"},
    "endo": {"endo_virsta", "endo_denova", "endo_handoc", "endo_micro", "endo_tte", "endo_pet"},
    "active_tb": {"tb_cxr", "tb_ct"},
    "tb_uveitis": {"tbu_phenotype", "tbu_endemicity", "tbu_tst", "tbu_igra", "tbu_chest_imaging"},
}


def _try_import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency optional
        raise LLMParserError("OpenAI SDK not installed. Run `pip install openai` or reinstall requirements.") from exc
    return OpenAI


def _extract_json(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        # strip fenced blocks
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMParserError("LLM response did not contain a JSON object.")
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise LLMParserError(f"LLM returned invalid JSON: {exc}") from exc


def _build_catalog_context(store: InMemoryModuleStore) -> Dict[str, Any]:
    modules: List[Dict[str, Any]] = []
    for summary in store.list_summaries():
        m = store.get(summary.id)
        if not m:
            continue
        modules.append(
            {
                "id": m.id,
                "name": m.name,
                "pretestPresets": [{"id": p.id, "label": p.label} for p in m.pretest_presets],
                "items": [{"id": i.id, "label": i.label, "category": i.category} for i in m.items],
            }
        )
    return {"modules": modules}


def _build_output_schema() -> Dict[str, Any]:
    schema = LLMExtractionPayload.model_json_schema(by_alias=True)
    schema.pop("title", None)
    return schema


def _build_instructions(catalog: Dict[str, Any]) -> str:
    output_schema = _build_output_schema()
    return (
        "You extract clinical free text into a ProbID structured extraction JSON.\n"
        "Return JSON only (no markdown, no prose).\n"
        "Your job is to translate user language into the exact ProbID inputs that can be passed into AnalyzeRequest.\n"
        "Use only the moduleId, presetId, and finding ids present in the provided catalog.\n"
        "If unsure, omit a finding instead of guessing.\n"
        "Use finding states only when the text clearly supports them: present, absent, or unknown.\n"
        "Do not mark sibling findings as absent just because one alternative is present; if a test was not mentioned, omit it.\n"
        "Set orderedFindingIds in the same order the findings are mentioned.\n"
        "Infer presetId from care setting whenever the text mentions outpatient, ED, inpatient, ICU, ventilation, early admission, or later hospital onset.\n"
        "Capture host and epidemiologic risk factors when they are stated, because they change the pretest probability.\n"
        "Use moduleHint and presetHint as priors when they are provided, but do not force them if the text clearly points elsewhere.\n"
        "If the text conflicts with a supplied hint, keep the best-supported interpretation and note the conflict in ambiguities.\n"
        "Do not fabricate probidControls unless the text explicitly describes score components or risk-modifier toggles.\n"
        "Only populate missingFindingIds with up to 5 high-yield missing findings that would materially change the probability estimate.\n"
        "Use ignoredClues for clinically meaningful details that do not map cleanly into the available catalog.\n"
        "Set confidence to low when the syndrome or major findings are ambiguous.\n"
        "Output JSON must validate against this schema:\n"
        + json.dumps(output_schema, ensure_ascii=True)
        + "\n\n"
        "Return the extraction payload only. Do not explain your reasoning.\n\n"
        "CATALOG:\n"
        + json.dumps(catalog, ensure_ascii=True)
    )


def _item_labels_for_ids(store: InMemoryModuleStore, module_id: str | None, item_ids: List[str]) -> List[str]:
    if not module_id or not item_ids:
        return []
    module = store.get(module_id)
    if not module:
        return []
    labels = {item.id: item.label for item in module.items}
    rendered: List[str] = []
    for item_id in item_ids:
        if item_id in labels:
            rendered.append(labels[item_id])
    return rendered


def _is_status_item(item_id: str, label: str | None) -> bool:
    item_id_lower = item_id.lower()
    label_lower = (label or "").lower()
    return any(token in item_id_lower for token in STATUS_ID_TOKENS) or any(
        token in label_lower for token in STATUS_LABEL_TOKENS
    )


def _text_has_setting_signal(text: str) -> bool:
    text_norm = f" {normalize(text)} "
    return any(f" {token} " in text_norm for token in SETTING_SIGNAL_TOKENS)


def _apply_modality_conflict_cleanup(text: str, parsed_request: AnalyzeRequest) -> List[str]:
    if parsed_request.module_id not in {"active_tb", "endo"} or not parsed_request.findings:
        return []

    text_norm = f" {normalize(text)} "
    findings = dict(parsed_request.findings)
    notes: List[str] = []

    if parsed_request.module_id == "active_tb":
        mentions_ct = any(
            token in text_norm
            for token in (
                " chest ct ",
                " ct chest ",
                " ct scan chest ",
                " ct scan of the chest ",
            )
        )
        mentions_cxr = any(
            token in text_norm
            for token in (
                " cxr ",
                " chest xray ",
                " chest x ray ",
                " chest radiograph ",
            )
        )

        if findings.get("tb_ct_suggestive") == "present" and findings.get("tb_cxr_suggestive") == "present":
            if mentions_ct and not mentions_cxr:
                findings.pop("tb_cxr_suggestive", None)
                parsed_request.ordered_finding_ids = [
                    item_id for item_id in parsed_request.ordered_finding_ids if item_id != "tb_cxr_suggestive"
                ]
                notes.append("Removed chest X-ray because the text only described a chest CT result.")
            elif mentions_cxr and not mentions_ct:
                findings.pop("tb_ct_suggestive", None)
                parsed_request.ordered_finding_ids = [
                    item_id for item_id in parsed_request.ordered_finding_ids if item_id != "tb_ct_suggestive"
                ]
                notes.append("Removed chest CT because the text only described a chest X-ray result.")

    if parsed_request.module_id == "endo":
        mentions_tte = " tte " in text_norm or " transthoracic echo " in text_norm
        mentions_tee = " tee " in text_norm or " transesophageal echo " in text_norm

        if "endo_tee" in findings and mentions_tee and not mentions_tte:
            if "endo_tte" in findings:
                findings.pop("endo_tte", None)
                parsed_request.ordered_finding_ids = [
                    item_id for item_id in parsed_request.ordered_finding_ids if item_id != "endo_tte"
                ]
                notes.append("Removed TTE because the text only described a TEE result.")
            if "endo_tte_na" in findings:
                findings.pop("endo_tte_na", None)
                parsed_request.ordered_finding_ids = [
                    item_id for item_id in parsed_request.ordered_finding_ids if item_id != "endo_tte_na"
                ]
        elif "endo_tte" in findings and mentions_tte and not mentions_tee:
            if "endo_tee" in findings:
                findings.pop("endo_tee", None)
                parsed_request.ordered_finding_ids = [
                    item_id for item_id in parsed_request.ordered_finding_ids if item_id != "endo_tee"
                ]
                notes.append("Removed TEE because the text only described a TTE result.")

    parsed_request.findings = findings
    if not parsed_request.ordered_finding_ids and parsed_request.findings:
        parsed_request.ordered_finding_ids = list(parsed_request.findings.keys())
    return notes


def _normalize_group_findings(*, store: InMemoryModuleStore, parsed_request: AnalyzeRequest) -> List[str]:
    module = store.get(parsed_request.module_id or "")
    if module is None or not parsed_request.findings:
        return []

    items_by_id = {item.id: item for item in module.items}
    findings = dict(parsed_request.findings)
    removed_neutral_state = 0
    removed_status_conflict = 0
    removed_exclusive_conflict = 0

    for item_id, state in list(findings.items()):
        item = items_by_id.get(item_id)
        if item is None:
            continue
        if _is_status_item(item_id, item.label) and state != "present":
            findings.pop(item_id, None)
            removed_neutral_state += 1

    groups: Dict[str, List[str]] = {}
    for item in module.items:
        if item.group:
            groups.setdefault(item.group, []).append(item.id)

    exclusive_groups = EXCLUSIVE_RESULT_GROUPS.get(module.id, set())
    for group_id, group_item_ids in groups.items():
        present_ids = [item_id for item_id in group_item_ids if findings.get(item_id) == "present"]
        if not present_ids:
            continue

        concrete_present_ids = [
            item_id
            for item_id in present_ids
            if not _is_status_item(item_id, (items_by_id.get(item_id).label if items_by_id.get(item_id) else ""))
        ]
        if concrete_present_ids:
            for item_id in list(group_item_ids):
                item = items_by_id.get(item_id)
                if item is None or item_id not in findings:
                    continue
                if _is_status_item(item_id, item.label):
                    findings.pop(item_id, None)
                    removed_status_conflict += 1

        if group_id in exclusive_groups:
            for item_id in list(group_item_ids):
                if findings.get(item_id) not in {"absent", "unknown"}:
                    continue
                findings.pop(item_id, None)
                removed_exclusive_conflict += 1

    parsed_request.findings = findings
    parsed_request.ordered_finding_ids = [item_id for item_id in parsed_request.ordered_finding_ids if item_id in findings]
    if not parsed_request.ordered_finding_ids and parsed_request.findings:
        parsed_request.ordered_finding_ids = list(parsed_request.findings.keys())
    notes: List[str] = []
    if removed_neutral_state:
        notes.append("Ignored neutral 'completed/unknown' flags that do not change the calculation.")
    if removed_status_conflict or removed_exclusive_conflict:
        notes.append("Kept only the clinically meaningful result when the same test group had conflicting entries.")
    return notes


def _merge_rule_based_clues(
    *,
    store: InMemoryModuleStore,
    text: str,
    parsed_request: AnalyzeRequest,
    module_hint: str | None,
    preset_hint: str | None,
    include_explanation: bool,
) -> List[str]:
    rule_result = parse_text_to_request(
        store=store,
        text=text,
        module_hint=module_hint,
        preset_hint=preset_hint,
        include_explanation=include_explanation,
    )
    rule_request = rule_result.parsed_request
    if rule_request is None:
        return []

    notes: List[str] = []
    if not parsed_request.module_id and rule_request.module_id:
        parsed_request.module_id = rule_request.module_id
        notes.append(f"Filled module from rule parser: '{rule_request.module_id}'.")

    module = store.get(parsed_request.module_id or "")
    if module is None:
        return notes

    if rule_request.module_id == module.id and rule_request.preset_id:
        default_preset_id = module.pretest_presets[0].id if module.pretest_presets else None
        if not parsed_request.preset_id:
            parsed_request.preset_id = rule_request.preset_id
            notes.append(f"Filled preset from rule parser: '{rule_request.preset_id}'.")
        elif (
            default_preset_id
            and parsed_request.preset_id == default_preset_id
            and rule_request.preset_id != parsed_request.preset_id
            and _text_has_setting_signal(text)
        ):
            parsed_request.preset_id = rule_request.preset_id
            notes.append(f"Adjusted preset from setting words: '{rule_request.preset_id}'.")

    if rule_request.module_id != module.id:
        return notes

    items_by_id = {item.id: item for item in module.items}
    added_host_labels: List[str] = []
    for item_id, state in rule_request.findings.items():
        item = items_by_id.get(item_id)
        if item is None or item.category != "host":
            continue
        if item_id in parsed_request.findings:
            continue
        parsed_request.findings[item_id] = state
        parsed_request.ordered_finding_ids.append(item_id)
        added_host_labels.append(item.label)

    if added_host_labels:
        notes.append("Added host risk factors from rule parser: " + ", ".join(added_host_labels))

    return notes


def parse_text_with_openai(
    *,
    store: InMemoryModuleStore,
    text: str,
    module_hint: str | None = None,
    preset_hint: str | None = None,
    include_explanation: bool = True,
    parser_model: str | None = None,
) -> ParseTextResult:
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
    catalog = _build_catalog_context(store)
    instructions = _build_instructions(catalog)

    user_payload = {
        "text": text,
        "moduleHint": module_hint,
        "presetHint": preset_hint,
        "includeExplanation": include_explanation,
    }

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(user_payload, ensure_ascii=True),
        )
    except Exception as exc:  # pragma: no cover - depends on runtime network/API
        raise LLMParserError(f"OpenAI request failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMParserError("OpenAI response did not include text output.")

    payload = _extract_json(output_text)
    if "includeExplanation" not in payload:
        payload["includeExplanation"] = include_explanation

    try:
        extracted = LLMExtractionPayload.model_validate(payload)
    except Exception as exc:
        raise LLMParserError(f"LLM JSON did not match extraction schema: {exc}") from exc

    if module_hint and not extracted.module_id:
        extracted.module_id = module_hint
    if preset_hint and not extracted.preset_id:
        extracted.preset_id = preset_hint

    request_payload = extracted.model_dump(
        by_alias=True,
        exclude_none=True,
        exclude={"confidence", "ambiguities", "missing_finding_ids", "ignored_clues"},
    )

    try:
        parsed_request = AnalyzeRequest.model_validate(request_payload)
    except Exception as exc:
        raise LLMParserError(f"LLM extraction could not be converted into AnalyzeRequest: {exc}") from exc

    merge_notes = _merge_rule_based_clues(
        store=store,
        text=text,
        parsed_request=parsed_request,
        module_hint=module_hint,
        preset_hint=preset_hint,
        include_explanation=include_explanation,
    )
    normalization_notes = _normalize_group_findings(store=store, parsed_request=parsed_request)
    modality_notes = _apply_modality_conflict_cleanup(text, parsed_request)

    # Keep output deterministic if the LLM omitted ordering.
    if not parsed_request.ordered_finding_ids and parsed_request.findings:
        parsed_request.ordered_finding_ids = list(parsed_request.findings.keys())

    understood, warnings, requires_confirmation = summarize_parsed_request(store, parsed_request)
    warnings.extend(normalization_notes)
    warnings.extend(modality_notes)
    warnings.extend(merge_notes)
    if extracted.confidence == "low":
        warnings.append(f"LLM extraction confidence: {extracted.confidence}")
        requires_confirmation = True
    if extracted.ambiguities:
        warnings.append("LLM ambiguities: " + "; ".join(extracted.ambiguities))
        requires_confirmation = True
    if extracted.ignored_clues:
        warnings.append("LLM ignored clues: " + "; ".join(extracted.ignored_clues))
        requires_confirmation = True
    if extracted.missing_finding_ids:
        missing_labels = _item_labels_for_ids(store, parsed_request.module_id, extracted.missing_finding_ids)
        if missing_labels:
            warnings.append("LLM suggests clarifying: " + ", ".join(missing_labels))
        else:
            warnings.append("LLM suggests clarifying finding ids: " + ", ".join(extracted.missing_finding_ids))
        requires_confirmation = True
    if module_hint:
        warnings.append(f"moduleHint provided: '{module_hint}'")
    if preset_hint:
        warnings.append(f"presetHint provided: '{preset_hint}'")

    if parsed_request.module_id is None and parsed_request.module is None:
        warnings.append("LLM did not produce a moduleId.")
        requires_confirmation = True

    return ParseTextResult(
        parsed_request=parsed_request,
        understood=understood if parsed_request else empty_understanding(),
        warnings=warnings,
        requires_confirmation=requires_confirmation,
        parser_name=f"openai-{model}-extract-v3",
    )
