from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (  # noqa: E402
    _analyze_internal,
    _assistant_case_prompt_options,
    _assistant_case_sections_for_module,
    app,
    store,
)
from app.schemas import AnalyzeRequest  # noqa: E402


NEW_SYNDROME_CASES = {
    "septic_arthritis": {
        "preset": "sa_low",
        "findings": [
            "sa_sym_monoarthritis",
            "sa_exam_painful_rom",
            "sa_synovial_wbc_high",
            "sa_gram_stain_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "monoarthritis, fever, painful range of motion, unable to bear weight"},
            {"selection": "continue_case_draft"},
            {"message": "ESR elevated, CRP elevated, synovial WBC over 50000"},
            {"selection": "continue_case_draft"},
            {"message": "gram stain positive, blood cultures positive"},
            {"selection": "continue_case_draft"},
            {"message": "ultrasound effusion present"},
            {"selection": "continue_case_draft"},
        ],
    },
    "bacterial_meningitis": {
        "preset": "bm_low",
        "findings": [
            "bm_vital_fever",
            "bm_exam_neck_stiffness",
            "bm_csf_glucose_low",
            "bm_csf_gram_stain_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "fever, headache, neck stiffness, altered mental status"},
            {"selection": "continue_case_draft"},
            {"message": "CSF pleocytosis, CSF neutrophil predominance, CSF protein elevated, CSF glucose low"},
            {"selection": "continue_case_draft"},
            {"message": "CSF gram stain positive, blood cultures positive"},
            {"selection": "continue_case_draft"},
            {"message": "CT not done"},
            {"selection": "continue_case_draft"},
        ],
    },
    "encephalitis": {
        "preset": "enc_low",
        "findings": [
            "enc_exam_ams",
            "enc_exam_seizure",
            "enc_hsv_pcr_positive",
            "enc_mri_temporal_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "fever, altered mental status, behavioral change, seizure"},
            {"selection": "continue_case_draft"},
            {"message": "CSF pleocytosis, CSF lymphocytic predominance, CSF protein elevated"},
            {"selection": "continue_case_draft"},
            {"message": "HSV PCR positive"},
            {"selection": "continue_case_draft"},
            {"message": "MRI temporal lobe positive"},
            {"selection": "continue_case_draft"},
        ],
    },
    "spinal_epidural_abscess": {
        "preset": "sea_low",
        "findings": [
            "sea_sym_back_pain",
            "sea_exam_neuro_deficit",
            "sea_esr_high",
            "sea_mri_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "back pain, fever, spinal tenderness, neurologic deficit"},
            {"selection": "continue_case_draft"},
            {"message": "ESR elevated, CRP elevated"},
            {"selection": "continue_case_draft"},
            {"message": "blood cultures positive"},
            {"selection": "continue_case_draft"},
            {"message": "MRI positive"},
            {"selection": "continue_case_draft"},
        ],
    },
    "brain_abscess": {
        "preset": "ba_low",
        "findings": [
            "ba_exam_focal_deficit",
            "ba_exam_seizure",
            "ba_crp_high",
            "ba_mri_dwi_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "headache, fever, focal deficit, seizure"},
            {"selection": "continue_case_draft"},
            {"message": "CRP elevated, WBC elevated"},
            {"selection": "continue_case_draft"},
            {"message": "blood cultures positive"},
            {"selection": "continue_case_draft"},
            {"message": "MRI with diffusion restriction"},
            {"selection": "continue_case_draft"},
        ],
    },
    "necrotizing_soft_tissue_infection": {
        "preset": "nsti_low",
        "findings": [
            "nsti_sym_pain_out_of_proportion",
            "nsti_sym_rapid_progression",
            "nsti_vital_hypotension",
            "nsti_ct_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "pain out of proportion, rapid progression, hypotension, bullae, crepitus"},
            {"selection": "continue_case_draft"},
            {"message": "WBC elevated, CRP elevated, LRINEC high"},
            {"selection": "continue_case_draft"},
            {"message": "blood cultures positive"},
            {"selection": "continue_case_draft"},
            {"message": "CT compatible with NSTI"},
            {"selection": "continue_case_draft"},
        ],
    },
    "tb_uveitis": {
        "preset": "ophthalmology_visit",
        "findings": [
            "tbu_phenotype_choroiditis_tuberculoma",
            "tbu_endemicity_endemic",
            "tbu_tst_positive",
            "tbu_igra_positive",
            "tbu_chest_imaging_positive",
        ],
        "assistant_turns": [
            {"selection": "continue_to_case"},
            {"message": "This is an ophthalmology visit for choroidal tuberculoma in a patient from a TB endemic region."},
            {"selection": "continue_case_draft"},
            {"message": "Tuberculin skin test positive and QuantiFERON positive."},
            {"selection": "continue_case_draft"},
            {"message": "Chest x ray positive for healed or active TB signs."},
            {"selection": "continue_case_draft"},
        ],
    },
}


def _run_direct_checks() -> list[str]:
    failures: list[str] = []
    for module_id, spec in NEW_SYNDROME_CASES.items():
        module = store.get(module_id)
        if module is None:
            failures.append(f"{module_id}: module missing from store")
            continue
        findings = {item_id: "present" for item_id in spec["findings"]}
        req = AnalyzeRequest(
            moduleId=module_id,
            presetId=spec["preset"],
            findings=findings,
            orderedFindingIds=list(spec["findings"]),
            includeExplanation=True,
        )
        result = _analyze_internal(req)
        if result.combined_lr == 1:
            failures.append(f"{module_id}: combined LR remained neutral for representative findings")
        if not result.applied_findings:
            failures.append(f"{module_id}: no applied findings in direct analysis")
    return failures


def _run_assistant_checks() -> list[str]:
    failures: list[str] = []
    client = TestClient(app)
    for module_id, spec in NEW_SYNDROME_CASES.items():
        initial = client.post("/v1/assistant/turn", json={"message": module_id.replace("_", " ")})
        if initial.status_code != 200:
            failures.append(f"{module_id}: initial assistant turn failed with {initial.status_code}")
            continue
        state = initial.json()["state"]

        preset_resp = client.post("/v1/assistant/turn", json={"state": state, "selection": spec["preset"]})
        if preset_resp.status_code != 200:
            failures.append(f"{module_id}: preset turn failed with {preset_resp.status_code}")
            continue
        state = preset_resp.json()["state"]

        for turn in spec["assistant_turns"]:
            payload = {"state": state, **turn}
            resp = client.post("/v1/assistant/turn", json=payload)
            if resp.status_code != 200:
                failures.append(f"{module_id}: assistant turn failed with {resp.status_code}")
                state = None
                break
            state = resp.json()["state"]
        if state is None:
            continue

        run_resp = client.post("/v1/assistant/turn", json={"state": state, "selection": "run_assessment"})
        if run_resp.status_code != 200:
            failures.append(f"{module_id}: run_assessment failed with {run_resp.status_code}")
            continue
        payload = run_resp.json()
        analysis = ((payload.get("analysis") or {}).get("analysis") or {})
        if not analysis:
            failures.append(f"{module_id}: assistant returned no analysis payload")
            continue
        if analysis.get("combinedLR") == 1:
            failures.append(f"{module_id}: assistant combined LR remained neutral")
        if not analysis.get("appliedFindings"):
            failures.append(f"{module_id}: assistant returned no applied findings")
    return failures


def _run_interface_checks() -> list[str]:
    failures: list[str] = []
    for module_id in NEW_SYNDROME_CASES:
        module = store.get(module_id)
        if module is None:
            failures.append(f"{module_id}: module missing for interface check")
            continue
        sections = _assistant_case_sections_for_module(module, None)
        if sections != ["exam_vitals", "lab", "micro", "imaging"]:
            failures.append(f"{module_id}: unexpected assistant sections {sections}")
        for section in sections:
            options = _assistant_case_prompt_options(module, None, section_override=section)
            insert_options = [opt for opt in options if opt.value.startswith("insert_text:")]
            if not insert_options:
                failures.append(f"{module_id}: no quick-add options for section {section}")
            if options[-1].value != "continue_case_draft":
                failures.append(f"{module_id}: section {section} missing continue action")
    return failures


def main() -> int:
    failures = [*_run_direct_checks(), *_run_assistant_checks(), *_run_interface_checks()]
    if failures:
        print("New syndrome smoke test failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("All new syndrome smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
