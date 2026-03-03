from __future__ import annotations

from ..schemas import HarmInputs, SyndromeModule


DEMO_MODULES: list[SyndromeModule] = [
    SyndromeModule(
        id="cap_demo",
        name="CAP (Demo)",
        description="Example community-acquired pneumonia module used until real ProbID module data is imported.",
        pretestPresets=[
            {"id": "ed_adult", "label": "ED adult with cough/fever", "p": 0.18},
            {"id": "primary_care", "label": "Primary care respiratory symptoms", "p": 0.08},
        ],
        defaultHarms=HarmInputs(unnecessary_treatment=1.0, missed_diagnosis=5.0),
        items=[
            {
                "id": "fever",
                "label": "Fever",
                "category": "vital",
                "lrPos": 1.8,
                "lrNeg": 0.7,
                "notes": "Demo placeholder values; replace with ProbID evidence-backed LRs.",
            },
            {
                "id": "focal_crackles",
                "label": "Focal crackles",
                "category": "exam",
                "lrPos": 2.2,
                "lrNeg": 0.8,
            },
            {
                "id": "cxr_infiltrate",
                "label": "CXR infiltrate",
                "category": "imaging",
                "lrPos": 8.5,
                "lrNeg": 0.2,
            },
            {
                "id": "procalcitonin_high",
                "label": "Procalcitonin elevated",
                "category": "lab",
                "lrPos": 3.0,
                "lrNeg": 0.5,
            },
        ],
    ),
    SyndromeModule(
        id="endo_demo",
        name="Endocarditis (Demo)",
        description="Example infective endocarditis module with placeholder inputs for API development.",
        pretestPresets=[
            {"id": "sab_bacteremia", "label": "S. aureus bacteremia context", "p": 0.12},
            {"id": "persistent_bacteremia", "label": "Persistent bacteremia / embolic signs", "p": 0.22},
        ],
        defaultHarms=HarmInputs(unnecessary_treatment=2.5, missed_diagnosis=10.0),
        items=[
            {"id": "new_murmur", "label": "New murmur", "category": "exam", "lrPos": 2.6, "lrNeg": 0.8},
            {"id": "embolic_phenomena", "label": "Embolic phenomena", "category": "exam", "lrPos": 3.8, "lrNeg": 0.7},
            {"id": "echo_vegetation", "label": "Echo vegetation", "category": "imaging", "lrPos": 15.0, "lrNeg": 0.15},
            {"id": "multiple_positive_cultures", "label": "Multiple positive blood cultures", "category": "micro", "lrPos": 4.5, "lrNeg": 0.4},
        ],
    ),
]

