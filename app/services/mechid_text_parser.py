from __future__ import annotations

import re
from typing import Dict, List, Optional

from .mechid_engine import canonical_antibiotic_aliases, list_mechid_organisms, normalize_organism


SUSC_STATES = {
    "susceptible": "Susceptible",
    "sensitive": "Susceptible",
    "intermediate": "Intermediate",
    "resistant": "Resistant",
}


ORGANISM_ALIASES = {
    "e coli": "Escherichia coli",
    "e. coli": "Escherichia coli",
    "ecoli": "Escherichia coli",
    "klebsiella": "Klebsiella pneumoniae",
    "pseudomonas": "Pseudomonas aeruginosa",
    "acinetobacter": "Acinetobacter baumannii complex",
    "mrsa": "Staphylococcus aureus",
    "staph aureus": "Staphylococcus aureus",
    "staphylococcus aureus": "Staphylococcus aureus",
    "s aureus": "Staphylococcus aureus",
    "vre": "Enterococcus faecium",
    "pneumococcus": "Streptococcus pneumoniae",
    "group b strep": "β-hemolytic Streptococcus (GAS/GBS)",
    "group b streptococcus": "β-hemolytic Streptococcus (GAS/GBS)",
    "gbs": "β-hemolytic Streptococcus (GAS/GBS)",
    "strep agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
    "strep ag": "β-hemolytic Streptococcus (GAS/GBS)",
    "streptococcus agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
    "s agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
}


PHENOTYPE_HINTS = {
    "mrsa": "MRSA",
    "mssa": "MSSA",
    "vre": "VRE",
    "esbl": "ESBL",
    "cre": "CRE",
}

SYNDROME_HINTS = (
    ("cystitis", "Uncomplicated cystitis", "Cystitis"),
    ("uti", "Complicated UTI / pyelonephritis", "UTI / pyelonephritis"),
    ("pyelo", "Complicated UTI / pyelonephritis", "Pyelonephritis"),
    ("pyelonephritis", "Complicated UTI / pyelonephritis", "Pyelonephritis"),
    ("bacteremia", "Bloodstream infection", "Bloodstream infection"),
    ("bloodstream", "Bloodstream infection", "Bloodstream infection"),
    ("pneumonia", "Pneumonia (HAP/VAP or severe CAP)", "Pneumonia"),
    ("hap", "Pneumonia (HAP/VAP or severe CAP)", "HAP/VAP"),
    ("vap", "Pneumonia (HAP/VAP or severe CAP)", "HAP/VAP"),
    ("aspiration", "Pneumonia (HAP/VAP or severe CAP)", "Aspiration pneumonia"),
    ("intra-abdominal", "Intra-abdominal infection", "Intra-abdominal infection"),
    ("peritonitis", "Intra-abdominal infection", "Peritonitis"),
    ("cholangitis", "Intra-abdominal infection", "Biliary infection"),
    ("diverticulitis", "Intra-abdominal infection", "Intra-abdominal infection"),
    ("meningitis", "CNS infection", "Meningitis"),
    ("cns", "CNS infection", "CNS infection"),
    ("osteomyelitis", "Bone/joint infection", "Osteomyelitis"),
    ("bone infection", "Bone/joint infection", "Osteomyelitis"),
    ("septic arthritis", "Bone/joint infection", "Septic arthritis"),
    ("joint infection", "Bone/joint infection", "Septic arthritis"),
    ("joint", "Bone/joint infection", "Bone/joint infection"),
    ("prosthetic joint", "Bone/joint infection", "Prosthetic joint infection"),
    ("pji", "Bone/joint infection", "Prosthetic joint infection"),
    ("diabetic foot", "Other deep-seated / high-inoculum focus", "Diabetic foot infection"),
    ("dfi", "Other deep-seated / high-inoculum focus", "Diabetic foot infection"),
    ("wound infection", "Other deep-seated / high-inoculum focus", "Wound infection"),
    ("soft tissue infection", "Other deep-seated / high-inoculum focus", "Soft tissue infection"),
    ("ssti", "Other deep-seated / high-inoculum focus", "Soft tissue infection"),
    ("deep abscess", "Other deep-seated / high-inoculum focus", "Deep abscess"),
    ("deep seated", "Other deep-seated / high-inoculum focus", "Deep-seated infection"),
)

SEVERITY_HINTS = {
    "septic shock": "Severe / septic shock",
    "shock": "Severe / septic shock",
    "critically ill": "Severe / septic shock",
    "severe": "Severe / septic shock",
    "bacteremic shock": "Severe / septic shock",
}

ORAL_PREFERENCE_HINTS = (
    "oral option",
    "oral options",
    "oral antibiotic",
    "oral antibiotics",
    "by mouth",
    "po option",
    "po therapy",
    "po antibiotic",
    "step down",
    "step-down",
    "oral step down",
    "oral step-down",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/+\-.,;: ]+", " ", text.lower())).strip()


def _find_organisms(text_norm: str) -> List[str]:
    matches: List[tuple[int, str]] = []
    seen: set[str] = set()

    def _record(term: str, canonical: str) -> None:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        match = pattern.search(text_norm)
        if match and canonical not in seen:
            seen.add(canonical)
            matches.append((match.start(), canonical))

    for alias, canonical in sorted(ORGANISM_ALIASES.items(), key=lambda entry: len(entry[0]), reverse=True):
        _record(alias, canonical)
    for organism in list_mechid_organisms():
        _record(organism.lower(), organism)

    matches.sort(key=lambda item: item[0])
    return [canonical for _, canonical in matches]


def _find_organism(text_norm: str) -> str | None:
    organisms = _find_organisms(text_norm)
    if len(organisms) == 1:
        return organisms[0]
    return None


def _find_phenotype_hints(text_norm: str) -> List[str]:
    hints: List[str] = []
    for token, label in PHENOTYPE_HINTS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text_norm):
            hints.append(label)
    return hints


def _extract_state_segments(text_norm: str) -> Dict[str, str]:
    captured: Dict[str, str] = {}
    patterns = [
        (
            r"(?:resistant|resistance) to (.+?)(?=(?:,?\s+(?:susceptible|sensitive|intermediate|resistant|resistance)\s+to\b)|[.;]|$)",
            "Resistant",
        ),
        (
            r"(?:susceptible|sensitive) to (.+?)(?=(?:,?\s+(?:susceptible|sensitive|intermediate|resistant|resistance)\s+to\b)|[.;]|$)",
            "Susceptible",
        ),
        (
            r"intermediate to (.+?)(?=(?:,?\s+(?:susceptible|sensitive|intermediate|resistant|resistance)\s+to\b)|[.;]|$)",
            "Intermediate",
        ),
    ]
    for pattern, state in patterns:
        for match in re.finditer(pattern, text_norm):
            segment = match.group(1)
            if not segment:
                continue
            captured[segment] = state
    return captured


def _extract_antibiotic_results(text_norm: str, organism: str) -> Dict[str, str]:
    aliases = canonical_antibiotic_aliases(organism)
    findings: Dict[str, str] = {}

    for segment, state in _extract_state_segments(text_norm).items():
        for alias, antibiotic in aliases.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", segment):
                findings[antibiotic] = state

    for alias, antibiotic in aliases.items():
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])(?:\s+(?:is|was|reported as))?\s+(susceptible|sensitive|intermediate|resistant)\b"
        )
        for match in pattern.finditer(text_norm):
            state = SUSC_STATES.get(match.group(1))
            if state:
                findings[antibiotic] = state

        reverse = re.compile(
            rf"\b(susceptible|sensitive|intermediate|resistant)\s+(?:to\s+)?(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        )
        for match in reverse.finditer(text_norm):
            state = SUSC_STATES.get(match.group(1))
            if state:
                findings[antibiotic] = state
    return findings


def _infer_tx_context(text_norm: str) -> Dict[str, str]:
    syndrome = "Not specified"
    severity = "Not specified"
    focus_detail = "Not specified"
    oral_preference = False
    for token, label, detail in SYNDROME_HINTS:
        if token in text_norm:
            syndrome = label
            focus_detail = detail
            break
    for token, label in SEVERITY_HINTS.items():
        if token in text_norm:
            severity = label
            break
    oral_preference = any(token in text_norm for token in ORAL_PREFERENCE_HINTS)
    return {
        "syndrome": syndrome,
        "severity": severity,
        "focusDetail": focus_detail,
        "oralPreference": oral_preference,
    }


def parse_mechid_text(text: str) -> Dict[str, object]:
    text_norm = _normalize_text(text)
    organisms = _find_organisms(text_norm)
    organism = organisms[0] if len(organisms) == 1 else None
    phenotype_hints = _find_phenotype_hints(text_norm)
    warnings: List[str] = []
    if not organisms:
        warnings.append("Could not confidently identify an organism from the text.")
        return {
            "organism": None,
            "mentionedOrganisms": [],
            "resistancePhenotypes": phenotype_hints,
            "susceptibilityResults": {},
            "txContext": _infer_tx_context(text_norm),
            "warnings": warnings,
            "requiresConfirmation": True,
        }
    if len(organisms) > 1:
        warnings.append("I detected more than one organism, so I cannot run single-isolate MechID inference yet.")

    try:
        normalized_org = normalize_organism(organism) if organism is not None else None
    except Exception:
        normalized_org = organism
    results = _extract_antibiotic_results(text_norm, normalized_org) if normalized_org else {}
    if not results:
        warnings.append("No susceptibility results were confidently extracted from the text.")
    return {
        "organism": normalized_org,
        "mentionedOrganisms": organisms,
        "resistancePhenotypes": phenotype_hints,
        "susceptibilityResults": results,
        "txContext": _infer_tx_context(text_norm),
        "warnings": warnings,
        "requiresConfirmation": organism is None or not results,
    }
