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
    "s aureus": "Staphylococcus aureus",
    "vre": "Enterococcus faecium",
    "pneumococcus": "Streptococcus pneumoniae",
}

SYNDROME_HINTS = {
    "cystitis": "Uncomplicated cystitis",
    "uti": "Complicated UTI / pyelonephritis",
    "pyelo": "Complicated UTI / pyelonephritis",
    "pyelonephritis": "Complicated UTI / pyelonephritis",
    "bacteremia": "Bloodstream infection",
    "bloodstream": "Bloodstream infection",
    "pneumonia": "Pneumonia (HAP/VAP or severe CAP)",
    "hap": "Pneumonia (HAP/VAP or severe CAP)",
    "vap": "Pneumonia (HAP/VAP or severe CAP)",
    "intra-abdominal": "Intra-abdominal infection",
    "peritonitis": "Intra-abdominal infection",
    "meningitis": "CNS infection",
    "cns": "CNS infection",
    "osteomyelitis": "Bone/joint infection",
    "bone infection": "Bone/joint infection",
    "septic arthritis": "Bone/joint infection",
    "joint infection": "Bone/joint infection",
    "joint": "Bone/joint infection",
    "prosthetic joint": "Bone/joint infection",
    "pji": "Bone/joint infection",
    "diabetic foot": "Other deep-seated / high-inoculum focus",
    "dfi": "Other deep-seated / high-inoculum focus",
    "wound infection": "Other deep-seated / high-inoculum focus",
    "soft tissue infection": "Other deep-seated / high-inoculum focus",
    "ssti": "Other deep-seated / high-inoculum focus",
    "deep abscess": "Other deep-seated / high-inoculum focus",
    "deep seated": "Other deep-seated / high-inoculum focus",
}

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


def _find_organism(text_norm: str) -> str | None:
    for alias, canonical in ORGANISM_ALIASES.items():
        if alias in text_norm:
            return canonical
    for organism in list_mechid_organisms():
        if organism.lower() in text_norm:
            return organism
    return None


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
    oral_preference = False
    for token, label in SYNDROME_HINTS.items():
        if token in text_norm:
            syndrome = label
            break
    for token, label in SEVERITY_HINTS.items():
        if token in text_norm:
            severity = label
            break
    oral_preference = any(token in text_norm for token in ORAL_PREFERENCE_HINTS)
    return {"syndrome": syndrome, "severity": severity, "oralPreference": oral_preference}


def parse_mechid_text(text: str) -> Dict[str, object]:
    text_norm = _normalize_text(text)
    organism = _find_organism(text_norm)
    warnings: List[str] = []
    if organism is None:
        warnings.append("Could not confidently identify an organism from the text.")
        return {
            "organism": None,
            "susceptibilityResults": {},
            "txContext": _infer_tx_context(text_norm),
            "warnings": warnings,
            "requiresConfirmation": True,
        }

    try:
        normalized_org = normalize_organism(organism)
    except Exception:
        normalized_org = organism
    results = _extract_antibiotic_results(text_norm, normalized_org)
    if not results:
        warnings.append("No susceptibility results were confidently extracted from the text.")
    return {
        "organism": normalized_org,
        "susceptibilityResults": results,
        "txContext": _infer_tx_context(text_norm),
        "warnings": warnings,
        "requiresConfirmation": organism is None or not results,
    }
