from __future__ import annotations

import re
from typing import Dict, List, Optional

from .mechid_engine import canonical_antibiotic_aliases, list_mechid_organisms, normalize_organism, resolve_antibiotic_name


SUSC_STATES = {
    "susceptible": "Susceptible",
    "sensitive": "Susceptible",
    "intermediate": "Intermediate",
    "resistant": "Resistant",
    "resistance": "Resistant",
}


ORGANISM_ALIASES = {
    "e coli": "Escherichia coli",
    "e. coli": "Escherichia coli",
    "ecoli": "Escherichia coli",
    "klebsiella": "Klebsiella pneumoniae",
    "enterobacter cloacae": "Enterobacter cloacae complex",
    "e cloacae": "Enterobacter cloacae complex",
    "e. cloacae": "Enterobacter cloacae complex",
    "enterobacter cloacae complex": "Enterobacter cloacae complex",
    "pseudomonas": "Pseudomonas aeruginosa",
    "acinetobacter": "Acinetobacter baumannii complex",
    "cons": "Coagulase-negative Staphylococcus",
    "coagulase negative staphylococcus": "Coagulase-negative Staphylococcus",
    "coagulase-negative staphylococcus": "Coagulase-negative Staphylococcus",
    "mrsa": "Staphylococcus aureus",
    "staph aureus": "Staphylococcus aureus",
    "staphylococcus aureus": "Staphylococcus aureus",
    "s aureus": "Staphylococcus aureus",
    "staph epidermidis": "Coagulase-negative Staphylococcus",
    "staphylococcus epidermidis": "Coagulase-negative Staphylococcus",
    "s epidermidis": "Coagulase-negative Staphylococcus",
    "staph lugdunensis": "Staphylococcus lugdunensis",
    "staphylococcus lugdunensis": "Staphylococcus lugdunensis",
    "s lugdunensis": "Staphylococcus lugdunensis",
    "e faecium": "Enterococcus faecium",
    "e. faecium": "Enterococcus faecium",
    "enterococcus faecium": "Enterococcus faecium",
    "e faecalis": "Enterococcus faecalis",
    "e. faecalis": "Enterococcus faecalis",
    "enterococcus faecalis": "Enterococcus faecalis",
    "e gallinarum": "Enterococcus gallinarum",
    "e. gallinarum": "Enterococcus gallinarum",
    "enterococcus gallinarum": "Enterococcus gallinarum",
    "vre": "Enterococcus faecium",
    "pneumococcus": "Streptococcus pneumoniae",
    "strep pneumo": "Streptococcus pneumoniae",
    "streptococcus pneumoniae": "Streptococcus pneumoniae",
    "s pneumoniae": "Streptococcus pneumoniae",
    "vgs": "Viridans group streptococci (VGS)",
    "viridans strep": "Viridans group streptococci (VGS)",
    "group b strep": "β-hemolytic Streptococcus (GAS/GBS)",
    "group b streptococcus": "β-hemolytic Streptococcus (GAS/GBS)",
    "gbs": "β-hemolytic Streptococcus (GAS/GBS)",
    "strep agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
    "strep ag": "β-hemolytic Streptococcus (GAS/GBS)",
    "streptococcus agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
    "s agalactiae": "β-hemolytic Streptococcus (GAS/GBS)",
}


PHENOTYPE_PATTERNS = (
    (r"\bmrsa\b|\bmethicillin[- ]resistant staphylococcus aureus\b|\boxacillin[- ]resistant staphylococcus aureus\b", "MRSA"),
    (r"\bmssa\b|\bmethicillin[- ]susceptible staphylococcus aureus\b|\boxacillin[- ]susceptible staphylococcus aureus\b", "MSSA"),
    (
        r"\bmrse\b|\bmr[- ]?cons\b|\bmethicillin[- ]resistant coagulase[- ]negative staphylococc(?:us|i)\b|\bmethicillin[- ]resistant staphylococcus epidermidis\b",
        "MR-CoNS",
    ),
    (
        r"\bms[- ]?cons\b|\bmethicillin[- ]susceptible coagulase[- ]negative staphylococc(?:us|i)\b|\bmethicillin[- ]susceptible staphylococcus epidermidis\b",
        "MS-CoNS",
    ),
    (r"\bvre\b|\bvancomycin[- ]resistant enterococc(?:us|i)\b", "VRE"),
    (r"\bvrsa\b|\bvancomycin[- ]resistant staphylococcus aureus\b", "VRSA"),
    (
        r"\bprsp\b|\bpenicillin[- ](?:resistant|non[- ]susceptible) streptococcus pneumoniae\b|\bpenicillin[- ](?:resistant|non[- ]susceptible) pneumococc(?:us|i)\b",
        "Penicillin-resistant pneumococcus",
    ),
    (r"\besbl\b", "ESBL"),
    (r"\bcre\b", "CRE"),
    (r"\b(?:bla[- ]?)?kpc\b", "KPC carbapenemase"),
    (r"\b(?:bla[- ]?)?ndm\b", "NDM carbapenemase"),
    (r"\b(?:bla[- ]?)?vim\b", "VIM carbapenemase"),
    (r"\b(?:bla[- ]?)?imp(?:[- ]?type)?\b", "IMP carbapenemase"),
    (r"\boxa(?:[- ]?48(?:[- ]?like)?)\b|\boxa48(?:[- ]?like)?\b", "OXA-48-like carbapenemase"),
    (r"\bmbl\b|\bmetallo[- ]beta[- ]lactamase\b", "MBL carbapenemase"),
)


PHENOTYPE_ORGANISM_HINTS = {
    "MRSA": "Staphylococcus aureus",
    "MSSA": "Staphylococcus aureus",
    "MR-CoNS": "Coagulase-negative Staphylococcus",
    "MS-CoNS": "Coagulase-negative Staphylococcus",
    "VRE": "Enterococcus faecium",
    "VRE gallinarum": "Enterococcus gallinarum",
    "VRSA": "Staphylococcus aureus",
    "Penicillin-resistant pneumococcus": "Streptococcus pneumoniae",
}


PHENOTYPE_AST_DEFAULTS = {
    "MRSA": (("Nafcillin/Oxacillin", "Resistant"), ("Cefoxitin", "Resistant")),
    "MSSA": (("Nafcillin/Oxacillin", "Susceptible"), ("Cefoxitin", "Susceptible")),
    "MR-CoNS": (("Nafcillin/Oxacillin", "Resistant"), ("Cefoxitin", "Resistant")),
    "MS-CoNS": (("Nafcillin/Oxacillin", "Susceptible"), ("Cefoxitin", "Susceptible")),
    "VRE": (("Vancomycin", "Resistant"),),
    "VRSA": (("Vancomycin", "Resistant"),),
    "Penicillin-resistant pneumococcus": (("Penicillin", "Resistant"),),
}

SYNDROME_HINTS = (
    ("infective endocarditis", "Bloodstream infection", "Endocarditis"),
    ("endocarditis", "Bloodstream infection", "Endocarditis"),
    ("valve infection", "Bloodstream infection", "Endocarditis"),
    ("prosthetic valve", "Bloodstream infection", "Endocarditis"),
    ("mechanical valve", "Bloodstream infection", "Endocarditis"),
    ("bioprosthetic valve", "Bloodstream infection", "Endocarditis"),
    ("prosthetic aortic valve", "Bloodstream infection", "Endocarditis"),
    ("prosthetic mitral valve", "Bloodstream infection", "Endocarditis"),
    ("tavr", "Bloodstream infection", "Endocarditis"),
    ("tavi", "Bloodstream infection", "Endocarditis"),
    ("transcatheter aortic valve", "Bloodstream infection", "Endocarditis"),
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
    ("prosthetic joint infection", "Bone/joint infection", "Prosthetic joint infection"),
    ("prosthetic joint", "Bone/joint infection", "Prosthetic joint infection"),
    ("pji", "Bone/joint infection", "Prosthetic joint infection"),
    ("osteomyelitis", "Bone/joint infection", "Osteomyelitis"),
    ("bone infection", "Bone/joint infection", "Osteomyelitis"),
    ("septic arthritis", "Bone/joint infection", "Septic arthritis"),
    ("joint infection", "Bone/joint infection", "Septic arthritis"),
    ("joint", "Bone/joint infection", "Bone/joint infection"),
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
    for pattern, label in PHENOTYPE_PATTERNS:
        if re.search(pattern, text_norm):
            hints.append(label)
    return hints


def infer_phenotype_defaults(
    organism: str | None,
    phenotype_hints: List[str],
) -> tuple[str | None, Dict[str, str], List[str]]:
    resolved_organism = organism
    warnings: List[str] = []
    phenotype_defaults: Dict[str, str] = {}
    unique_hints = list(dict.fromkeys(phenotype_hints))

    for hint in unique_hints:
        hint_organism = PHENOTYPE_ORGANISM_HINTS.get(hint)
        if hint_organism is None:
            continue
        if resolved_organism is None:
            resolved_organism = hint_organism
            warnings.append(f"Inferred organism from phenotype label: {hint}.")
            continue
        try:
            current_normalized = normalize_organism(resolved_organism)
            hint_normalized = normalize_organism(hint_organism)
        except Exception:
            current_normalized = resolved_organism
            hint_normalized = hint_organism
        if current_normalized != hint_normalized:
            warnings.append(
                f"Phenotype label {hint} suggests {hint_organism}, which does not match the extracted organism {resolved_organism}."
            )

    if resolved_organism is None:
        return None, phenotype_defaults, warnings

    for hint in unique_hints:
        for antibiotic_name, state in PHENOTYPE_AST_DEFAULTS.get(hint, ()):
            antibiotic = resolve_antibiotic_name(resolved_organism, antibiotic_name)
            if antibiotic is None:
                continue
            phenotype_defaults.setdefault(antibiotic, state)

    if phenotype_defaults:
        derived_calls = ", ".join(f"{drug} {state}" for drug, state in phenotype_defaults.items())
        warnings.append(f"Applied phenotype-derived susceptibility defaults: {derived_calls}.")

    return resolved_organism, phenotype_defaults, warnings


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
    ordered_aliases = sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True)

    def _segment_matches(segment: str) -> List[str]:
        matches: List[tuple[int, int, str]] = []
        for alias, antibiotic in ordered_aliases:
            for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", segment):
                start, end = match.span()
                if any(existing_start <= start and end <= existing_end for existing_start, existing_end, _ in matches):
                    continue
                matches.append((start, end, antibiotic))
                break
        matches.sort(key=lambda item: item[0])
        deduped: List[str] = []
        for _, _, antibiotic in matches:
            if antibiotic not in deduped:
                deduped.append(antibiotic)
        return deduped

    for segment, state in _extract_state_segments(text_norm).items():
        for antibiotic in _segment_matches(segment):
            findings[antibiotic] = state

    for alias, _antibiotic in ordered_aliases:
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])(?:\s+(?:is|was|reported as))?\s+(susceptible|sensitive|intermediate|resistant|resistance)\b"
        )
        for match in pattern.finditer(text_norm):
            state = SUSC_STATES.get(match.group(1))
            antibiotic = resolve_antibiotic_name(organism, alias)
            if state and antibiotic:
                findings[antibiotic] = state

        adjective = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?:[- ]resistant|[- ]intermediate|[- ]susceptible|[- ]sensitive)\b"
        )
        for match in adjective.finditer(text_norm):
            token = match.group(0).rsplit("-", 1)[-1].rsplit(" ", 1)[-1]
            state = SUSC_STATES.get(token)
            antibiotic = resolve_antibiotic_name(organism, alias)
            if state and antibiotic:
                findings[antibiotic] = state

        reverse = re.compile(
            rf"\b(susceptible|sensitive|intermediate|resistant)\s+(?:to\s+)?(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        )
        for match in reverse.finditer(text_norm):
            state = SUSC_STATES.get(match.group(1))
            antibiotic = resolve_antibiotic_name(organism, alias)
            if state and antibiotic:
                findings[antibiotic] = state
    return findings


def _infer_tx_context(text_norm: str) -> Dict[str, str]:
    syndrome = "Not specified"
    severity = "Not specified"
    focus_detail = "Not specified"
    oral_preference = False
    carbapenemase_result = "Not specified"
    carbapenemase_class = "Not specified"
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
    if re.search(r"\b(?:carbapenemase|cp[- ]?cre)\b.*\b(?:negative|not detected)\b", text_norm) or re.search(
        r"\b(?:negative|not detected)\b.*\bcarbapenemase\b",
        text_norm,
    ):
        carbapenemase_result = "Negative"
    elif re.search(r"\bcarbapenemase\b.*\b(?:pending|not tested)\b", text_norm) or re.search(
        r"\b(?:pending|not tested)\b.*\bcarbapenemase\b",
        text_norm,
    ):
        carbapenemase_result = "Not tested / pending"
    elif re.search(r"\b(?:carbapenemase|cp[- ]?cre)\b.*\b(?:positive|detected|present)\b", text_norm) or re.search(
        r"\b(?:positive|detected|present)\b.*\b(?:for\s+)?carbapenemase\b",
        text_norm,
    ):
        carbapenemase_result = "Positive"

    class_patterns = (
        (r"\b(?:bla[- ]?)?kpc\b", "KPC"),
        (r"\boxa(?:[- ]?48(?:[- ]?like)?)\b|\boxa48(?:[- ]?like)?\b", "OXA-48-like"),
        (r"\b(?:bla[- ]?)?ndm\b", "NDM"),
        (r"\b(?:bla[- ]?)?vim\b", "VIM"),
        (r"\b(?:bla[- ]?)?imp(?:[- ]?type)?\b", "IMP"),
    )
    for pattern, label in class_patterns:
        if re.search(pattern, text_norm):
            carbapenemase_class = label
            if carbapenemase_result == "Not specified":
                carbapenemase_result = "Positive"
            break
    if carbapenemase_class == "Not specified" and re.search(r"\b(?:mbl|metallo beta lactamase|metallo-beta-lactamase)\b", text_norm):
        carbapenemase_class = "Other / Unknown"
        if carbapenemase_result == "Not specified":
            carbapenemase_result = "Positive"
    return {
        "syndrome": syndrome,
        "severity": severity,
        "focusDetail": focus_detail,
        "oralPreference": oral_preference,
        "carbapenemaseResult": carbapenemase_result,
        "carbapenemaseClass": carbapenemase_class,
    }


def parse_mechid_text(text: str) -> Dict[str, object]:
    text_norm = _normalize_text(text)
    organisms = _find_organisms(text_norm)
    phenotype_hints = _find_phenotype_hints(text_norm)
    warnings: List[str] = []
    organism = organisms[0] if len(organisms) == 1 else None
    if len(organisms) > 1:
        warnings.append("I detected more than one organism, so I cannot run single-isolate MechID inference yet.")

    try:
        normalized_org = normalize_organism(organism) if organism is not None else None
    except Exception:
        normalized_org = organism
    normalized_org, phenotype_defaults, phenotype_warnings = infer_phenotype_defaults(normalized_org, phenotype_hints)
    warnings.extend(phenotype_warnings)

    if not organisms and normalized_org is None:
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

    results = _extract_antibiotic_results(text_norm, normalized_org) if normalized_org else {}
    for antibiotic, state in phenotype_defaults.items():
        results.setdefault(antibiotic, state)
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
