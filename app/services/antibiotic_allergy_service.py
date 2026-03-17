from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..schemas import (
    AntibioticAllergyAnalyzeRequest,
    AntibioticAllergyAnalyzeResponse,
    AntibioticAllergyEntry,
    AntibioticAllergyFollowUpQuestion,
    AntibioticAllergyRecommendation,
    AntibioticAllergyTextAnalyzeRequest,
    AntibioticAllergyTextAnalyzeResponse,
    AntibioticAllergyTextParsedRequest,
    ReferenceEntry,
)


@dataclass(frozen=True)
class AgentInfo:
    name: str
    drug_class: str
    beta_subclass: str | None = None
    side_chain_group: str | None = None


AGENT_INFO: Dict[str, AgentInfo] = {
    "penicillin": AgentInfo("Penicillin", "beta_lactam", "penicillin", None),
    "penicillin_g": AgentInfo("Penicillin G", "beta_lactam", "penicillin", None),
    "amoxicillin": AgentInfo("Amoxicillin", "beta_lactam", "penicillin", "aminopenicillin_amoxicillin"),
    "amoxicillin_clavulanate": AgentInfo("Amoxicillin/Clavulanate", "beta_lactam", "penicillin", "aminopenicillin_amoxicillin"),
    "ampicillin": AgentInfo("Ampicillin", "beta_lactam", "penicillin", "aminopenicillin_ampicillin"),
    "ampicillin_sulbactam": AgentInfo("Ampicillin/Sulbactam", "beta_lactam", "penicillin", "aminopenicillin_ampicillin"),
    "nafcillin": AgentInfo("Nafcillin", "beta_lactam", "penicillin", None),
    "oxacillin": AgentInfo("Oxacillin", "beta_lactam", "penicillin", None),
    "nafcillin_oxacillin": AgentInfo("Nafcillin/Oxacillin", "beta_lactam", "penicillin", None),
    "piperacillin_tazobactam": AgentInfo("Piperacillin/Tazobactam", "beta_lactam", "penicillin", None),
    "cefazolin": AgentInfo("Cefazolin", "beta_lactam", "cephalosporin", "cefazolin_unique"),
    "cephalexin": AgentInfo("Cephalexin", "beta_lactam", "cephalosporin", "aminopenicillin_ampicillin"),
    "ceftriaxone": AgentInfo("Ceftriaxone", "beta_lactam", "cephalosporin", "ceftriaxone_group"),
    "cefotaxime": AgentInfo("Cefotaxime", "beta_lactam", "cephalosporin", "ceftriaxone_group"),
    "cefepime": AgentInfo("Cefepime", "beta_lactam", "cephalosporin", "ceftriaxone_group"),
    "ceftazidime": AgentInfo("Ceftazidime", "beta_lactam", "cephalosporin", "aztreonam_group"),
    "ceftaroline": AgentInfo("Ceftaroline", "beta_lactam", "cephalosporin", None),
    "aztreonam": AgentInfo("Aztreonam", "beta_lactam", "monobactam", "aztreonam_group"),
    "meropenem": AgentInfo("Meropenem", "beta_lactam", "carbapenem", None),
    "ertapenem": AgentInfo("Ertapenem", "beta_lactam", "carbapenem", None),
    "imipenem_cilastatin": AgentInfo("Imipenem/Cilastatin", "beta_lactam", "carbapenem", None),
    "vancomycin": AgentInfo("Vancomycin", "glycopeptide"),
    "vancomycin_iv": AgentInfo("Vancomycin", "glycopeptide"),
    "linezolid": AgentInfo("Linezolid", "oxazolidinone"),
    "daptomycin": AgentInfo("Daptomycin", "lipopeptide"),
    "tmp_smx": AgentInfo("Trimethoprim/Sulfamethoxazole", "sulfonamide"),
    "trimethoprim_sulfamethoxazole": AgentInfo("Trimethoprim/Sulfamethoxazole", "sulfonamide"),
    "ciprofloxacin": AgentInfo("Ciprofloxacin", "fluoroquinolone"),
    "levofloxacin": AgentInfo("Levofloxacin", "fluoroquinolone"),
    "moxifloxacin": AgentInfo("Moxifloxacin", "fluoroquinolone"),
    "doxycycline": AgentInfo("Doxycycline", "tetracycline"),
    "tetracycline_doxycycline": AgentInfo("Doxycycline", "tetracycline"),
    "clindamycin": AgentInfo("Clindamycin", "lincosamide"),
    "metronidazole": AgentInfo("Metronidazole", "nitroimidazole"),
    "nitrofurantoin": AgentInfo("Nitrofurantoin", "nitrofuran"),
    "fosfomycin": AgentInfo("Fosfomycin", "phosphonic_acid"),
    "rifampin": AgentInfo("Rifampin", "rifamycin"),
}

AGENT_ALIASES: Dict[str, str] = {
    "penicillin": "penicillin",
    "penicillin g": "penicillin_g",
    "pcn": "penicillin",
    "amoxicillin": "amoxicillin",
    "augmentin": "amoxicillin_clavulanate",
    "amoxicillin clavulanate": "amoxicillin_clavulanate",
    "amoxicillin/clavulanate": "amoxicillin_clavulanate",
    "ampicillin": "ampicillin",
    "unasyn": "ampicillin_sulbactam",
    "ampicillin sulbactam": "ampicillin_sulbactam",
    "ampicillin/sulbactam": "ampicillin_sulbactam",
    "nafcillin": "nafcillin",
    "oxacillin": "oxacillin",
    "nafcillin oxacillin": "nafcillin_oxacillin",
    "zosyn": "piperacillin_tazobactam",
    "piperacillin tazobactam": "piperacillin_tazobactam",
    "piperacillin/tazobactam": "piperacillin_tazobactam",
    "cefazolin": "cefazolin",
    "ancef": "cefazolin",
    "cephalexin": "cephalexin",
    "keflex": "cephalexin",
    "ceftriaxone": "ceftriaxone",
    "rocephin": "ceftriaxone",
    "cefotaxime": "cefotaxime",
    "cefepime": "cefepime",
    "ceftazidime": "ceftazidime",
    "ceftaroline": "ceftaroline",
    "aztreonam": "aztreonam",
    "meropenem": "meropenem",
    "ertapenem": "ertapenem",
    "imipenem": "imipenem_cilastatin",
    "imipenem cilastatin": "imipenem_cilastatin",
    "imipenem/cilastatin": "imipenem_cilastatin",
    "vancomycin": "vancomycin",
    "vanc": "vancomycin",
    "linezolid": "linezolid",
    "zyvox": "linezolid",
    "daptomycin": "daptomycin",
    "dapto": "daptomycin",
    "trimethoprim/sulfamethoxazole": "tmp_smx",
    "trimethoprim sulfamethoxazole": "tmp_smx",
    "trimethoprim-sulfamethoxazole": "tmp_smx",
    "tmp smx": "tmp_smx",
    "tmp-smx": "tmp_smx",
    "tmp/smx": "tmp_smx",
    "bactrim": "tmp_smx",
    "ciprofloxacin": "ciprofloxacin",
    "cipro": "ciprofloxacin",
    "levofloxacin": "levofloxacin",
    "levaquin": "levofloxacin",
    "moxifloxacin": "moxifloxacin",
    "doxycycline": "doxycycline",
    "doxy": "doxycycline",
    "clindamycin": "clindamycin",
    "cleocin": "clindamycin",
    "metronidazole": "metronidazole",
    "flagyl": "metronidazole",
    "nitrofurantoin": "nitrofurantoin",
    "macrobid": "nitrofurantoin",
    "fosfomycin": "fosfomycin",
    "monurol": "fosfomycin",
    "rifampin": "rifampin",
}

REACTION_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\b(?:sjs|stevens[- ]johnson|ten|dress|agesp|blistering|mucosal)\b", "scar", "delayed"),
    (r"\b(?:hepatitis|liver injury|nephritis|kidney injury)\b", "organ_injury", "delayed"),
    (r"\b(?:serum sickness)\b", "serum_sickness_like", "delayed"),
    (r"\b(?:hemolytic anemia)\b", "hemolytic_anemia", "delayed"),
    (r"\b(?:anaphylaxis|hypotension|bronchospasm)\b", "anaphylaxis", "immediate"),
    (r"\b(?:angioedema)\b", "angioedema", "immediate"),
    (r"\b(?:hives|urticaria)\b", "urticaria", "immediate"),
    (r"\b(?:maculopapular|morbilliform|delayed rash|rash)\b", "benign_delayed_rash", "delayed"),
    (r"\b(?:nausea|vomiting|diarrhea|gi upset|stomach upset)\b", "isolated_gi", "unknown"),
    (r"\b(?:headache)\b", "headache", "unknown"),
    (r"\b(?:family history)\b", "family_history_only", "unknown"),
]

ALLERGY_KEYWORDS = ("allergy", "allergic", "reaction", "anaphylaxis", "hives", "urticaria", "rash", "angioedema")

REFERENCE_LIST = [
    ReferenceEntry(
        context="Practice parameter: beta-lactam allergy evaluation and cross-reactivity",
        citation="Khan et al. Drug allergy: A 2022 practice parameter update. J Allergy Clin Immunol. 2022.",
        url="https://doi.org/10.1016/j.jaci.2022.08.028",
    ),
    ReferenceEntry(
        context="Practice update: practical incorporation of beta-lactam cross-reactivity guidance",
        citation="Ramsey et al. Drug allergy practice parameter updates to incorporate into your clinical practice. J Allergy Clin Immunol Pract. 2023.",
        url="https://doi.org/10.1016/j.jaip.2022.12.002",
    ),
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/+\-.,;: ]+", " ", text.lower())).strip()


def normalize_antibiotic_name(text: str) -> str | None:
    key = _normalize_text(text)
    if not key:
        return None
    if key in AGENT_ALIASES:
        return AGENT_ALIASES[key]
    for alias, canonical in sorted(AGENT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if key == alias:
            return canonical
    return None


def _display_name(agent_id: str) -> str:
    info = AGENT_INFO.get(agent_id)
    return info.name if info is not None else agent_id.replace("_", " ").title()


def _reaction_bucket(entry: AntibioticAllergyEntry) -> str:
    if entry.reaction_type in {"intolerance", "isolated_gi", "headache", "family_history_only"}:
        return "nonallergic"
    if entry.reaction_type == "benign_delayed_rash":
        return "low"
    if entry.reaction_type in {"urticaria", "angioedema"}:
        return "immediate"
    if entry.reaction_type == "anaphylaxis":
        return "anaphylaxis"
    if entry.reaction_type in {"scar", "organ_injury", "serum_sickness_like", "hemolytic_anemia"}:
        return "severe_delayed"
    return "unknown"


def _same_side_chain(agent_a: str, agent_b: str) -> bool:
    info_a = AGENT_INFO.get(agent_a)
    info_b = AGENT_INFO.get(agent_b)
    if info_a is None or info_b is None:
        return False
    return bool(info_a.side_chain_group and info_a.side_chain_group == info_b.side_chain_group)


def _assess_beta_lactam_candidate(candidate_id: str, culprit_id: str, bucket: str) -> Tuple[str, str]:
    candidate = AGENT_INFO.get(candidate_id)
    culprit = AGENT_INFO.get(culprit_id)
    if candidate is None or culprit is None:
        return "caution", "I could not fully map the beta-lactam cross-reactivity for this pair, so I would verify the culprit and side-chain relationship before relying on it."

    if bucket == "severe_delayed":
        return "avoid", "Because the reported reaction sounds like a severe delayed immune phenotype, I would avoid beta-lactams related to the culprit until allergy expertise clarifies what is truly safe."

    if candidate.beta_subclass == "carbapenem" and culprit.beta_subclass in {"penicillin", "cephalosporin"}:
        return "preferred", "Carbapenems are usually still reasonable despite penicillin or cephalosporin allergy histories when the prior reaction was not a severe delayed syndrome."

    if candidate.beta_subclass == "monobactam":
        if culprit.side_chain_group == "aztreonam_group" or culprit_id == "ceftazidime":
            return "caution", "Aztreonam is the classic beta-lactam exception because it can cross-react with ceftazidime."
        return "preferred", "Aztreonam is usually safe despite other beta-lactam allergy histories except for the ceftazidime cross-reactivity exception."

    if culprit.beta_subclass == "monobactam":
        if candidate.side_chain_group == "aztreonam_group" or candidate_id == "ceftazidime":
            return "caution", "Ceftazidime shares the key side chain cross-reactivity exception with aztreonam."
        return "preferred", "A remote aztreonam allergy does not usually block unrelated beta-lactams."

    if culprit.beta_subclass == "penicillin" and candidate.beta_subclass == "cephalosporin":
        if _same_side_chain(candidate_id, culprit_id):
            if bucket in {"immediate", "anaphylaxis", "unknown"}:
                return "avoid", "This cephalosporin shares a relevant side-chain pattern with the culprit penicillin, so I would avoid routine use after an immediate or unclear allergic history."
            return "caution", "This cephalosporin shares a side-chain pattern with the culprit penicillin, so I would only use it with added caution."
        return "preferred", "A cephalosporin with a dissimilar side chain is usually still reasonable despite penicillin allergy, including many immediate penicillin reactions."

    if culprit.beta_subclass == "cephalosporin" and candidate.beta_subclass == "penicillin":
        if _same_side_chain(candidate_id, culprit_id):
            if bucket in {"immediate", "anaphylaxis", "unknown"}:
                return "avoid", "This penicillin shares a relevant side-chain pattern with the culprit cephalosporin, so I would not treat it as a routine substitute after an immediate or unclear history."
            return "caution", "This penicillin shares a side-chain pattern with the culprit cephalosporin, so I would use extra caution."
        if bucket == "anaphylaxis":
            return "caution", "Penicillin can still be possible after cephalosporin allergy, but I would not call it routine after a cephalosporin anaphylaxis history."
        return "preferred", "Penicillin is often still usable after a non-severe cephalosporin allergy history when there is no side-chain concern."

    if culprit.beta_subclass == candidate.beta_subclass == "cephalosporin":
        if _same_side_chain(candidate_id, culprit_id):
            if bucket in {"immediate", "anaphylaxis", "unknown"}:
                return "avoid", "A cephalosporin with a similar side chain is not my first substitute after an immediate or unclear cephalosporin allergy history."
            return "caution", "A cephalosporin with a similar side chain deserves extra caution."
        if bucket in {"immediate", "anaphylaxis"}:
            return "caution", "A dissimilar cephalosporin may still be possible, but I would not call it routine after a clearly immediate cephalosporin reaction."
        return "preferred", "A cephalosporin with a dissimilar side chain is often still usable after a low-risk cephalosporin allergy history."

    if culprit.beta_subclass == candidate.beta_subclass == "penicillin":
        if bucket == "nonallergic":
            return "preferred", "This history sounds more like an intolerance or side effect than a true penicillin allergy."
        if bucket == "low":
            return "caution", "Another penicillin may still be possible, but I would not treat the same beta-lactam subclass as automatically safe after a true rash history."
        return "avoid", "Another penicillin is not my preferred substitute after an immediate or unclear penicillin allergy history."

    return "caution", "This beta-lactam pair is not a classic high-confidence substitution rule, so I would confirm the exact culprit and reaction details before relying on it."


def _pairwise_assessment(candidate_id: str, entry: AntibioticAllergyEntry) -> Tuple[int, AntibioticAllergyRecommendation]:
    culprit_id = normalize_antibiotic_name(entry.reported_agent)
    bucket = _reaction_bucket(entry)
    candidate_name = _display_name(candidate_id)
    culprit_name = _display_name(culprit_id) if culprit_id else entry.reported_agent
    base_trigger = [f"{culprit_name}: {entry.reaction_type}"]

    if culprit_id is None:
        recommendation = "caution" if bucket != "nonallergic" else "preferred"
        summary = f"{candidate_name} needs context because I could not confidently normalize the reported culprit."
        rationale = "The allergy label is not mapped to a specific agent in the API catalog, so I would verify the exact antibiotic before using it to drive selection."
        score = 1 if recommendation == "caution" else 0
        return score, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate_name,
            recommendation=recommendation,
            summary=summary,
            rationale=rationale,
            triggeredBy=base_trigger,
        )

    culprit = AGENT_INFO.get(culprit_id)
    candidate = AGENT_INFO.get(candidate_id)
    if candidate is None:
        return 1, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate_name,
            recommendation="caution",
            summary=f"{candidate_name} is not yet fully mapped in the allergy engine.",
            rationale="I could not map the candidate agent to a complete cross-reactivity profile yet.",
            triggeredBy=base_trigger,
        )

    if bucket == "nonallergic":
        return 0, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate.name,
            recommendation="preferred",
            summary=f"{candidate_name} remains reasonable because the reported history sounds more like a side effect than a true immune allergy.",
            rationale="Isolated GI upset, headache, or family history alone should not usually block the best antibiotic choice.",
            triggeredBy=base_trigger,
        )

    if candidate_id == culprit_id:
        if entry.reaction_type == "scar":
            return 2, AntibioticAllergyRecommendation(
                agent=candidate_name,
                normalizedAgent=candidate.name,
                recommendation="avoid",
                summary=f"{candidate_name} should not be used again after a reported severe cutaneous reaction such as SJS/TEN or DRESS.",
                rationale="A culprit antibiotic associated with SJS/TEN, DRESS, or a similarly severe delayed immune reaction should not be re-exposed as routine care.",
                triggeredBy=base_trigger,
            )
        recommendation = "avoid" if bucket in {"immediate", "anaphylaxis", "severe_delayed", "unknown"} else "caution"
        score = 2 if recommendation == "avoid" else 1
        return score, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate.name,
            recommendation=recommendation,
            summary=f"{candidate_name} is not my preferred choice against a reported allergy to the same drug.",
            rationale="If the same antibiotic truly caused the reaction, I would not rely on it until the label is clarified or delabeled.",
            triggeredBy=base_trigger,
        )

    if candidate.drug_class == "beta_lactam" and culprit is not None and culprit.drug_class == "beta_lactam":
        recommendation, rationale = _assess_beta_lactam_candidate(candidate_id, culprit_id, bucket)
        score = {"preferred": 0, "caution": 1, "avoid": 2}[recommendation]
        return score, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate.name,
            recommendation=recommendation,
            summary=f"{candidate_name} is {recommendation.replace('_', ' ')} in the setting of reported {culprit_name} allergy.",
            rationale=rationale,
            triggeredBy=base_trigger,
        )

    if culprit is not None and candidate.drug_class == culprit.drug_class:
        recommendation = "caution" if bucket in {"low", "immediate", "unknown"} else "avoid"
        score = 1 if recommendation == "caution" else 2
        return score, AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate.name,
            recommendation=recommendation,
            summary=f"{candidate_name} is not automatically interchangeable with the culprit because it sits in the same non-beta-lactam class.",
            rationale="Cross-reactivity within non-beta-lactam antibiotic classes is less standardized than beta-lactam side-chain rules, so I would be cautious when the same class caused the reaction.",
            triggeredBy=base_trigger,
        )

    return 0, AntibioticAllergyRecommendation(
        agent=candidate_name,
        normalizedAgent=candidate.name,
        recommendation="preferred",
        summary=f"{candidate_name} is not directly blocked by the reported {culprit_name} allergy history.",
        rationale="This candidate is outside the culprit drug class or lacks a meaningful mapped cross-reactivity signal in the current rule set.",
        triggeredBy=base_trigger,
    )


def _collapse_recommendations(
    candidate_id: str,
    entries: List[AntibioticAllergyEntry],
) -> AntibioticAllergyRecommendation:
    best_score = -1
    selected: AntibioticAllergyRecommendation | None = None
    combined_triggers: List[str] = []
    for entry in entries:
        score, recommendation = _pairwise_assessment(candidate_id, entry)
        combined_triggers.extend(recommendation.triggered_by)
        if selected is None or score > best_score:
            best_score = score
            selected = recommendation
    if selected is None:
        candidate_name = _display_name(candidate_id)
        return AntibioticAllergyRecommendation(
            agent=candidate_name,
            normalizedAgent=candidate_name,
            recommendation="preferred",
            summary=f"{candidate_name} is not blocked by the current allergy list.",
            rationale="No mapped culprit in the current request conflicts with this candidate.",
            triggeredBy=[],
        )
    selected.triggered_by = list(dict.fromkeys(combined_triggers))
    return selected


def _dedupe_candidates(candidates: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in candidates:
        canonical = normalize_antibiotic_name(item)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def _is_tolerated_context(text_norm: str, alias: str, start: int, end: int) -> bool:
    before = text_norm[max(0, start - 40) : start]
    after = text_norm[end : min(len(text_norm), end + 40)]
    joined = before + alias + after
    patterns = (
        rf"(?:tolerated|has tolerated|previously tolerated|received|has received|took|has taken)\s+{re.escape(alias)}\b",
        rf"{re.escape(alias)}\s+(?:without issue|without issues|without a reaction|without problems|previously tolerated|later tolerated)\b",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def _tolerance_supports_lower_risk(tolerated_ids: List[str], entries: List[AntibioticAllergyEntry]) -> bool:
    if not tolerated_ids or not entries:
        return False
    tolerated_infos = [AGENT_INFO.get(agent_id) for agent_id in tolerated_ids]
    for entry in entries:
        culprit_id = normalize_antibiotic_name(entry.reported_agent)
        if culprit_id is None:
            continue
        culprit_info = AGENT_INFO.get(culprit_id)
        if culprit_info is None:
            continue
        if any(info is not None and info.drug_class == culprit_info.drug_class for info in tolerated_infos):
            return True
    return False


def analyze_antibiotic_allergy(req: AntibioticAllergyAnalyzeRequest) -> AntibioticAllergyAnalyzeResponse:
    candidates = _dedupe_candidates(req.candidate_agents)
    tolerated = _dedupe_candidates(req.tolerated_agents)
    entries = req.allergy_entries
    recommendations = [_collapse_recommendations(candidate_id, entries) for candidate_id in candidates]
    recommendations.sort(key=lambda item: {"avoid": 2, "caution": 1, "preferred": 0}[item.recommendation], reverse=True)

    warnings: List[str] = []
    general_advice: List[str] = []
    delabeling: List[str] = []
    follow_up: List[AntibioticAllergyFollowUpQuestion] = []
    overall_risk = "low"

    buckets = [_reaction_bucket(entry) for entry in entries]
    if any(bucket == "severe_delayed" for bucket in buckets):
        overall_risk = "severe_delayed"
        general_advice.append(
            "A severe delayed history such as SJS/TEN, DRESS, serum-sickness-like reaction, immune hemolysis, or organ injury should usually block casual beta-lactam reuse until allergy expertise clarifies the safe options."
        )
        if any(entry.reaction_type == "scar" for entry in entries):
            general_advice.append(
                "If a specific antibiotic caused SJS/TEN, that culprit should not be used again."
            )
    elif any(bucket == "anaphylaxis" for bucket in buckets):
        overall_risk = "high_immediate"
        general_advice.append(
            "An anaphylaxis history does not block every beta-lactam, but it does make same-drug and same-side-chain substitutions unsafe as routine choices."
        )
    elif any(bucket in {"immediate", "unknown"} for bucket in buckets):
        overall_risk = "intermediate"
    else:
        overall_risk = "low"

    if any(bucket == "nonallergic" for bucket in buckets):
        general_advice.append(
            "Minor side effects such as isolated nausea, diarrhea, or headache do not by themselves preclude reuse of the same or a related antibiotic."
        )
    if any(bucket == "nonallergic" for bucket in buckets):
        delabeling.append(
            "At least one listed reaction reads more like intolerance or side effect than a true allergy, so delabeling or careful clarification could reopen better antibiotic options."
        )

    if tolerated:
        general_advice.append(
            f"Tolerance of {', '.join(_display_name(agent_id) for agent_id in tolerated[:3])} after the original label lowers concern for a broad class allergy and should be weighed heavily when choosing therapy."
        )
        delabeling.append(
            "If the patient has already tolerated a related antibiotic since the original label, that history can strongly support de-risking or delabeling the allergy."
        )
        if overall_risk == "intermediate" and all(bucket in {"unknown", "nonallergic"} for bucket in buckets) and _tolerance_supports_lower_risk(tolerated, entries):
            overall_risk = "low"

    if any(entry.reaction_type == "unknown" for entry in entries):
        follow_up.append(
            AntibioticAllergyFollowUpQuestion(
                id="clarify_reaction",
                prompt="What exactly happened with the allergy label, such as hives, anaphylaxis, a delayed rash, or only GI upset?",
                reason="The exact reaction phenotype changes how much cross-reactivity matters."
            )
        )
    if any("penicillin" in _normalize_text(entry.reported_agent) and entry.reaction_type == "unknown" for entry in entries):
        follow_up.append(
            AntibioticAllergyFollowUpQuestion(
                id="clarify_penicillin_culprit",
                prompt="Was the culprit penicillin, amoxicillin, ampicillin, or an unknown childhood label?",
                reason="The exact culprit helps determine whether side-chain cross-reactivity is relevant."
            )
        )
    if any(entry.reaction_type == "unknown" for entry in entries):
        follow_up.append(
            AntibioticAllergyFollowUpQuestion(
                id="clarify_tolerated_related_agents",
                prompt="Has the patient taken any similar antibiotics since then without a reaction, such as another penicillin, cephalosporin, or carbapenem?",
                reason="Tolerance of related agents can substantially lower concern that the recorded label reflects a true ongoing allergy."
            )
        )

    if not candidates:
        warnings.append(
            "No candidate antibiotics were supplied, so this response focuses on allergy phenotype and cross-reactivity rather than selecting between therapies."
        )
        general_advice.append(
            "This API becomes most useful once MechID or the clinician supplies a short candidate list such as cefazolin, ceftriaxone, meropenem, linezolid, or TMP-SMX."
        )

    if req.infection_context:
        general_advice.append(
            f"Keep the infection context in view: {req.infection_context}. Allergy history should guide safe substitutions, but it should not force a weaker antibiotic when a non-cross-reactive preferred option still exists."
        )

    summary = "I did not receive enough allergy detail to assess antibiotic compatibility yet."
    if entries and candidates:
        avoids = [item.agent for item in recommendations if item.recommendation == "avoid"]
        cautions = [item.agent for item in recommendations if item.recommendation == "caution"]
        preferred = [item.agent for item in recommendations if item.recommendation == "preferred"]
        parts: List[str] = []
        if preferred:
            parts.append(f"Most usable options here are {', '.join(preferred[:3])}")
        if cautions:
            parts.append(f"use extra caution with {', '.join(cautions[:3])}")
        if avoids:
            parts.append(f"avoid routine use of {', '.join(avoids[:3])}")
        if parts:
            summary = "; ".join(parts) + "."
    elif entries:
        summary = (
            "I can stratify the reported allergy phenotype now, but I still need the candidate antibiotics or a MechID-generated treatment shortlist to tell you which agent is best."
        )

    return AntibioticAllergyAnalyzeResponse(
        summary=summary,
        overallRisk=overall_risk,
        recommendations=recommendations,
        generalAdvice=list(dict.fromkeys(general_advice)),
        delabelingOpportunities=list(dict.fromkeys(delabeling)),
        followUpQuestions=follow_up,
        warnings=warnings,
        references=REFERENCE_LIST,
    )


def _detect_reaction_from_window(window: str) -> Tuple[str, str]:
    for pattern, reaction_type, timing in REACTION_PATTERNS:
        if re.search(pattern, window):
            return reaction_type, timing
    return "unknown", "unknown"


def _is_allergy_context(text_norm: str, alias: str, start: int, end: int) -> bool:
    before = text_norm[max(0, start - 40) : start]
    after = text_norm[end : min(len(text_norm), end + 40)]
    joined = before + alias + after
    patterns = (
        rf"(?:allergic to|allergy to|reaction to|anaphylaxis to|hives to|urticaria to|rash to)\s+{re.escape(alias)}",
        rf"{re.escape(alias)}\s+(?:allergy|reaction)\b",
        rf"{re.escape(alias)}\s+(?:anaphylaxis|angioedema|hives|urticaria|rash|dress|sjs|ten)\b",
        rf"{re.escape(alias)}\s+(?:caused|led to|triggered|associated with|associated|related to)\b",
        rf"(?:anaphylaxis|angioedema|hives|urticaria|rash|dress|sjs|ten)\s+(?:with|from)\s+{re.escape(alias)}",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def _is_candidate_context(text_norm: str, alias: str, start: int, end: int) -> bool:
    before = text_norm[max(0, start - 40) : start]
    after = text_norm[end : min(len(text_norm), end + 40)]
    joined = before + alias + after
    patterns = (
        rf"(?:can i use|could i use|can we use|should i use|would you use|what about|use)\s+{re.escape(alias)}\b",
        rf"{re.escape(alias)}\s+(?:again|instead)\b",
    )
    return any(re.search(pattern, joined) for pattern in patterns)


def parse_antibiotic_allergy_text(req: AntibioticAllergyTextAnalyzeRequest) -> AntibioticAllergyTextAnalyzeResponse:
    text = req.text.strip()
    text_norm = _normalize_text(text)
    candidate_agents: List[str] = []
    tolerated_agents: List[str] = []
    allergy_entries: List[AntibioticAllergyEntry] = []
    warnings: List[str] = []

    matches: List[Tuple[int, int, str, str]] = []
    for alias, canonical in sorted(AGENT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text_norm):
            matches.append((match.start(), match.end(), alias, canonical))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    used_spans: List[Tuple[int, int]] = []
    seen_allergy_keys: set[Tuple[str, str, str]] = set()
    seen_candidates: set[str] = set()
    seen_tolerated: set[str] = set()

    for start, end, alias, canonical in matches:
        if any(start < span_end and end > span_start for span_start, span_end in used_spans):
            continue
        used_spans.append((start, end))
        window = text_norm[max(0, start - 40) : min(len(text_norm), end + 40)]
        allergy_context = _is_allergy_context(text_norm, alias, start, end)
        candidate_context = _is_candidate_context(text_norm, alias, start, end)
        tolerated_context = _is_tolerated_context(text_norm, alias, start, end)
        if allergy_context:
            reaction_type, timing = _detect_reaction_from_window(window)
            key = (canonical, reaction_type, timing)
            if key not in seen_allergy_keys:
                seen_allergy_keys.add(key)
                allergy_entries.append(
                    AntibioticAllergyEntry(
                        reportedAgent=_display_name(canonical),
                        reactionType=reaction_type,
                        timing=timing,
                    )
                )
        if tolerated_context and canonical not in seen_tolerated:
            seen_tolerated.add(canonical)
            tolerated_agents.append(_display_name(canonical))
        if candidate_context or (not allergy_context and not tolerated_context):
            if canonical not in seen_candidates:
                seen_candidates.add(canonical)
                candidate_agents.append(_display_name(canonical))

    if "best antibiotic" in text_norm and not candidate_agents:
        warnings.append("I did not detect a candidate antibiotic list, so the analysis will focus on compatibility rather than choosing between regimens.")

    parsed = AntibioticAllergyTextParsedRequest(
        candidateAgents=candidate_agents,
        toleratedAgents=tolerated_agents,
        allergyEntries=allergy_entries,
        infectionContext=text,
    )
    analysis = analyze_antibiotic_allergy(
        AntibioticAllergyAnalyzeRequest(
            candidateAgents=candidate_agents,
            toleratedAgents=tolerated_agents,
            allergyEntries=allergy_entries,
            infectionContext=text,
        )
    )
    requires_confirmation = not allergy_entries
    if requires_confirmation:
        warnings.append("I did not confidently detect a specific antibiotic allergy label from the text.")

    return AntibioticAllergyTextAnalyzeResponse(
        text=text,
        parsedRequest=parsed,
        warnings=warnings,
        requiresConfirmation=requires_confirmation,
        analysis=analysis,
    )
