from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .immunoid_regimens import IMMUNOID_REGIMENS
from .immunoid_source_catalog import IMMUNOID_SOURCE_AGENTS


@dataclass(frozen=True)
class SourceDef:
    citation: str
    url: str


@dataclass(frozen=True)
class AgentDef:
    id: str
    name: str
    drug_class: str
    risk_tags: tuple[str, ...]
    groups: tuple[str, ...]


def _agent(
    agent_id: str,
    name: str,
    drug_class: str,
    risk_tags: tuple[str, ...],
    *groups: str,
) -> AgentDef:
    return AgentDef(
        id=agent_id,
        name=name,
        drug_class=drug_class,
        risk_tags=risk_tags,
        groups=groups,
    )


SOURCES: Dict[str, SourceDef] = {
    "asco_idsa_antimicrobial": SourceDef(
        citation=(
            "Taplitz RA, Kennedy EB, Bow EJ, et al. Antimicrobial Prophylaxis for Adult "
            "Patients With Cancer-Related Immunosuppression: ASCO and IDSA Clinical "
            "Practice Guideline Update. J Clin Oncol. 2018;36(30):3043-3054."
        ),
        url="https://doi.org/10.1200/JCO.18.00374",
    ),
    "eular_screening": SourceDef(
        citation=(
            "Fragoulis GE, Nikiphorou E, Larsen J, et al. 2022 EULAR recommendations "
            "for screening and prophylaxis of chronic and opportunistic infections in "
            "adults with autoimmune inflammatory rheumatic diseases. Ann Rheum Dis. "
            "2023;82(6):742-753."
        ),
        url="https://doi.org/10.1136/ard-2022-223335",
    ),
    "aga_hbv": SourceDef(
        citation=(
            "AGA Clinical Practice Guideline on the Prevention and Treatment of "
            "Hepatitis B Virus Reactivation in At-Risk Individuals. Gastroenterology. 2025."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/39863345/",
    ),
    "cdc_hbv_testing": SourceDef(
        citation="CDC. Clinical Testing and Diagnosis for Hepatitis B. Updated January 31, 2025.",
        url="https://www.cdc.gov/hepatitis-b/hcp/diagnosis-testing/index.html",
    ),
    "cdc_tb_ltbi": SourceDef(
        citation="CDC. Latent Tuberculosis Infection: A Guide for Primary Health Care Providers. Updated 2024.",
        url="https://www.cdc.gov/tb/hcp/clinical-overview/latent-tuberculosis-infection.html",
    ),
    "cdc_tb_risk_country": SourceDef(
        citation="CDC. Risk Factors: People Born in or Who Travel to Places Where TB is Common. Updated January 17, 2025.",
        url="https://www.cdc.gov/tb/risk-factors/country.html",
    ),
    "cdc_strongyloides": SourceDef(
        citation="CDC. Clinical Care of Strongyloides. Updated July 3, 2024.",
        url="https://www.cdc.gov/strongyloides/hcp/clinical-care/index.html",
    ),
    "strongyloides_review": SourceDef(
        citation=(
            "Wiener HJ, Strymish J, Camargo JF. A Practical Approach to Screening for "
            "Strongyloides stercoralis. Open Forum Infect Dis. 2023;10(5):ofad150."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/37222401/",
    ),
    "cdc_cocci": SourceDef(
        citation="CDC. Testing Algorithm for Coccidioidomycosis. Updated May 10, 2024.",
        url="https://www.cdc.gov/valley-fever/hcp/testing-algorithm/index.html",
    ),
    "cdc_histoplasma_maps": SourceDef(
        citation="CDC. Areas with Histoplasmosis. Updated April 24, 2024.",
        url="https://www.cdc.gov/histoplasmosis/data-research/maps/index.html",
    ),
    "cdc_histoplasma_environment": SourceDef(
        citation="CDC/NIOSH. Histoplasma in the Environment: An Overview. Updated December 17, 2024.",
        url="https://www.cdc.gov/niosh/histoplasmosis/about/environment.html",
    ),
    "mmwr_eculizumab": SourceDef(
        citation=(
            "McNamara LA, Topaz N, Wang X, et al. High Risk for Invasive Meningococcal "
            "Disease Among Patients Receiving Eculizumab Despite Receipt of "
            "Meningococcal Vaccine. MMWR Morb Mortal Wkly Rep. 2017;66(27):734-737."
        ),
        url="https://doi.org/10.15585/mmwr.mm6627e1",
    ),
}


AGENTS: Dict[str, AgentDef] = {
    "prednisone_20": _agent(
        "prednisone_20",
        "Prednisone >= 20 mg/day",
        "Corticosteroid (systemic)",
        ("PJP", "TB (reactivation)", "Strongyloides", "VZV/HSV"),
        "high_dose_steroid",
        "pjp_context",
        "tb_context",
        "strongy_context",
        "cocci_context",
    ),
    "prednisolone_20": _agent(
        "prednisolone_20",
        "Prednisolone >= 20 mg/day",
        "Corticosteroid (systemic)",
        ("PJP", "TB (reactivation)", "Strongyloides", "VZV/HSV"),
        "high_dose_steroid",
        "pjp_context",
        "tb_context",
        "strongy_context",
        "cocci_context",
    ),
    "methylpred_16": _agent(
        "methylpred_16",
        "Methylprednisolone >= 16 mg/day",
        "Corticosteroid (systemic)",
        ("PJP", "TB (reactivation)", "Strongyloides", "VZV/HSV"),
        "high_dose_steroid",
        "pjp_context",
        "tb_context",
        "strongy_context",
        "cocci_context",
    ),
    "hydrocortisone_80": _agent(
        "hydrocortisone_80",
        "Hydrocortisone >= 80 mg/day",
        "Corticosteroid (systemic)",
        ("PJP", "TB (reactivation)", "Strongyloides", "VZV/HSV"),
        "high_dose_steroid",
        "pjp_context",
        "tb_context",
        "strongy_context",
        "cocci_context",
    ),
    "dexamethasone_3": _agent(
        "dexamethasone_3",
        "Dexamethasone >= 3 mg/day",
        "Corticosteroid (systemic)",
        ("PJP", "TB (reactivation)", "Strongyloides", "VZV/HSV"),
        "high_dose_steroid",
        "pjp_context",
        "tb_context",
        "strongy_context",
        "cocci_context",
    ),
    "infliximab": _agent(
        "infliximab",
        "Infliximab",
        "TNF blocker",
        ("TB (reactivation)", "Endemic fungi", "Listeria"),
        "tb_screen",
        "cocci_context",
        "histo_context",
    ),
    "etanercept": _agent(
        "etanercept",
        "Etanercept",
        "TNF blocker",
        ("TB (reactivation)", "Endemic fungi", "Listeria"),
        "tb_screen",
        "cocci_context",
        "histo_context",
    ),
    "adalimumab": _agent(
        "adalimumab",
        "Adalimumab",
        "TNF blocker",
        ("TB (reactivation)", "Endemic fungi", "Listeria"),
        "tb_screen",
        "cocci_context",
        "histo_context",
    ),
    "certolizumab": _agent(
        "certolizumab",
        "Certolizumab pegol",
        "TNF blocker",
        ("TB (reactivation)", "Endemic fungi", "Listeria"),
        "tb_screen",
        "cocci_context",
        "histo_context",
    ),
    "golimumab": _agent(
        "golimumab",
        "Golimumab",
        "TNF blocker",
        ("TB (reactivation)", "Endemic fungi", "Listeria"),
        "tb_screen",
        "cocci_context",
        "histo_context",
    ),
    "tofacitinib": _agent(
        "tofacitinib",
        "Tofacitinib",
        "JAK inhibitor",
        ("TB (reactivation)", "VZV/HSV", "PJP"),
        "tb_screen",
        "cocci_context",
        "histo_context",
        "pjp_context",
    ),
    "baricitinib": _agent(
        "baricitinib",
        "Baricitinib",
        "JAK inhibitor",
        ("TB (reactivation)", "VZV/HSV", "PJP"),
        "tb_screen",
        "cocci_context",
        "histo_context",
        "pjp_context",
    ),
    "upadacitinib": _agent(
        "upadacitinib",
        "Upadacitinib",
        "JAK inhibitor",
        ("TB (reactivation)", "VZV/HSV", "PJP"),
        "tb_screen",
        "cocci_context",
        "histo_context",
        "pjp_context",
    ),
    "ruxolitinib": _agent(
        "ruxolitinib",
        "Ruxolitinib",
        "JAK inhibitor",
        ("TB (reactivation)", "VZV/HSV", "PJP"),
        "tb_screen",
        "cocci_context",
        "histo_context",
        "pjp_context",
    ),
    "rituximab": _agent(
        "rituximab",
        "Rituximab",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "PJP", "Encapsulated bacteria"),
        "anti_cd20",
        "pjp_context",
    ),
    "obinutuzumab": _agent(
        "obinutuzumab",
        "Obinutuzumab",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "PJP", "Encapsulated bacteria"),
        "anti_cd20",
        "pjp_context",
    ),
    "ocrelizumab": _agent(
        "ocrelizumab",
        "Ocrelizumab",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "Encapsulated bacteria"),
        "anti_cd20",
    ),
    "ofatumumab": _agent(
        "ofatumumab",
        "Ofatumumab",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "Encapsulated bacteria"),
        "anti_cd20",
    ),
    "ofatumumab_kesimpta": _agent(
        "ofatumumab_kesimpta",
        "Ofatumumab (Kesimpta)",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "Encapsulated bacteria"),
        "anti_cd20",
    ),
    "ublituximab": _agent(
        "ublituximab",
        "Ublituximab",
        "Monoclonal antibody (anti-CD20)",
        ("HBV reactivation", "Encapsulated bacteria"),
        "anti_cd20",
    ),
    "fludarabine": _agent(
        "fludarabine",
        "Fludarabine",
        "Antimetabolite (purine analog)",
        ("PJP", "VZV/HSV", "CMV"),
        "pjp_high_risk",
    ),
    "cladribine": _agent(
        "cladribine",
        "Cladribine",
        "Antimetabolite (purine analog)",
        ("PJP", "VZV/HSV", "CMV"),
        "pjp_high_risk",
    ),
    "pentostatin": _agent(
        "pentostatin",
        "Pentostatin",
        "Antimetabolite (purine analog)",
        ("PJP", "VZV/HSV", "CMV"),
        "pjp_high_risk",
    ),
    "antithymocyte_globulin": _agent(
        "antithymocyte_globulin",
        "Anti-thymocyte globulin (ATG)",
        "Lymphocyte-depleting antibody",
        ("PJP", "CMV", "Strongyloides"),
        "pjp_high_risk",
        "strongy_context",
    ),
    "mycophenolate_mofetil": _agent(
        "mycophenolate_mofetil",
        "Mycophenolate mofetil",
        "Transplant / immunosuppressive (antimetabolite)",
        ("PJP", "CMV"),
        "strongy_context",
    ),
    "tacrolimus": _agent(
        "tacrolimus",
        "Tacrolimus",
        "Transplant / immunosuppressive (calcineurin inhibitor)",
        ("PJP", "Nocardia"),
        "strongy_context",
    ),
    "cyclophosphamide": _agent(
        "cyclophosphamide",
        "Cyclophosphamide",
        "Alkylating agent",
        ("Neutropenia-related infections", "Bacterial (general)"),
        "cytotoxic_chemo",
    ),
    "bendamustine": _agent(
        "bendamustine",
        "Bendamustine",
        "Alkylating agent",
        ("Neutropenia-related infections", "PJP", "Bacterial (general)"),
        "cytotoxic_chemo",
        "pjp_context",
    ),
    "cytarabine": _agent(
        "cytarabine",
        "Cytarabine",
        "Antimetabolite",
        ("Neutropenia-related infections", "Invasive mold (Aspergillus)"),
        "cytotoxic_chemo",
    ),
    "doxorubicin": _agent(
        "doxorubicin",
        "Doxorubicin",
        "Cytotoxic chemotherapy (anthracycline)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "daunorubicin": _agent(
        "daunorubicin",
        "Daunorubicin",
        "Cytotoxic chemotherapy (anthracycline)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "idarubicin": _agent(
        "idarubicin",
        "Idarubicin",
        "Cytotoxic chemotherapy (anthracycline)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "epirubicin": _agent(
        "epirubicin",
        "Epirubicin",
        "Cytotoxic chemotherapy (anthracycline)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "etoposide": _agent(
        "etoposide",
        "Etoposide",
        "Cytotoxic chemotherapy (topoisomerase II inhibitor)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "teniposide": _agent(
        "teniposide",
        "Teniposide",
        "Cytotoxic chemotherapy (topoisomerase II inhibitor)",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "melphalan": _agent(
        "melphalan",
        "Melphalan",
        "Alkylating agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "busulfan": _agent(
        "busulfan",
        "Busulfan",
        "Alkylating agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "thiotepa": _agent(
        "thiotepa",
        "Thiotepa",
        "Alkylating agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "temozolomide": _agent(
        "temozolomide",
        "Temozolomide",
        "Alkylating agent",
        ("Neutropenia-related infections", "PJP"),
        "cytotoxic_chemo",
        "pjp_context",
    ),
    "cisplatin": _agent(
        "cisplatin",
        "Cisplatin",
        "Alkylating-like platinum agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "carboplatin": _agent(
        "carboplatin",
        "Carboplatin",
        "Alkylating-like platinum agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "oxaliplatin": _agent(
        "oxaliplatin",
        "Oxaliplatin",
        "Alkylating-like platinum agent",
        ("Neutropenia-related infections",),
        "cytotoxic_chemo",
    ),
    "eculizumab": _agent(
        "eculizumab",
        "Eculizumab",
        "Complement inhibitor (C5)",
        ("Encapsulated bacteria",),
        "complement_c5",
    ),
    "ravulizumab": _agent(
        "ravulizumab",
        "Ravulizumab",
        "Complement inhibitor (C5)",
        ("Encapsulated bacteria",),
        "complement_c5",
    ),
}

for agent_id, source_entry in IMMUNOID_SOURCE_AGENTS.items():
    existing = AGENTS.get(agent_id)
    groups = existing.groups if existing is not None else tuple(source_entry.get("groups", ()))
    AGENTS[agent_id] = _agent(
        agent_id,
        source_entry["name"],
        source_entry["drug_class"],
        tuple(source_entry.get("risk_tags", ())),
        *groups,
    )


def _reference_entry(context: str, source_key: str) -> Dict[str, str]:
    source = SOURCES[source_key]
    return {"context": context, "citation": source.citation, "url": source.url}


def _exposure_value(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "unknown"
    return str(value)


def list_immunoid_agents() -> List[Dict[str, Any]]:
    agents = [
        {
            "id": agent.id,
            "name": agent.name,
            "drugClass": agent.drug_class,
            "riskTags": list(agent.risk_tags),
        }
        for agent in AGENTS.values()
    ]
    return sorted(agents, key=lambda item: item["name"].lower())


def list_immunoid_regimens() -> List[Dict[str, Any]]:
    regimens = []
    for regimen_id, entry in IMMUNOID_REGIMENS.items():
        component_ids = list(entry.get("component_agent_ids", ()))
        component_names = [AGENTS[agent_id].name for agent_id in component_ids if agent_id in AGENTS]
        regimens.append(
            {
                "id": regimen_id,
                "name": entry["name"],
                "componentAgentIds": component_ids,
                "componentAgentNames": component_names,
            }
        )
    return sorted(regimens, key=lambda item: item["name"].lower())


def analyze_immunoid(payload: Dict[str, Any]) -> Dict[str, Any]:
    requested_regimen_ids = payload.get("selected_regimen_ids") or []
    selected_regimen_ids: List[str] = []
    seen_regimen_ids: set[str] = set()
    for regimen_id in requested_regimen_ids:
        if regimen_id not in seen_regimen_ids:
            selected_regimen_ids.append(regimen_id)
            seen_regimen_ids.add(regimen_id)

    supported_regimens = [IMMUNOID_REGIMENS[regimen_id] for regimen_id in selected_regimen_ids if regimen_id in IMMUNOID_REGIMENS]
    unsupported_regimen_ids = [regimen_id for regimen_id in selected_regimen_ids if regimen_id not in IMMUNOID_REGIMENS]
    requested_ids = payload.get("selected_agent_ids") or []
    selected_ids: List[str] = []
    seen_ids: set[str] = set()
    for regimen in supported_regimens:
        for agent_id in regimen.get("component_agent_ids", ()):
            if agent_id not in seen_ids:
                selected_ids.append(agent_id)
                seen_ids.add(agent_id)
    for agent_id in requested_ids:
        if agent_id not in seen_ids:
            selected_ids.append(agent_id)
            seen_ids.add(agent_id)

    selected_agents = [AGENTS[agent_id] for agent_id in selected_ids if agent_id in AGENTS]
    unsupported_agent_ids = [agent_id for agent_id in selected_ids if agent_id not in AGENTS]
    group_set = {group for agent in selected_agents for group in agent.groups}
    risk_flags = sorted({tag for agent in selected_agents for tag in agent.risk_tags})

    recommendations: List[Dict[str, Any]] = []
    follow_up_questions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    rec_ids: set[str] = set()
    question_ids: set[str] = set()

    if unsupported_regimen_ids:
        warnings.append(
            "Some selected regimens are not yet mapped in the API ImmunoID rule set and were ignored for rule firing."
        )

    if unsupported_agent_ids:
        warnings.append(
            "Some selected agents are not yet mapped in the API ImmunoID rule set and were ignored for rule firing."
        )

    def triggered_by(groups: set[str] | None = None, ids: set[str] | None = None) -> List[str]:
        matched: List[str] = []
        for agent in selected_agents:
            if groups and group_set.isdisjoint(groups) and not ids:
                continue
            if groups and not set(agent.groups).intersection(groups):
                if not ids or agent.id not in ids:
                    continue
            if ids and agent.id not in ids and not (groups and set(agent.groups).intersection(groups)):
                continue
            matched.append(agent.name)
        return matched

    def add_recommendation(
        rec_id: str,
        *,
        title: str,
        category: str,
        priority: str,
        summary: str,
        rationale: str,
        triggered_by_names: List[str],
        citations: List[Dict[str, str]],
    ) -> None:
        if rec_id in rec_ids:
            return
        rec_ids.add(rec_id)
        recommendations.append(
            {
                "id": rec_id,
                "title": title,
                "category": category,
                "priority": priority,
                "summary": summary,
                "rationale": rationale,
                "triggeredBy": triggered_by_names,
                "citations": citations,
            }
        )

    def add_question(
        question_id: str,
        *,
        prompt: str,
        reason: str,
        related_recommendation_ids: List[str] | None = None,
    ) -> None:
        if question_id in question_ids:
            return
        question_ids.add(question_id)
        follow_up_questions.append(
            {
                "id": question_id,
                "prompt": prompt,
                "reason": reason,
                "relatedRecommendationIds": related_recommendation_ids or [],
            }
        )

    hbv_hbsag = payload.get("hbv_hbsag", "unknown")
    hbv_anti_hbc = payload.get("hbv_anti_hbc", "unknown")
    tb_screen_result = payload.get("tb_screen_result", "unknown")
    tb_endemic_exposure = payload.get("tb_endemic_exposure")
    regimen_defaults = [regimen.get("defaults", {}) for regimen in supported_regimens]
    planned_steroid_duration_days = payload.get("planned_steroid_duration_days")
    steroid_duration_source = "provided"
    if planned_steroid_duration_days is None:
        regimen_steroid_days = [entry.get("planned_steroid_duration_days") for entry in regimen_defaults if entry.get("planned_steroid_duration_days") is not None]
        if regimen_steroid_days and len(set(regimen_steroid_days)) == 1:
            planned_steroid_duration_days = regimen_steroid_days[0]
            steroid_duration_source = "regimen"
    strongyloides_exposure = payload.get("strongyloides_exposure")
    strongyloides_igg = payload.get("strongyloides_igg", "unknown")
    coccidioides_exposure = payload.get("coccidioides_exposure")
    histoplasma_exposure = payload.get("histoplasma_exposure")
    anticipated_neutropenia = payload.get("anticipated_prolonged_profound_neutropenia")
    anticipated_neutropenia_source = "provided"
    if anticipated_neutropenia is None:
        regimen_neutropenia = [
            entry.get("anticipated_prolonged_profound_neutropenia")
            for entry in regimen_defaults
            if entry.get("anticipated_prolonged_profound_neutropenia") is not None
        ]
        if regimen_neutropenia and len(set(regimen_neutropenia)) == 1:
            anticipated_neutropenia = regimen_neutropenia[0]
            anticipated_neutropenia_source = "regimen"
    exposure_summary: List[Dict[str, str]] = []

    def add_exposure_summary(
        item_id: str,
        label: str,
        value: Any,
        *,
        include_when_unknown: bool = True,
        source: str = "provided",
    ) -> None:
        rendered = _exposure_value(value)
        if rendered == "unknown" and not include_when_unknown:
            return
        exposure_summary.append(
            {
                "id": item_id,
                "label": label,
                "value": rendered,
                "source": source,
            }
        )

    if "anti_cd20" in group_set:
        add_exposure_summary("hbv_hbsag", "HBsAg", hbv_hbsag)
        add_exposure_summary("hbv_anti_hbc", "HBV anti-HBc", hbv_anti_hbc)
        add_exposure_summary("hbv_anti_hbs", "HBV anti-HBs", payload.get("hbv_anti_hbs", "unknown"))

    if "tb_screen" in group_set or "tb_context" in group_set:
        add_exposure_summary("tb_screen_result", "TB screen result", tb_screen_result)
        if "tb_context" in group_set:
            add_exposure_summary("tb_endemic_exposure", "TB-endemic geography risk", tb_endemic_exposure)

    if "high_dose_steroid" in group_set:
        add_exposure_summary(
            "planned_steroid_duration_days",
            "Planned steroid duration",
            (
                f"{planned_steroid_duration_days} days"
                if isinstance(planned_steroid_duration_days, int)
                else None
            ),
            source=steroid_duration_source,
        )

    if "strongy_context" in group_set:
        add_exposure_summary("strongyloides_exposure", "Strongyloides-endemic exposure", strongyloides_exposure)
        add_exposure_summary("strongyloides_igg", "Strongyloides IgG", strongyloides_igg)

    if "cocci_context" in group_set:
        add_exposure_summary("coccidioides_exposure", "Coccidioides-endemic exposure", coccidioides_exposure)

    if "histo_context" in group_set:
        add_exposure_summary("histoplasma_exposure", "Histoplasma geography/exposure", histoplasma_exposure)

    if "cytotoxic_chemo" in group_set:
        add_exposure_summary(
            "anticipated_prolonged_profound_neutropenia",
            "Expected prolonged profound neutropenia",
            anticipated_neutropenia,
            source=anticipated_neutropenia_source,
        )

    if "anti_cd20" in group_set:
        if "unknown" in {hbv_hbsag, hbv_anti_hbc, payload.get("hbv_anti_hbs", "unknown")}:
            add_recommendation(
                "hbv_screen_before_anti_cd20",
                title="Order baseline HBV serologies before anti-CD20 therapy",
                category="screening",
                priority="high",
                summary="Obtain HBsAg, total anti-HBc, and anti-HBs before anti-CD20 therapy or the next cycle.",
                rationale="Anti-CD20 therapy carries clinically important HBV reactivation risk, and prophylaxis decisions depend on complete baseline serologies.",
                triggered_by_names=triggered_by(groups={"anti_cd20"}),
                citations=[
                    _reference_entry("HBV screening before anti-CD20 therapy", "cdc_hbv_testing"),
                    _reference_entry("HBV reactivation prevention", "aga_hbv"),
                ],
            )
        if hbv_hbsag == "positive" or hbv_anti_hbc == "positive":
            add_recommendation(
                "hbv_prophylaxis_high_risk",
                title="Plan HBV antiviral prophylaxis before anti-CD20 therapy",
                category="prophylaxis",
                priority="high",
                summary="Positive HBsAg or anti-HBc should trigger HBV-directed prophylaxis planning before B-cell-depleting therapy.",
                rationale="Anti-CD20 therapy is a high-risk setting for HBV reactivation; this should not be left to passive monitoring alone.",
                triggered_by_names=triggered_by(groups={"anti_cd20"}),
                citations=[
                    _reference_entry("HBV reactivation prevention in anti-CD20 therapy", "aga_hbv"),
                    _reference_entry("HBV testing framework", "cdc_hbv_testing"),
                ],
            )

    if "tb_screen" in group_set:
        if tb_screen_result in {"unknown", "indeterminate"}:
            add_recommendation(
                "tb_screen_before_biologic",
                title="Screen for latent TB before high-risk biologic or JAK therapy",
                category="screening",
                priority="high",
                summary="Obtain a TB IGRA before therapy if screening is not already documented; repeat or clarify indeterminate results.",
                rationale="TNF blockers and JAK inhibitors can reactivate latent TB, and screening should precede treatment.",
                triggered_by_names=triggered_by(groups={"tb_screen"}),
                citations=[
                    _reference_entry("TB screening before biologic/JAK therapy", "eular_screening"),
                    _reference_entry("Latent TB evaluation", "cdc_tb_ltbi"),
                ],
            )
        elif tb_screen_result == "positive":
            add_recommendation(
                "tb_positive_manage_before_immunosuppression",
                title="Address latent TB before or alongside immunosuppression",
                category="referral",
                priority="high",
                summary="A positive TB screen should trigger latent TB evaluation and treatment planning before ongoing high-risk immunosuppression.",
                rationale="Once latent TB is identified, therapy should be coordinated rather than ignored while immunosuppression proceeds unchecked.",
                triggered_by_names=triggered_by(groups={"tb_screen"}),
                citations=[
                    _reference_entry("TB screening before biologic/JAK therapy", "eular_screening"),
                    _reference_entry("Latent TB treatment framework", "cdc_tb_ltbi"),
                ],
            )

    if "pjp_high_risk" in group_set:
        add_recommendation(
            "pjp_prophylaxis_high_risk_agents",
            title="Plan PJP prophylaxis for strongly lymphocyte-depleting therapy",
            category="prophylaxis",
            priority="high",
            summary="Strongly lymphocyte-depleting agents such as purine analogs or ATG generally justify PJP prophylaxis while risk persists.",
            rationale="These agents are recurrently associated with clinically important opportunistic infection risk, including PJP.",
            triggered_by_names=triggered_by(groups={"pjp_high_risk"}),
            citations=[
                _reference_entry("Cancer-related prophylaxis", "asco_idsa_antimicrobial"),
                _reference_entry("Opportunistic infection prophylaxis in immunosuppression", "eular_screening"),
            ],
        )

    pjp_combo_agents = [
        agent.name
        for agent in selected_agents
        if set(agent.groups).intersection({"pjp_context", "pjp_high_risk"})
    ]
    if len(pjp_combo_agents) >= 2:
        add_recommendation(
            "pjp_prophylaxis_combination_review",
            title="Review PJP prophylaxis for combination immunosuppression",
            category="prophylaxis",
            priority="high",
            summary=(
                "When multiple PJP-relevant immunosuppressive exposures are layered together, "
                "prophylaxis should be reviewed explicitly rather than assumed unnecessary."
            ),
            rationale=(
                "PJP risk rises with combined lymphocyte-directed or steroid-based immunosuppression, "
                "including combinations such as anti-CD20 therapy plus steroids or other additional agents."
            ),
            triggered_by_names=pjp_combo_agents,
            citations=[
                _reference_entry("Cancer-related prophylaxis", "asco_idsa_antimicrobial"),
                _reference_entry("Opportunistic infection prophylaxis in immunosuppression", "eular_screening"),
            ],
        )

    if "high_dose_steroid" in group_set:
        if planned_steroid_duration_days is None:
            add_question(
                "steroid_duration",
                prompt="Will systemic steroid exposure at this threshold last 28 days or longer?",
                reason="PJP prophylaxis becomes much more relevant when moderate-high dose systemic steroids are prolonged.",
                related_recommendation_ids=["pjp_prophylaxis_steroids"],
            )
        elif planned_steroid_duration_days >= 28:
            add_recommendation(
                "pjp_prophylaxis_steroids",
                title="Consider PJP prophylaxis for prolonged high-dose steroids",
                category="prophylaxis",
                priority="high",
                summary="Prednisone-equivalent exposure at this threshold for 4 weeks or longer should prompt PJP prophylaxis review, especially with combination immunosuppression.",
                rationale="Duration matters; prolonged systemic steroids are a common trigger for PJP prophylaxis decisions.",
                triggered_by_names=triggered_by(groups={"high_dose_steroid"}),
                citations=[
                    _reference_entry("Cancer-related prophylaxis", "asco_idsa_antimicrobial"),
                    _reference_entry("Steroid-associated opportunistic infection prophylaxis", "eular_screening"),
                ],
            )

    if "strongy_context" in group_set:
        if strongyloides_exposure is None:
            add_question(
                "strongyloides_exposure",
                prompt=(
                    "Is there birth, residence, or substantial travel exposure in Strongyloides-endemic settings "
                    "(for example Latin America, the Caribbean, sub-Saharan Africa, Southeast Asia, Oceania, or parts of the southeastern US/Appalachia)?"
                ),
                reason="Steroids and transplant-level immunosuppression can precipitate Strongyloides hyperinfection when relevant epidemiology is present.",
                related_recommendation_ids=["strongyloides_screen", "strongyloides_treat_before_immunosuppression"],
            )
        elif strongyloides_exposure:
            if strongyloides_igg == "unknown":
                add_recommendation(
                    "strongyloides_screen",
                    title="Order Strongyloides IgG before major immunosuppression",
                    category="screening",
                    priority="high",
                    summary="Relevant endemic exposure plus steroids or transplant-level immunosuppression should trigger Strongyloides serologic screening before treatment when feasible.",
                    rationale="Missing Strongyloides before immunosuppression can lead to hyperinfection syndrome, especially with corticosteroids.",
                    triggered_by_names=triggered_by(groups={"strongy_context"}),
                    citations=[
                        _reference_entry("Strongyloides screening before immunosuppression", "cdc_strongyloides"),
                        _reference_entry("Practical Strongyloides screening framework", "strongyloides_review"),
                    ],
                )
            elif strongyloides_igg == "positive":
                add_recommendation(
                    "strongyloides_treat_before_immunosuppression",
                    title="Treat or urgently coordinate Strongyloides management before escalating immunosuppression",
                    category="referral",
                    priority="high",
                    summary="A positive Strongyloides screen should trigger treatment planning before substantial immunosuppression whenever possible.",
                    rationale="Positive serology in this context should not be ignored because steroids can precipitate hyperinfection and dissemination.",
                    triggered_by_names=triggered_by(groups={"strongy_context"}),
                    citations=[
                        _reference_entry("Strongyloides management", "cdc_strongyloides"),
                        _reference_entry("Practical Strongyloides screening framework", "strongyloides_review"),
                    ],
                )

    if "tb_context" in group_set:
        if tb_screen_result in {"unknown", "indeterminate"}:
            if tb_endemic_exposure is None:
                add_question(
                    "tb_endemic_exposure",
                    prompt=(
                        "Was the patient born in, did they live in, or have they had substantial travel to places where TB is common "
                        "(for example parts of Asia, Africa, Latin America, or other higher-incidence settings), or other clear TB epidemiologic risk?"
                    ),
                    reason="Prolonged systemic steroids raise the stakes of latent TB, and epidemiology helps decide when risk-based testing should be pushed harder.",
                    related_recommendation_ids=["tb_risk_based_screen_review"],
                )
            elif tb_endemic_exposure:
                add_recommendation(
                    "tb_risk_based_screen_review",
                    title="Review latent TB testing before prolonged high-dose steroids when epidemiology is relevant",
                    category="screening",
                    priority="moderate",
                    summary=(
                        "Birth, residence, or substantial travel in TB-higher-incidence settings should lower the threshold for IGRA testing "
                        "before prolonged high-dose steroids, or prompt clarification when a prior result is indeterminate."
                    ),
                    rationale=(
                        "CDC recommends TB testing based on risk assessment, and missing latent TB becomes more consequential once prolonged systemic steroids are added."
                    ),
                    triggered_by_names=triggered_by(groups={"tb_context"}),
                    citations=[
                        _reference_entry("TB risk-based testing framework", "cdc_tb_risk_country"),
                        _reference_entry("Latent TB evaluation", "cdc_tb_ltbi"),
                    ],
                )
        elif tb_screen_result == "positive":
            add_recommendation(
                "tb_positive_manage_steroid_context",
                title="Coordinate latent TB management before prolonged high-dose steroids",
                category="referral",
                priority="high",
                summary="A positive TB screen in this setting should trigger latent TB treatment planning rather than proceeding with prolonged high-dose steroids without a plan.",
                rationale="Once latent TB is identified, immunosuppression should proceed with an explicit management strategy instead of passive observation.",
                triggered_by_names=triggered_by(groups={"tb_context"}),
                citations=[
                    _reference_entry("TB risk-based testing framework", "cdc_tb_risk_country"),
                    _reference_entry("Latent TB treatment framework", "cdc_tb_ltbi"),
                ],
            )

    if "cocci_context" in group_set:
        if coccidioides_exposure is None:
            add_question(
                "coccidioides_exposure",
                prompt=(
                    "Does the patient live in or have meaningful travel/residence exposure to Coccidioides-endemic areas "
                    "(for example Arizona, California Central Valley, New Mexico, west Texas, southern Nevada, Utah, Washington state, or northern Mexico)?"
                ),
                reason="Geography meaningfully changes endemic fungal risk and can alter the pre-immunosuppression checklist.",
                related_recommendation_ids=["cocci_context_review"],
            )
        elif coccidioides_exposure:
            add_recommendation(
                "cocci_context_review",
                title="Adjust the workup for coccidioidomycosis exposure risk",
                category="context",
                priority="moderate",
                summary="Endemic exposure should lower the threshold for coccidioidomycosis testing and may justify baseline serology or ID review before potent immunosuppression depending on local practice.",
                rationale="This is not a universal national screening rule, but endemic geography materially changes pre-immunosuppression risk assessment.",
                triggered_by_names=triggered_by(groups={"cocci_context"}),
                citations=[_reference_entry("Coccidioidomycosis testing framework", "cdc_cocci")],
            )

    if "histo_context" in group_set:
        if histoplasma_exposure is None:
            add_question(
                "histoplasma_exposure",
                prompt=(
                    "Is there residence or substantial travel in Histoplasma-endemic areas "
                    "(especially the Ohio or Mississippi River valleys, broader central/eastern US, or Central/South America), "
                    "or cave, bat, bird-dropping, demolition, excavation, or dusty-soil exposure?"
                ),
                reason="Histoplasma risk is geography- and exposure-dependent rather than a universal baseline screening issue.",
                related_recommendation_ids=["histoplasma_context_review"],
            )
        elif histoplasma_exposure:
            add_recommendation(
                "histoplasma_context_review",
                title="Adjust the workup for histoplasmosis exposure risk",
                category="context",
                priority="moderate",
                summary=(
                    "Relevant Histoplasma geography or bat, bird-dropping, cave, demolition, or excavation exposure should lower the threshold "
                    "for Histoplasma testing and ID review if compatible symptoms develop during potent immunosuppression."
                ),
                rationale=(
                    "Routine national baseline screening is not established, but endemic geography and aerosolizing exposures materially change pretest probability."
                ),
                triggered_by_names=triggered_by(groups={"histo_context"}),
                citations=[
                    _reference_entry("Histoplasmosis geography", "cdc_histoplasma_maps"),
                    _reference_entry("Environmental Histoplasma exposure", "cdc_histoplasma_environment"),
                ],
            )

    if "cytotoxic_chemo" in group_set:
        if anticipated_neutropenia is None:
            add_question(
                "prolonged_profound_neutropenia",
                prompt="Is prolonged profound neutropenia expected from this regimen?",
                reason="Antibacterial and antifungal prophylaxis depends more on expected depth and duration of neutropenia than on the drug name alone.",
                related_recommendation_ids=["neutropenia_prophylaxis_review"],
            )
        elif anticipated_neutropenia:
            add_recommendation(
                "neutropenia_prophylaxis_review",
                title="Review antibacterial and antifungal prophylaxis for prolonged profound neutropenia",
                category="prophylaxis",
                priority="high",
                summary="When prolonged profound neutropenia is expected, confirm regimen-based antibacterial prophylaxis and mold-active antifungal prophylaxis according to local oncology protocols.",
                rationale="For cytotoxic regimens, prophylaxis decisions are driven by the expected neutropenia profile rather than the drug class label alone.",
                triggered_by_names=triggered_by(groups={"cytotoxic_chemo"}),
                citations=[_reference_entry("Cancer-related antimicrobial prophylaxis", "asco_idsa_antimicrobial")],
            )

    if "complement_c5" in group_set:
        add_recommendation(
            "meningococcal_protocol_complement",
            title="Confirm meningococcal vaccination and complement-inhibitor infection protocol",
            category="prophylaxis",
            priority="high",
            summary="Patients receiving terminal complement inhibition need product-specific meningococcal vaccination review and center-specific infection prophylaxis planning.",
            rationale="Vaccination alone does not eliminate invasive meningococcal risk with C5 blockade.",
            triggered_by_names=triggered_by(groups={"complement_c5"}),
            citations=[_reference_entry("Complement inhibition and meningococcal risk", "mmwr_eculizumab")],
        )

    status = "needs_more_info" if follow_up_questions else "complete"
    selected_regimens_payload = []
    for regimen_id in selected_regimen_ids:
        regimen = IMMUNOID_REGIMENS.get(regimen_id)
        if regimen is None:
            continue
        component_ids = list(regimen.get("component_agent_ids", ()))
        selected_regimens_payload.append(
            {
                "id": regimen_id,
                "name": regimen["name"],
                "componentAgentIds": component_ids,
                "componentAgentNames": [AGENTS[agent_id].name for agent_id in component_ids if agent_id in AGENTS],
            }
        )
    selected_agents_payload = [
        {
            "id": agent.id,
            "name": agent.name,
            "drugClass": agent.drug_class,
            "riskTags": list(agent.risk_tags),
        }
        for agent in selected_agents
    ]
    return {
        "status": status,
        "selectedRegimens": selected_regimens_payload,
        "selectedAgents": selected_agents_payload,
        "unsupportedRegimenIds": unsupported_regimen_ids,
        "unsupportedAgentIds": unsupported_agent_ids,
        "riskFlags": risk_flags,
        "recommendations": recommendations,
        "followUpQuestions": follow_up_questions,
        "exposureSummary": exposure_summary,
        "warnings": warnings,
    }
