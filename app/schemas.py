from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FindingState = Literal["present", "absent", "unknown"]
Recommendation = Literal["observe", "test", "treat"]
ASTResult = Literal["Susceptible", "Intermediate", "Resistant"]


class SourceRef(BaseModel):
    short: str
    year: Optional[int] = None
    url: Optional[str] = None


class LRItem(BaseModel):
    id: str
    label: str
    category: Optional[str] = None
    group: Optional[str] = None
    lr_pos: Optional[float] = Field(default=None, alias="lrPos")
    lr_neg: Optional[float] = Field(default=None, alias="lrNeg")
    notes: Optional[str] = None
    source: Optional[SourceRef] = None

    model_config = {"populate_by_name": True}


class PretestPreset(BaseModel):
    id: str
    label: str
    p: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None
    source: Optional[SourceRef] = None


class HarmInputs(BaseModel):
    unnecessary_treatment: float = Field(default=1.0, gt=0, alias="unnecessaryTx")
    missed_diagnosis: float = Field(default=4.0, gt=0, alias="missedDx")

    model_config = {"populate_by_name": True}


class SyndromeModule(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    pretest_presets: List[PretestPreset] = Field(alias="pretestPresets")
    items: List[LRItem]
    default_harms: Optional[HarmInputs] = Field(default=None, alias="defaultHarms")

    model_config = {"populate_by_name": True}


class VAPRiskModifiersInput(BaseModel):
    enabled: bool = False
    selected_ids: List[str] = Field(default_factory=list, alias="selectedIds")

    model_config = {"populate_by_name": True}


class EndoRiskModifiersInput(BaseModel):
    enabled: bool = False
    selected_ids: List[str] = Field(default_factory=list, alias="selectedIds")

    model_config = {"populate_by_name": True}


class VirstaInput(BaseModel):
    enabled: bool = False
    emboli: bool = False
    meningitis: bool = False
    intracardiac_device: bool = Field(default=False, alias="intracardiacDevice")
    prior_endocarditis: bool = Field(default=False, alias="priorEndocarditis")
    native_valve_disease: bool = Field(default=False, alias="nativeValveDisease")
    ivdu: bool = False
    persistent_bacteremia_48h: bool = Field(default=False, alias="persistentBacteremia48h")
    vertebral_osteomyelitis: bool = Field(default=False, alias="vertebralOsteomyelitis")
    acquisition: Literal["nosocomial", "community_or_nhca"] = "nosocomial"
    severe_sepsis_shock: bool = Field(default=False, alias="severeSepsisShock")
    crp_gt_190: bool = Field(default=False, alias="crpGt190")

    model_config = {"populate_by_name": True}


class DenovaInput(BaseModel):
    enabled: bool = False
    duration_7d: bool = Field(default=False, alias="duration7d")
    embolization: bool = False
    num_positive_2: bool = Field(default=False, alias="numPositive2")
    origin_unknown: bool = Field(default=False, alias="originUnknown")
    valve_disease: bool = Field(default=False, alias="valveDisease")
    auscultation_murmur: bool = Field(default=False, alias="auscultationMurmur")

    model_config = {"populate_by_name": True}


class HandocInput(BaseModel):
    enabled: bool = False
    heart_murmur_valve: bool = Field(default=False, alias="heartMurmurValve")
    species: Literal[
        "unspecified_other",
        "s_anginosus_group",
        "s_gallolyticus_bovis_group",
        "s_mutans_group",
        "s_sanguinis_group",
        "s_mitis_oralis_group",
        "s_salivarius_group",
    ] = "unspecified_other"
    num_positive_2: bool = Field(default=False, alias="numPositive2")
    duration_7d: bool = Field(default=False, alias="duration7d")
    only_one_species: bool = Field(default=False, alias="onlyOneSpecies")
    community_acquired: bool = Field(default=False, alias="communityAcquired")

    model_config = {"populate_by_name": True}


class EndoScoresInput(BaseModel):
    virsta: Optional[VirstaInput] = None
    denova: Optional[DenovaInput] = None
    handoc: Optional[HandocInput] = None


class ProbIDControlsInput(BaseModel):
    vap_risk_modifiers: Optional[VAPRiskModifiersInput] = Field(default=None, alias="vapRiskModifiers")
    endo_risk_modifiers: Optional[EndoRiskModifiersInput] = Field(default=None, alias="endoRiskModifiers")
    endo_scores: Optional[EndoScoresInput] = Field(default=None, alias="endoScores")

    model_config = {"populate_by_name": True}


class AnalyzeRequest(BaseModel):
    module_id: Optional[str] = Field(default=None, alias="moduleId")
    module: Optional[SyndromeModule] = None
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    pretest_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, alias="pretestProbability")
    pretest_odds_multiplier: float = Field(default=1.0, gt=0, alias="pretestOddsMultiplier")
    findings: Dict[str, FindingState] = Field(default_factory=dict)
    ordered_finding_ids: List[str] = Field(default_factory=list, alias="orderedFindingIds")
    harms: Optional[HarmInputs] = None
    probid_controls: Optional[ProbIDControlsInput] = Field(default=None, alias="probidControls")
    include_explanation: bool = Field(default=True, alias="includeExplanation")

    model_config = {"populate_by_name": True}


class StepwiseUpdate(BaseModel):
    id: str
    label: str
    state: FindingState
    lr_used: float = Field(alias="lrUsed")
    p_after: float = Field(alias="pAfter")

    model_config = {"populate_by_name": True}


class AppliedFinding(BaseModel):
    id: str
    label: str
    state: FindingState
    lr_used: float = Field(alias="lrUsed")
    impact_score: float = Field(alias="impactScore")

    model_config = {"populate_by_name": True}


class DecisionThresholds(BaseModel):
    observe_probability: float = Field(alias="observeProbability")
    treat_probability: float = Field(alias="treatProbability")

    model_config = {"populate_by_name": True}


class PretestSummary(BaseModel):
    base_probability: float = Field(alias="baseProbability")
    adjusted_probability: float = Field(alias="adjustedProbability")
    preset_id: Optional[str] = Field(default=None, alias="presetId")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    module_id: str = Field(alias="moduleId")
    module_name: str = Field(alias="moduleName")
    pretest: PretestSummary
    combined_lr: float = Field(alias="combinedLR")
    posttest_probability: float = Field(alias="posttestProbability")
    thresholds: DecisionThresholds
    recommendation: Recommendation
    recommendation_summary: Optional[str] = Field(default=None, alias="recommendationSummary")
    recommended_next_steps: List[str] = Field(default_factory=list, alias="recommendedNextSteps")
    confidence: float = Field(ge=0.0, le=1.0)
    applied_findings: List[AppliedFinding] = Field(alias="appliedFindings")
    stepwise: List[StepwiseUpdate]
    reasons: List[str]
    risk_flags: List[str] = Field(alias="riskFlags")
    explanation_for_user: Optional[str] = Field(default=None, alias="explanationForUser")

    model_config = {"populate_by_name": True}


class ModuleSummary(BaseModel):
    id: str
    name: str
    item_count: int = Field(alias="itemCount")
    preset_count: int = Field(alias="presetCount")

    model_config = {"populate_by_name": True}


class RegisterModulesRequest(BaseModel):
    modules: List[SyndromeModule]


class RegisterModulesResponse(BaseModel):
    registered: int
    ids: List[str]


class TextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    module_hint: Optional[str] = Field(default=None, alias="moduleHint")
    preset_hint: Optional[str] = Field(default=None, alias="presetHint")
    parser_strategy: Literal["auto", "rule", "local", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")
    run_analyze: bool = Field(default=True, alias="runAnalyze")
    include_explanation: bool = Field(default=True, alias="includeExplanation")

    model_config = {"populate_by_name": True}


class ParsedUnderstanding(BaseModel):
    module_id: Optional[str] = Field(default=None, alias="moduleId")
    module_name: Optional[str] = Field(default=None, alias="moduleName")
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    preset_label: Optional[str] = Field(default=None, alias="presetLabel")
    findings_present: List[str] = Field(default_factory=list, alias="findingsPresent")
    findings_absent: List[str] = Field(default_factory=list, alias="findingsAbsent")
    unknown_mentions: List[str] = Field(default_factory=list, alias="unknownMentions")

    model_config = {"populate_by_name": True}


class ReferenceEntry(BaseModel):
    context: str
    citation: str
    url: Optional[str] = None


class TextAnalyzeResponse(BaseModel):
    parser: str
    text: str
    parser_fallback_used: bool = Field(default=False, alias="parserFallbackUsed")
    parsed_request: Optional[AnalyzeRequest] = Field(default=None, alias="parsedRequest")
    understood: ParsedUnderstanding
    warnings: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")
    references: List[ReferenceEntry] = Field(default_factory=list)
    analysis: Optional[AnalyzeResponse] = None

    model_config = {"populate_by_name": True}


class ParserTrainingExample(BaseModel):
    text: str = Field(min_length=1)
    module_id: Optional[str] = Field(default=None, alias="moduleId")
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    findings: Dict[str, FindingState] = Field(default_factory=dict)
    ordered_finding_ids: List[str] = Field(default_factory=list, alias="orderedFindingIds")
    notes: Optional[str] = None

    model_config = {"populate_by_name": True}


AssistantStage = Literal[
    "select_module",
    "select_preset",
    "select_endo_blood_culture_context",
    "select_pretest_factors",
    "describe_case",
    "confirm_case",
    "mechid_describe",
    "mechid_confirm",
    "done",
]


class AssistantOption(BaseModel):
    value: str
    label: str
    description: Optional[str] = None
    insert_text: Optional[str] = Field(default=None, alias="insertText")
    absent_text: Optional[str] = Field(default=None, alias="absentText")

    model_config = {"populate_by_name": True}


class AssistantState(BaseModel):
    stage: AssistantStage = "select_module"
    workflow: Literal["probid", "mechid"] = "probid"
    module_id: Optional[str] = Field(default=None, alias="moduleId")
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    endo_blood_culture_context: Optional[
        Literal["staph", "strep", "enterococcus", "other_unknown_pending"]
    ] = Field(default=None, alias="endoBloodCultureContext")
    endo_score_factor_ids: List[str] = Field(default_factory=list, alias="endoScoreFactorIds")
    case_section: Optional[
        Literal["exam_vitals", "lab", "micro", "imaging"]
    ] = Field(default=None, alias="caseSection")
    case_text: Optional[str] = Field(default=None, alias="caseText")
    mechid_text: Optional[str] = Field(default=None, alias="mechidText")
    pretest_factor_ids: List[str] = Field(default_factory=list, alias="pretestFactorIds")
    pretest_factor_labels: List[str] = Field(default_factory=list, alias="pretestFactorLabels")
    parser_strategy: Literal["auto", "rule", "local", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")

    model_config = {"populate_by_name": True}


class AssistantTurnRequest(BaseModel):
    state: Optional[AssistantState] = None
    message: Optional[str] = None
    selection: Optional[str] = None
    parser_strategy: Optional[Literal["auto", "rule", "local", "openai"]] = Field(default=None, alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: Optional[bool] = Field(default=None, alias="allowFallback")

    model_config = {"populate_by_name": True}


class AssistantTurnResponse(BaseModel):
    assistant_name: str = Field(default="Uncertainty Assistant", alias="assistantName")
    assistant_message: str = Field(alias="assistantMessage")
    state: AssistantState
    options: List[AssistantOption] = Field(default_factory=list)
    analysis: Optional[TextAnalyzeResponse] = None
    mechid_analysis: Optional["MechIDTextAnalyzeResponse"] = Field(default=None, alias="mechidAnalysis")
    tips: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MechIDTxContext(BaseModel):
    syndrome: str = "Not specified"
    severity: str = "Not specified"


class MechIDAnalyzeRequest(BaseModel):
    organism: str = Field(min_length=1)
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: Optional[MechIDTxContext] = Field(default=None, alias="txContext")

    model_config = {"populate_by_name": True}


class MechIDResultRow(BaseModel):
    antibiotic: str
    result: ASTResult
    source: Literal["user", "cascade_rule", "intrinsic_rule"]


class MechIDAnalyzeResponse(BaseModel):
    organism: str
    panel: List[str] = Field(default_factory=list)
    submitted_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="submittedResults")
    inferred_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="inferredResults")
    final_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="finalResults")
    rows: List[MechIDResultRow] = Field(default_factory=list)
    mechanisms: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)
    favorable_signals: List[str] = Field(default_factory=list, alias="favorableSignals")
    therapy_notes: List[str] = Field(default_factory=list, alias="therapyNotes")
    references: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MechIDTextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    parser_strategy: Literal["auto", "rule", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")

    model_config = {"populate_by_name": True}


class MechIDTextParsedRequest(BaseModel):
    organism: Optional[str] = None
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: MechIDTxContext = Field(default_factory=MechIDTxContext, alias="txContext")

    model_config = {"populate_by_name": True}


class MechIDTextAnalyzeResponse(BaseModel):
    parser: str = "rule-based-v1"
    text: str
    parsed_request: Optional[MechIDTextParsedRequest] = Field(default=None, alias="parsedRequest")
    warnings: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")
    parser_fallback_used: bool = Field(default=False, alias="parserFallbackUsed")
    analysis: Optional[MechIDAnalyzeResponse] = None

    model_config = {"populate_by_name": True}
