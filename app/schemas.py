from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


class NextBestTest(BaseModel):
    id: str
    label: str
    category: Optional[str] = None
    probability_if_positive: float = Field(alias="probabilityIfPositive")
    probability_if_negative: float = Field(alias="probabilityIfNegative")
    probability_swing: float = Field(alias="probabilitySwing")
    source_short: Optional[str] = Field(default=None, alias="sourceShort")

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


class ClinicalScoreResult(BaseModel):
    """Result of a validated clinical prediction rule calculation."""
    score_name: str = Field(alias="scoreName")
    score_value: Optional[int] = Field(default=None, alias="scoreValue")
    risk_class: Optional[str] = Field(default=None, alias="riskClass")
    interpretation: str
    recommendation: str
    components_met: List[str] = Field(default_factory=list, alias="componentsMet")
    components_not_met: List[str] = Field(default_factory=list, alias="componentsNotMet")
    source: str

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
    treatment_duration_guidance: List[str] = Field(default_factory=list, alias="treatmentDurationGuidance")
    monitoring_recommendations: List[str] = Field(default_factory=list, alias="monitoringRecommendations")
    confidence: float = Field(ge=0.0, le=1.0)
    applied_findings: List[AppliedFinding] = Field(alias="appliedFindings")
    stepwise: List[StepwiseUpdate]
    reasons: List[str]
    risk_flags: List[str] = Field(alias="riskFlags")
    explanation_for_user: Optional[str] = Field(default=None, alias="explanationForUser")
    next_best_tests: List[NextBestTest] = Field(default_factory=list, alias="nextBestTests")
    clinical_scores: List[ClinicalScoreResult] = Field(default_factory=list, alias="clinicalScores")

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


class CalibrationOutcome(BaseModel):
    """A single outcome record for calibration tracking."""
    module_id: str = Field(alias="moduleId")
    module_name: str = Field(alias="moduleName")
    predicted_probability: float = Field(alias="predictedProbability")
    actual_outcome: bool = Field(alias="actualOutcome")  # True = syndrome confirmed, False = ruled out
    clinical_scores: List[ClinicalScoreResult] = Field(default_factory=list, alias="clinicalScores")
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    findings_snapshot: Optional[Dict[str, str]] = Field(default=None, alias="findingsSnapshot")
    notes: Optional[str] = None
    timestamp: Optional[str] = None  # ISO format, set by server

    model_config = {"populate_by_name": True}


class CalibrationBucket(BaseModel):
    """A single bin in a calibration curve."""
    bin_lower: float = Field(alias="binLower")
    bin_upper: float = Field(alias="binUpper")
    bin_midpoint: float = Field(alias="binMidpoint")
    predicted_mean: float = Field(alias="predictedMean")
    observed_rate: float = Field(alias="observedRate")
    count: int
    outcome_true: int = Field(alias="outcomeTrue")
    outcome_false: int = Field(alias="outcomeFalse")

    model_config = {"populate_by_name": True}


class ModuleCalibrationStats(BaseModel):
    """Calibration statistics for a single module."""
    module_id: str = Field(alias="moduleId")
    module_name: str = Field(alias="moduleName")
    total_outcomes: int = Field(alias="totalOutcomes")
    overall_accuracy: float = Field(alias="overallAccuracy")
    brier_score: float = Field(alias="brierScore")
    calibration_curve: List[CalibrationBucket] = Field(alias="calibrationCurve")
    mean_predicted: float = Field(alias="meanPredicted")
    mean_observed: float = Field(alias="meanObserved")
    overconfidence_bias: float = Field(alias="overconfidenceBias")

    model_config = {"populate_by_name": True}


class CalibrationReport(BaseModel):
    """Full calibration report across all modules."""
    total_outcomes: int = Field(alias="totalOutcomes")
    modules: List[ModuleCalibrationStats]
    overall_brier_score: float = Field(alias="overallBrierScore")
    generated_at: str = Field(alias="generatedAt")

    model_config = {"populate_by_name": True}


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


ImmunoSerologyState = Literal["positive", "negative", "unknown"]
ImmunoTBScreenState = Literal["positive", "negative", "indeterminate", "unknown"]
ImmunoAnalyzeStatus = Literal["complete", "needs_more_info"]
ImmunoRecommendationCategory = Literal["screening", "prophylaxis", "monitoring", "referral", "context"]
ImmunoRecommendationPriority = Literal["high", "moderate"]
ImmunoExposureSource = Literal["provided", "text", "selection", "regimen"]


class ImmunoAgentSummary(BaseModel):
    id: str
    name: str
    drug_class: str = Field(alias="drugClass")
    risk_tags: List[str] = Field(default_factory=list, alias="riskTags")
    aliases: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ImmunoAgentListResponse(BaseModel):
    agents: List[ImmunoAgentSummary] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ImmunoRegimenSummary(BaseModel):
    id: str
    name: str
    component_agent_ids: List[str] = Field(default_factory=list, alias="componentAgentIds")
    component_agent_names: List[str] = Field(default_factory=list, alias="componentAgentNames")

    model_config = {"populate_by_name": True}


class ImmunoRegimenListResponse(BaseModel):
    regimens: List[ImmunoRegimenSummary] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ImmunoAnalyzeRequest(BaseModel):
    selected_regimen_ids: List[str] = Field(default_factory=list, alias="selectedRegimenIds")
    selected_agent_ids: List[str] = Field(default_factory=list, alias="selectedAgentIds")
    planned_steroid_duration_days: Optional[int] = Field(default=None, ge=0, alias="plannedSteroidDurationDays")
    anticipated_prolonged_profound_neutropenia: Optional[bool] = Field(
        default=None,
        alias="anticipatedProlongedProfoundNeutropenia",
    )
    hbv_hbsag: ImmunoSerologyState = Field(default="unknown", alias="hbvHbsAg")
    hbv_anti_hbc: ImmunoSerologyState = Field(default="unknown", alias="hbvAntiHbc")
    hbv_anti_hbs: ImmunoSerologyState = Field(default="unknown", alias="hbvAntiHbs")
    tb_screen_result: ImmunoTBScreenState = Field(default="unknown", alias="tbScreenResult")
    tb_endemic_exposure: Optional[bool] = Field(default=None, alias="tbEndemicExposure")
    strongyloides_exposure: Optional[bool] = Field(default=None, alias="strongyloidesExposure")
    strongyloides_igg: ImmunoSerologyState = Field(default="unknown", alias="strongyloidesIgg")
    coccidioides_exposure: Optional[bool] = Field(default=None, alias="coccidioidesExposure")
    histoplasma_exposure: Optional[bool] = Field(default=None, alias="histoplasmaExposure")

    model_config = {"populate_by_name": True}


class ImmunoRecommendation(BaseModel):
    id: str
    title: str
    category: ImmunoRecommendationCategory
    priority: ImmunoRecommendationPriority
    summary: str
    rationale: str
    triggered_by: List[str] = Field(default_factory=list, alias="triggeredBy")
    citations: List[ReferenceEntry] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ImmunoFollowUpQuestion(BaseModel):
    id: str
    prompt: str
    reason: str
    related_recommendation_ids: List[str] = Field(default_factory=list, alias="relatedRecommendationIds")

    model_config = {"populate_by_name": True}


class ImmunoExposureSummaryItem(BaseModel):
    id: str
    label: str
    value: str
    source: ImmunoExposureSource = "provided"

    model_config = {"populate_by_name": True}


class ImmunoAnalyzeResponse(BaseModel):
    status: ImmunoAnalyzeStatus
    selected_regimens: List[ImmunoRegimenSummary] = Field(default_factory=list, alias="selectedRegimens")
    selected_agents: List[ImmunoAgentSummary] = Field(default_factory=list, alias="selectedAgents")
    unsupported_regimen_ids: List[str] = Field(default_factory=list, alias="unsupportedRegimenIds")
    unsupported_agent_ids: List[str] = Field(default_factory=list, alias="unsupportedAgentIds")
    risk_flags: List[str] = Field(default_factory=list, alias="riskFlags")
    recommendations: List[ImmunoRecommendation] = Field(default_factory=list)
    follow_up_questions: List[ImmunoFollowUpQuestion] = Field(default_factory=list, alias="followUpQuestions")
    exposure_summary: List[ImmunoExposureSummaryItem] = Field(default_factory=list, alias="exposureSummary")
    warnings: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


AntibioticAllergyReactionType = Literal[
    "unknown",
    "intolerance",
    "isolated_gi",
    "headache",
    "family_history_only",
    "benign_delayed_rash",
    "urticaria",
    "angioedema",
    "anaphylaxis",
    "scar",
    "organ_injury",
    "serum_sickness_like",
    "hemolytic_anemia",
]
AntibioticAllergyTiming = Literal["unknown", "immediate", "delayed"]
AntibioticAllergyRecommendationLevel = Literal["preferred", "caution", "avoid"]


class AntibioticAllergyEntry(BaseModel):
    reported_agent: str = Field(alias="reportedAgent", min_length=1)
    reaction_type: AntibioticAllergyReactionType = Field(default="unknown", alias="reactionType")
    timing: AntibioticAllergyTiming = "unknown"
    verified: bool = False
    notes: Optional[str] = None

    model_config = {"populate_by_name": True}


class AntibioticAllergyRecommendation(BaseModel):
    agent: str
    normalized_agent: str = Field(alias="normalizedAgent")
    recommendation: AntibioticAllergyRecommendationLevel
    summary: str
    rationale: str
    triggered_by: List[str] = Field(default_factory=list, alias="triggeredBy")

    model_config = {"populate_by_name": True}


class AntibioticAllergyFollowUpQuestion(BaseModel):
    id: str
    prompt: str
    reason: str


class AntibioticAllergyAnalyzeRequest(BaseModel):
    candidate_agents: List[str] = Field(default_factory=list, alias="candidateAgents")
    tolerated_agents: List[str] = Field(default_factory=list, alias="toleratedAgents")
    allergy_entries: List[AntibioticAllergyEntry] = Field(default_factory=list, alias="allergyEntries")
    infection_context: Optional[str] = Field(default=None, alias="infectionContext")

    model_config = {"populate_by_name": True}


class AntibioticAllergyAnalyzeResponse(BaseModel):
    summary: str
    overall_risk: str = Field(alias="overallRisk")
    recommendations: List[AntibioticAllergyRecommendation] = Field(default_factory=list)
    general_advice: List[str] = Field(default_factory=list, alias="generalAdvice")
    delabeling_opportunities: List[str] = Field(default_factory=list, alias="delabelingOpportunities")
    follow_up_questions: List[AntibioticAllergyFollowUpQuestion] = Field(default_factory=list, alias="followUpQuestions")
    warnings: List[str] = Field(default_factory=list)
    references: List[ReferenceEntry] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class AntibioticAllergyTextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)


class AntibioticAllergyTextParsedRequest(BaseModel):
    candidate_agents: List[str] = Field(default_factory=list, alias="candidateAgents")
    tolerated_agents: List[str] = Field(default_factory=list, alias="toleratedAgents")
    allergy_entries: List[AntibioticAllergyEntry] = Field(default_factory=list, alias="allergyEntries")
    infection_context: Optional[str] = Field(default=None, alias="infectionContext")

    model_config = {"populate_by_name": True}


class AntibioticAllergyTextAnalyzeResponse(BaseModel):
    parser: str = "rule-based-v1"
    text: str
    parsed_request: Optional[AntibioticAllergyTextParsedRequest] = Field(default=None, alias="parsedRequest")
    warnings: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")
    analysis: Optional[AntibioticAllergyAnalyzeResponse] = None

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
    "select_syndrome_module",
    "select_consult_focus",
    "select_preset",
    "select_endo_blood_culture_context",
    "select_pretest_factors",
    "describe_case",
    "confirm_case",
    "mechid_describe",
    "mechid_confirm",
    "doseid_describe",
    "allergyid_describe",
    "immunoid_select_agents",
    "immunoid_collect_context",
    "done",
]


class AssistantOption(BaseModel):
    value: str
    label: str
    description: Optional[str] = None
    insert_text: Optional[str] = Field(default=None, alias="insertText")
    absent_text: Optional[str] = Field(default=None, alias="absentText")

    model_config = {"populate_by_name": True}


class SessionPatientContext(BaseModel):
    """Patient-level demographics and allergy notes that persist across all workflows in a session."""

    age_years: Optional[int] = Field(default=None, alias="ageYears")
    sex: Optional[Literal["male", "female"]] = None
    total_body_weight_kg: Optional[float] = Field(default=None, alias="totalBodyWeightKg")
    height_cm: Optional[float] = Field(default=None, alias="heightCm")
    serum_creatinine_mg_dl: Optional[float] = Field(default=None, alias="serumCreatinineMgDl")
    crcl_ml_min: Optional[float] = Field(default=None, alias="crclMlMin")
    renal_mode: Literal["standard", "ihd", "crrt"] = Field(default="standard", alias="renalMode")
    allergy_text: Optional[str] = Field(default=None, alias="allergyText")

    model_config = {"populate_by_name": True}

    def is_empty(self) -> bool:
        return (
            self.age_years is None
            and self.sex is None
            and self.total_body_weight_kg is None
            and self.height_cm is None
            and self.serum_creatinine_mg_dl is None
            and self.crcl_ml_min is None
            and self.renal_mode == "standard"
            and not self.allergy_text
        )


class AssistantState(BaseModel):
    stage: AssistantStage = "select_module"
    workflow: Literal["probid", "mechid", "immunoid", "doseid", "allergyid"] = "probid"
    module_id: Optional[str] = Field(default=None, alias="moduleId")
    preset_id: Optional[str] = Field(default=None, alias="presetId")
    pending_intake_text: Optional[str] = Field(default=None, alias="pendingIntakeText")
    pending_followup_workflow: Optional[Literal["probid", "mechid", "immunoid", "doseid"]] = Field(default=None, alias="pendingFollowupWorkflow")
    pending_followup_text: Optional[str] = Field(default=None, alias="pendingFollowupText")
    endo_blood_culture_context: Optional[
        Literal["staph", "strep", "enterococcus", "other_unknown_pending"]
    ] = Field(default=None, alias="endoBloodCultureContext")
    endo_score_factor_ids: List[str] = Field(default_factory=list, alias="endoScoreFactorIds")
    case_section: Optional[
        Literal["exam_vitals", "lab", "micro", "imaging"]
    ] = Field(default=None, alias="caseSection")
    case_text: Optional[str] = Field(default=None, alias="caseText")
    probid_cached_case_result: Optional[Dict[str, Any]] = Field(default=None, alias="probidCachedCaseResult")
    mechid_text: Optional[str] = Field(default=None, alias="mechidText")
    doseid_text: Optional[str] = Field(default=None, alias="doseidText")
    allergyid_text: Optional[str] = Field(default=None, alias="allergyidText")
    immunoid_selected_regimen_ids: List[str] = Field(default_factory=list, alias="immunoidSelectedRegimenIds")
    immunoid_selected_agent_ids: List[str] = Field(default_factory=list, alias="immunoidSelectedAgentIds")
    immunoid_planned_steroid_duration_days: Optional[int] = Field(default=None, alias="immunoidPlannedSteroidDurationDays")
    immunoid_anticipated_prolonged_profound_neutropenia: Optional[bool] = Field(
        default=None,
        alias="immunoidAnticipatedProlongedProfoundNeutropenia",
    )
    immunoid_hbv_hbsag: ImmunoSerologyState = Field(default="unknown", alias="immunoidHbvHbsAg")
    immunoid_hbv_anti_hbc: ImmunoSerologyState = Field(default="unknown", alias="immunoidHbvAntiHbc")
    immunoid_hbv_anti_hbs: ImmunoSerologyState = Field(default="unknown", alias="immunoidHbvAntiHbs")
    immunoid_tb_screen_result: ImmunoTBScreenState = Field(default="unknown", alias="immunoidTbScreenResult")
    immunoid_tb_endemic_exposure: Optional[bool] = Field(default=None, alias="immunoidTbEndemicExposure")
    immunoid_strongyloides_exposure: Optional[bool] = Field(default=None, alias="immunoidStrongyloidesExposure")
    immunoid_strongyloides_igg: ImmunoSerologyState = Field(default="unknown", alias="immunoidStrongyloidesIgg")
    immunoid_coccidioides_exposure: Optional[bool] = Field(default=None, alias="immunoidCoccidioidesExposure")
    immunoid_histoplasma_exposure: Optional[bool] = Field(default=None, alias="immunoidHistoplasmaExposure")
    immunoid_signal_sources: Dict[str, ImmunoExposureSource] = Field(default_factory=dict, alias="immunoidSignalSources")
    pretest_factor_ids: List[str] = Field(default_factory=list, alias="pretestFactorIds")
    pretest_factor_labels: List[str] = Field(default_factory=list, alias="pretestFactorLabels")
    parser_strategy: Literal["auto", "rule", "local", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")
    patient_context: Optional[SessionPatientContext] = Field(default=None, alias="patientContext")
    # Cross-module consult thread — carries clinical context across module transitions
    established_syndrome: Optional[str] = Field(default=None, alias="establishedSyndrome")
    consult_organisms: List[str] = Field(default_factory=list, alias="consultOrganisms")
    # Compact result snapshots — populated at each module completion, consumed by consult summary
    last_probid_summary: Optional[Dict] = Field(default=None, alias="lastProbidSummary")
    last_mechid_summary: Optional[Dict] = Field(default=None, alias="lastMechidSummary")
    last_doseid_summary: Optional[Dict] = Field(default=None, alias="lastDoseidSummary")
    last_allergy_summary: Optional[Dict] = Field(default=None, alias="lastAllergySummary")
    # Institutional antibiogram — loaded once per session, used to localise empiric therapy advice
    institutional_antibiogram: Optional[Dict] = Field(default=None, alias="institutionalAntibiogram")
    # HIV consult context — carries ART, viral load, CD4, resistance, and special population data
    hiv_context: Optional[Dict] = Field(default=None, alias="hivContext")
    # Clarifying question follow-up — saves intent + original message so the answer routes back correctly
    pending_intent: Optional[str] = Field(default=None, alias="pendingIntent")
    pending_intent_context: Optional[str] = Field(default=None, alias="pendingIntentContext")

    model_config = {"populate_by_name": True}


class AssistantTurnRequest(BaseModel):
    state: Optional[AssistantState] = None
    message: Optional[str] = None
    selection: Optional[str] = None
    parser_strategy: Optional[Literal["auto", "rule", "local", "openai"]] = Field(default=None, alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: Optional[bool] = Field(default=None, alias="allowFallback")

    model_config = {"populate_by_name": True}


class DoseIDAssistantPatientContext(BaseModel):
    age_years: Optional[int] = Field(default=None, alias="ageYears")
    sex: Optional[Literal["male", "female"]] = None
    total_body_weight_kg: Optional[float] = Field(default=None, alias="totalBodyWeightKg")
    height_cm: Optional[float] = Field(default=None, alias="heightCm")
    serum_creatinine_mg_dl: Optional[float] = Field(default=None, alias="serumCreatinineMgDl")
    crcl_ml_min: Optional[float] = Field(default=None, alias="crclMlMin")
    renal_mode: Literal["standard", "ihd", "crrt"] = Field(default="standard", alias="renalMode")

    model_config = {"populate_by_name": True}


class DoseIDAssistantAnalysis(BaseModel):
    medications: List[str] = Field(default_factory=list)
    patient_context: DoseIDAssistantPatientContext = Field(alias="patientContext")
    recommendations: List[DoseIDDoseRecommendation] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list, alias="missingInputs")
    follow_up_questions: List["DoseIDFollowUpQuestion"] = Field(default_factory=list, alias="followUpQuestions")
    provisional: bool = False
    provisional_reasons: List[str] = Field(default_factory=list, alias="provisionalReasons")

    model_config = {"populate_by_name": True}


AssistantAuthoritativeField = Literal["analysis", "mechidAnalysis", "immunoidAnalysis", "doseidAnalysis", "allergyidAnalysis"]


class AssistantInterfaceContract(BaseModel):
    interaction_model_role: Literal["llm_interface"] = Field(default="llm_interface", alias="interactionModelRole")
    deterministic_results_authoritative: bool = Field(default=True, alias="deterministicResultsAuthoritative")
    llm_can_change_deterministic_results: bool = Field(default=False, alias="llmCanChangeDeterministicResults")
    narration_source: Literal["deterministic_only", "llm_grounded_on_deterministic_payload"] = Field(
        default="deterministic_only",
        alias="narrationSource",
    )
    authoritative_workflow: Optional[Literal["probid", "mechid", "immunoid", "doseid", "allergyid"]] = Field(
        default=None,
        alias="authoritativeWorkflow",
    )
    authoritative_stage: Optional[str] = Field(default=None, alias="authoritativeStage")
    authoritative_result_fields: List[AssistantAuthoritativeField] = Field(
        default_factory=list,
        alias="authoritativeResultFields",
    )
    notes: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class AssistantTurnResponse(BaseModel):
    assistant_name: str = Field(default="ID Consultant Assistant", alias="assistantName")
    assistant_message: str = Field(alias="assistantMessage")
    assistant_narration_refined: bool = Field(default=False, alias="assistantNarrationRefined")
    state: AssistantState
    options: List[AssistantOption] = Field(default_factory=list)
    analysis: Optional[TextAnalyzeResponse] = None
    mechid_analysis: Optional["MechIDTextAnalyzeResponse"] = Field(default=None, alias="mechidAnalysis")
    immunoid_analysis: Optional[ImmunoAnalyzeResponse] = Field(default=None, alias="immunoidAnalysis")
    doseid_analysis: Optional[DoseIDAssistantAnalysis] = Field(default=None, alias="doseidAnalysis")
    allergyid_analysis: Optional[AntibioticAllergyAnalyzeResponse] = Field(default=None, alias="allergyidAnalysis")
    assistant_contract: Optional[AssistantInterfaceContract] = Field(default=None, alias="assistantContract")
    tips: List[str] = Field(default_factory=list)
    suggested_placeholder: Optional[str] = Field(default=None, alias="suggestedPlaceholder")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _populate_assistant_contract(self) -> "AssistantTurnResponse":
        if self.assistant_contract is not None:
            return self

        authoritative_result_fields: List[AssistantAuthoritativeField] = []
        if self.analysis is not None:
            authoritative_result_fields.append("analysis")
        if self.mechid_analysis is not None:
            authoritative_result_fields.append("mechidAnalysis")
        if self.immunoid_analysis is not None:
            authoritative_result_fields.append("immunoidAnalysis")
        if self.doseid_analysis is not None:
            authoritative_result_fields.append("doseidAnalysis")
        if self.allergyid_analysis is not None:
            authoritative_result_fields.append("allergyidAnalysis")

        notes = [
            "The language model may clarify, route, or rephrase, but it must not alter deterministic findings or recommendations.",
            "When authoritativeResultFields is non-empty, those structured payloads are the source of truth for the consult state and results.",
        ]
        if self.assistant_narration_refined:
            notes.append("assistantMessage was refined from deterministic payloads by the narration layer.")
        else:
            notes.append("assistantMessage is the direct deterministic or workflow-generated fallback message.")

        self.assistant_contract = AssistantInterfaceContract(
            narrationSource=(
                "llm_grounded_on_deterministic_payload"
                if self.assistant_narration_refined
                else "deterministic_only"
            ),
            authoritativeWorkflow=self.state.workflow,
            authoritativeStage=self.state.stage,
            authoritativeResultFields=authoritative_result_fields,
            notes=notes,
        )
        return self


class DoseIDPatientInput(BaseModel):
    age_years: int = Field(alias="ageYears", ge=18, le=120)
    sex: Literal["male", "female"]
    total_body_weight_kg: float = Field(alias="totalBodyWeightKg", gt=0)
    height_cm: float = Field(alias="heightCm", gt=0)
    serum_creatinine_mg_dl: float = Field(alias="serumCreatinineMgDl", gt=0)

    model_config = {"populate_by_name": True}


class DoseIDCalculatePatientInput(BaseModel):
    age_years: Optional[int] = Field(default=None, alias="ageYears", ge=18, le=120)
    sex: Optional[Literal["male", "female"]] = None
    total_body_weight_kg: Optional[float] = Field(default=None, alias="totalBodyWeightKg", gt=0)
    height_cm: Optional[float] = Field(default=None, alias="heightCm", gt=0)
    serum_creatinine_mg_dl: Optional[float] = Field(default=None, alias="serumCreatinineMgDl", gt=0)
    crcl_ml_min: Optional[float] = Field(default=None, alias="crclMlMin", gt=0)

    model_config = {"populate_by_name": True}


class DoseIDFollowUpQuestion(BaseModel):
    id: str
    prompt: str
    reason: str


class DoseIDIndicationOption(BaseModel):
    id: str
    label: str


class DoseIDDoseWeight(BaseModel):
    basis: Literal["tbw", "ibw", "adjbw", "lbw"]
    kg: float


class DoseIDMedicationSelection(BaseModel):
    medication_id: str = Field(alias="medicationId", min_length=1)
    indication_id: Optional[str] = Field(default=None, alias="indicationId")

    model_config = {"populate_by_name": True}


class DoseIDMedicationCatalogEntry(BaseModel):
    id: str
    name: str
    category: str
    indications: List[DoseIDIndicationOption] = Field(default_factory=list)
    source_pages: str = Field(alias="sourcePages")

    model_config = {"populate_by_name": True}


class DoseIDDoseRecommendation(BaseModel):
    medication_id: str = Field(alias="medicationId")
    medication_name: str = Field(alias="medicationName")
    category: str
    indication_id: str = Field(alias="indicationId")
    indication_label: str = Field(alias="indicationLabel")
    regimen: str
    renal_bucket: str = Field(alias="renalBucket")
    notes: List[str] = Field(default_factory=list)
    source_pages: str = Field(alias="sourcePages")
    dose_weight: Optional[DoseIDDoseWeight] = Field(default=None, alias="doseWeight")

    model_config = {"populate_by_name": True}


class DoseIDCalculateRequest(BaseModel):
    patient: DoseIDCalculatePatientInput
    renal_mode: Literal["standard", "ihd", "crrt"] = Field(default="standard", alias="renalMode")
    selections: List[DoseIDMedicationSelection] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DoseIDCalculateResponse(BaseModel):
    status: Literal["ready", "needs_more_info"] = "ready"
    recommendations: List[DoseIDDoseRecommendation] = Field(default_factory=list)
    patient_context: DoseIDAssistantPatientContext = Field(alias="patientContext")
    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list, alias="missingInputs")
    follow_up_questions: List[DoseIDFollowUpQuestion] = Field(default_factory=list, alias="followUpQuestions")

    model_config = {"populate_by_name": True}


class DoseIDTextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    parser_strategy: Literal["auto", "rule", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")

    model_config = {"populate_by_name": True}


class DoseIDTextParsedRequest(BaseModel):
    medications: List[DoseIDMedicationSelection] = Field(default_factory=list)
    patient_context: DoseIDAssistantPatientContext = Field(alias="patientContext")

    model_config = {"populate_by_name": True}


class DoseIDTextAnalyzeResponse(BaseModel):
    parser: str = "rule-based-v1"
    text: str
    parsed_request: Optional[DoseIDTextParsedRequest] = Field(default=None, alias="parsedRequest")
    warnings: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")
    parser_fallback_used: bool = Field(default=False, alias="parserFallbackUsed")
    analysis: Optional[DoseIDAssistantAnalysis] = None

    model_config = {"populate_by_name": True}


class DoseIDCatalogResponse(BaseModel):
    medications: List[DoseIDMedicationCatalogEntry] = Field(default_factory=list)


class MechIDDoseContext(BaseModel):
    patient: DoseIDPatientInput
    renal_mode: Literal["standard", "ihd", "crrt"] = Field(default="standard", alias="renalMode")
    max_suggestions: int = Field(default=3, alias="maxSuggestions", ge=1, le=6)

    model_config = {"populate_by_name": True}


class MechIDTxContext(BaseModel):
    syndrome: str = "Not specified"
    severity: str = "Not specified"
    focus_detail: str = Field(default="Not specified", alias="focusDetail")
    oral_preference: bool = Field(default=False, alias="oralPreference")
    carbapenemase_result: str = Field(default="Not specified", alias="carbapenemaseResult")
    carbapenemase_class: str = Field(default="Not specified", alias="carbapenemaseClass")

    model_config = {"populate_by_name": True}


class MechIDAnalyzeRequest(BaseModel):
    organism: str = Field(min_length=1)
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: Optional[MechIDTxContext] = Field(default=None, alias="txContext")
    dose_context: Optional[MechIDDoseContext] = Field(default=None, alias="doseContext")

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
    treatment_duration_guidance: List[str] = Field(default_factory=list, alias="treatmentDurationGuidance")
    monitoring_recommendations: List[str] = Field(default_factory=list, alias="monitoringRecommendations")
    dosing_recommendations: List[DoseIDDoseRecommendation] = Field(default_factory=list, alias="dosingRecommendations")
    references: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MechIDTextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    parser_strategy: Literal["auto", "rule", "openai"] = Field(default="auto", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")

    model_config = {"populate_by_name": True}


class MechIDImageAnalyzeRequest(BaseModel):
    image_data_url: str = Field(min_length=1, alias="imageDataUrl")
    filename: Optional[str] = None
    parser_model: Optional[str] = Field(default=None, alias="parserModel")

    model_config = {"populate_by_name": True}


class AntibiogramUploadRequest(BaseModel):
    image_data_url: str = Field(min_length=1, alias="imageDataUrl")
    filename: Optional[str] = None
    state: Optional["AssistantState"] = None

    model_config = {"populate_by_name": True}


class MechIDTextParsedRequest(BaseModel):
    organism: Optional[str] = None
    mentioned_organisms: List[str] = Field(default_factory=list, alias="mentionedOrganisms")
    resistance_phenotypes: List[str] = Field(default_factory=list, alias="resistancePhenotypes")
    susceptibility_results: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResults")
    tx_context: MechIDTxContext = Field(default_factory=MechIDTxContext, alias="txContext")

    model_config = {"populate_by_name": True}


class MechIDProvisionalAdvice(BaseModel):
    summary: str
    recommended_options: List[str] = Field(default_factory=list, alias="recommendedOptions")
    oral_options: List[str] = Field(default_factory=list, alias="oralOptions")
    missing_susceptibilities: List[str] = Field(default_factory=list, alias="missingSusceptibilities")
    notes: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MechIDTextAnalyzeResponse(BaseModel):
    parser: str = "rule-based-v1"
    text: str
    parsed_request: Optional[MechIDTextParsedRequest] = Field(default=None, alias="parsedRequest")
    warnings: List[str] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False, alias="requiresConfirmation")
    parser_fallback_used: bool = Field(default=False, alias="parserFallbackUsed")
    analysis: Optional[MechIDAnalyzeResponse] = None
    provisional_advice: Optional[MechIDProvisionalAdvice] = Field(default=None, alias="provisionalAdvice")

    model_config = {"populate_by_name": True}


class MechIDImageAnalyzeResponse(BaseModel):
    parser: str = "openai-mechid-image"
    image_filename: Optional[str] = Field(default=None, alias="imageFilename")
    source_summary: Optional[str] = Field(default=None, alias="sourceSummary")
    mechid_result: MechIDTextAnalyzeResponse = Field(alias="mechidResult")

    model_config = {"populate_by_name": True}


class MechIDTrainerParsedExpectation(BaseModel):
    organism: Optional[str] = None
    syndrome: Optional[str] = None
    severity: Optional[str] = None
    focus_detail: Optional[str] = Field(default=None, alias="focusDetail")
    oral_preference: Optional[bool] = Field(default=None, alias="oralPreference")
    carbapenemase_result: Optional[str] = Field(default=None, alias="carbapenemaseResult")
    carbapenemase_class: Optional[str] = Field(default=None, alias="carbapenemaseClass")
    mentioned_organisms_contains: List[str] = Field(default_factory=list, alias="mentionedOrganismsContains")
    resistance_phenotypes_contains: List[str] = Field(default_factory=list, alias="resistancePhenotypesContains")
    susceptibility_results_subset: Dict[str, ASTResult] = Field(default_factory=dict, alias="susceptibilityResultsSubset")

    model_config = {"populate_by_name": True}


class MechIDTrainerEvalCase(BaseModel):
    id: str
    text: str
    parser_strategy: str = Field(default="rule", alias="parserStrategy")
    assistant_guidance: Optional[str] = Field(default=None, alias="assistantGuidance")
    assistant_review_target: Optional[str] = Field(default=None, alias="assistantReviewTarget")
    assistant_final_target: Optional[str] = Field(default=None, alias="assistantFinalTarget")
    expected_requires_confirmation: Optional[bool] = Field(default=None, alias="expectedRequiresConfirmation")
    expected_parsed: Optional[MechIDTrainerParsedExpectation] = Field(default=None, alias="expectedParsed")
    expected_analysis_present: Optional[bool] = Field(default=None, alias="expectedAnalysisPresent")
    expected_provisional_present: Optional[bool] = Field(default=None, alias="expectedProvisionalPresent")
    expected_mechanisms_contains: List[str] = Field(default_factory=list, alias="expectedMechanismsContains")
    expected_therapy_notes_contains: List[str] = Field(default_factory=list, alias="expectedTherapyNotesContains")
    expected_final_results_subset: Dict[str, ASTResult] = Field(default_factory=dict, alias="expectedFinalResultsSubset")
    expected_recommended_options_contains: List[str] = Field(default_factory=list, alias="expectedRecommendedOptionsContains")
    expected_oral_options_contains: List[str] = Field(default_factory=list, alias="expectedOralOptionsContains")
    expected_missing_susceptibilities_contains: List[str] = Field(default_factory=list, alias="expectedMissingSusceptibilitiesContains")
    assistant_review_contains: List[str] = Field(default_factory=list, alias="assistantReviewContains")
    assistant_final_contains: List[str] = Field(default_factory=list, alias="assistantFinalContains")
    notes: Optional[str] = None

    model_config = {"populate_by_name": True}


class MechIDTrainerEvalPatch(BaseModel):
    id: Optional[str] = None
    parser_strategy: Optional[str] = Field(default=None, alias="parserStrategy")
    assistant_guidance: Optional[str] = Field(default=None, alias="assistantGuidance")
    assistant_review_target: Optional[str] = Field(default=None, alias="assistantReviewTarget")
    assistant_final_target: Optional[str] = Field(default=None, alias="assistantFinalTarget")
    expected_requires_confirmation: Optional[bool] = Field(default=None, alias="expectedRequiresConfirmation")
    expected_parsed: Optional[MechIDTrainerParsedExpectation] = Field(default=None, alias="expectedParsed")
    expected_analysis_present: Optional[bool] = Field(default=None, alias="expectedAnalysisPresent")
    expected_provisional_present: Optional[bool] = Field(default=None, alias="expectedProvisionalPresent")
    expected_mechanisms_contains: Optional[List[str]] = Field(default=None, alias="expectedMechanismsContains")
    expected_therapy_notes_contains: Optional[List[str]] = Field(default=None, alias="expectedTherapyNotesContains")
    expected_final_results_subset: Optional[Dict[str, ASTResult]] = Field(default=None, alias="expectedFinalResultsSubset")
    expected_recommended_options_contains: Optional[List[str]] = Field(default=None, alias="expectedRecommendedOptionsContains")
    expected_oral_options_contains: Optional[List[str]] = Field(default=None, alias="expectedOralOptionsContains")
    expected_missing_susceptibilities_contains: Optional[List[str]] = Field(default=None, alias="expectedMissingSusceptibilitiesContains")
    assistant_review_contains: Optional[List[str]] = Field(default=None, alias="assistantReviewContains")
    assistant_final_contains: Optional[List[str]] = Field(default=None, alias="assistantFinalContains")
    notes: Optional[str] = None

    model_config = {"populate_by_name": True}


class MechIDTrainerPreviewRequest(BaseModel):
    text: str = Field(min_length=1)
    correction_text: Optional[str] = Field(default=None, alias="correctionText")
    recommendation_text: Optional[str] = Field(default=None, alias="recommendationText")
    parser_strategy: Literal["auto", "rule", "openai"] = Field(default="rule", alias="parserStrategy")
    parser_model: Optional[str] = Field(default=None, alias="parserModel")
    allow_fallback: bool = Field(default=True, alias="allowFallback")

    model_config = {"populate_by_name": True}


class MechIDTrainerPreviewResponse(BaseModel):
    mechid_result: MechIDTextAnalyzeResponse = Field(alias="mechidResult")
    assistant_review_message: str = Field(alias="assistantReviewMessage")
    assistant_review_refined: bool = Field(default=False, alias="assistantReviewRefined")
    assistant_final_message: str = Field(alias="assistantFinalMessage")
    assistant_final_refined: bool = Field(default=False, alias="assistantFinalRefined")
    draft_case: MechIDTrainerEvalCase = Field(alias="draftCase")
    correction_applied: bool = Field(default=False, alias="correctionApplied")
    correction_warning: Optional[str] = Field(default=None, alias="correctionWarning")
    recommendation_applied: bool = Field(default=False, alias="recommendationApplied")
    recommendation_warning: Optional[str] = Field(default=None, alias="recommendationWarning")

    model_config = {"populate_by_name": True}


class MechIDTrainerSaveRequest(BaseModel):
    draft_case: MechIDTrainerEvalCase = Field(alias="draftCase")

    model_config = {"populate_by_name": True}


class MechIDTrainerSaveResponse(BaseModel):
    saved: bool = True
    path: str
    case_id: str = Field(alias="caseId")
    total_cases: int = Field(alias="totalCases")

    model_config = {"populate_by_name": True}


class MechIDTrainerCaseSummary(BaseModel):
    id: str
    text_preview: str = Field(alias="textPreview")
    parser_strategy: str = Field(alias="parserStrategy")

    model_config = {"populate_by_name": True}


class MechIDTrainerCaseListResponse(BaseModel):
    cases: List[MechIDTrainerCaseSummary] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MechIDTrainerDeleteResponse(BaseModel):
    deleted: bool = True
    case_id: str = Field(alias="caseId")
    total_cases: int = Field(alias="totalCases")

    model_config = {"populate_by_name": True}


class MechIDTrainerEvaluateRequest(BaseModel):
    draft_case: MechIDTrainerEvalCase = Field(alias="draftCase")
    check_assistant: bool = Field(default=False, alias="checkAssistant")

    model_config = {"populate_by_name": True}


class MechIDTrainerEvaluateResponse(BaseModel):
    passed: bool
    case_id: str = Field(alias="caseId")
    failures: List[str] = Field(default_factory=list)
    success: int
    total: int
    parsed_checks: int = Field(alias="parsedChecks")
    parsed_passes: int = Field(alias="parsedPasses")
    analysis_checks: int = Field(alias="analysisChecks")
    analysis_passes: int = Field(alias="analysisPasses")
    provisional_checks: int = Field(alias="provisionalChecks")
    provisional_passes: int = Field(alias="provisionalPasses")
    assistant_checks: int = Field(alias="assistantChecks")
    assistant_passes: int = Field(alias="assistantPasses")

    model_config = {"populate_by_name": True}
