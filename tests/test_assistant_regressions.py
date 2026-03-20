import unittest
from unittest import mock

from backend.app.main import (
    _assistant_case_text_for_parser,
    _assistant_build_doseid_analysis,
    _assistant_detect_doseid_medication_ids,
    _assistant_is_doseid_intent,
    assistant_turn,
    store,
)
from backend.app.schemas import AssistantTurnRequest, AntibioticAllergyTextAnalyzeRequest
from backend.app.services.antibiotic_allergy_service import parse_antibiotic_allergy_text


MECHID_SDD_MESSAGE = (
    "Please interpret this susceptibility pattern. "
    "E coli bloodstream isolate. Ceftriaxone resistant, cefepime susceptible dose dependent, "
    "piperacillin-tazobactam susceptible, meropenem susceptible, ciprofloxacin resistant."
)

MECHID_AST_MESSAGE = (
    "E coli bloodstream isolate. Ceftriaxone resistant, cefepime susceptible, "
    "piperacillin-tazobactam susceptible, meropenem susceptible, ciprofloxacin resistant."
)


class AssistantRegressionTests(unittest.TestCase):
    def test_mechid_ast_message_does_not_route_to_doseid(self) -> None:
        self.assertFalse(_assistant_is_doseid_intent(MECHID_SDD_MESSAGE))

        response = assistant_turn(AssistantTurnRequest(message=MECHID_SDD_MESSAGE))

        self.assertEqual(response.state.workflow, "mechid")
        self.assertEqual(response.state.stage, "mechid_confirm")
        self.assertIsNotNone(response.mechid_analysis)

    def test_mechid_describe_turn_no_longer_crashes_on_tx_context_snapshot(self) -> None:
        start = assistant_turn(AssistantTurnRequest(selection="mechid"))

        response = assistant_turn(
            AssistantTurnRequest(
                state=start.state,
                message=MECHID_AST_MESSAGE,
            )
        )

        self.assertEqual(response.state.workflow, "mechid")
        self.assertEqual(response.state.stage, "mechid_confirm")
        self.assertIsNotNone(response.mechid_analysis)
        self.assertIn("add_more_details", [option.value for option in response.options])

    def test_allergy_followup_from_vague_prompt_does_not_crash(self) -> None:
        start = assistant_turn(
            AssistantTurnRequest(
                message="Can you help with antibiotic allergy compatibility? The patient says they had a penicillin allergy."
            )
        )

        response = assistant_turn(
            AssistantTurnRequest(
                state=start.state,
                message="Remote childhood amoxicillin rash only, no anaphylaxis, now needs cefazolin.",
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)

    def test_immunoid_prompt_with_quantiferon_not_hijacked_by_biomarker_shortcut(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "Can you review immunosuppression prophylaxis? "
                    "Rituximab plus prednisone 20 mg daily planned for 6 weeks. "
                    "HBsAg negative, anti-HBc positive, anti-HBs negative. QuantiFERON negative."
                )
            )
        )

        self.assertEqual(response.state.workflow, "immunoid")
        self.assertEqual(response.state.stage, "immunoid_collect_context")
        self.assertIsNotNone(response.immunoid_analysis)

    def test_explicit_doseid_prompts_with_indication_language_stay_in_doseid(self) -> None:
        prompts = (
            "Can you help with levofloxacin dosing? 74 kg adult, CrCl 36 mL/min, pneumonia.",
            "Can you help with fluconazole dosing? 80 kg adult, CrCl 35 mL/min, candidemia.",
            "Can you help with acyclovir IV dosing? 78 kg adult, CrCl 41 mL/min, HSV encephalitis.",
            "Can you help with ertapenem dosing? 79 kg adult, CrCl 29 mL/min, ESBL urinary infection.",
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                response = assistant_turn(AssistantTurnRequest(message=prompt))
                self.assertEqual(response.state.workflow, "doseid")
                self.assertEqual(response.state.stage, "doseid_describe")
                self.assertIsNotNone(response.doseid_analysis)

    def test_humanized_doseid_prompt_with_sentence_punctuation_still_detects_drug(self) -> None:
        message = (
            "Quick dosing gut-check on Cefepime. 72 kg adult patient, creatinine clearance 45 mL/min, "
            "severe infection. What would you do?"
        )

        self.assertEqual(_assistant_detect_doseid_medication_ids(message), ["cefepime"])

        response = assistant_turn(AssistantTurnRequest(message=message))

        self.assertEqual(response.state.workflow, "doseid")
        self.assertEqual(response.state.stage, "doseid_describe")
        self.assertIsNotNone(response.doseid_analysis)
        self.assertIn("Cefepime", response.doseid_analysis.medications)

    def test_q_in_quickly_does_not_trigger_false_doseid_on_allergy_case(self) -> None:
        message = (
            "This one sounds more convincing for IgE: amoxicillin caused hives and throat tightness "
            "pretty quickly. We are thinking about cefepime now."
        )

        self.assertFalse(_assistant_is_doseid_intent(message))

        response = assistant_turn(AssistantTurnRequest(message=message))

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)

    def test_guided_probid_describe_turn_does_not_jump_to_allergyid(self) -> None:
        cases = (
            ("spinal_epidural_abscess", "sea_low", "back pain, fever, spinal tenderness, neurologic deficit"),
            (
                "necrotizing_soft_tissue_infection",
                "nsti_low",
                "pain out of proportion, rapid progression, hypotension, bullae, crepitus",
            ),
        )

        for module_id, preset_id, message in cases:
            with self.subTest(module_id=module_id):
                state = assistant_turn(AssistantTurnRequest()).state
                state = assistant_turn(AssistantTurnRequest(state=state, selection="probid")).state
                state = assistant_turn(AssistantTurnRequest(state=state, selection=module_id)).state
                state = assistant_turn(AssistantTurnRequest(state=state, selection=preset_id)).state
                state = assistant_turn(AssistantTurnRequest(state=state, selection="continue_to_case")).state

                response = assistant_turn(AssistantTurnRequest(state=state, message=message))

                self.assertEqual(response.state.workflow, "probid")
                self.assertEqual(response.state.module_id, module_id)
                self.assertEqual(response.state.stage, "describe_case")
                self.assertIn("Added that", response.assistant_message)

    def test_specific_klebsiella_species_do_not_collapse_into_polymicrobial_parse(self) -> None:
        cases = (
            (
                "Klebsiella oxytoca",
                "Please interpret this susceptibility pattern. Klebsiella oxytoca bloodstream isolate. "
                "Ceftriaxone resistant, cefepime susceptible, meropenem susceptible, ciprofloxacin resistant.",
            ),
            (
                "Klebsiella aerogenes",
                "Please interpret this susceptibility pattern. Klebsiella aerogenes bloodstream isolate. "
                "Ceftriaxone resistant, cefepime susceptible, piperacillin-tazobactam resistant, meropenem susceptible.",
            ),
        )

        for organism, prompt in cases:
            with self.subTest(organism=organism):
                response = assistant_turn(AssistantTurnRequest(message=prompt))

                self.assertEqual(response.state.workflow, "mechid")
                self.assertEqual(response.state.stage, "mechid_confirm")
                self.assertIsNotNone(response.mechid_analysis)
                self.assertIsNotNone(response.mechid_analysis.parsed_request)
                self.assertEqual(response.mechid_analysis.parsed_request.organism, organism)
                self.assertTrue(response.mechid_analysis.parsed_request.susceptibility_results)

    def test_conversational_pneumonia_treatment_question_does_not_jump_to_mechid(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="Could you help me think through what antibiotics you would start for what looks like pneumonia?"
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "cap")

    def test_immunoid_context_prompt_not_hijacked_by_generic_prophylaxis_handler(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "Can you help me sort prophylaxis for rituximab plus prednisone 20 mg daily for 6 weeks? "
                    "Hep B surface antigen negative, core antibody positive, surface antibody negative, Quantiferon negative."
                )
            )
        )

        self.assertEqual(response.state.workflow, "immunoid")
        self.assertEqual(response.state.stage, "immunoid_collect_context")
        self.assertIsNotNone(response.immunoid_analysis)

    def test_generic_allergy_history_request_routes_to_allergyid_not_delabeling(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "Can you help me untangle an antibiotic allergy list? The chart just says penicillin allergy "
                    "and I do not really trust it."
                )
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)

    def test_prefixed_generic_allergy_history_request_stays_in_allergyid(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "Phone call summary from family was basically: Can you help me untangle an antibiotic allergy list? "
                    "The chart just says penicillin allergy and I do not really trust it."
                )
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)

    def test_prefixed_vancomycin_infusion_reaction_stays_in_allergyid(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "Phone call summary from family was basically: The chart says vancomycin allergy, but the story "
                    "sounds like flushing and itching during the infusion rather than a real allergy. "
                    "Is ceftriaxone still fine?"
                )
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)
        self.assertIn("Ceftriaxone", {item.agent for item in response.allergyid_analysis.recommendations})

    def test_allergy_followup_now_needs_candidate_drug_survives_merge(self) -> None:
        start = assistant_turn(
            AssistantTurnRequest(
                message="Chart allergy story is messy. Can you help me sort an antibiotic allergy list? The chart mostly just says penicillin allergy."
            )
        )

        response = assistant_turn(
            AssistantTurnRequest(
                state=start.state,
                message="Trying to clean up a probably-wrong allergy label before rounds. Immediate hives and throat tightness after penicillin, now needs cefepime.",
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)
        self.assertIn("Cefepime", {item.agent for item in response.allergyid_analysis.recommendations})

    def test_allergy_followup_can_i_still_use_candidate_drug_survives_merge(self) -> None:
        start = assistant_turn(
            AssistantTurnRequest(
                message="Trying to clean up a probably-wrong allergy label before rounds. Can you help me sort an antibiotic allergy list? The chart mostly just says penicillin allergy."
            )
        )

        response = assistant_turn(
            AssistantTurnRequest(
                state=start.state,
                message="Sorry this is copied straight from signout. Vancomycin infusion reaction only, can I still use ceftriaxone?",
            )
        )

        self.assertEqual(response.state.workflow, "allergyid")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.allergyid_analysis)
        self.assertIn("Ceftriaxone", {item.agent for item in response.allergyid_analysis.recommendations})

    def test_conversational_spinal_epidural_abscess_route_does_not_trip_followup_testing(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="Quick steer before rounds: Back pain plus fever plus new weakness, and I am worried about a spinal epidural abscess."
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "spinal_epidural_abscess")
        self.assertIn(response.state.stage, {"select_preset", "confirm_case"})
        self.assertNotIn("TEE", response.assistant_message)

    def test_spinal_epidural_abscess_mri_positive_shorthand_is_normalized_for_guided_parse(self) -> None:
        module = store.get("spinal_epidural_abscess")
        self.assertIsNotNone(module)

        normalized = _assistant_case_text_for_parser(
            module,
            "Let me give the case in chunks.\nMicro-wise, MRI positive",
        )

        self.assertIn("MRI spine shows epidural abscess or phlegmon", normalized)

    def test_humanized_guided_spinal_epidural_abscess_case_reaches_done(self) -> None:
        request = AssistantTurnRequest()
        response = assistant_turn(request)
        response = assistant_turn(AssistantTurnRequest(state=response.state, selection="probid"))
        response = assistant_turn(AssistantTurnRequest(state=response.state, selection="spinal_epidural_abscess"))
        response = assistant_turn(AssistantTurnRequest(state=response.state, selection="sea_low"))
        response = assistant_turn(AssistantTurnRequest(state=response.state, selection="continue_to_case"))

        for message in (
            "Let me give the case in chunks: back pain, fever, spinal tenderness, neurologic deficit",
            "A little more color: ESR elevated, CRP elevated",
            "Sorry, another detail I should have mentioned: blood cultures positive",
            "Micro-wise, MRI positive",
        ):
            response = assistant_turn(AssistantTurnRequest(state=response.state, message=message))
            if response.state.stage == "describe_case":
                response = assistant_turn(AssistantTurnRequest(state=response.state, selection="continue_case_draft"))

        response = assistant_turn(AssistantTurnRequest(state=response.state, selection="run_assessment"))

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "spinal_epidural_abscess")
        self.assertEqual(response.state.stage, "done")
        self.assertIsNotNone(response.analysis)
        self.assertTrue(response.analysis.analysis.applied_findings)

    def test_run_assessment_uses_cached_probid_parse_instead_of_reparsing(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="ED patient with fever, cough, hypoxemia, and chest xray consolidation. Please help me assess pneumonia."
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.stage, "confirm_case")
        self.assertIsNotNone(response.state.probid_cached_case_result)

        with mock.patch("backend.app.main._assistant_parse_case_text", side_effect=AssertionError("should not reparse")):
            final = assistant_turn(AssistantTurnRequest(state=response.state, selection="run_assessment"))

        self.assertEqual(final.state.workflow, "probid")
        self.assertEqual(final.state.stage, "done")
        self.assertIsNotNone(final.analysis)
        self.assertIsNotNone(final.analysis.analysis)


class AllergyParserRegressionTests(unittest.TestCase):
    def test_no_anaphylaxis_phrase_does_not_upgrade_remote_rash(self) -> None:
        parsed = parse_antibiotic_allergy_text(
            AntibioticAllergyTextAnalyzeRequest(
                text="Remote childhood amoxicillin rash only, no anaphylaxis, now needs cefazolin."
            )
        )

        self.assertEqual(len(parsed.parsed_request.allergy_entries), 1)
        entry = parsed.parsed_request.allergy_entries[0]
        self.assertEqual(entry.reported_agent, "Amoxicillin")
        self.assertEqual(entry.reaction_type, "benign_delayed_rash")
        self.assertEqual(entry.timing, "delayed")
        self.assertEqual(parsed.analysis.overall_risk, "low")


class DoseIDRegressionTests(unittest.TestCase):
    def test_ripe_uses_total_body_weight_for_pyrazinamide_and_ethambutol(self) -> None:
        result = _assistant_build_doseid_analysis(
            "RIPE dosing for 62 kg, CrCl 35, female, 165 cm."
        )

        recommendations = {item.medication_name: item for item in result.recommendations}

        pyrazinamide = recommendations["Pyrazinamide"]
        ethambutol = recommendations["Ethambutol"]

        self.assertIsNotNone(pyrazinamide.dose_weight)
        self.assertIsNotNone(ethambutol.dose_weight)
        self.assertEqual(pyrazinamide.dose_weight.basis, "tbw")
        self.assertEqual(ethambutol.dose_weight.basis, "tbw")
        self.assertAlmostEqual(pyrazinamide.dose_weight.kg, 62.0)
        self.assertAlmostEqual(ethambutol.dose_weight.kg, 62.0)
        self.assertEqual(pyrazinamide.regimen, "1200 mg PO daily")
        self.assertEqual(ethambutol.regimen, "1200 mg PO daily")


if __name__ == "__main__":
    unittest.main()
