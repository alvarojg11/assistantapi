import unittest

from backend.app.main import (
    ALLERGYID_ASSISTANT_ID,
    DOSEID_ASSISTANT_ID,
    IMMUNOID_ASSISTANT_ID,
    _assistant_detect_consult_intent,
    _assistant_explicit_non_syndrome_workflow_request,
    _assistant_explicit_syndrome_module_request,
    _select_module_from_turn,
    assistant_turn,
)
from backend.app.schemas import AssistantTurnRequest


class AssistantRoutingTests(unittest.TestCase):
    def test_explicit_syndrome_request_routes_to_endocarditis(self) -> None:
        self.assertEqual(
            _assistant_explicit_syndrome_module_request("please evaluate for endocarditis"),
            "endo",
        )

    def test_free_text_question_routes_to_pneumonia_syndrome(self) -> None:
        selected = _select_module_from_turn(
            AssistantTurnRequest(message="does this patient have pneumonia?")
        )
        self.assertEqual(selected, "cap")

    def test_explicit_dosing_request_routes_to_doseid(self) -> None:
        self.assertEqual(
            _assistant_explicit_non_syndrome_workflow_request(
                "can you help with antimicrobial dosing?"
            ),
            DOSEID_ASSISTANT_ID,
        )

    def test_explicit_allergy_request_routes_to_allergyid(self) -> None:
        self.assertEqual(
            _assistant_explicit_non_syndrome_workflow_request(
                "can you help with allergy compatibility?"
            ),
            ALLERGYID_ASSISTANT_ID,
        )

    def test_explicit_immunoid_request_routes_to_immunoid(self) -> None:
        self.assertEqual(
            _assistant_explicit_non_syndrome_workflow_request(
                "can you help with immunosuppression prophylaxis?"
            ),
            IMMUNOID_ASSISTANT_ID,
        )

    def test_treatment_start_consult_intent_detected(self) -> None:
        self.assertEqual(
            _assistant_detect_consult_intent(
                "should id start antifungal treatment for this patient?"
            ),
            "treatment_decision",
        )

    def test_hold_therapy_consult_intent_detected(self) -> None:
        self.assertEqual(
            _assistant_detect_consult_intent(
                "can I hold off on antifungal therapy for now?"
            ),
            "treatment_decision",
        )

    def test_therapy_selection_consult_intent_detected(self) -> None:
        self.assertEqual(
            _assistant_detect_consult_intent(
                "what antifungal would you use for possible mold infection?"
            ),
            "therapy_selection",
        )

    def test_pneumonia_therapy_selection_consult_intent_detected(self) -> None:
        self.assertEqual(
            _assistant_detect_consult_intent(
                "what would you use for this pneumonia?"
            ),
            "therapy_selection",
        )

    def test_vancomycin_treatment_decision_consult_intent_detected(self) -> None:
        self.assertEqual(
            _assistant_detect_consult_intent(
                "should I start vancomycin?"
            ),
            "treatment_decision",
        )

    def test_generic_antifungal_treatment_question_prompts_fungal_clarification(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="should ID start antifungal treatment for this patient with possible fungal infection?"
            )
        )

        self.assertEqual(response.state.stage, "select_syndrome_module")
        self.assertIn("Bottom line:", response.assistant_message)
        self.assertIn("What I still need:", response.assistant_message)
        self.assertIn("fungal treatment consult", response.assistant_message.lower())
        self.assertIn("inv_candida", [option.value for option in response.options])
        self.assertIn("inv_mold", [option.value for option in response.options])

    def test_specific_mold_treatment_question_routes_into_invasive_mold_pathway(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "should I start mold-active therapy for this neutropenic patient with pulmonary nodules and positive galactomannan?"
                )
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "inv_mold")
        self.assertIn("mold-active therapy", response.assistant_message.lower())

    def test_hold_antifungal_question_prompts_fungal_clarification(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="can I hold off on antifungal therapy for now in this patient with possible fungal infection?"
            )
        )

        self.assertEqual(response.state.stage, "select_syndrome_module")
        self.assertIn("fungal treatment consult", response.assistant_message.lower())
        self.assertIn("inv_candida", [option.value for option in response.options])
        self.assertIn("inv_mold", [option.value for option in response.options])

    def test_specific_mold_therapy_selection_question_routes_into_invasive_mold_pathway(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message=(
                    "what antifungal would you use for a neutropenic patient with pulmonary nodules and positive galactomannan?"
                )
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "inv_mold")
        self.assertIn("therapy-selection consult", response.assistant_message.lower())

    def test_pneumonia_therapy_selection_question_routes_into_pneumonia_lane(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="what would you use for this pneumonia?"
            )
        )

        self.assertEqual(response.state.workflow, "probid")
        self.assertEqual(response.state.module_id, "cap")
        self.assertIn("therapy-selection consult", response.assistant_message.lower())

    def test_vancomycin_question_prompts_for_syndrome_source(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="should I start vancomycin?"
            )
        )

        self.assertEqual(response.state.stage, "select_syndrome_module")
        self.assertIn("Bottom line:", response.assistant_message)
        self.assertIn("What I would do now:", response.assistant_message)
        self.assertIn("vancomycin", response.assistant_message.lower())
        self.assertIn("syndrome", response.assistant_message.lower())

    def test_hold_antibiotics_question_prompts_for_syndrome_source(self) -> None:
        response = assistant_turn(
            AssistantTurnRequest(
                message="can I hold antibiotics for now?"
            )
        )

        self.assertEqual(response.state.stage, "select_syndrome_module")
        self.assertIn("treatment consult", response.assistant_message.lower())
        self.assertIn("source", response.assistant_message.lower())


if __name__ == "__main__":
    unittest.main()
