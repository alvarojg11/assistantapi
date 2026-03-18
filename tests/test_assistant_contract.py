import unittest

from backend.app.schemas import AssistantState, AssistantTurnResponse, DoseIDAssistantAnalysis


class AssistantContractTests(unittest.TestCase):
    def test_contract_is_auto_populated_for_plain_response(self) -> None:
        response = AssistantTurnResponse(
            assistantMessage="Ready for the next question.",
            state=AssistantState(),
        )

        self.assertIsNotNone(response.assistant_contract)
        assert response.assistant_contract is not None
        self.assertEqual(response.assistant_contract.interaction_model_role, "llm_interface")
        self.assertTrue(response.assistant_contract.deterministic_results_authoritative)
        self.assertFalse(response.assistant_contract.llm_can_change_deterministic_results)
        self.assertEqual(response.assistant_contract.authoritative_workflow, "probid")
        self.assertEqual(response.assistant_contract.authoritative_stage, "select_module")
        self.assertEqual(response.assistant_contract.authoritative_result_fields, [])
        self.assertEqual(response.assistant_contract.narration_source, "deterministic_only")

    def test_contract_tracks_refined_narration_and_authoritative_fields(self) -> None:
        response = AssistantTurnResponse(
            assistantMessage="I refined the dosing explanation.",
            assistantNarrationRefined=True,
            state=AssistantState(workflow="doseid", stage="doseid_describe"),
            doseidAnalysis=DoseIDAssistantAnalysis(
                medications=["cefepime"],
                patientContext={"renalMode": "standard"},
            ),
        )

        self.assertIsNotNone(response.assistant_contract)
        assert response.assistant_contract is not None
        self.assertEqual(response.assistant_contract.authoritative_workflow, "doseid")
        self.assertEqual(response.assistant_contract.authoritative_stage, "doseid_describe")
        self.assertEqual(response.assistant_contract.authoritative_result_fields, ["doseidAnalysis"])
        self.assertEqual(
            response.assistant_contract.narration_source,
            "llm_grounded_on_deterministic_payload",
        )


if __name__ == "__main__":
    unittest.main()
