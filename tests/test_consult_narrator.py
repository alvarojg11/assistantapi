import unittest
from unittest.mock import patch

from backend.app.schemas import DoseIDAssistantAnalysis, DoseIDAssistantPatientContext
from backend.app.services.consult_narrator import (
    _build_grounding_envelope,
    narrate_doseid_assistant_message,
)


class ConsultNarratorTests(unittest.TestCase):
    def test_grounding_envelope_marks_deterministic_payload_authoritative(self) -> None:
        envelope = _build_grounding_envelope(
            workflow="doseid",
            stage="assistant",
            fallback_message="fallback",
            deterministic_payload={"doseidAnalysis": {"medications": ["cefepime"]}},
            examples=[{"input": "example", "output": "example output"}],
            extra_context={"moduleLabel": "DoseID"},
        )

        self.assertEqual(envelope["assistantContract"]["interactionModelRole"], "llm_interface")
        self.assertTrue(envelope["assistantContract"]["deterministicResultsAuthoritative"])
        self.assertFalse(envelope["assistantContract"]["llmCanChangeDeterministicResults"])
        self.assertEqual(envelope["task"]["fallbackMessage"], "fallback")
        self.assertIn("deterministicPayload", envelope)
        self.assertIn("styleExamples", envelope)
        self.assertIn("context", envelope)

    def test_doseid_narration_uses_standardized_grounding_payload(self) -> None:
        captured: dict = {}

        def fake_call_consult_model(*, prompt, payload, model=None):
            captured["prompt"] = prompt
            captured["payload"] = payload
            return "Narrated answer"

        doseid_result = DoseIDAssistantAnalysis(
            medications=["cefepime"],
            patientContext=DoseIDAssistantPatientContext(renalMode="standard"),
        )

        with patch("backend.app.services.consult_narrator.consult_narration_enabled", return_value=True):
            with patch("backend.app.services.consult_narrator._call_consult_model", side_effect=fake_call_consult_model):
                message, refined = narrate_doseid_assistant_message(
                    doseid_result=doseid_result,
                    fallback_message="fallback",
                )

        self.assertEqual(message, "Narrated answer")
        self.assertTrue(refined)
        self.assertIn("assistantContract", captured["payload"])
        self.assertIn("task", captured["payload"])
        self.assertIn("deterministicPayload", captured["payload"])
        self.assertEqual(captured["payload"]["task"]["workflow"], "doseid")
        self.assertEqual(captured["payload"]["task"]["stage"], "assistant")
        self.assertEqual(captured["payload"]["task"]["fallbackMessage"], "fallback")
        self.assertIn("doseidAnalysis", captured["payload"]["deterministicPayload"])
        self.assertIn("deterministicPayload", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
