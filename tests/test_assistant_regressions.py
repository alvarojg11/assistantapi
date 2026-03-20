import unittest

from backend.app.main import (
    _assistant_build_doseid_analysis,
    _assistant_is_doseid_intent,
    assistant_turn,
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
