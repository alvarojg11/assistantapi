import unittest

from backend.app.services.mechid_llm_parser import MechIDLLMExtractionPayload, _canonicalize_extraction


class MechIDImageFallbackTests(unittest.TestCase):
    def test_named_carbapenemase_resistance_phenotype_sets_tx_context(self) -> None:
        payload = MechIDLLMExtractionPayload(
            organism="Klebsiella pneumoniae",
            resistancePhenotypes=["KPC"],
            susceptibilityResults={"Meropenem": "Resistant"},
            txContext={"carbapenemaseResult": "Not specified", "carbapenemaseClass": "Not specified"},
        )

        normalized = _canonicalize_extraction(payload)
        tx_context = normalized["txContext"]

        self.assertEqual(tx_context["carbapenemaseResult"], "Positive")
        self.assertEqual(tx_context["carbapenemaseClass"], "KPC")

    def test_visible_text_fallback_recovers_carbapenemase_class(self) -> None:
        payload = MechIDLLMExtractionPayload(
            organism="Escherichia coli",
            susceptibilityResults={
                "Meropenem": "Resistant",
                "Aztreonam": "Resistant",
                "Cefiderocol": "Susceptible",
            },
            txContext={"carbapenemaseResult": "Not specified", "carbapenemaseClass": "Not specified"},
            visibleText=(
                "LAB AST REPORT\n"
                "ORGANISM: ESCHERICHIA COLI\n"
                "MEROPENEM R\n"
                "AZTREONAM R\n"
                "CEFIDEROCOL S\n"
                "NDM POSITIVE\n"
            ),
        )

        normalized = _canonicalize_extraction(payload)
        tx_context = normalized["txContext"]

        self.assertEqual(tx_context["carbapenemaseResult"], "Positive")
        self.assertEqual(tx_context["carbapenemaseClass"], "NDM")
        self.assertIn("NDM carbapenemase", normalized["resistancePhenotypes"])
        self.assertTrue(
            any("Visible text fallback" in warning and "carbapenemase class" in warning.lower() for warning in normalized["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
