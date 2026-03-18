import unittest

from backend.app.main import _assistant_rewrite_doseid_followup_reply


class DoseIDFollowUpRewriteTests(unittest.TestCase):
    def test_multi_field_reply_collects_demographics_and_creatinine(self) -> None:
        rewritten = _assistant_rewrite_doseid_followup_reply(
            "foscarnet",
            "female, 65, 70 kg, 165 cm, creatinine 1.2",
        )

        self.assertIn("age 65", rewritten)
        self.assertIn("female", rewritten)
        self.assertIn("70 kg", rewritten)
        self.assertIn("165 cm", rewritten)
        self.assertIn("serum creatinine 1.2", rewritten)

    def test_multi_field_reply_does_not_misread_age_as_renal_value(self) -> None:
        rewritten = _assistant_rewrite_doseid_followup_reply(
            "cefepime",
            "female, 65, creatinine 1.2",
        )

        self.assertIn("age 65", rewritten)
        self.assertIn("female", rewritten)
        self.assertIn("serum creatinine 1.2", rewritten)
        self.assertNotIn("crcl 65", rewritten)

    def test_explicit_single_field_reply_keeps_creatinine_label(self) -> None:
        rewritten = _assistant_rewrite_doseid_followup_reply(
            "foscarnet",
            "creatinine 1.2",
        )

        self.assertEqual(rewritten, "serum creatinine 1.2")

    def test_gender_label_and_apostrophe_height_are_understood(self) -> None:
        rewritten = _assistant_rewrite_doseid_followup_reply(
            "foscarnet",
            "gender: F, age 42, wt 60 kg, height 5'8\", cr 0.9",
        )

        self.assertIn("age 42", rewritten)
        self.assertIn("female", rewritten)
        self.assertIn("60 kg", rewritten)
        self.assertIn("5 ft 8 in", rewritten)
        self.assertIn("serum creatinine 0.9", rewritten)


if __name__ == "__main__":
    unittest.main()
