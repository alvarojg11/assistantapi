import importlib.util
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


def _load_evaluate_assistant_cases_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_assistant_cases.py"
    module_name = "backend.scripts.evaluate_assistant_cases"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class EvaluateAssistantCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_evaluate_assistant_cases_module()

    def test_body_stability_snapshot_ignores_narration_and_option_labels(self) -> None:
        first = {
            "assistantMessage": "First phrasing",
            "assistantNarrationRefined": "More words",
            "state": {"workflow": "doseid", "stage": "doseid_describe"},
            "options": [{"value": "restart", "label": "Start over"}],
            "doseidAnalysis": {
                "medications": ["Cefepime"],
                "followUpQuestions": ["Any dialysis?"],
            },
        }
        second = {
            "assistantMessage": "Completely different phrasing",
            "assistantNarrationRefined": "Different narration",
            "state": {"workflow": "doseid", "stage": "doseid_describe"},
            "options": [{"value": "restart", "label": "Begin again"}],
            "doseidAnalysis": {
                "medications": ["Cefepime"],
                "followUpQuestions": ["Any dialysis?"],
            },
        }

        self.assertEqual(
            self.module._body_stability_snapshot(first),
            self.module._body_stability_snapshot(second),
        )

    def test_build_stability_failures_detects_structured_drift(self) -> None:
        baseline = self.module.CaseRunResult(
            passed=True,
            turn_count=1,
            transcript=["TURN 1 REQUEST {}", "TURN 1 RESPONSE status=200 body={}"],
            stability_snapshots=['{"state":{"stage":"done","workflow":"probid"}}'],
        )
        current = self.module.CaseRunResult(
            passed=True,
            turn_count=1,
            transcript=["TURN 1 REQUEST {}", "TURN 1 RESPONSE status=200 body={}"],
            stability_snapshots=['{"state":{"stage":"select_preset","workflow":"probid"}}'],
        )

        failures = self.module._build_stability_failures(
            case_id="case_a [run 2]",
            baseline_run=baseline,
            current_run=current,
            baseline_label="run 1",
            current_label="run 2",
        )

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].turn_index, 1)
        self.assertIn("stability drift", failures[0].reason)
        self.assertIn("run 1", failures[0].transcript[-2])
        self.assertIn("run 2", failures[0].transcript[-1])

    def test_remote_eval_client_returns_599_on_timeout(self) -> None:
        client = self.module.RemoteEvalClient(
            base_url="https://example.com",
            timeout=1.0,
            insecure=False,
        )

        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            status, body, raw_text = client.post("/v1/assistant/turn", {"message": "hello"})

        self.assertEqual(status, 599)
        self.assertIsInstance(body, dict)
        self.assertIn("TimeoutError", raw_text)
        self.assertIn("TimeoutError", body["error"])

    def test_remote_eval_client_returns_599_on_url_error(self) -> None:
        client = self.module.RemoteEvalClient(
            base_url="https://example.com",
            timeout=1.0,
            insecure=False,
        )

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("network down")):
            status, body, raw_text = client.post("/v1/assistant/turn", {"message": "hello"})

        self.assertEqual(status, 599)
        self.assertIsInstance(body, dict)
        self.assertIn("URLError", raw_text)
        self.assertIn("URLError", body["error"])


if __name__ == "__main__":
    unittest.main()
