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
