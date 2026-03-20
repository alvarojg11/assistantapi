from __future__ import annotations

import argparse
import copy
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_DATASET = BACKEND_ROOT / "app" / "data" / "assistant_eval_cases.json"
DEFAULT_REMOTE_BASE_URL = "https://assistantapi-production.up.railway.app"


class EvalClient:
    def post(self, path: str, payload: Dict[str, Any]) -> tuple[int, Any, str]:
        raise NotImplementedError


class LocalEvalClient(EvalClient):
    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        self._client = TestClient(app)

    def post(self, path: str, payload: Dict[str, Any]) -> tuple[int, Any, str]:
        response = self._client.post(path, json=payload)
        raw_text = response.text
        body: Any = None
        if raw_text:
            try:
                body = response.json()
            except Exception:
                body = None
        return response.status_code, body, raw_text


class RemoteEvalClient(EvalClient):
    def __init__(self, *, base_url: str, timeout: float, insecure: bool) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()

    def post(self, path: str, payload: Dict[str, Any]) -> tuple[int, Any, str]:
        url = f"{self._base_url}{path if path.startswith('/') else '/' + path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=self._ctx, timeout=self._timeout) as response:
                raw_text = response.read().decode("utf-8")
                body: Any = None
                if raw_text:
                    try:
                        body = json.loads(raw_text)
                    except Exception:
                        body = None
                return response.status, body, raw_text
        except urllib.error.HTTPError as err:
            raw_text = err.read().decode("utf-8", errors="replace")
            body: Any = None
            if raw_text:
                try:
                    body = json.loads(raw_text)
                except Exception:
                    body = None
            return err.code, body, raw_text
        except (urllib.error.URLError, TimeoutError, socket.timeout) as err:
            raw_text = f"Remote request failed: {type(err).__name__}: {err}"
            return 599, {"error": raw_text}, raw_text


@dataclass
class CaseFailure:
    case_id: str
    turn_index: int
    reason: str
    transcript: List[str] = field(default_factory=list)


@dataclass
class EvalStats:
    total_cases: int = 0
    passed_cases: int = 0
    total_turns: int = 0
    passed_turns: int = 0
    failures: List[CaseFailure] = field(default_factory=list)


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        raise ValueError(f"Dataset must be a list of cases: {path}")
    return loaded


def _json_path_get(value: Any, path: str) -> Any:
    current = value
    for segment in path.split("."):
        if isinstance(current, list):
            if not segment.isdigit():
                raise KeyError(f"Expected list index at '{segment}' for path '{path}'")
            index = int(segment)
            current = current[index]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            raise KeyError(f"Cannot descend into '{segment}' for path '{path}'")
    return current


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, dict, tuple, set)):
        return len(value) > 0
    return True


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return all(_contains(actual, item) for item in expected)
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, list):
        return any(_contains(item, expected) for item in actual)
    if isinstance(actual, dict):
        return any(_contains(item, expected) for item in actual.values())
    return actual == expected


def _option_values(body: Dict[str, Any]) -> List[str]:
    options = body.get("options") or []
    return [str(option.get("value")) for option in options if isinstance(option, dict) and option.get("value") is not None]


def _check_list_item_rule(body: Dict[str, Any], rule: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    path = str(rule.get("path") or "")
    if not path:
        return ["list_item_checks entry is missing 'path'"]

    try:
        items = _json_path_get(body, path)
    except Exception as exc:
        return [f"{path}: could not resolve list path ({exc})"]

    if not isinstance(items, list):
        return [f"{path}: expected a list, got {type(items).__name__}"]

    where = rule.get("where") or {}
    matching_items: List[Any] = []
    for item in items:
        try:
            if all(_json_path_get(item, item_path) == expected for item_path, expected in where.items()):
                matching_items.append(item)
        except Exception:
            continue

    if not matching_items:
        return [f"{path}: no list item matched selector {where}"]

    selected = matching_items[0]
    for item_path, expected in (rule.get("equals") or {}).items():
        try:
            actual = _json_path_get(selected, item_path)
        except Exception as exc:
            failures.append(f"{path}: selected item missing '{item_path}' ({exc})")
            continue
        if actual != expected:
            failures.append(f"{path}: expected selected item {item_path} == {expected!r}, got {actual!r}")

    for item_path, expected in (rule.get("contains") or {}).items():
        try:
            actual = _json_path_get(selected, item_path)
        except Exception as exc:
            failures.append(f"{path}: selected item missing '{item_path}' ({exc})")
            continue
        if not _contains(actual, expected):
            failures.append(f"{path}: expected selected item {item_path} to contain {expected!r}, got {actual!r}")

    for item_path in (rule.get("nonempty_paths") or []):
        try:
            actual = _json_path_get(selected, item_path)
        except Exception as exc:
            failures.append(f"{path}: selected item missing '{item_path}' ({exc})")
            continue
        if not _is_nonempty(actual):
            failures.append(f"{path}: expected selected item {item_path} to be non-empty")

    return failures


def _check_expectations(body: Dict[str, Any], expect: Dict[str, Any]) -> List[str]:
    failures: List[str] = []

    any_of = expect.get("any_of") or []
    if any_of:
        branch_reasons: List[str] = []
        matched = False
        for index, branch in enumerate(any_of, start=1):
            branch_failures = _check_expectations(body, branch)
            if not branch_failures:
                matched = True
                break
            branch_reasons.append(f"branch {index}: {branch_failures[0]}")
        if not matched:
            preview = "; ".join(branch_reasons[:3])
            failures.append(f"any_of: no branches matched ({preview})")

    expected_state = expect.get("state") or {}
    actual_state = body.get("state") or {}
    for key, expected in expected_state.items():
        actual = actual_state.get(key)
        if actual != expected:
            failures.append(f"state.{key}: expected {expected!r}, got {actual!r}")

    options = _option_values(body)
    for value in expect.get("options_include", []) or []:
        if value not in options:
            failures.append(f"options: expected to include {value!r}, got {options!r}")
    for value in expect.get("options_exclude", []) or []:
        if value in options:
            failures.append(f"options: expected to exclude {value!r}, got {options!r}")

    message = str(body.get("assistantMessage") or "")
    for token in expect.get("message_contains", []) or []:
        if token not in message:
            failures.append(f"assistantMessage: expected to contain {token!r}")
    for token in expect.get("message_not_contains", []) or []:
        if token in message:
            failures.append(f"assistantMessage: expected not to contain {token!r}")

    for path, expected in (expect.get("json_equals") or {}).items():
        try:
            actual = _json_path_get(body, path)
        except Exception as exc:
            failures.append(f"{path}: could not resolve path ({exc})")
            continue
        if actual != expected:
            failures.append(f"{path}: expected {expected!r}, got {actual!r}")

    for path, expected in (expect.get("json_contains") or {}).items():
        try:
            actual = _json_path_get(body, path)
        except Exception as exc:
            failures.append(f"{path}: could not resolve path ({exc})")
            continue
        if not _contains(actual, expected):
            failures.append(f"{path}: expected to contain {expected!r}, got {actual!r}")

    for path in expect.get("nonempty_paths", []) or []:
        try:
            actual = _json_path_get(body, path)
        except Exception as exc:
            failures.append(f"{path}: could not resolve path ({exc})")
            continue
        if not _is_nonempty(actual):
            failures.append(f"{path}: expected non-empty value")

    for rule in expect.get("list_item_checks", []) or []:
        failures.extend(_check_list_item_rule(body, rule))

    return failures


def _render_body_for_transcript(body: Any, raw_text: str, limit: int = 600) -> str:
    if body is not None:
        rendered = json.dumps(body, indent=2, ensure_ascii=True)
    else:
        rendered = raw_text
    rendered = rendered.strip()
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _run_case(
    client: EvalClient,
    case: Dict[str, Any],
    *,
    show_transcripts: bool,
    stop_on_failure: bool,
) -> tuple[bool, int, List[CaseFailure]]:
    case_id = str(case.get("id") or "unnamed_case")
    turns = case.get("turns") or []
    endpoint_default = str(case.get("endpoint") or "/v1/assistant/turn")
    state: Any = None
    transcript: List[str] = []
    failures: List[CaseFailure] = []

    for index, turn in enumerate(turns, start=1):
        request_payload = copy.deepcopy(turn.get("request") or {})
        carry_state = turn.get("carry_state", True)
        endpoint = str(turn.get("endpoint") or endpoint_default)
        if carry_state and state is not None and "state" not in request_payload:
            request_payload["state"] = state

        transcript.append(f"TURN {index} REQUEST {json.dumps(request_payload, ensure_ascii=True)}")
        status_code, body, raw_text = client.post(endpoint, request_payload)
        transcript.append(
            f"TURN {index} RESPONSE status={status_code} body={_render_body_for_transcript(body, raw_text)}"
        )

        expected_status = int((turn.get("expect") or {}).get("http_status", 200))
        if status_code != expected_status:
            failures.append(
                CaseFailure(
                    case_id=case_id,
                    turn_index=index,
                    reason=f"expected HTTP {expected_status}, got {status_code}",
                    transcript=list(transcript),
                )
            )
            if stop_on_failure:
                break
            return False, index, failures

        if body is None:
            failures.append(
                CaseFailure(
                    case_id=case_id,
                    turn_index=index,
                    reason="response was not valid JSON",
                    transcript=list(transcript),
                )
            )
            if stop_on_failure:
                break
            return False, index, failures

        expectation_failures = _check_expectations(body, turn.get("expect") or {})
        if expectation_failures:
            for reason in expectation_failures:
                failures.append(
                    CaseFailure(
                        case_id=case_id,
                        turn_index=index,
                        reason=reason,
                        transcript=list(transcript),
                    )
                )
            if stop_on_failure:
                break
            return False, index, failures

        if turn.get("save_state", True):
            state = body.get("state")

    passed = not failures
    if passed and show_transcripts:
        print(f"PASS {case_id}")
        for line in transcript:
            print(f"  {line}")
        print()
    return passed, len(turns), failures


def _build_client(args: argparse.Namespace) -> EvalClient:
    if args.target == "remote":
        return RemoteEvalClient(
            base_url=args.base_url,
            timeout=args.timeout,
            insecure=args.insecure,
        )
    return LocalEvalClient()


def _iter_selected_cases(cases: Sequence[Dict[str, Any]], needle: str | None) -> Iterable[Dict[str, Any]]:
    if not needle:
        yield from cases
        return
    lowered = needle.lower()
    for case in cases:
        case_id = str(case.get("id") or "").lower()
        description = str(case.get("description") or "").lower()
        if lowered in case_id or lowered in description:
            yield case


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay and validate multi-turn assistant cases against the local app or a live base URL."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to assistant evaluation dataset JSON/JSONL (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--target",
        choices=["local", "remote"],
        default="local",
        help="Evaluate against the in-process FastAPI app or a remote base URL.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_REMOTE_BASE_URL,
        help=f"Remote base URL when --target remote (default: {DEFAULT_REMOTE_BASE_URL})",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run cases whose id or description contains this substring.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the selected case set N times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Remote request timeout in seconds.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for remote requests.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failure details and transcripts.",
    )
    parser.add_argument(
        "--show-transcripts",
        action="store_true",
        help="Print full transcripts for passing cases too.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop at the first failed turn.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    cases = list(_iter_selected_cases(_load_cases(args.dataset), args.filter))
    if not cases:
        raise SystemExit("No cases matched the selected dataset/filter.")

    client = _build_client(args)
    stats = EvalStats()

    for round_index in range(args.repeat):
        for case in cases:
            case_id = str(case.get("id") or "unnamed_case")
            decorated_id = case_id if args.repeat == 1 else f"{case_id} [run {round_index + 1}]"
            case_copy = dict(case)
            case_copy["id"] = decorated_id
            stats.total_cases += 1
            passed, turn_count, failures = _run_case(
                client,
                case_copy,
                show_transcripts=args.show_transcripts,
                stop_on_failure=args.stop_on_failure,
            )
            stats.total_turns += turn_count
            if passed:
                stats.passed_cases += 1
                stats.passed_turns += turn_count
                if not args.show_transcripts:
                    print(f"PASS {decorated_id}")
            else:
                stats.failures.extend(failures)
                stats.passed_turns += max(0, turn_count - 1)
                print(f"FAIL {decorated_id}")
                if args.show_failures:
                    for failure in failures:
                        print(f"  turn {failure.turn_index}: {failure.reason}")
                    if failures and failures[0].transcript:
                        print("  transcript:")
                        for line in failures[0].transcript:
                            print(f"    {line}")
            if failures and args.stop_on_failure:
                break
        if stats.failures and args.stop_on_failure:
            break

    print()
    print(
        f"Cases: {stats.passed_cases}/{stats.total_cases} passed "
        f"({(stats.passed_cases / stats.total_cases * 100.0) if stats.total_cases else 0.0:.1f}%)"
    )
    print(
        f"Turns: {stats.passed_turns}/{stats.total_turns} passed "
        f"({(stats.passed_turns / stats.total_turns * 100.0) if stats.total_turns else 0.0:.1f}%)"
    )
    if stats.failures and args.show_failures:
        print(f"Failures logged: {len(stats.failures)}")

    return 1 if stats.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
