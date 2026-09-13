"""Shared official-simulator transport for the active Q3/Q4 controllers.

The controller-level retry loop treats only ``None`` as an indeterminate
transport failure.  A formed HTTP response, including 4xx/5xx, is returned as
a dictionary so it is not retried as though the connection had disappeared.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
from pathlib import Path
import socket
import sys
import threading
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_ROBOT_ID = "202619058021"
REQUEST_TIMEOUT_S = 5.0
DEFAULT_MAX_REAL_DURATION_S = 1200.0

BASE_URL = os.environ.get("CUMCM_SIM_URL", DEFAULT_BASE_URL).rstrip("/")
ROBOT_ID = os.environ.get("CUMCM_ROBOT_ID", DEFAULT_ROBOT_ID)

_log_lock = threading.Lock()
_request_attempts: dict[str, int] = {}
_jsonl_path: Path | None = None
_console_path: Path | None = None
_console_tee = None


def _contains_forbidden_identifier_character(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


def _validate_identifier(name: str, value: str, maximum_bytes: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    byte_length = len(value.encode("utf-8"))
    if not 1 <= byte_length <= maximum_bytes:
        raise ValueError(f"{name} must contain 1..{maximum_bytes} UTF-8 bytes")
    if _contains_forbidden_identifier_character(value):
        raise ValueError(f"{name} contains a control or format character")
    return value


def _validate_base_url(value: str) -> str:
    parsed = urlparse(str(value))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("CUMCM_SIM_URL must be an HTTP loopback address")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("CUMCM_SIM_URL must not contain a path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("CUMCM_SIM_URL contains an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("CUMCM_SIM_URL must include a valid port")
    return str(value).rstrip("/")


def configure(*, run_name="controller", base_url=None, robot_id=None, log_path=None):
    """Validate runtime settings and prepare per-run audit log paths."""
    global BASE_URL, ROBOT_ID, _jsonl_path, _console_path
    BASE_URL = _validate_base_url(BASE_URL if base_url is None else base_url)
    ROBOT_ID = _validate_identifier(
        "robot_id", ROBOT_ID if robot_id is None else robot_id, 64
    )
    safe_run_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(run_name)
    ).strip("_") or "controller"

    if log_path is None:
        configured = os.environ.get("CUMCM_RUN_LOG")
        if configured:
            candidate = Path(configured)
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            candidate = (
                Path.cwd()
                / "logs"
                / f"{safe_run_name}_{stamp}_{os.getpid()}.jsonl"
            )
    else:
        candidate = Path(log_path)
    candidate = candidate.expanduser().resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("a", encoding="utf-8", newline="\n"):
        pass
    _jsonl_path = candidate
    _console_path = candidate.with_suffix(".console.log")
    _request_attempts.clear()
    return {
        "base_url": BASE_URL,
        "robot_id": ROBOT_ID,
        "jsonl_path": str(_jsonl_path),
        "console_path": str(_console_path),
    }


def _redact_payload(payload):
    if not isinstance(payload, dict):
        return payload
    redacted = dict(payload)
    if "robot_id" in redacted:
        redacted["robot_id"] = "<redacted>"
    return redacted


def _write_event(event: dict) -> None:
    if _jsonl_path is None:
        return
    record = {"logged_at_utc": datetime.now(timezone.utc).isoformat(), **event}
    try:
        line = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with _log_lock:
            with _jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()
    except Exception as exc:
        print(f"simulator audit log write failed: {exc}", file=sys.stderr)


class _ConsoleTee:
    def __init__(self, original, path: Path):
        self._original = original
        self._stream = path.open("a", encoding="utf-8", newline="")
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "replace")

    def write(self, text):
        result = self._original.write(text)
        self._stream.write(text)
        self._stream.flush()
        return result

    def flush(self):
        self._original.flush()
        self._stream.flush()

    def isatty(self):
        return bool(getattr(self._original, "isatty", lambda: False)())

    def close_log(self):
        if sys.stdout is self:
            sys.stdout = self._original
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


def install_console_log() -> str:
    """Mirror subsequent stdout output to a durable per-run console log."""
    global _console_tee
    if _console_path is None:
        raise RuntimeError("configure() must be called before install_console_log()")
    if _console_tee is None:
        _console_tee = _ConsoleTee(sys.stdout, _console_path)
        sys.stdout = _console_tee
        atexit.register(_console_tee.close_log)
    return str(_console_path)


def runtime_limit_from_enter(response, *, reserve_s=2.0):
    """Return the usable duration from /enter without a 16/17-minute cap."""
    value = response.get("remaining_real_duration_s") if isinstance(response, dict) else None
    try:
        remaining = float(value)
    except (TypeError, ValueError):
        print(
            "warning: /enter response has no valid remaining_real_duration_s; "
            "using the documented 1200 s fallback"
        )
        remaining = DEFAULT_MAX_REAL_DURATION_S
    if not (remaining >= 0.0) or remaining == float("inf"):
        print(
            "warning: /enter response has invalid remaining_real_duration_s; "
            "using the documented 1200 s fallback"
        )
        remaining = DEFAULT_MAX_REAL_DURATION_S
    return max(0.0, remaining - max(0.0, float(reserve_s)))


def base(request_id):
    request_id = _validate_identifier("request_id", str(request_id), 128)
    return {"arena_id": "default", "robot_id": ROBOT_ID, "request_id": request_id}


def _decode_response(body: bytes, http_status: int) -> dict:
    try:
        response = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "accepted": False,
            "_http_status": int(http_status),
            "_client_error": f"invalid_json_response: {exc}",
        }
    if not isinstance(response, dict):
        return {
            "accepted": False,
            "_http_status": int(http_status),
            "_client_error": "response_is_not_a_json_object",
        }
    normalized = dict(response)
    normalized["_http_status"] = int(http_status)
    if http_status != 200:
        normalized["accepted"] = False
    elif not isinstance(normalized.get("accepted"), bool):
        normalized["accepted"] = False
        normalized["_client_error"] = "accepted_is_not_boolean"
    return normalized


def _attempt_number(payload) -> int:
    request_id = str(payload.get("request_id", "")) if isinstance(payload, dict) else ""
    with _log_lock:
        attempt = _request_attempts.get(request_id, 0) + 1
        _request_attempts[request_id] = attempt
    return attempt


def post(path, payload):
    """Send one attempt; return None only for a transport-level failure."""
    attempt = _attempt_number(payload)
    request_id = payload.get("request_id") if isinstance(payload, dict) else None
    event_base = {
        "path": path,
        "request_id": request_id,
        "attempt": attempt,
        "payload": _redact_payload(payload),
    }
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        response = {"accepted": False, "_client_error": f"invalid_request_json: {exc}"}
        _write_event({**event_base, "outcome": "client_error", "response": response})
        return response

    request = Request(
        BASE_URL + path,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_S) as http_response:
            status = int(http_response.getcode())
            body = http_response.read()
    except HTTPError as exc:
        status = int(exc.code)
        try:
            body = exc.read()
        except Exception:
            body = b""
        response = _decode_response(body, status)
        _write_event(
            {
                **event_base,
                "outcome": "http_response",
                "http_status": status,
                "response": response,
            }
        )
        return response
    except (
        URLError,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        RemoteDisconnected,
        IncompleteRead,
        OSError,
    ) as exc:
        _write_event(
            {
                **event_base,
                "outcome": "transport_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return None
    except Exception as exc:
        response = {
            "accepted": False,
            "_client_error": f"unexpected_client_error: {type(exc).__name__}: {exc}",
        }
        _write_event({**event_base, "outcome": "client_error", "response": response})
        return response

    response = _decode_response(body, status)
    _write_event(
        {
            **event_base,
            "outcome": "http_response",
            "http_status": status,
            "response": response,
        }
    )
    return response


def measure(x, y, channel, req_id):
    payload = base(req_id)
    payload["position"] = {"x": x, "y": y}
    payload["channel"] = channel
    return post("/measure", payload)
