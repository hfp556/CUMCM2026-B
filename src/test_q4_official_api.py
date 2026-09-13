import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.new import Q3_active_v2 as q3
from src.new import Q4_active as q4
from src.new import api_utils as api


class _FakeResponse:
    def __init__(self, status, body):
        self._status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self._status

    def read(self):
        return self._body


class Q4OfficialApiTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.log_path = Path(self._temporary.name) / "q4.jsonl"
        api.configure(
            base_url="http://127.0.0.1:2026",
            robot_id="202619058021",
            log_path=self.log_path,
        )

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _measure_payload(request_id="measure-fixed"):
        payload = api.base(request_id)
        payload["position"] = {"x": 12.0, "y": -8.0}
        payload["channel"] = 4
        return payload

    def test_success_checks_http_and_business_status_and_redacts_log(self):
        body = json.dumps(
            {
                "accepted": True,
                "real_timestamp_ms": 1,
                "virtual_time_s": 5,
                "measure_result": "no_signal",
            }
        ).encode("utf-8")
        with patch.object(api, "urlopen", return_value=_FakeResponse(200, body)):
            response = api.post("/measure", self._measure_payload())

        self.assertTrue(response["accepted"])
        self.assertEqual(200, response["_http_status"])
        record = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual("<redacted>", record["payload"]["robot_id"])
        self.assertEqual("measure-fixed", record["payload"]["request_id"])

    def test_transport_failure_retries_the_identical_payload_and_request_id(self):
        payload = self._measure_payload("same-id")
        success_body = json.dumps(
            {
                "accepted": True,
                "real_timestamp_ms": 2,
                "virtual_time_s": 5,
                "measure_result": "no_signal",
            }
        ).encode("utf-8")
        calls = [URLError("temporary disconnect"), _FakeResponse(200, success_body)]
        previous_post = q3.post
        try:
            q3.post = api.post
            with patch.object(api, "urlopen", side_effect=calls) as mocked_urlopen:
                with patch.object(q3.time, "sleep"):
                    response = q3._post("/measure", payload)
        finally:
            q3.post = previous_post

        self.assertTrue(response["accepted"])
        self.assertEqual(2, mocked_urlopen.call_count)
        sent_payloads = [
            json.loads(call.args[0].data.decode("utf-8"))
            for call in mocked_urlopen.call_args_list
        ]
        self.assertEqual(sent_payloads[0], sent_payloads[1])
        self.assertEqual("same-id", sent_payloads[0]["request_id"])

    def test_formed_http_error_is_returned_and_not_retried(self):
        payload = self._measure_payload("http-error")
        body = json.dumps(
            {"accepted": False, "real_timestamp_ms": 3, "virtual_time_s": 0}
        ).encode("utf-8")
        error = HTTPError(
            "http://127.0.0.1:2026/measure",
            409,
            "Conflict",
            None,
            io.BytesIO(body),
        )
        previous_post = q3.post
        try:
            q3.post = api.post
            with patch.object(api, "urlopen", side_effect=error) as mocked_urlopen:
                response = q3._post("/measure", payload)
        finally:
            q3.post = previous_post

        self.assertFalse(response["accepted"])
        self.assertEqual(409, response["_http_status"])
        self.assertEqual(1, mocked_urlopen.call_count)

    def test_runtime_guard_uses_enter_remaining_duration(self):
        self.assertEqual(
            1198,
            q4._runtime_limit_from_enter({"remaining_real_duration_s": 1200}),
        )
        self.assertEqual(
            598,
            q4._runtime_limit_from_enter({"remaining_real_duration_s": 600}),
        )
        self.assertEqual(
            8,
            q4._runtime_limit_from_enter({"remaining_real_duration_s": 10}),
        )
        self.assertEqual(
            0,
            q4._runtime_limit_from_enter({"remaining_real_duration_s": 1}),
        )


if __name__ == "__main__":
    unittest.main()
