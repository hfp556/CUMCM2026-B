"""Official-API-compatible local simulator with a browser experiment console.

The server deliberately has no third-party web dependency.  The simulation
rules are shared with ``src/q3_phase3_experiments.py`` so batch experiments and
interactive debugging exercise the same virtual world.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional
from urllib.parse import urlparse
import webbrowser


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q3_phase3_experiments import (  # noqa: E402
    OfflineSimulator,
    Scenario,
    Source,
    make_scenario,
)


VERSION_SCRIPTS = {
    "fast2": ROOT / "src" / "new" / "Q3_fast2.py",
    "phase1": ROOT / "src" / "experiments" / "q3" / "Q3_phase1.py",
    "active": ROOT / "src" / "new" / "Q3_active_v2.py",
}
ERROR_MODES = {"hash", "minus_one", "zero", "plus_one"}
BASE_FIELDS = {"arena_id", "robot_id", "request_id"}
ACTION_FIELDS = BASE_FIELDS | {"position", "channel"}


class RequestProblem(ValueError):
    """An HTTP 400 problem in an official simulator request."""


class InjectedTransportFailure(RuntimeError):
    """A deliberate connection drop used to test retry behavior."""


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _source_dict(source: Source) -> dict[str, Any]:
    return {
        "channel": int(source.channel),
        "position": [float(source.position[0]), float(source.position[1])],
        "reception_radius": float(source.reception_radius),
    }


def _scenario_from_payload(payload: Mapping[str, Any]) -> Scenario:
    seed = int(payload.get("seed", 20260913))
    raw_sources = payload.get("sources")
    if raw_sources is None:
        count = int(payload.get("source_count", 12))
        if not 10 <= count <= 16:
            raise ValueError("source_count 必须在 10 到 16 之间")
        return make_scenario(seed, n=count)

    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= 16:
        raise ValueError("sources 必须是含 1 到 16 个信号源的数组")
    sources: list[Source] = []
    channels: set[int] = set()
    for index, raw in enumerate(raw_sources, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"第 {index} 个信号源格式无效")
        channel = int(raw["channel"])
        position = raw["position"]
        reception = float(raw["reception_radius"])
        if channel in channels or not 1 <= channel <= 20:
            raise ValueError("信号源频道必须唯一且位于 1..20")
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or not all(_finite_number(value) for value in position)
        ):
            raise ValueError(f"第 {index} 个信号源坐标无效")
        x, y = float(position[0]), float(position[1])
        if math.hypot(x, y) > OfflineSimulator.TARGET_RADIUS_M + 1e-9:
            raise ValueError("信号源必须位于半径 1800 米的目标区域内")
        if not 1000.0 <= reception <= 1500.0:
            raise ValueError("有效接收半径必须在 1000 到 1500 米之间")
        sources.append(Source(channel, (x, y), reception))
        channels.add(channel)
    return Scenario(seed, tuple(sources))


class PlatformState:
    """Thread-safe simulator, process runner, and observable event history."""

    MAX_HISTORY = 4000
    MAX_OUTPUT_LINES = 4000

    def __init__(
        self,
        *,
        seed: int = 20260913,
        source_count: int = 12,
        error_mode: str = "hash",
        fail_transport_every: Optional[int] = None,
        force_first_clear_fail: bool = False,
    ) -> None:
        self.lock = threading.RLock()
        self.transactions: list[dict[str, Any]] = []
        self.output_lines: list[dict[str, Any]] = []
        self.process: Optional[subprocess.Popen[str]] = None
        self.run_status = "idle"
        self.run_mode = "single"
        self.run_version: Optional[str] = None
        self.run_returncode: Optional[int] = None
        self.run_started_at: Optional[float] = None
        self.stop_requested = False
        self.batch: dict[str, Any] = self._empty_batch()
        self.base_url = "http://127.0.0.1:2026"
        self.generation = 0
        self.robot_id: Optional[str] = None
        self.finished = False
        self.config: dict[str, Any] = {}
        self.scenario = Scenario(0, tuple())
        self.simulator = OfflineSimulator(self.scenario)
        self.configure(
            {
                "seed": seed,
                "source_count": source_count,
                "error_mode": error_mode,
                "fail_transport_every": fail_transport_every,
                "force_first_clear_fail": force_first_clear_fail,
            }
        )

    @staticmethod
    def _empty_batch() -> dict[str, Any]:
        return {
            "total": 0,
            "completed": 0,
            "current_index": 0,
            "current_seed": None,
            "seeds": [],
            "results": [],
        }

    def _ensure_not_running(self) -> None:
        if self.run_status in {"starting", "running", "stopping"}:
            raise RuntimeError("实验正在运行，请先停止当前程序")

    def _install_scenario(
        self,
        payload: Mapping[str, Any],
        *,
        clear_output: bool,
        reset_run: bool,
    ) -> None:
        scenario = _scenario_from_payload(payload)
        error_mode = str(payload.get("error_mode", "hash"))
        if error_mode not in ERROR_MODES:
            raise ValueError(f"error_mode 必须是 {sorted(ERROR_MODES)} 之一")
        raw_interval = payload.get("fail_transport_every")
        interval = None if raw_interval in (None, "", 0, "0") else int(raw_interval)
        if interval is not None and interval < 2:
            raise ValueError("网络故障间隔必须为空或至少为 2")
        force_fail = bool(payload.get("force_first_clear_fail", False))

        self.scenario = scenario
        self.simulator = OfflineSimulator(
            scenario,
            error_mode=error_mode,
            fail_transport_every=interval,
            force_first_clear_fail=force_fail,
        )
        self.config = {
            "seed": int(scenario.seed),
            "source_count": len(scenario.sources),
            "error_mode": error_mode,
            "fail_transport_every": interval,
            "force_first_clear_fail": force_fail,
            "custom_sources": "sources" in payload,
        }
        self.transactions = []
        if clear_output:
            self.output_lines = []
        self.robot_id = None
        self.finished = False
        if reset_run:
            self.run_status = "idle"
            self.run_mode = "single"
            self.run_version = None
            self.run_returncode = None
            self.run_started_at = None
            self.stop_requested = False
            self.batch = self._empty_batch()
        self.generation += 1

    def configure(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Create or import a scenario and reset every observable action."""

        with self.lock:
            self._ensure_not_running()
            self._install_scenario(payload, clear_output=True, reset_run=True)
            return self.snapshot()

    @staticmethod
    def _validate_request(path: str, payload: Mapping[str, Any]) -> None:
        if path not in {"/enter", "/measure", "/clear", "/exit"}:
            raise RequestProblem("未知接口")
        expected = ACTION_FIELDS if path in {"/measure", "/clear"} else BASE_FIELDS
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            details = []
            if missing:
                details.append(f"缺少字段: {', '.join(missing)}")
            if extra:
                details.append(f"未知字段: {', '.join(extra)}")
            raise RequestProblem("；".join(details))
        for name in ("robot_id", "request_id"):
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise RequestProblem(f"{name} 必须是非空字符串")
            if any(ord(char) < 32 for char in value):
                raise RequestProblem(f"{name} 不能包含控制字符")
        if path in {"/measure", "/clear"}:
            position = payload.get("position")
            if not isinstance(position, Mapping) or set(position) != {"x", "y"}:
                raise RequestProblem("position 必须且只能包含 x、y")
            if not all(_finite_number(position.get(axis)) for axis in ("x", "y")):
                raise RequestProblem("position.x 和 position.y 必须是有限数值")
            if any(abs(float(position[axis])) > 2_000_000 for axis in ("x", "y")):
                raise RequestProblem("坐标分量绝对值不得超过 2000000")
            channel = payload.get("channel")
            if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
                raise RequestProblem("channel 必须是 1..20 的整数")

    def _record_transaction(
        self,
        path: str,
        payload: Mapping[str, Any],
        response: Optional[Mapping[str, Any]],
        *,
        http_status: int = 200,
    ) -> None:
        self.transactions.append(
            {
                "sequence": len(self.transactions) + 1,
                "wall_time": time.strftime("%H:%M:%S"),
                "path": path,
                "request_id": payload.get("request_id"),
                "payload": copy.deepcopy(dict(payload)),
                "response": None if response is None else copy.deepcopy(dict(response)),
                "http_status": int(http_status),
            }
        )
        if len(self.transactions) > self.MAX_HISTORY:
            del self.transactions[: len(self.transactions) - self.MAX_HISTORY]

    def record_bad_request(self, path: str, payload: Mapping[str, Any], message: str) -> None:
        with self.lock:
            self._record_transaction(
                path,
                payload,
                {"accepted": False, "error": message},
                http_status=400,
            )

    def handle_official(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one official endpoint call or raise a transport drop."""

        self._validate_request(path, payload)
        with self.lock:
            request_id = str(payload["request_id"])
            cached = request_id in self.simulator._request_cache

            if payload["arena_id"] != "default":
                response = {"accepted": False, "virtual_time_s": 0, "error": "invalid_arena_id"}
            elif self.robot_id is not None and payload["robot_id"] != self.robot_id:
                response = {"accepted": False, "virtual_time_s": 0, "error": "robot_id_mismatch"}
            elif path == "/enter" and self.finished and not cached:
                response = {"accepted": False, "virtual_time_s": 0, "error": "test_finished"}
            elif path == "/enter" and self.simulator.entered and not cached:
                response = {"accepted": False, "virtual_time_s": 0, "error": "already_entered"}
            else:
                response = self.simulator.post(path, payload)
                if response is None:
                    self._record_transaction(path, payload, None, http_status=0)
                    raise InjectedTransportFailure("已注入一次网络中断")
                if path == "/enter" and response.get("accepted") is True:
                    self.robot_id = str(payload["robot_id"])
                    response.update(
                        {
                            "max_virtual_duration_s": 360000,
                            "max_real_duration_s": 1200,
                            "remaining_real_duration_s": 1200,
                        }
                    )
                if path == "/exit" and response.get("accepted") is True:
                    self.finished = True
                    response["exit_reason"] = "user_exit"

            self._record_transaction(path, payload, response)
            return response

    def _append_output(self, line: str, stream: str = "stdout") -> None:
        with self.lock:
            self.output_lines.append(
                {
                    "sequence": len(self.output_lines) + 1,
                    "wall_time": time.strftime("%H:%M:%S"),
                    "stream": stream,
                    "text": line.rstrip("\r\n"),
                }
            )
            if len(self.output_lines) > self.MAX_OUTPUT_LINES:
                del self.output_lines[: len(self.output_lines) - self.MAX_OUTPUT_LINES]

    def start_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Start one whitelisted Q3 entry point in a background subprocess."""

        if str(payload.get("run_mode", "single")) == "batch":
            return self.start_batch(payload)

        version = str(payload.get("version", "active"))
        if version not in VERSION_SCRIPTS:
            raise ValueError(f"未知程序版本: {version}")
        with self.lock:
            self._ensure_not_running()
            if not bool(payload.get("reuse_scenario", False)):
                self.configure(payload)
            elif self.simulator.entered or self.finished or self.simulator.actions:
                # Reusing means reusing source truth, not stale robot state.
                config = dict(self.config)
                config["sources"] = [_source_dict(item) for item in self.scenario.sources]
                self.configure(config)

            script = VERSION_SCRIPTS[version]
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONUTF8": "1",
                    "PYTHONUNBUFFERED": "1",
                    "CUMCM_SIM_URL": self.base_url,
                }
            )
            self.run_status = "starting"
            self.run_mode = "single"
            self.run_version = version
            self.run_returncode = None
            self.run_started_at = time.time()
            self.stop_requested = False
            self.batch = self._empty_batch()
            self._append_output(f"▶ 启动 {script.name}", "platform")
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(script)],
                    cwd=str(script.parent),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except Exception:
                self.run_status = "failed"
                self.process = None
                raise
            self.run_status = "running"
            generation = self.generation
            thread = threading.Thread(
                target=self._watch_process,
                args=(self.process, generation),
                name=f"q3-{version}-output",
                daemon=True,
            )
            thread.start()
            return self.snapshot()

    def start_batch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run one selected Q3 version over twelve deterministic seeds."""

        version = str(payload.get("version", "active"))
        if version not in VERSION_SCRIPTS:
            raise ValueError(f"未知程序版本: {version}")
        base_seed = int(payload.get("seed", 20260913))
        seeds = list(range(base_seed, base_seed + 12))
        with self.lock:
            self._ensure_not_running()
            # Validate all shared scenario options before launching the worker.
            validation_payload = dict(payload)
            validation_payload["seed"] = seeds[0]
            validation_payload.pop("sources", None)
            _scenario_from_payload(validation_payload)
            error_mode = str(payload.get("error_mode", "hash"))
            if error_mode not in ERROR_MODES:
                raise ValueError(f"error_mode 必须是 {sorted(ERROR_MODES)} 之一")
            raw_interval = payload.get("fail_transport_every")
            interval = None if raw_interval in (None, "", 0, "0") else int(raw_interval)
            if interval is not None and interval < 2:
                raise ValueError("网络故障间隔必须为空或至少为 2")

            self.output_lines = []
            self.transactions = []
            self.run_status = "starting"
            self.run_mode = "batch"
            self.run_version = version
            self.run_returncode = None
            self.run_started_at = time.time()
            self.stop_requested = False
            self.batch = {
                "total": 12,
                "completed": 0,
                "current_index": 0,
                "current_seed": None,
                "seeds": seeds,
                "results": [],
            }
            self._append_output(
                f"▶ 批量运行 {VERSION_SCRIPTS[version].name}：12 个种子 {seeds[0]}..{seeds[-1]}",
                "platform",
            )
            thread = threading.Thread(
                target=self._batch_worker,
                args=(version, dict(payload), seeds),
                name=f"q3-{version}-batch",
                daemon=True,
            )
            self.run_status = "running"
            thread.start()
            return self.snapshot()

    def _batch_worker(
        self,
        version: str,
        payload: dict[str, Any],
        seeds: list[int],
    ) -> None:
        script = VERSION_SCRIPTS[version]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "CUMCM_SIM_URL": self.base_url,
            }
        )
        for index, seed in enumerate(seeds, start=1):
            with self.lock:
                if self.stop_requested:
                    break
                scenario_payload = dict(payload)
                scenario_payload["seed"] = seed
                scenario_payload.pop("sources", None)
                self._install_scenario(
                    scenario_payload,
                    clear_output=False,
                    reset_run=False,
                )
                self.batch["current_index"] = index
                self.batch["current_seed"] = seed
                self._append_output(
                    f"── 第 {index}/12 局 · seed {seed} ──",
                    "platform",
                )
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(script)],
                        cwd=str(script.parent),
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                except Exception as exc:
                    self.batch["results"].append(
                        {
                            "seed": seed,
                            "returncode": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self.batch["completed"] = index
                    continue
                self.process = process

            assert process.stdout is not None
            for line in process.stdout:
                self._append_output(line)
            returncode = process.wait()
            with self.lock:
                summary = self.simulator.summary()
                successful = bool(
                    returncode == 0
                    and summary["cleared_count"] == summary["source_count"]
                )
                self.batch["results"].append(
                    {
                        "seed": seed,
                        "returncode": int(returncode),
                        "successful": successful,
                        **summary,
                    }
                )
                self.batch["completed"] = index
                self.process = None
                self._append_output(
                    f"局 {index}/12 完成：{summary['cleared_count']}/{summary['source_count']} 清除，"
                    f"虚拟用时 {summary['total_virtual_time_s']:.1f} s",
                    "platform",
                )
                if self.stop_requested:
                    break

        with self.lock:
            self.process = None
            self.batch["current_seed"] = None
            summary = self._batch_summary()
            self.run_returncode = (
                0
                if self.batch["completed"] == 12 and summary["process_failures"] == 0
                else (None if self.stop_requested else 1)
            )
            self.run_status = "stopped" if self.stop_requested else "completed"
            if summary["average_virtual_time_s"] is not None:
                self._append_output(
                    f"■ 批量完成：{summary['completed']}/12 局，平均虚拟用时 "
                    f"{summary['average_virtual_time_s']:.1f} s，完整清除 "
                    f"{summary['successful']}/{summary['completed']} 局",
                    "platform",
                )

    def _batch_summary(self) -> dict[str, Any]:
        results = self.batch["results"]
        timed = [item for item in results if "total_virtual_time_s" in item]

        def average(name: str) -> Optional[float]:
            values = [float(item[name]) for item in timed if name in item]
            return None if not values else sum(values) / len(values)

        return {
            "total": int(self.batch["total"]),
            "completed": int(self.batch["completed"]),
            "current_index": int(self.batch["current_index"]),
            "current_seed": self.batch["current_seed"],
            "seeds": list(self.batch["seeds"]),
            "successful": sum(bool(item.get("successful")) for item in results),
            "process_failures": sum(item.get("returncode") not in (0,) for item in results),
            "average_virtual_time_s": average("total_virtual_time_s"),
            "average_move_m": average("total_move_m"),
            "average_action_count": (
                None
                if not timed
                else sum(
                    int(item.get("measure_count", 0))
                    + int(item.get("clear_attempt_count", 0))
                    for item in timed
                )
                / len(timed)
            ),
            "results": copy.deepcopy(results),
        }

    def _watch_process(self, process: subprocess.Popen[str], generation: int) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._append_output(line)
        returncode = process.wait()
        with self.lock:
            if generation != self.generation or self.process is not process:
                return
            self.run_returncode = int(returncode)
            if self.stop_requested:
                self.run_status = "stopped"
            else:
                self.run_status = "completed" if returncode == 0 else "failed"
            label = "完成" if returncode == 0 else "结束"
            self._append_output(f"■ 程序{label}，退出码 {returncode}", "platform")
            self.process = None

    def stop_run(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                self.stop_requested = True
                self.run_status = "stopping"
                process.terminate()
                self._append_output("■ 已请求停止程序", "platform")
            elif self.run_mode == "batch" and self.run_status in {"starting", "running"}:
                self.stop_requested = True
                self.run_status = "stopping"
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            summary = self.simulator.summary()
            return {
                "generation": self.generation,
                "config": copy.deepcopy(self.config),
                "scenario": {
                    "seed": int(self.scenario.seed),
                    "sources": [_source_dict(item) for item in self.scenario.sources],
                },
                "session": {
                    "entered": bool(self.simulator.entered),
                    "finished": bool(self.finished),
                    "robot_id": self.robot_id,
                    "position": [float(value) for value in self.simulator.position],
                    "current_channel": int(self.simulator.current_channel),
                },
                "run": {
                    "status": self.run_status,
                    "mode": self.run_mode,
                    "version": self.run_version,
                    "returncode": self.run_returncode,
                    "started_at": self.run_started_at,
                },
                "batch": self._batch_summary(),
                "summary": summary,
                "actions": copy.deepcopy(self.simulator.actions),
                "transactions": copy.deepcopy(self.transactions),
                "output": copy.deepcopy(self.output_lines),
            }

    def export(self) -> dict[str, Any]:
        result = self.snapshot()
        result["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        result["format"] = "cumcm-q3-local-run-v1"
        return result


class LocalTestServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: PlatformState):
        super().__init__(address, LocalTestHandler)
        self.platform_state = state
        state.base_url = f"http://127.0.0.1:{self.server_address[1]}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RequestProblem(f"JSON 含重复字段: {key}")
        result[key] = value
    return result


class LocalTestHandler(BaseHTTPRequestHandler):
    server: LocalTestServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_bytes(
        self,
        data: bytes,
        *,
        status: int = 200,
        content_type: str = "application/octet-stream",
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if headers:
            for name, value in headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _send_json(
        self,
        value: Mapping[str, Any],
        *,
        status: int = 200,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(
            data,
            status=status,
            content_type="application/json; charset=utf-8",
            headers=headers,
        )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestProblem("Content-Length 无效") from exc
        if not 0 < length <= 2_000_000:
            raise RequestProblem("请求体为空或超过 2 MB")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestProblem("请求体不是合法 UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise RequestProblem("请求体必须是 JSON 对象")
        return value

    def _static(self, name: str, content_type: str) -> None:
        try:
            data = (STATIC_DIR / name).read_bytes()
        except FileNotFoundError:
            self._send_json({"error": "not_found"}, status=404)
            return
        self._send_bytes(data, content_type=content_type)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._static("index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._static("styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._static("app.js", "text/javascript; charset=utf-8")
        elif path == "/api/health":
            self._send_json({"ok": True, "service": "cumcm-q3-local-platform"})
        elif path == "/api/state":
            self._send_json(self.server.platform_state.snapshot())
        elif path == "/api/export":
            payload = self.server.platform_state.export()
            filename = f"q3-run-seed-{payload['scenario']['seed']}.json"
            self._send_json(
                payload,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload: dict[str, Any] = {}
        try:
            payload = self._read_json()
            if path in {"/enter", "/measure", "/clear", "/exit"}:
                response = self.server.platform_state.handle_official(path, payload)
                self._send_json(response)
            elif path == "/api/scenario":
                self._send_json(self.server.platform_state.configure(payload))
            elif path == "/api/run":
                self._send_json(self.server.platform_state.start_run(payload), status=202)
            elif path == "/api/stop":
                self._send_json(self.server.platform_state.stop_run())
            else:
                self._send_json({"error": "not_found"}, status=404)
        except InjectedTransportFailure:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        except RequestProblem as exc:
            if path in {"/enter", "/measure", "/clear", "/exit"}:
                self.server.platform_state.record_bad_request(path, payload, str(exc))
            self._send_json({"accepted": False, "error": str(exc)}, status=400)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:  # keep the local console responsive during debugging
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUMCM Q3 本地测试平台")
    parser.add_argument("--port", type=int, default=2026, help="监听端口，默认 2026")
    parser.add_argument("--seed", type=int, default=20260913, help="初始随机种子")
    parser.add_argument("--sources", type=int, default=12, help="初始信号源数量 10..16")
    parser.add_argument(
        "--error-mode",
        choices=sorted(ERROR_MODES),
        default="hash",
        help="测向误差模式",
    )
    parser.add_argument("--open", action="store_true", help="启动后打开浏览器")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state = PlatformState(
        seed=args.seed,
        source_count=args.sources,
        error_mode=args.error_mode,
    )
    try:
        server = LocalTestServer(("127.0.0.1", args.port), state)
    except OSError as exc:
        print(f"无法启动：127.0.0.1:{args.port} 不可用（{exc}）", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"CUMCM Q3 本地测试平台已启动：{url}")
    print("现有 Q3 程序可直接连接；按 Ctrl+C 关闭平台。")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在关闭本地测试平台……")
    finally:
        state.stop_run()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
