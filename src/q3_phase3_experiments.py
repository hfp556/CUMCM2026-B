"""Small, reproducible offline harness for the three real Q3 entry points.

The harness deliberately calls each entry point's ``main``.  It only replaces
the transport function with :class:`OfflineSimulator.post`, so discovery,
planning, fallback, and clear actions remain the implementation under test.
This first version is intentionally small: it provides the shared simulator,
scenario generation, one-version runner, and a smoke experiment.  Larger
Phase 3 sweeps can build on these primitives without touching the Q3 entries.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import argparse
import copy
import hashlib
import importlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import time as _time
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_NEW = ROOT / "src" / "new"
SRC_EXPERIMENTS_Q3 = ROOT / "src" / "experiments" / "q3"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))
if str(SRC_EXPERIMENTS_Q3) not in sys.path:
    sys.path.insert(0, str(SRC_EXPERIMENTS_Q3))


@dataclass(frozen=True)
class Source:
    """One omnidirectional offline source; channel numbers are 1..20."""

    channel: int
    position: tuple[float, float]
    reception_radius: float

    def point(self) -> np.ndarray:
        return np.asarray(self.position, dtype=float)


@dataclass(frozen=True)
class Scenario:
    seed: int
    sources: tuple[Source, ...]


def make_scenario(
    seed: int,
    *,
    n: Optional[int] = None,
    channels: Optional[Sequence[int]] = None,
) -> Scenario:
    """Generate uniformly distributed source locations in the target disk.

    The radius uses ``1800*sqrt(U)`` so the spatial density is uniform.  A
    fixed seed makes all three version runs share exactly the same scene.
    """

    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(10, 17) if n is None else n)
    if not 1 <= count <= 16:
        raise ValueError("n must be in [1, 16] for the offline harness")
    if channels is None:
        channel_values = tuple(range(1, count + 1))
    else:
        channel_values = tuple(int(value) for value in channels)
        if len(channel_values) != count or len(set(channel_values)) != count:
            raise ValueError("channels must be distinct and match n")
        if any(value < 1 or value > 20 for value in channel_values):
            raise ValueError("channels must be in 1..20")

    sources = []
    for channel in channel_values:
        radius = 1800.0 * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        point = (radius * math.cos(angle), radius * math.sin(angle))
        reception = float(rng.uniform(1000.0, 1500.0))
        sources.append(Source(channel, point, reception))
    return Scenario(int(seed), tuple(sources))


class OfflineSimulator:
    """Faithful local implementation of the four Q3 HTTP endpoints.

    ``post`` has the same shape as ``api_utils.post``.  It implements request
    idempotency, accepted/current-position/current-channel semantics, the
    exact virtual-time rules, fixed two-decimal bearing errors, and optional
    one-shot transport failures for later adversarial tests.
    """

    SPEED_MPS = 5.0
    MEASURE_TIME_S = 5.0
    CHANNEL_SWITCH_TIME_S = 1.0
    CLEAR_FAIL_TIME_S = 3.0
    CLEAR_SUCCESS_TIME_S = 5.0
    NEAR_RADIUS_M = 5.0
    CLEAR_RADIUS_M = 20.0
    TARGET_RADIUS_M = 1800.0
    MAX_COORDINATE_ABS = 2_000_000.0

    def __init__(
        self,
        scenario: Scenario | Iterable[Source],
        *,
        error_mode: str = "hash",
        fail_transport_every: Optional[int] = None,
        force_first_clear_fail: bool = False,
    ) -> None:
        if isinstance(scenario, Scenario):
            self.seed = int(scenario.seed)
            source_values = scenario.sources
        else:
            source_values = tuple(scenario)
            self.seed = 0
        self.sources = {int(item.channel): item for item in source_values}
        if len(self.sources) != len(tuple(source_values)):
            raise ValueError("source channels must be distinct")
        self.error_mode = str(error_mode)
        self.fail_transport_every = fail_transport_every
        self.force_first_clear_fail = bool(force_first_clear_fail)

        self.entered = False
        self.position = np.array([0.0, 0.0], dtype=float)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.actions: list[dict] = []
        self.discovered: set[int] = set()
        self.cleared: set[int] = set()
        self.first_discovery_time: dict[int, float] = {}
        self.clear_time: dict[int, float] = {}
        self._request_cache: dict[str, tuple[str, dict, dict]] = {}
        self._transport_injected: set[str] = set()
        self._new_action_requests = 0
        self._forced_clear_channels: set[int] = set()
        self.transport_failures = 0
        self.transport_retries = 0

    @staticmethod
    def _payload_signature(path: str, payload: Mapping) -> str:
        # A stable signature is enough to reject accidental request-id reuse
        # while keeping response replay independent of object identity.
        import json

        return json.dumps([path, payload], sort_keys=True, separators=(",", ":"))

    def _response(self, **fields) -> dict:
        result = {
            "accepted": True,
            "real_timestamp_ms": int(1_000_000_000_000 + len(self.actions)),
            "virtual_time_s": round(float(self.virtual_time_s), 6),
        }
        result.update(fields)
        return result

    def _valid_position(self, value) -> bool:
        point = np.asarray(value, dtype=float)
        return bool(
            point.shape == (2,)
            and np.all(np.isfinite(point))
            and np.all(np.abs(point) <= self.MAX_COORDINATE_ABS)
        )

    def direction_error(self, source: Source, position: Sequence[float]) -> float:
        """Return the deterministic error for ``(seed, source, round(pos,2))``."""

        if self.error_mode in {"minus_one", "-1", "negative"}:
            return -1.0
        if self.error_mode in {"plus_one", "+1", "positive"}:
            return 1.0
        if self.error_mode in {"zero", "0"}:
            return 0.0
        point = np.asarray(position, dtype=float)
        key = (
            f"{self.seed}|{source.channel}|{round(float(point[0]), 2):.2f}|"
            f"{round(float(point[1]), 2):.2f}"
        ).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return -1.0 + 2.0 * unit

    def _bearing(self, source: Source, position: np.ndarray) -> float:
        delta = source.point() - position
        true_bearing = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
        measured = round((true_bearing + self.direction_error(source, position)) % 360.0, 2)
        return round(measured % 360.0, 2)

    def _advance_position(self, point: np.ndarray, channel: int, path: str) -> tuple[float, float]:
        movement = float(np.linalg.norm(point - self.position))
        switch = (
            self.CHANNEL_SWITCH_TIME_S
            if path == "/measure" and int(channel) != self.current_channel
            else 0.0
        )
        self.position = point.copy()
        if path == "/measure":
            self.current_channel = int(channel)
        return movement, switch

    def _record_action(
        self,
        *,
        path: str,
        channel: int,
        point: np.ndarray,
        response: dict,
        before: float,
        movement: float,
        switch: float,
        action_time: float,
        source_distance: Optional[float],
    ) -> None:
        action = {
                "path": path,
                "channel": int(channel),
                "position": [float(point[0]), float(point[1])],
                "accepted": bool(response.get("accepted") is True),
                "result": response.get("measure_result", response.get("clear_result")),
                "virtual_time_before_s": float(before),
                "virtual_time_after_s": float(self.virtual_time_s),
                "movement_m": float(movement),
                "channel_switch_s": float(switch),
                "action_time_s": float(action_time),
                "source_distance_m": None if source_distance is None else float(source_distance),
            }
        if "svd_deg" in response:
            action["svd_deg"] = float(response["svd_deg"])
        self.actions.append(action)

    def _measure(self, payload: Mapping) -> dict:
        point = np.array(
            [float(payload["position"]["x"]), float(payload["position"]["y"])],
            dtype=float,
        )
        channel = int(payload["channel"])
        before = self.virtual_time_s
        movement, switch = self._advance_position(point, channel, "/measure")
        source = self.sources.get(channel)
        distance = None if source is None else float(np.linalg.norm(source.point() - point))
        if source is None or channel in self.cleared or distance is None or distance > source.reception_radius:
            result = "no_signal"
            fields = {"measure_result": result}
        elif distance <= self.NEAR_RADIUS_M:
            result = "near"
            fields = {"measure_result": result}
            self.discovered.add(channel)
        else:
            result = "direction"
            fields = {"measure_result": result, "svd_deg": self._bearing(source, point)}
            self.discovered.add(channel)
        if channel in self.discovered and channel not in self.first_discovery_time:
            # Discovery is attributed to completion of the accepted measure.
            self.first_discovery_time[channel] = before + movement / self.SPEED_MPS + switch + self.MEASURE_TIME_S
        action_time = movement / self.SPEED_MPS + switch + self.MEASURE_TIME_S
        self.virtual_time_s += action_time
        response = self._response(**fields)
        self._record_action(
            path="/measure",
            channel=channel,
            point=point,
            response=response,
            before=before,
            movement=movement,
            switch=switch,
            action_time=action_time,
            source_distance=distance,
        )
        return response

    def _clear(self, payload: Mapping) -> dict:
        point = np.array(
            [float(payload["position"]["x"]), float(payload["position"]["y"])],
            dtype=float,
        )
        channel = int(payload["channel"])
        before = self.virtual_time_s
        movement, switch = self._advance_position(point, channel, "/clear")
        source = self.sources.get(channel)
        distance = None if source is None else float(np.linalg.norm(source.point() - point))
        forced = self.force_first_clear_fail and channel not in self._forced_clear_channels
        if forced:
            self._forced_clear_channels.add(channel)
        success = bool(
            source is not None
            and channel not in self.cleared
            and distance is not None
            and distance <= self.CLEAR_RADIUS_M
            and not forced
        )
        if success:
            result = "success"
            action_time = movement / self.SPEED_MPS + self.CLEAR_SUCCESS_TIME_S
            self.cleared.add(channel)
        else:
            result = "no_target_in_range"
            action_time = movement / self.SPEED_MPS + self.CLEAR_FAIL_TIME_S
        self.virtual_time_s += action_time
        response = self._response(clear_result=result)
        self._record_action(
            path="/clear",
            channel=channel,
            point=point,
            response=response,
            before=before,
            movement=movement,
            switch=switch,
            action_time=action_time,
            source_distance=distance,
        )
        if success and channel not in self.clear_time:
            self.clear_time[channel] = self.virtual_time_s
        return response

    def post(self, path: str, payload: Mapping) -> Optional[dict]:
        """Transport-compatible endpoint; ``None`` means a network failure."""

        path = str(path)
        request_id = str(payload.get("request_id", ""))
        signature = self._payload_signature(path, payload)
        cached = self._request_cache.get(request_id)
        if cached is not None:
            cached_path, cached_signature, cached_response = cached
            if cached_path == path and cached_signature == signature:
                return copy.deepcopy(cached_response)
            return {"accepted": False, "virtual_time_s": 0, "error": "request_id_reuse"}

        if path in {"/measure", "/clear"}:
            self._new_action_requests += 1
            if (
                self.fail_transport_every
                and self.fail_transport_every > 0
                and self._new_action_requests % int(self.fail_transport_every) == 0
                and request_id not in self._transport_injected
            ):
                # Do not consume an action.  The real Q3 _post will retry the
                # same request_id and this next call will execute once.
                self._transport_injected.add(request_id)
                self.transport_failures += 1
                return None
            if request_id in self._transport_injected:
                self.transport_retries += 1

        if path == "/enter":
            self.entered = True
            response = self._response()
        elif not self.entered:
            response = {"accepted": False, "virtual_time_s": 0, "error": "not_entered"}
        elif path == "/measure":
            try:
                point = [payload["position"]["x"], payload["position"]["y"]]
                channel = int(payload["channel"])
                valid = self._valid_position(point) and 1 <= channel <= 20
            except (KeyError, TypeError, ValueError):
                valid = False
            response = self._measure(payload) if valid else {
                "accepted": False,
                "virtual_time_s": 0,
                "error": "invalid_measure",
            }
        elif path == "/clear":
            try:
                point = [payload["position"]["x"], payload["position"]["y"]]
                channel = int(payload["channel"])
                valid = self._valid_position(point) and 1 <= channel <= 20
            except (KeyError, TypeError, ValueError):
                valid = False
            response = self._clear(payload) if valid else {
                "accepted": False,
                "virtual_time_s": 0,
                "error": "invalid_clear",
            }
        elif path == "/exit":
            self.entered = False
            response = self._response()
        else:
            response = {"accepted": False, "virtual_time_s": 0, "error": "unknown_path"}

        self._request_cache[request_id] = (path, signature, copy.deepcopy(response))
        return copy.deepcopy(response)

    def summary(self) -> dict:
        source_channels = set(self.sources)
        source_times = {
            channel: self.clear_time[channel] - self.first_discovery_time[channel]
            for channel in source_channels
            if channel in self.clear_time and channel in self.first_discovery_time
        }
        measures = [item for item in self.actions if item["path"] == "/measure" and item["accepted"]]
        clears = [item for item in self.actions if item["path"] == "/clear" and item["accepted"]]
        outside = [
            item for item in measures
            if math.hypot(item["position"][0], item["position"][1]) > self.TARGET_RADIUS_M + 1e-9
        ]
        return {
            "source_count": len(source_channels),
            "discovered_count": len(self.discovered & source_channels),
            "cleared_count": len(self.cleared & source_channels),
            "discovered_channels": sorted(self.discovered & source_channels),
            "cleared_channels": sorted(self.cleared & source_channels),
            "missed_discovery_channels": sorted(source_channels - self.discovered),
            "uncleared_channels": sorted(source_channels - self.cleared),
            "total_virtual_time_s": float(self.virtual_time_s),
            "total_move_m": float(sum(item["movement_m"] for item in self.actions if item["accepted"])),
            "measure_count": len(measures),
            "clear_attempt_count": len(clears),
            "clear_failure_count": sum(item["result"] == "no_target_in_range" for item in clears),
            "source_time_s": {str(k): float(v) for k, v in sorted(source_times.items())},
            "outside_measure_count": len(outside),
            "transport_failures": int(self.transport_failures),
            "transport_retries": int(self.transport_retries),
        }


def _load_version(version: str):
    names = {
        "fast2": "Q3_fast2",
        "phase1": "Q3_phase1",
        "active": "Q3_active",
    }
    try:
        return importlib.import_module(names[str(version)])
    except KeyError as exc:
        raise ValueError(f"unknown Q3 version: {version}") from exc


def _reset_module_state(module) -> None:
    module.CURRENT_POS = np.array([0.0, 0.0], dtype=float)
    module.LAST_VT = 0.0
    if hasattr(module, "_phase1"):
        module._phase1.CURRENT_POS = np.array([0.0, 0.0], dtype=float)
        module._phase1.LAST_VT = 0.0


@contextmanager
def _patched_entry(
    module,
    simulator: OfflineSimulator,
    counters: dict,
    *,
    force_first_process_failure: bool = False,
):
    """Patch transport/clock only, while counting actual fallback calls."""

    saved = []

    def replace(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    replace(module, "post", simulator.post)
    # Q3_active delegates all requests to its Phase 1 module.
    if hasattr(module, "_phase1"):
        replace(module._phase1, "post", simulator.post)

    # The source modules use time.sleep only between transport retries.  A
    # fixed wall clock disables the real-time guard while virtual time remains
    # entirely simulator-controlled.
    replace(_time, "sleep", lambda _seconds: None)
    replace(_time, "time", lambda: 1_000_000.0)

    wrapped = []

    def wrap(obj, name, counter_name):
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)

        def counted(*args, **kwargs):
            counters[counter_name] = counters.get(counter_name, 0) + 1
            return original(*args, **kwargs)

        replace(obj, name, counted)
        wrapped.append((obj, name))

    # Wrapping the module functions does not alter their internals.  For the
    # active entry, its fallback implementation lives in Phase 1, so count
    # those names in both modules.
    wrap(module, "homing_clear", "homing_fallback_calls")
    wrap(module, "bracket_clear", "bracket_fallback_calls")
    wrap(module, "iterative_clear", "iterative_fallback_calls")
    if hasattr(module, "homing_bracket_fallback"):
        wrap(module, "homing_bracket_fallback", "homing_bracket_fallback_calls")
    if hasattr(module, "_phase1"):
        wrap(module._phase1, "homing_clear", "homing_fallback_calls")
        wrap(module._phase1, "bracket_clear", "bracket_fallback_calls")
        wrap(module._phase1, "iterative_clear", "iterative_fallback_calls")

    if hasattr(module, "process_channel"):
        original = getattr(module, "process_channel")
        state = {"done": False}

        def counted_process(*args, **kwargs):
            counters["process_calls"] = counters.get("process_calls", 0) + 1
            process_ids = counters.setdefault("_process_channel_ids", [])
            if args:
                first = args[0]
                process_ids.append(int(getattr(first, "channel", first)))
            if force_first_process_failure and not state["done"]:
                state["done"] = True
                counters["forced_process_failures"] = counters.get(
                    "forced_process_failures", 0
                ) + 1
                if args and hasattr(args[0], "last_failure_reason"):
                    args[0].last_failure_reason = "forced_first_processing_failure"
                return False
            return original(*args, **kwargs)

        replace(module, "process_channel", counted_process)

    try:
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def light_active_config(module=None):
    """Return the explicitly labelled light configuration for smoke/sweeps."""

    if module is None:
        module = _load_version("active")
    if not hasattr(module, "ActiveConfig"):
        return None
    return module.ActiveConfig(
        source_sample_count=2,
        error_samples=(-1.0, 0.0, 1.0),
        spacing_m=250.0,
        circle_segments=16,
        max_constraint_vertices=64,
        max_candidates=24,
        max_steps=4,
        time_limit_s=1_000_000.0,
    )


def run_version(
    version: str,
    scenario: Scenario,
    *,
    simulator: Optional[OfflineSimulator] = None,
    active_config=None,
    planner=None,
    force_first_process_failure: bool = False,
) -> dict:
    """Run one real Q3 ``main`` against the shared offline simulator."""

    module = _load_version(version)
    sim = OfflineSimulator(scenario) if simulator is None else simulator
    if sim.sources.keys() != {item.channel for item in scenario.sources}:
        raise ValueError("simulator sources must match the scenario")
    _reset_module_state(module)
    counters: dict[str, int] = {}
    output = io.StringIO()
    exception_text = None
    with _patched_entry(
        module,
        sim,
        counters,
        force_first_process_failure=force_first_process_failure,
    ):
        with redirect_stdout(output):
            try:
                if str(version) == "active":
                    # Production runs must use the entry point's declared
                    # default.  Callers must pass light_active_config(module)
                    # explicitly when they want the sensitivity setting.
                    config = (
                        module.DEFAULT_ACTIVE_CONFIG
                        if active_config is None
                        else active_config
                    )
                    module.main(active_config=config, planner=planner)
                else:
                    module.main()
            except Exception as exc:  # preserve partial metrics for adversarial cases
                exception_text = f"{type(exc).__name__}: {exc}"
    process_ids = counters.pop("_process_channel_ids", [])
    counters["channel_retry_count"] = max(0, len(process_ids) - len(set(process_ids)))
    result = sim.summary()
    result.update(
        {
            "version": str(version),
            "seed": int(scenario.seed),
            "stdout_captured": True,
            "stdout_line_count": len(output.getvalue().splitlines()),
            "stdout_excerpt": output.getvalue()[:500] + ("..." if len(output.getvalue()) > 500 else ""),
            "process_fallback_counts": counters,
            "completed_without_exception": exception_text is None,
            "exception": exception_text,
            "active_config_mode": (
                "production_default"
                if str(version) == "active" and active_config is None
                else ("custom" if str(version) == "active" else None)
            ),
            "retry_count": int(
                counters.get("channel_retry_count", 0) + sim.transport_retries
            ),
        }
    )
    if planner is not None and hasattr(planner, "summary"):
        result["planner_summary"] = planner.summary()
    result["any_missed_channel"] = bool(result["uncleared_channels"] or result["missed_discovery_channels"])
    return result


def smoke_experiment(seed: int = 20260913) -> dict:
    """Run one fixed small scene through all three real entry points."""

    scenario = make_scenario(seed, n=10)
    runs = {}
    for version in ("fast2", "phase1", "active"):
        config = light_active_config() if version == "active" else None
        runs[version] = run_version(version, scenario, active_config=config)
    return {
        "seed": int(seed),
        "source_count": len(scenario.sources),
        "sources": [
            {
                "channel": source.channel,
                "position": [float(source.position[0]), float(source.position[1])],
                "reception_radius": float(source.reception_radius),
            }
            for source in scenario.sources
        ],
        "runs": runs,
    }


def _metric_stats(values: Iterable[float]) -> dict:
    """Deterministic distribution summary used for every requested metric."""

    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def aggregate_runs(runs: Sequence[Mapping]) -> dict:
    """Aggregate raw per-game runner records without hiding missing games."""

    records = list(runs)
    game_count = len(records)
    source_count = int(sum(int(item.get("source_count", 0)) for item in records))
    discovered = int(sum(int(item.get("discovered_count", 0)) for item in records))
    cleared = int(sum(int(item.get("cleared_count", 0)) for item in records))
    successful = [
        item
        for item in records
        if int(item.get("cleared_count", 0)) == int(item.get("source_count", 0))
    ]
    per_source = []
    for item in records:
        per_source.extend(float(value) for value in item.get("source_time_s", {}).values())

    def total(name: str) -> int:
        return int(sum(int(item.get(name, 0)) for item in records))

    fallback_totals = {}
    for item in records:
        for key, value in item.get("process_fallback_counts", {}).items():
            fallback_totals[key] = fallback_totals.get(key, 0) + int(value)

    return {
        "game_count": game_count,
        "source_count": source_count,
        "discovered_count": discovered,
        "cleared_count": cleared,
        "discovery_rate": None if source_count == 0 else discovered / source_count,
        "clear_rate": None if source_count == 0 else cleared / source_count,
        "games_with_any_missed_channel": int(
            sum(bool(item.get("any_missed_channel", False)) for item in records)
        ),
        "any_missed_channel_game_rate": (
            None
            if game_count == 0
            else sum(bool(item.get("any_missed_channel", False)) for item in records)
            / game_count
        ),
        "successful_game_count": len(successful),
        "successful_game_rate": None if game_count == 0 else len(successful) / game_count,
        "successful_total_virtual_time_s": _metric_stats(
            item["total_virtual_time_s"] for item in successful
        ),
        "per_source_time_s": _metric_stats(per_source),
        "total_virtual_time_s": _metric_stats(
            item["total_virtual_time_s"] for item in records
        ),
        "total_move_m": {
            "sum": float(sum(float(item.get("total_move_m", 0.0)) for item in records)),
            "mean_per_game": None if game_count == 0 else float(
                np.mean([float(item.get("total_move_m", 0.0)) for item in records])
            ),
        },
        "measure_count": {
            "sum": total("measure_count"),
            "mean_per_game": None if game_count == 0 else total("measure_count") / game_count,
        },
        "clear_attempt_count": {
            "sum": total("clear_attempt_count"),
            "mean_per_game": None if game_count == 0 else total("clear_attempt_count") / game_count,
        },
        "clear_failure_count": {
            "sum": total("clear_failure_count"),
            "mean_per_game": None if game_count == 0 else total("clear_failure_count") / game_count,
        },
        "transport_failures": {
            "sum": total("transport_failures"),
            "mean_per_game": None if game_count == 0 else total("transport_failures") / game_count,
        },
        "transport_retries": {
            "sum": total("transport_retries"),
            "mean_per_game": None if game_count == 0 else total("transport_retries") / game_count,
        },
        "channel_retry_count": {
            "sum": total("channel_retry_count"),
            "mean_per_game": None if game_count == 0 else total("channel_retry_count") / game_count,
        },
        "retry_count": {
            "sum": total("retry_count"),
            "mean_per_game": None if game_count == 0 else total("retry_count") / game_count,
        },
        "outside_measure_count": {
            "sum": total("outside_measure_count"),
            "mean_per_game": None if game_count == 0 else total("outside_measure_count") / game_count,
        },
        "fallback_totals": fallback_totals,
        "completed_without_exception_count": int(
            sum(bool(item.get("completed_without_exception", False)) for item in records)
        ),
    }


def _source_dict(source: Source) -> dict:
    return {
        "channel": int(source.channel),
        "position": [float(source.position[0]), float(source.position[1])],
        "reception_radius": float(source.reception_radius),
    }


def _run_record(scenario: Scenario, version: str, **kwargs) -> dict:
    result = run_version(version, scenario, **kwargs)
    result["scenario_sources"] = [_source_dict(source) for source in scenario.sources]
    return result


def run_monte_carlo(
    seeds: Sequence[int],
    *,
    versions: Sequence[str] = ("fast2", "phase1", "active"),
    active_config=None,
    active_label: str = "production_default",
) -> dict:
    """Run paired scenes and retain each raw game/version result."""

    seed_values = [int(seed) for seed in seeds]
    by_version = {str(version): [] for version in versions}
    scenes = []
    for seed in seed_values:
        scenario = make_scenario(seed)
        scenes.append(
            {
                "seed": int(seed),
                "source_count": len(scenario.sources),
                "sources": [_source_dict(source) for source in scenario.sources],
            }
        )
        for version in versions:
            result = _run_record(
                scenario,
                str(version),
                active_config=active_config if str(version) == "active" else None,
            )
            result["active_config_label"] = active_label if str(version) == "active" else None
            by_version[str(version)].append(result)
    return {
        "seeds": seed_values,
        "scene_count": len(seed_values),
        "pairing": "same seed, source positions and reception radii for each version",
        "active_config_label": active_label,
        "scenes": scenes,
        "raw_by_version": by_version,
        "summary_by_version": {
            version: aggregate_runs(records) for version, records in by_version.items()
        },
    }


class FairCandidatePoolPlanner:
    """Use one generated robust pool, optionally filtering only r<=1800."""

    def __init__(self, target_only: bool):
        self.target_only = bool(target_only)
        self.calls = 0
        self.pool_sizes: list[int] = []
        self.outside_pool_sizes: list[int] = []
        self.selected_points: list[list[float] | None] = []
        self.selected_outside: list[bool] = []

    def __call__(self, omega, current_position, **kwargs):
        active = _load_version("active")
        budgets = tuple(float(value) for value in kwargs.get("budgets_m", (100, 200, 300, 400)))
        max_budget = max(budgets)
        pool = active.generate_robust_candidates(
            omega,
            current_position,
            max_budget,
            spacing_m=float(kwargs.get("spacing_m", 150.0)),
            min_reception_radius=float(kwargs.get("min_reception_radius", 1000.0)),
            target_radius=None,
            circle_segments=int(kwargs.get("circle_segments", 16)),
            max_candidates=int(kwargs.get("max_candidates", 80)),
        )
        pool = np.asarray(pool, dtype=float).reshape((-1, 2)) if len(pool) else np.empty((0, 2))
        outside = np.linalg.norm(pool, axis=1) > 1800.0 + 1e-9 if len(pool) else np.zeros(0, dtype=bool)
        allowed = pool[~outside] if self.target_only else pool
        self.calls += 1
        self.pool_sizes.append(int(len(pool)))
        self.outside_pool_sizes.append(int(np.sum(outside)))
        result = active.solve_pareto(
            omega,
            current_position,
            budgets_m=budgets,
            candidate_points=allowed,
            source_sample_count=int(kwargs.get("source_sample_count", 5)),
            error_samples=tuple(kwargs.get("error_samples", (-1.0, 0.0, 1.0))),
            spacing_m=float(kwargs.get("spacing_m", 150.0)),
            min_reception_radius=float(kwargs.get("min_reception_radius", 1000.0)),
            max_reception_radius=float(kwargs.get("max_reception_radius", 1500.0)),
            angle_error_deg=float(kwargs.get("angle_error_deg", 1.0)),
            circle_segments=int(kwargs.get("circle_segments", 16)),
            target_radius=1800.0 if self.target_only else None,
            max_candidates=int(kwargs.get("max_candidates", 80)),
        )
        decision = active.select_active_candidate(result, budgets)
        point = None if decision is None else np.asarray(decision.point, dtype=float)
        self.selected_points.append(None if point is None else point.tolist())
        self.selected_outside.append(
            bool(point is not None and float(np.linalg.norm(point)) > 1800.0 + 1e-9)
        )
        return result

    def summary(self) -> dict:
        return {
            "target_only": self.target_only,
            "planner_calls": int(self.calls),
            "candidate_pool_sizes": list(self.pool_sizes),
            "outside_candidates_in_shared_pool": list(self.outside_pool_sizes),
            "selected_points": list(self.selected_points),
            "selected_outside": list(self.selected_outside),
            "selected_outside_count": int(sum(self.selected_outside)),
            "fairness": "unrestricted pool generated once per posterior/current state; target-only filters r>1800 only",
        }


def _bearing_trace(simulator: OfflineSimulator, channel: int) -> dict:
    directions = [
        item
        for item in simulator.actions
        if item["path"] == "/measure"
        and item["channel"] == int(channel)
        and item["accepted"]
        and item.get("result") == "direction"
        and "svd_deg" in item
    ]
    angles = [float(item["svd_deg"]) for item in directions]
    max_angle = 0.0
    max_oriented_angle = 0.0
    for first in angles:
        for second in angles:
            diff = abs((first - second + 180.0) % 360.0 - 180.0)
            max_oriented_angle = max(max_oriented_angle, diff)
            # A line intersection is unoriented: bearings separated by 179°
            # describe almost the same line, not a near-right-angle crossing.
            max_angle = max(max_angle, min(diff, 180.0 - diff))
    observations = [np.asarray(item["position"], dtype=float) for item in directions]
    bearings = angles
    posterior_sequence = []
    if observations:
        phase1 = _load_version("phase1")
        for index in range(1, len(observations) + 1):
            polygon = phase1.region_of(observations[:index], bearings[:index])
            if len(polygon) == 0:
                posterior_sequence.append({"observation_count": index, "rho_m": None, "diameter_m": None})
                continue
            _, rho = phase1.minimum_enclosing_circle(polygon)
            diameter, _ = phase1.convex_diameter(polygon)
            posterior_sequence.append(
                {
                    "observation_count": index,
                    "rho_m": float(rho),
                    "diameter_m": float(diameter),
                }
            )
    valid = [item for item in posterior_sequence if item["rho_m"] is not None]
    return {
        "direction_count": len(directions),
        "crossing_angle_deg_max_pair": float(max_angle),
        "oriented_bearing_span_deg_max_pair": float(max_oriented_angle),
        "posterior_sequence": posterior_sequence,
        "initial_rho_m": None if not valid else valid[0]["rho_m"],
        "final_rho_m": None if not valid else valid[-1]["rho_m"],
        "initial_diameter_m": None if not valid else valid[0]["diameter_m"],
        "final_diameter_m": None if not valid else valid[-1]["diameter_m"],
    }


def _fixed_source(channel: int, radius: float, angle_deg: float, reception: float) -> Source:
    angle = math.radians(float(angle_deg))
    return Source(
        int(channel),
        (float(radius) * math.cos(angle), float(radius) * math.sin(angle)),
        float(reception),
    )


def slender_cases() -> list[dict]:
    """Fixed near-collinear cases for the original homing/bracket comparison."""

    def observations(angle_deg, first_radius, second_radius, lateral=0.0):
        angle = math.radians(float(angle_deg))
        radial = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-radial[1], radial[0]])
        first = first_radius * radial - float(lateral) * normal
        second = second_radius * radial + float(lateral) * normal
        return (tuple(first), tuple(second))

    return [
        {"case_id": "east_parallel_minus", "scenario": Scenario(61001, (_fixed_source(1, 1500, 0, 1500),)), "error_mode": "minus_one", "observations": observations(0, 300, 500)},
        {"case_id": "east_parallel_plus", "scenario": Scenario(61002, (_fixed_source(1, 1500, 0, 1500),)), "error_mode": "plus_one", "observations": observations(0, 300, 500)},
        {"case_id": "east_near_parallel_hash", "scenario": Scenario(61003, (_fixed_source(1, 1700, 0, 1500),)), "error_mode": "hash", "observations": observations(0, 600, 800, 5)},
        {"case_id": "north_parallel_minus", "scenario": Scenario(61004, (_fixed_source(1, 1700, 90, 1500),)), "error_mode": "minus_one", "observations": observations(90, 600, 800)},
        {"case_id": "diagonal_near_parallel_plus", "scenario": Scenario(61005, (_fixed_source(1, 1790, 25, 1500),)), "error_mode": "plus_one", "observations": observations(25, 700, 900, 5)},
        {"case_id": "opposite_diagonal_hash", "scenario": Scenario(61006, (_fixed_source(1, 1790, 205, 1500),)), "error_mode": "hash", "observations": observations(205, 700, 900, 5)},
    ]


def _initial_region_metrics(observations, bearings) -> dict:
    phase1 = _load_version("phase1")
    polygon = phase1.region_of(observations, bearings)
    if len(polygon) == 0:
        return {"diameter_m": None, "rho_m": None, "slender_aspect_ratio": None}
    diameter, _ = phase1.convex_diameter(polygon)
    _, rho = phase1.minimum_enclosing_circle(polygon)
    centered = polygon - np.mean(polygon, axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    spans = np.ptp(centered @ vectors, axis=0)
    narrow = max(float(np.min(spans)), 1e-12)
    return {
        "diameter_m": float(diameter),
        "rho_m": float(rho),
        "slender_aspect_ratio": float(np.max(spans) / narrow),
        "polygon_vertices": int(len(polygon)),
    }


def run_processing_case(version, scenario, observations, *, error_mode, active_config=None, planner=None) -> dict:
    """Run one real channel processor from the same pre-observed state."""

    module = _load_version(version)
    source = scenario.sources[0]
    points = [np.asarray(point, dtype=float) for point in observations]
    simulator = OfflineSimulator(scenario, error_mode=error_mode)
    bearings = [simulator._bearing(source, point) for point in points]
    simulator.entered = True
    simulator.position = points[-1].copy()
    simulator.current_channel = int(source.channel)
    simulator.discovered.add(int(source.channel))
    simulator.first_discovery_time[int(source.channel)] = 0.0
    _reset_module_state(module)
    module.CURRENT_POS = points[-1].copy()
    if hasattr(module, "_phase1"):
        module._phase1.CURRENT_POS = points[-1].copy()
    counters = {}
    output = io.StringIO()
    exception_text = None
    success = False
    with _patched_entry(module, simulator, counters):
        with redirect_stdout(output):
            try:
                if version == "fast2":
                    success = bool(module.process_channel(
                        int(source.channel), [point.copy() for point in points], list(bearings)
                    ))
                elif version == "active":
                    state = module.ChannelState(
                        int(source.channel), observations=[point.copy() for point in points],
                        bearings=list(bearings), last_position=points[-1].copy(),
                    )
                    config = module.DEFAULT_ACTIVE_CONFIG if active_config is None else active_config
                    success = bool(module.process_channel(
                        state, active_config=config, planner=planner
                    ))
                else:
                    raise ValueError("processing comparison supports fast2 and active")
            except Exception as exc:
                exception_text = f"{type(exc).__name__}: {exc}"
    counters.pop("_process_channel_ids", None)
    result = simulator.summary()
    result.update({
        "version": version,
        "process_returned_success": success,
        "completed_without_exception": exception_text is None,
        "exception": exception_text,
        "initial_observations": [point.tolist() for point in points],
        "initial_bearings_deg": [float(value) for value in bearings],
        "initial_region": _initial_region_metrics(points, bearings),
        "process_fallback_counts": counters,
        "stdout_excerpt": output.getvalue()[:500],
    })
    if planner is not None and hasattr(planner, "summary"):
        result["planner_summary"] = planner.summary()
    return result


def run_experiment6(cases: Optional[Sequence[Mapping]] = None) -> dict:
    records = []
    selected_cases = slender_cases() if cases is None else list(cases)
    for case in selected_cases:
        versions = {}
        for version in ("fast2", "active"):
            versions[version] = run_processing_case(
                version,
                case["scenario"],
                case["observations"],
                error_mode=case["error_mode"],
            )
        records.append(
            {
                "case_id": case["case_id"],
                "source": _source_dict(case["scenario"].sources[0]),
                "error_mode": case["error_mode"],
                "baseline_homing_step_m": 200.0,
                "baseline_bracket_end_m": 36.0,
                "active_budgets_m": [100.0, 200.0, 300.0, 400.0],
                "raw_by_version": versions,
            }
        )
    return {
        "description": "Controlled slender/near-collinear states; real fast2 and active process_channel actions, excluding discovery scan time.",
        "comparison": "fast2 original 200 m homing/bracket versus active controller",
        "active_config_label": "production_default",
        "case_count": len(records),
        "cases": records,
    }


def boundary_cases() -> list[dict]:
    radii = (1750.0, 1775.0, 1795.0, 1800.0)
    angles = (0.0, 23.0, 71.0, 137.0)
    errors = ("minus_one", "plus_one", "hash", "plus_one")
    cases = []
    for index, (radius, angle, error_mode) in enumerate(zip(radii, angles, errors), 1):
        cases.append(
            {
                "case_id": f"boundary_{index}_{int(radius)}_{int(angle)}",
                "scenario": Scenario(
                    62000 + index,
                    (_fixed_source(1, radius, angle, 1000.0),),
                ),
                "error_mode": error_mode,
            }
        )
    return cases


def run_experiment7(cases: Optional[Sequence[Mapping]] = None) -> dict:
    records = []
    config = _load_version("active").DEFAULT_ACTIVE_CONFIG
    selected_cases = boundary_cases() if cases is None else list(cases)
    for case in selected_cases:
        pair = {}
        for target_only, label in ((False, "allow_outside"), (True, "target_only")):
            simulator = OfflineSimulator(case["scenario"], error_mode=case["error_mode"])
            planner = FairCandidatePoolPlanner(target_only)
            pair[label] = _run_record(
                case["scenario"],
                "active",
                simulator=simulator,
                active_config=config,
                planner=planner,
            )
            pair[label]["boundary_trace"] = _bearing_trace(simulator, 1)
        allow = pair["allow_outside"]
        restricted = pair["target_only"]
        allow_time = float(allow["total_virtual_time_s"])
        restricted_time = float(restricted["total_virtual_time_s"])
        planner_summary = allow.get("planner_summary", {})
        selected_outside_count = int(planner_summary.get("selected_outside_count", 0))
        outside_used = selected_outside_count > 0
        records.append(
            {
                "case_id": case["case_id"],
                "source": _source_dict(case["scenario"].sources[0]),
                "error_mode": case["error_mode"],
                "fair_candidate_pool": "same per-state unrestricted pool; target-only filters only candidate norm>1800",
                "raw_pair": pair,
                "paired_time_delta_allow_minus_target_only_s": allow_time - restricted_time,
                "paired_time_saving_evidence": bool(outside_used and allow_time < restricted_time),
                "active_selected_outside_count": selected_outside_count,
                "outside_measure_used_in_allow": outside_used,
                "all_outside_measure_count_including_fallback": int(allow["outside_measure_count"]),
            }
        )
    allow_times = [item["raw_pair"]["allow_outside"]["total_virtual_time_s"] for item in records]
    restricted_times = [item["raw_pair"]["target_only"]["total_virtual_time_s"] for item in records]
    return {
        "description": "Boundary 1750-1800 m active-controller fair paired ablation.",
        "active_config_label": "production_default",
        "case_count": len(records),
        "cases": records,
        "paired_summary": {
            "case_count": len(records),
            "allow_outside_total_time": _metric_stats(allow_times),
            "target_only_total_time": _metric_stats(restricted_times),
            "allow_faster_case_count": int(sum(a < b for a, b in zip(allow_times, restricted_times))),
            "outside_used_case_count": int(sum(item["outside_measure_used_in_allow"] for item in records)),
            "time_saving_evidence_case_count": int(sum(item["paired_time_saving_evidence"] for item in records)),
        },
    }


def _clustered_one_side_sources(count: int = 10) -> tuple[Source, ...]:
    result = []
    for index in range(count):
        angle = -42.0 + index * (84.0 / max(1, count - 1))
        radius = 1250.0 + 45.0 * index
        reception = 1000.0 + 40.0 * (index % 5)
        result.append(_fixed_source(index + 1, radius, angle, reception))
    return tuple(result)


def _boundary_distributed_sources(count: int = 12) -> tuple[Source, ...]:
    result = []
    for index in range(count):
        angle = index * (360.0 / count)
        reception = 1000.0 + 45.0 * (index % 5)
        result.append(_fixed_source(index + 1, 1800.0, angle, reception))
    return tuple(result)


def adversarial_cases() -> list[dict]:
    worst_angle = math.degrees(math.pi / 7.0)
    cases = [
        {"case_id": "01_boundary_source", "scenario": Scenario(63101, (_fixed_source(1, 1800, 0, 1000),)), "error_mode": "hash"},
        {"case_id": "02_minimum_reception_radius", "scenario": Scenario(63102, (_fixed_source(1, 0, 0, 1000),)), "error_mode": "hash"},
        {"case_id": "03_heptagon_worst_scan_point", "scenario": Scenario(63103, (_fixed_source(1, 1800, worst_angle, 1000),)), "error_mode": "hash"},
        {"case_id": "04_nearly_parallel_bearings", "scenario": Scenario(63104, tuple(_fixed_source(i + 1, 1500, -2.0 + 2.0 * i, 1200) for i in range(3))), "error_mode": "hash"},
        {"case_id": "05_slender_region", "scenario": Scenario(63105, (_fixed_source(1, 1500, 0, 1000),)), "error_mode": "plus_one"},
        {"case_id": "06_mec_radius_near_twenty", "scenario": Scenario(63106, (_fixed_source(1, 1320, 0, 1050),)), "error_mode": "hash"},
        {"case_id": "07_extreme_bearing_errors", "scenario": Scenario(63107, (_fixed_source(1, 1700, 45, 1000),)), "error_modes": ("minus_one", "plus_one")},
        {"case_id": "08_forced_first_process_failure", "scenario": Scenario(63108, (_fixed_source(1, 1300, 0, 1100),)), "error_mode": "hash", "force_first_process_failure": True},
        {"case_id": "09_forced_first_clear_failure", "scenario": Scenario(63109, (_fixed_source(1, 1300, 180, 1100),)), "error_mode": "hash", "force_first_clear_fail": True},
        {"case_id": "10_intermittent_network_failure", "scenario": Scenario(63110, (_fixed_source(1, 1300, 90, 1100),)), "error_mode": "hash", "fail_transport_every": 17},
        {"case_id": "11_sources_one_side", "scenario": Scenario(63111, _clustered_one_side_sources()), "error_mode": "hash"},
        {"case_id": "12_sources_distributed_boundary", "scenario": Scenario(63112, _boundary_distributed_sources()), "error_mode": "hash"},
    ]
    return cases


def run_experiment12(cases: Optional[Sequence[Mapping]] = None) -> dict:
    records = []
    config = _load_version("active").DEFAULT_ACTIVE_CONFIG
    selected_cases = adversarial_cases() if cases is None else list(cases)
    for case in selected_cases:
        error_modes = case.get("error_modes", (case.get("error_mode", "hash"),))
        for variant, error_mode in enumerate(error_modes):
            versions = {}
            for version in ("fast2", "phase1", "active"):
                simulator = OfflineSimulator(
                    case["scenario"],
                    error_mode=error_mode,
                    fail_transport_every=case.get("fail_transport_every"),
                    force_first_clear_fail=case.get("force_first_clear_fail", False),
                )
                versions[version] = _run_record(
                    case["scenario"],
                    version,
                    simulator=simulator,
                    active_config=config if version == "active" else None,
                    force_first_process_failure=case.get("force_first_process_failure", False),
                )
                versions[version]["bounded"] = bool(
                    versions[version]["completed_without_exception"]
                    and len(simulator.actions) <= 10000
                    and simulator.virtual_time_s <= 360000.0
                )
            records.append(
                {
                    "case_id": case["case_id"],
                    "variant": int(variant),
                    "error_mode": error_mode,
                    "source_count": len(case["scenario"].sources),
                    "fault_injection": {
                        "force_first_process_failure": bool(case.get("force_first_process_failure", False)),
                        "force_first_clear_fail": bool(case.get("force_first_clear_fail", False)),
                        "fail_transport_every": case.get("fail_transport_every"),
                    },
                    "raw_by_version": versions,
                }
            )
    return {
        "description": "Twelve adversarial classes; each applicable class is run through all three real mains.",
        "active_config_label": "production_default",
        "thresholds": {"max_actions": 10000, "max_virtual_time_s": 360000.0},
        "cases": records,
        "summary_by_case": {
            case_id: {
                version: {
                    "runs": int(sum(1 for item in records if item["case_id"] == case_id)),
                    "bounded_count": int(sum(
                        item["raw_by_version"][version].get("bounded", False)
                        for item in records
                        if item["case_id"] == case_id
                    )),
                    "any_missed_channel": bool(any(
                        item["raw_by_version"][version].get("any_missed_channel", True)
                        for item in records
                        if item["case_id"] == case_id
                    )),
                }
                for version in ("fast2", "phase1", "active")
            }
            for case_id in sorted({item["case_id"] for item in records})
        },
    }


def _active_config_dict(config) -> Optional[dict]:
    if config is None:
        return None
    names = (
        "budgets_m",
        "source_sample_count",
        "error_samples",
        "spacing_m",
        "circle_segments",
        "max_constraint_vertices",
        "max_candidates",
        "min_reception_radius",
        "max_reception_radius",
        "angle_error_deg",
        "max_steps",
        "time_limit_s",
    )
    return {
        name: [float(value) for value in getattr(config, name)]
        if name in {"budgets_m", "error_samples"}
        else getattr(config, name)
        for name in names
    }


def _subset_monte_carlo(mc: Mapping, count: int) -> dict:
    """Keep the first ``count`` paired scenes from a raw MC object."""

    result = dict(mc)
    result["seeds"] = list(mc["seeds"][:count])
    result["scene_count"] = int(count)
    result["scenes"] = list(mc["scenes"][:count])
    result["raw_by_version"] = {
        version: list(records[:count])
        for version, records in mc["raw_by_version"].items()
    }
    result["summary_by_version"] = {
        version: aggregate_runs(records)
        for version, records in result["raw_by_version"].items()
    }
    return result


def run_phase3(mode: str = "quick", *, output_path: Optional[Path] = None) -> dict:
    """Run requested Phase 3 scope and write a JSON artifact."""

    mode = str(mode).lower()
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    if mode == "quick":
        seeds = list(range(20260913, 20260915))
        mc_all = run_monte_carlo(seeds, active_label="production_default")
        exp6 = run_experiment6(slender_cases()[:2])
        exp7 = run_experiment7(boundary_cases()[:2])
        exp12 = run_experiment12(adversarial_cases()[:4])
        monte_carlo = {
            "production_default_paired": mc_all,
            "production_default_active_config": _active_config_dict(
                _load_version("active").DEFAULT_ACTIVE_CONFIG
            ),
            "sensitivity_note": "Quick smoke only (2 paired seeds); use --full for Phase 3 estimates.",
        }
    else:
        # Phase 4's numerical planner is considerably more expensive than
        # the two deterministic controllers.  Keep a 30-scene paired sample
        # for fast2/phase1 and active-light, and a 12-scene production-default
        # active sample as the explicitly labelled primary active estimate.
        seeds30 = list(range(20260913, 20260943))
        seeds12 = seeds30[:12]
        baseline30 = run_monte_carlo(
            seeds30,
            versions=("fast2", "phase1"),
            active_label="not_applicable",
        )
        active_default12 = run_monte_carlo(
            seeds12,
            versions=("active",),
            active_label="production_default",
        )
        active_light30 = run_monte_carlo(
            seeds30,
            versions=("active",),
            active_config=light_active_config(),
            active_label="light_sensitivity",
        )
        monte_carlo = {
            "production_default_common_paired_12": {
                "description": "Same 12 scenes for fast2, phase1, and active production default.",
                "seeds": seeds12,
                "scenes": baseline30["scenes"][:12],
                "fast2": _subset_monte_carlo(baseline30, 12)["raw_by_version"]["fast2"],
                "phase1": _subset_monte_carlo(baseline30, 12)["raw_by_version"]["phase1"],
                "active": active_default12["raw_by_version"]["active"],
                "summary_by_version": {
                    "fast2": aggregate_runs(baseline30["raw_by_version"]["fast2"][:12]),
                    "phase1": aggregate_runs(baseline30["raw_by_version"]["phase1"][:12]),
                    "active": active_default12["summary_by_version"]["active"],
                },
            },
            "fast2_phase1_30": baseline30,
            "active_production_default_12": active_default12,
            "active_light_sensitivity_30": active_light30,
            "production_default_active_config": _active_config_dict(
                _load_version("active").DEFAULT_ACTIVE_CONFIG
            ),
            "light_active_config": _active_config_dict(light_active_config()),
            "sample_plan_reason": "Active production default was run on 12 scenes because its measured runtime is ~10 s/game; 30-scene light sensitivity is retained separately and never combined with the production-default estimate.",
        }
        exp6 = run_experiment6()
        exp7 = run_experiment7()
        exp12 = run_experiment12()

    result = {
        "schema": "q3_phase3_experiments.v2",
        "mode": mode,
        "method": {
            "entry_points": {
                "fast2": "src/new/Q3_fast2.py",
                "phase1": "src/experiments/q3/Q3_phase1.py",
                "active": "src/experiments/q3/Q3_active.py",
            },
            "transport": "same OfflineSimulator.post monkeypatched into each real main; no proxy controller",
            "source_sampling": "N random in [10,16], r=1800*sqrt(U), distinct channels, Ri uniform in [1000,1500]",
            "direction_error": "deterministic hash(seed, channel/source, round(position,2)) in [-1,1], rounded to two decimals; fixed modes used only in fixed/adversarial cases",
            "timing": "accepted measure=move/5+channel switch+5s; accepted clear failure=move/5+3s; accepted clear success=move/5+5s; clear does not switch channel",
            "stats": "discovery/clear are source-level; per-source time is clear completion minus first accepted discovery completion; successful-game totals are separately summarized",
            "outside_ablation": "paired same source/error scenario and same per-state unrestricted candidate generator; target-only filters candidate norm > 1800",
            "limits": "all sources are omnidirectional in this offline harness; no directional coverage cone is injected",
        },
        "monte_carlo": monte_carlo,
        "experiment6": exp6,
        "experiment7": exp7,
        "experiment12": exp12,
    }
    if output_path is None:
        output_path = SRC_EXPERIMENTS_Q3 / "q3_phase3_results.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


__all__ = [
    "FairCandidatePoolPlanner",
    "OfflineSimulator",
    "ROOT",
    "Scenario",
    "Source",
    "make_scenario",
    "light_active_config",
    "aggregate_runs",
    "adversarial_cases",
    "boundary_cases",
    "run_experiment6",
    "run_experiment7",
    "run_experiment12",
    "run_monte_carlo",
    "run_processing_case",
    "run_phase3",
    "slender_cases",
    "run_version",
    "smoke_experiment",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline Phase 3 Q3 experiments")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="run a small validation sample")
    mode_group.add_argument("--full", action="store_true", help="run the full Phase 3 sample plan")
    parser.add_argument(
        "--output",
        type=Path,
        default=SRC_EXPERIMENTS_Q3 / "q3_phase3_results.json",
        help="JSON output path",
    )
    args = parser.parse_args()
    selected_mode = "full" if args.full else "quick"
    artifact = run_phase3(selected_mode, output_path=args.output)
    if selected_mode == "quick":
        mc_count = artifact["monte_carlo"]["production_default_paired"]["scene_count"]
    else:
        mc_count = artifact["monte_carlo"]["production_default_common_paired_12"]["summary_by_version"]["active"]["game_count"]
    print(
        json.dumps(
            {
                "mode": artifact["mode"],
                "output": str(args.output),
                "mc": mc_count,
                "experiment6_cases": artifact["experiment6"].get("case_count"),
                "experiment7_cases": artifact["experiment7"].get("case_count"),
                "experiment12_rows": len(artifact["experiment12"].get("cases", [])),
            },
            ensure_ascii=False,
        )
    )
