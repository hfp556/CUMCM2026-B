"""Small deterministic directional-source harness for Q4 Active.

This is local experiment infrastructure, not an official-simulator model.  It
extends the repository's Q3 offline simulator with a fixed 180-degree transmit
half-plane and runs the real ``Q4_active.main`` through the same transport API.
"""

from contextlib import redirect_stdout
from dataclasses import dataclass
import argparse
import io
import json
import math

import numpy as np

from new import Q3_active_v2 as q3
from new import Q4_active as q4
from q3_phase3_experiments import OfflineSimulator, Scenario


@dataclass(frozen=True)
class DirectionalSource:
    channel: int
    position: tuple[float, float]
    reception_radius: float
    transmit_heading_deg: float | None

    def point(self):
        return np.asarray(self.position, dtype=float)


def make_directional_scenario(seed, n=10, directional_fraction=1.0):
    rng = np.random.default_rng(int(seed))
    sources = []
    for channel in range(1, int(n) + 1):
        radius = 1800.0 * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        position = (radius * math.cos(angle), radius * math.sin(angle))
        reception = float(rng.uniform(1000.0, 1500.0))
        heading = (
            float(rng.uniform(0.0, 360.0))
            if float(rng.random()) < float(directional_fraction)
            else None
        )
        sources.append(DirectionalSource(channel, position, reception, heading))
    return Scenario(int(seed), tuple(sources))


class DirectionalOfflineSimulator(OfflineSimulator):
    """Q3 simulator plus a fixed inclusive 180-degree transmit half-plane."""

    def __init__(self, scenario, **kwargs):
        super().__init__(scenario, **kwargs)
        self.directional_blind_measurements = 0

    @staticmethod
    def _directionally_visible(source, receiver):
        heading = getattr(source, "transmit_heading_deg", None)
        if heading is None:
            return True
        offset = np.asarray(receiver, dtype=float) - source.point()
        direction = np.array(
            [math.cos(math.radians(heading)), math.sin(math.radians(heading))]
        )
        return float(offset @ direction) >= -1e-9

    def _measure(self, payload):
        channel = int(payload["channel"])
        point = np.array(
            [float(payload["position"]["x"]), float(payload["position"]["y"])],
            dtype=float,
        )
        was_discovered = channel in self.discovered
        response = super()._measure(payload)
        source = self.sources.get(channel)
        if (
            response.get("measure_result") == "direction"
            and source is not None
            and not self._directionally_visible(source, point)
        ):
            self.directional_blind_measurements += 1
            response["measure_result"] = "no_signal"
            response.pop("svd_deg", None)
            action = self.actions[-1]
            action["result"] = "no_signal"
            action.pop("svd_deg", None)
            if not was_discovered:
                self.discovered.discard(channel)
                self.first_discovery_time.pop(channel, None)
        return response


def run_directional(seed, n=10, directional_fraction=1.0, echo=False):
    scenario = make_directional_scenario(
        seed, n=n, directional_fraction=directional_fraction
    )
    simulator = DirectionalOfflineSimulator(scenario)
    previous_post = q3.post
    previous_position = q3.CURRENT_POS.copy()
    previous_virtual_time = q3.LAST_VT
    q3.post = simulator.post
    q3.CURRENT_POS = np.array([0.0, 0.0])
    q3.LAST_VT = 0.0
    stream = io.StringIO()
    try:
        with redirect_stdout(stream):
            q4.main()
    finally:
        q3.post = previous_post
        q3.CURRENT_POS = previous_position
        q3.LAST_VT = previous_virtual_time
    output = stream.getvalue()
    if echo:
        print(output, end="")
    summary = simulator.summary()
    summary.update(
        {
            "seed": int(seed),
            "directional_fraction": float(directional_fraction),
            "directional_blind_measurements": int(
                simulator.directional_blind_measurements
            ),
            "reported_success": "success: all" in output,
            "directional_reacquisition_logs": output.count(
                "directional fallback"
            ),
        }
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260913])
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--directional-fraction", type=float, default=1.0)
    parser.add_argument("--echo", action="store_true")
    args = parser.parse_args(argv)
    results = [
        run_directional(
            seed,
            n=args.sources,
            directional_fraction=args.directional_fraction,
            echo=args.echo,
        )
        for seed in args.seeds
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
