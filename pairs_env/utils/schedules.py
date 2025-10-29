from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass
class ScheduleParams:
    total_notional: float
    T: float                            # total session seconds
    phi: Callable[[float], float] | None = None  # VWAP intraday density
    alpha: float = 0.6                  # IS front-load exponent (0<alpha<1 => front-loaded)
    pov: float = 0.10                   # POV participation rate

class Schedules:
    @staticmethod
    def twap(t: float, p: ScheduleParams) -> float:
        x = max(0.0, min(t / max(p.T, 1e-9), 1.0))
        return p.total_notional * x

    @staticmethod
    def vwap(t: float, p: ScheduleParams, steps: int = 600) -> float:
        if p.phi is None:
            raise ValueError("VWAP requires a phi(t) density")
        T = max(p.T, 1e-9)
        dt = T / steps
        acc = total = 0.0
        for i in range(steps):
            s = (i + 0.5) * dt
            w = p.phi(s / T)          # phi expects normalized time in [0,1]
            total += w * dt
            if s <= t:
                acc += w * dt
        frac = 0.0 if total <= 0 else acc / total
        return p.total_notional * max(0.0, min(frac, 1.0))

    @staticmethod
    def pov_increment(obs_notional_window: float, p: ScheduleParams) -> float:
        return p.pov * max(obs_notional_window, 0.0)

    @staticmethod
    def is_(t: float, p: ScheduleParams) -> float:
        x = max(0.0, min(t / max(p.T, 1e-9), 1.0))
        return p.total_notional * (x ** p.alpha)

def cumulative_target(mode: str, t: float, params: ScheduleParams) -> float:
    m = mode.upper()
    if m == "TWAP":
        return Schedules.twap(t, params)
    if m == "VWAP":
        return Schedules.vwap(t, params)
    if m == "IS":
        return Schedules.is_(t, params)
    if m == "POV":
        # For v0 we use TWAP as the backbone; POV increment is applied in the runner if you want.
        return Schedules.twap(t, params)
    raise ValueError(f"Unknown mode: {mode}")
