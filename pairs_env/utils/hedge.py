from __future__ import annotations
from dataclasses import dataclass

LOT = 1  # share granularity

def round_shares(qty: float) -> int:
    return int(round(qty / LOT) * LOT)

@dataclass
class HedgeRule:
    kind: str = "dollar"  # "dollar" or "beta"
    beta: float = 1.0     # used if kind == "beta"

    def split_delta(self, delta_notional: float, px_a: float, px_b: float):
        # +delta_notional => long spread => buy A, sell B
        beta = self.beta if self.kind == "beta" else 1.0
        notional_a = delta_notional
        notional_b = -beta * delta_notional
        qty_a = round_shares(notional_a / max(px_a, 1e-9))
        qty_b = round_shares(notional_b / max(px_b, 1e-9))
        return qty_a, qty_b
