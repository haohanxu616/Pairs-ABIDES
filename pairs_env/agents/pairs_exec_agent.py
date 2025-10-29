# --------------------------------------------------
# Minimal pairs execution agent (market orders v0)
# --------------------------------------------------
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "abides-core"))
sys.path.insert(0, str(ROOT / "abides-markets"))

from abides_markets.agents.trading_agent import TradingAgent
from pairs_env.utils.schedules import ScheduleParams, cumulative_target
from pairs_env.utils.hedge import HedgeRule

log = logging.getLogger("pairs_agent")

@dataclass
class PairsExecConfig:
    mode: str
    pair: Tuple[str, str]
    hedge_kind: str
    notional_total: float
    decision_dt: float
    T: float

class PairsExecutionAgent(TradingAgent):
    def __init__(self, id: int, starting_cash: float, config: PairsExecConfig, **kwargs):
        super().__init__(id, name="PairsExecAgent", type="PairsExecAgent", starting_cash=starting_cash, **kwargs)
        self.cfg = config
        self.hedge = HedgeRule(kind=config.hedge_kind, beta=1.0)
        self.params = ScheduleParams(total_notional=abs(config.notional_total), T=config.T)
        self.sign = 1.0 if config.notional_total >= 0 else -1.0
        self.cum_target_last = 0.0
        self.NS_PER_S = 1_000_000_000

    # ✅ Ensure kernel is set even if parent TradingAgent forgets to call super()
    def kernel_initializing(self, kernel) -> None:
        # Set the kernel reference ourselves
        self.kernel = kernel
        # If the parent defines kernel_initializing, still let it do its work (safe)
        try:
            super().kernel_initializing(kernel)
        except AttributeError:
            pass

    def kernel_starting(self, startTime: int) -> None:
        # Session is [startTime, startTime + T]
        self.session_start = startTime
        self.session_end = startTime + int(self.cfg.T * self.NS_PER_S)
        # Now safe because self.kernel is set
        self.set_wakeup(startTime)

    def wakeup(self, currentTime: int):
        t_sec = (currentTime - self.session_start) / float(self.NS_PER_S)
        t_sec = max(0.0, min(t_sec, self.cfg.T))

        cum = cumulative_target(self.cfg.mode, t_sec, self.params)
        delta = cum - self.cum_target_last
        self.cum_target_last = cum

        if abs(delta) > 1e-8:
            self._execute_delta(delta * self.sign)

        next_time = currentTime + int(self.cfg.decision_dt * self.NS_PER_S)
        if next_time <= self.session_end:
            self.set_wakeup(next_time)

    def _safe_mid(self, symbol: str) -> float:
        return 100.0

    def _execute_delta(self, delta_notional: float):
        sym_a, sym_b = self.cfg.pair
        px_a = self._safe_mid(sym_a)
        px_b = self._safe_mid(sym_b)
        qty_a, qty_b = self.hedge.split_delta(delta_notional, px_a, px_b)

        # Signed qty convention in TradingAgent: + => buy, - => sell
        if qty_a != 0:
            self.placeMarketOrder(sym_a, qty_a)   # NOTE: if your API is place_market_order, rename here
        if qty_b != 0:
            self.placeMarketOrder(sym_b, qty_b)

        log.info(f"Exec delta={delta_notional:.2f} -> {sym_a}:{qty_a}, {sym_b}:{qty_b}")
