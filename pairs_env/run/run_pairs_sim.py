# -----------------------------------------------
# Minimal runner for pairs execution simulation
# -----------------------------------------------
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

# Make sure local source trees are used instead of site-packages
ROOT = Path(__file__).resolve().parents[2]  # .../PAIRSABIDES-MAIN
sys.path.insert(0, str(ROOT))                          # so `pairs_env` is importable
sys.path.insert(0, str(ROOT / "abides-core"))
sys.path.insert(0, str(ROOT / "abides-markets"))

from abides_core.kernel import Kernel
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns  

# ABIDES-Markets agents (note the .agents. path)
from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_markets.agents.noise_agent import NoiseAgent
from abides_markets.agents.value_agent import ValueAgent

from pairs_env.agents.pairs_exec_agent import PairsExecutionAgent, PairsExecConfig

log = logging.getLogger("pairs_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
import numpy as np

# --- Minimal Oracle so ExchangeAgent can boot --------------------------------
class DummyOracle:
    """
    Minimal oracle:
      - get_daily_open_price(symbol, mkt_open) -> int cents
      - observe_price(symbol, t, sigma_n=0, random_state=None) -> int cents
    """
    def __init__(self, open_price_cents: int = 10000):
        self.open_price_cents = open_price_cents
        self.f_log: Dict[str, list] = {}  # optional

    def get_daily_open_price(self, symbol: str, mkt_open: NanosecondTime) -> int:
        return int(self.open_price_cents)

    def observe_price(                       # ← NEW
        self,
        symbol: str,
        current_time: NanosecondTime,
        sigma_n: float = 0.0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> int:
        """
        Return a noisy observation around the open price (in integer cents).
        Matches ValueAgent’s call signature.
        """
        rng = random_state or np.random.RandomState(42)
        noise = rng.normal(loc=0.0, scale=float(sigma_n))
        return int(round(self.open_price_cents + noise))


# --- Build agents ------------------------------------------------------------
def build_agents(symbols, start_time_ns, end_time_ns):
    agents = []
    next_id = 0

    # Exchange requires `symbols=` in your build
    exch = ExchangeAgent(
        id=next_id,
        name="EXCH",
        mkt_open=start_time_ns,
        mkt_close=end_time_ns,
        symbols=symbols,
    )
    agents.append(exch); next_id += 1

    # Symmetric lightweight background flow
    for sym in symbols:
        val = ValueAgent(id=next_id, symbol=sym, starting_cash=1_000_000)
        agents.append(val); next_id += 1
        for _ in range(2):
            nt = NoiseAgent(id=next_id, symbol=sym, starting_cash=1_000_000)
            nt.wakeup_time = start_time_ns    # ← initialize so comparisons won’t see None
            agents.append(nt); next_id += 1


    return agents, next_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["TWAP", "VWAP", "POV", "IS"], default="TWAP")
    parser.add_argument("--pair", type=str, default="SYM_A,SYM_B")
    parser.add_argument("--notional", type=float, default=1_000_000)
    parser.add_argument("--hedge", choices=["dollar", "beta"], default="dollar")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--dt", type=int, default=30, help="decision dt (seconds)")
    args = parser.parse_args()

    sym_a, sym_b = args.pair.split(",")
    symbols = [sym_a, sym_b]

    NS_PER_S = 1_000_000_000
    start_time_ns = str_to_ns("09:30:00")                    # 09:30:00 in ns
    end_time_ns   = start_time_ns + args.minutes * 60 * NS_PER_S


    agents, next_id = build_agents(symbols, start_time_ns, end_time_ns)

    cfg = PairsExecConfig(
        mode=args.mode,
        pair=(sym_a, sym_b),
        hedge_kind=args.hedge,
        notional_total=args.notional,
        decision_dt=float(args.dt),
        T=float(args.minutes * 60),
    )
    exec_agent = PairsExecutionAgent(id=next_id, starting_cash=10_000_000, config=cfg)
    agents.append(exec_agent)

    # ✅ Inject a simple oracle so ExchangeAgent can read open prices
    oracle = DummyOracle(open_price_cents=10000)  # $100.00

    kernel = Kernel(
        agents=agents,
        start_time=start_time_ns,
        stop_time=end_time_ns,
        default_computation_delay=0,
        custom_properties={"oracle": oracle},
    )

    log.info(f"Starting sim: mode={args.mode}, pair=({sym_a},{sym_b}), notional={args.notional}")
    kernel.run()                   # run() wraps initialize() + runner() + terminate()
    log.info("Simulation finished.")


if __name__ == "__main__":
    main()
