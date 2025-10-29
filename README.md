# Pairs-ABIDES

**Pairs-ABIDES** extends the [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public) simulation framework to model *synchronized multi-asset (pairs) execution* under realistic market microstructure dynamics.
It introduces deterministic rule-based experts, reinforcement learning agents, and a synchronization-aware mixture-of-experts (MoE-DPO) pipeline for interpretable optimal execution across correlated legs.

---

## Project Structure

```
pairs_abides/
│
├── env/              # Core simulation environment (ABIDES wrapper for pairs execution)
├── agents/           # Q-learning agents for each expert (per-leg & pair-level)
├── experts/          # Deterministic rule-based experts: LAE, SCE, OME, RAE
├── qlearn/           # Tabular Q-learning logic for Phase 1 pretraining
├── sip_sync/         # Stage 2: Synchronization-aware SIP ensemble (hedge ratio tracking)
├── prefs/            # Preference generation and trajectory ranking
├── moedpo/           # Stage 3: Mixture-of-Experts DPO finetuning
```

---

## Workflow Overview

1. **Stage 0 — Initialization**
   Load correlated asset pair, define hedge ratio ρ, simulation clock, and base ABIDES agents.

2. **Stage 1 — Expert Q-Learning Pretraining**
   Each expert (LAE, SCE, OME, RAE) is trained via tabular Q-learning to produce discrete execution policies in ABIDES-Gym.

3. **Stage 2 — SIP-Sync Optimization**
   Synchronization-aware optimization aligns both legs, balancing execution quality and hedge-ratio tracking.

4. **Stage 3 — MoE-DPO Finetuning**
   Distill Q-experts into differentiable policies and optimize preference-based mixtures for adaptive execution.

---

## Rule-Based Experts

| Expert  | Full Name                           | Primary Role                                         |
| ------- | ----------------------------------- | ---------------------------------------------------- |
| **LAE** | Liquidity-Adaptive Expert           | Responds to liquidity, spread, and crumbling risk    |
| **SCE** | Synchronization-Constrained Expert  | Maintains leg synchronization under hedge ratio ρ    |
| **OME** | Opportunistic Microstructure Expert | Accelerates fills in favorable microstructure states |
| **RAE** | Risk-Aversion / Stability Expert    | Reduces exposure under volatility or instability     |
