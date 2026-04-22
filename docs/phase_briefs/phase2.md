# Phase 2 Brief: Knowledge Layer

(Full brief saved verbatim — see conversation for complete text.)

Goal: Build the deterministic, rule-based modules that encode Random Battle knowledge:
set prediction, feature extraction, forme tracking, and damage calculation.

Deliverables:
1. pkmn/randbats data ingestion (knowledge/set_pool.py)
2. Bayesian set predictor (knowledge/set_predictor.py)
3. Feature extractor (knowledge/features.py)
4. Stateful forme tracking (knowledge/formes.py)
5. Damage calculator wrapper (knowledge/damage_calc.py)
6. Integration test (scripts/integration_test_phase2.py)

Key constraints:
- No neural networks, no training — all code is deterministic rule-based
- No replay parsing (Phase 3)
- Feature vector must be fixed-size; length documented in DECISIONS.md
- Damage calc: prefer Metamon's, then poke-env's, then @smogon/calc bridge
- Species name normalization centralized in set_pool.py
- Cache pkmn/randbats for 7 days; unit tests use pinned snapshot
