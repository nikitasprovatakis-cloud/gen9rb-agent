Gen 9 Random Battle Agent — Project Overview
Goal
Build a competitive agent for Pokemon Showdown's Gen 9 Random Battle format, trained via offline RL on high-ELO human replays and polished with laptop-scale self-play.
Design decisions (settled, do not re-litigate without explicit reason)

No LLM reasoning in the critical path. We use neural networks trained on outcome data.
Outcome-based training signal. V network learns from terminal rewards (+1 win, -1 loss). No HP shaping.
Advantage-weighted imitation handles credit assignment. Bad moves in losing trajectories get down-weighted automatically via V.
Replay filter. Gen 9 Random Battle, 1900+ match rating, 12-month window, tournament replays included.
Architecture. 20M-parameter transformer over trajectory, shared backbone for policy and value, inherits from Metamon.
Action space. Flat, 13 slots max (4 moves × {normal, tera} + up to 5 switches), illegal actions masked at logits.Set prediction. Bayesian updater with pkmn/randbats prior, no special-case Choice item handling.Belief encoding. Option C — expected-value features plus explicit probabilities for decision-critical binaries.Self-play. PPO, terminal rewards, diverse opponent pool (past checkpoints + heuristic baseline + Metamon Gen 9 checkpoint). Target 200-500k games on laptop RTX 3080.Search inference. MCTS-at-endgame as Phase 8 optional addition with explicit A/B comparison against pure-policy.The 10 phasesInfrastructure — Local Showdown, poke-env, Metamon fork, MaxDamageBot baseline. ✅ Complete.Knowledge layer — Set predictor, feature extractor, stateful-forme tracking, damage calc wrapper. Next.Replay ingestion — Scrape 1900+ Gen 9 RB replays, reconstruct first-person trajectories.Behavioral cloning — Train policy on all trajectories.Value learning — Train V on outcomes.Advantage-weighted offline RL — Refine policy using V-derived advantages.Self-play polish — Bounded laptop-scale RL refinement.Search-augmented inference — MCTS-at-endgame with A/B comparison.Opponent-specific adaptation — Tendency features from ~30 replays.Ladder evaluation — Deploy to public ladder with clearly-labeled bot account.Working patternThis project is designed collaboratively. Design decisions and phase briefs are produced via a back-and-forth between the developer and a senior Claude model. Claude Code receives per-phase briefs saved to docs/phase_briefs/phaseN.md. When Claude Code completes a phase, the developer relays results back to the senior Claude for review before moving to the next phase.
Claude Code should not skip ahead between phases, re-architect settled decisions, or make independent scope-expanding choices without flagging them first.
Key sources

Metamon (UT Austin RPL): https://github.com/UT-Austin-RPL/metamon
pkmn/randbats (set data): https://pkmn.github.io/randbats/
Pokemon Showdown: https://github.com/smogon/pokemon-showdown
poke-env (bot interface): shipped with Metamon, do not upgrade independently
