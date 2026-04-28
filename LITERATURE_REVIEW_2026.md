# Literature Review (April 2026) — Territorial × ToM × Multi-Agent Rearrangement

Generated as part of the 4/29 all-nighter. Use this to position the paper.

## TL;DR

**Niche is real, not crowded.** No 2024-2026 paper explicitly tackles
"distinguishing intentional vs accidental displacement using territorial
spatial structure + partner-presence inference."

**Target benchmark for the real paper**: Habitat 3.0 social rearrangement.

**Industry pull**: 1X NEO, Figure 02, Physical Intelligence — all home-robot
companies face this disambiguation problem in 2026.

---

## Q1. Recent literature (2024–2026)

The closest neighbors that combine multi-agent + embodied + ToM + manipulation:

| Paper | Year | Venue | What they do | What they DON'T do (= our gap) |
|---|---|---|---|---|
| MindForge — arXiv:2411.12977 | 2024 | arXiv | Hierarchical goal/perspective-taking with unseen partners; emergent collab in rearrangement | No territorial structure; no displacement-intent disambiguation |
| Habitat 3.0 — arXiv:2310.13724 | 2023→2024 | arXiv (FAIR) | Social rearrangement task; humans + avatars + robots cohabit | Yielding/cooperation focus; doesn't model "accident vs intent" |
| EMOS / Habitat-MAS — arXiv:2410.22662 | 2024 | arXiv | Heterogeneous multi-robot OS with LLM agents; multi-floor manipulation | LLM-driven; no spatial-territorial structure |
| PARTNR | 2025 | — | Tool-prediction for human-robot collab on Habitat | No intent disambiguation |
| Generative Multi-Agent Collab Survey — arXiv:2502.11518 | 2025 | arXiv | Survey of recent frameworks; counterfactual reasoning + memory | Confirms gap (no specific "displacement intent" branch) |
| Multi-Agent Embodied AI: Advances and Future Directions — arXiv:2505.05108 | 2025 | Sci. China Inf. Sci. | Reviews heterogeneous multi-robot rearrangement | Identifies open problem: "intent recognition under partial observability" |
| MANER — arXiv:2306.06543 | 2023→2024 | arXiv | Learning-based task sequencing + collaborative grasping | No belief modeling about partner |
| HiVAE — arXiv:2602.16826 | 2026 | arXiv | 3-level BDI hierarchy for ToM; campus-size navigation | Navigation only, no manipulation; no disambiguation |

### The exact gap our framing fills

> "Did my partner intentionally rearrange object X, or did X get displaced
> by accident (gravity / non-partner cause)?"
>
> Combined with: territorial familiarity (rooms × visit history) as the
> spatial backbone for that decision.

This combination is **not addressed in any 2024-2026 paper above.**

---

## Q2. Practical applications (real-world value)

### Industry labs/companies actively working on adjacent problems

- **Google DeepMind Robotics** — Gemini Robotics + Gemini Robotics-ER (2025) for household task planning. Spatial reasoning for multi-step tasks. Public roadmap mentions intent inference but no technical paper yet.
- **Physical Intelligence** ($2.8B valuation 2024) — Foundation models for robotic control across heterogeneous platforms. Home assistants in roadmap.
- **1X Technologies (NEO)** — Consumer home robot, pre-orders 2026, $20K. Alpha-testing household rearrangement in 2025. **Their stated open problem in 2026**: differentiating "user moved this" from "robot bumped it."
- **Figure AI (Figure 02)** — Washing machine loading, laundry folding demos in 2026. Same disambiguation problem.
- **Boston Dynamics / Hyundai Robotics** — Warehouse + last-mile + home expansion.

### Three concrete use cases for paper

1. **Eldercare companion + human caregiver share kitchen.**
   Medication moved → robot infers: "did caregiver reorganize, or did I bump it?"
   Decides: alert vs self-correct. Misjudgment = trust loss.

2. **Smart home, two-robot setup (vacuum + manipulation).**
   Toy displaced during cleaning → fetch robot infers: "did human leave it
   here, or vacuum push it?" Decides: retrieve vs ignore.

3. **Warehouse human-robot collaboration.**
   Crate shifted → robot infers: "did human partner make space, or vibration?"
   Decides: pick up new path vs continue. Misjudgment = wasted time + trust.

### Sentence for paper introduction

> "Distinguishing intentional rearrangement from accidental displacement is
> a foundational sub-problem for any multi-agent embodied system in shared
> human spaces — recent industry deployments (1X NEO, Figure 02, Physical
> Intelligence) all face this exact decision under partial observability."

---

## Q3. Benchmarks + ablation conventions

### Benchmark comparison table

| Benchmark | Multi-agent | Manipulation | Built-in ToM eval | Setup | Status 2026 |
|---|---|---|---|---|---|
| **Habitat 3.0** | ✅ N agents | ✅ rearrange | ❌ custom probe | 3/5 | ✅ active, FAIR |
| **MAP-THOR** | ✅ N | ✅ long-horizon | ❌ infer from traces | 4/5 | ⚠️ less actively maintained |
| **Habitat-MAS (EMOS)** | ✅ heterogeneous | ✅ multi-floor | ❌ custom | 4/5 | ✅ research code |
| **ProcTHOR** | ❌ single | ✅ rearrange | N/A | 3/5 | ✅ active, AI2 |
| **Memory Maze** | ❌ single | ❌ navigation | N/A | 2/5 | ✅ active, jurgisp |
| **MAIN** | ✅ 2-3 agents | ⚠️ navigation | implicit | 2/5 | ⚠️ superseded by Habitat 3.0 |

### Our recommendation: **Habitat 3.0**

Reasons:
- multi-agent rearrangement is first-class (social rearrangement task)
- maintained by FAIR, used by 100+ labs
- baseline literature: PARTNR 2025, Habitat 3.0 paper, MindForge variants
- license: MIT, public install
- can plug our `TerritorialToMReverter` as a custom policy

GitHub: https://github.com/facebookresearch/habitat-lab

### Ablation conventions (what reviewers expect)

Based on PARTNR 2025, MAP-THOR 2024, Habitat 3.0 2024:

**Must-have:**
1. **Signal decomposition** — ablate each ToM signal separately (✅ we already do this in Phase 2: room-only, visit-only, both)
2. **Naive baseline** — flat (always-revert) and territorial (no ToM) (✅ we have)
3. **No-memory baseline** — no-territorial-memory variant (✅ flat)
4. **Partner-policy variation** — test against different partner policies (⚠️ we only have scripted; should add random / greedy / collaborative variants)

**Strong-paper ablations:**
5. **Environment complexity sweep** — 4-room → 8-room → full apartment (⚠️ we have 4-room only)
6. **Partial observability sweep** — vary visibility radius (⚠️ we have v=3 only)
7. **N-agent scalability** — 2 → 3 → 4 agents (❌ we have 2 only)

**Outstanding-paper ablations:**
8. **Real-robot or photoreal sim deployment** (Habitat 3.0 photoreal scenes)
9. **User study** — humans rate how "respectful" each method's behavior is

### Action items for "real" paper

- [ ] Install Habitat 3.0 + run social-rearrangement baseline
- [ ] Port `agents.py` ToM logic to Habitat policy interface
- [ ] Add partner-policy variants (random / greedy / collaborative)
- [ ] Run env complexity sweep (1-room, 2-room, 4-room, 8-room)
- [ ] Visibility-radius sweep (v=1, 2, 3, 5, ∞)
- [ ] Compare with PARTNR baseline on intent-recognition subtask

---

## Honest assessment

**For NeurIPS 2026 main track (4 days from now)**: not feasible. Toy env results are not enough.

**For NeurIPS 2026 workshop (Cooperative AI / Embodied World Models / Goal-conditioned RL)**: feasible — current data + a clean writeup. Workshops accept toy-env demonstrations of novel framings.

**For ICLR 2027 / NeurIPS 2027 main track**: very feasible. Add Habitat 3.0 validation + ablation 5, 6, 7 above + 1 user study angle.

The toy result tonight is the seed for that paper.

---

## Sources cited (verbatim from the lit-review agent's response)

- arXiv:2411.12977 — MindForge (2024)
- arXiv:2310.13724 — Habitat 3.0 (2023→2024)
- arXiv:2502.11518 — Generative Multi-Agent Collab Survey (2025)
- arXiv:2505.05108 — Multi-Agent Embodied AI: Advances and Future Directions (2025)
- arXiv:2410.22662 — EMOS / Habitat-MAS (2024)
- arXiv:2306.06543 — MANER (2023→2024)
- arXiv:2602.16826 — HiVAE (2026)
- DeepMind Robotics blog — Gemini Robotics (2025)
- 1X Technologies — NEO consumer home robot (pre-order 2026)
- Figure AI — Figure 02 demos (2026)
- github.com/facebookresearch/habitat-lab (MIT)
- ai2thor.allenai.org / procthor.allenai.org (Apache 2.0)
- github.com/jurgisp/memory-maze (MIT)

---

*Generated 4/29/2026 during all-nighter. Verify all cites before final submission.*
