# Related Work: Territorial Belief in Active Spatial Exploration

This document condenses `multi_agent_territory_litreview.md` (40+ refs, 7
buckets) into the 1.5-page form expected for a course-project proposal,
re-organised around four threads that meet at our research question.

## 1 · From passive spatial reasoning to active belief construction

Early "spatial reasoning" benchmarks for foundation models — bAbI
(Weston et al., 2015), StepGame (Shi et al., 2022), SpartQA
(Mirzaee et al., 2021) — frame the problem as *static* relational
inference over a textual scene. A second wave evaluated single- and
multi-image perception: SpatialVLM (Chen et al., 2024), 3DSRBench
(Ma et al., 2024), MMSI-Bench (Yang et al., 2025c). A third introduced
*cognitive-map prediction* as auxiliary supervision: VSI-Bench
(Yang et al., 2025a) showed that asking models to predict an explicit
N×M map *during* video QA improves task accuracy; MindCube
(Yin et al., 2025) extended this to multi-view layouts. All of these,
however, remain *disembodied* — the agent reasons over a fixed log of
observations supplied by the benchmark.

**Theory of Space** (Zhang et al., ICLR 2026), the core reference for
this project, breaks with that pattern. ToS frames spatial intelligence
as the *active* construction, revision, and exploitation of an internal
belief: the agent itself decides what to observe next, externalises its
cognitive map at every step (probing), and is tested under a *false-belief*
perturbation that secretly relocates objects. The two characteristic
failures ToS reports — *belief drift* (correct facts overwritten in later
turns) and *belief inertia* (post-perturbation pull-back to obsolete
priors) — are the failure modes our territorial belief is designed to
address. Active-exploration baselines in ToS rely on either prompted
foundation models (GPT-5.2, Gemini-3-Pro, Claude-4.5-Sonnet, Qwen3-VL,
GLM-4.6V, InternVL-3.5) or rule-based proxy agents (`Scout`, `Strategist`)
that traverse with AC-3 constraint propagation. **No published baseline
imposes a structural unit between the agent and the global map** — that
gap is the entry point for our work.

## 2 · Cognitive maps in neuroscience and bio-inspired AI

The mammalian brain does not store space as a flat coordinate list.
Place cells (O'Keefe & Dostrovsky, 1971) encode region-confined
"locations"; grid cells (Hafting et al., 2005) provide a hexagonal basis
in entorhinal cortex; together they implement a multi-scale, hierarchical
spatial code (Stensola et al., 2012). Banino et al. (Nature, 2018)
showed grid-like representations *spontaneously emerge* in deep RL agents
trained on velocity-only navigation, indicating that this code is a
generic optimum for path integration. Recent extensions place this
biology in modern AI: the *Motivational Cognitive Maps* (MoHA, bioRxiv
2025) framework modulates the spatial map by motivational signals;
RoboMemory (arXiv 2508.01415) integrates spatial, temporal, and episodic
memory in a multi-memory architecture; ESWM (Episodic Spatial World
Model, 2025) reconstructs space from sparse episodes for transfer.
These works converge on the claim that *region-level latent units
improve sample efficiency and transfer*, but none yet expose those
units to the kind of probing ToS introduces. **Our project applies the
ToS probing methodology to a deliberately bio-inspired region-level
representation.**

## 3 · Territoriality, familiarity, and ownership

In behavioural ecology, *territory* is the spatial region an organism
defends, traverses, or returns to (Burt 1943; Powell & Mitchell 2012);
*home range* is the spatial unit at which familiarity, resource value,
and recovery are organised. The biology literature emphasises two
properties our project borrows: (i) familiarity is *cumulative* and
*recency-weighted*, not just a binary "visited" flag; (ii) territory
boundaries are *gradient*, with rapid update at unfamiliar perimeters
and slow update at the familiar core. Neither property is reflected in
current AI work on spatial exploration. Count-based bonuses (RND,
Burda et al., 2018; ICM, Pathak et al., 2017) treat visitation density
as an *exploration signal* but not as a *structural property of regions*.
Multi-agent partitioning approaches — Voronoi-based coverage
(Cortés et al., 2004), MAGEC (GNN multi-robot patrolling, 2024),
Cooperative Patrol Routing (arXiv 2501.08020) — *assign* regions to
agents but never let the partition itself emerge from familiarity.
**The dual-source territoriality framing — physical (room walls) +
familiarity (visit-weighted) — that our representation realises has
no direct precedent in the foundation-model literature.**

## 4 · World models, episodic memory, and long-horizon spatial tasks

A complementary line of work asks whether spatial reasoning can be
delegated to a learned latent world model. DreamerV3 (Hafner et al., 2023)
and the Memory Maze benchmark (Pasukonis & Hafner, ICLR 2023) demonstrate
that latent world models substantially outperform model-free methods on
long-horizon memory tasks; Recall-to-Imagine (R2I, ICLR 2024) augments
this with episodic retrieval, achieving superhuman performance on Memory
Maze. These methods, however, treat the latent space as monolithic — no
explicit factorisation between "the room I am in" and "the rooms I have
seen". GAWM (Global-Aware World Model, 2025) and the multi-modal
multi-agent world model of Pixels-to-Cooperation (arXiv 2511.01310)
extend the latent-world-model idea to multi-agent settings but again
without territorial structure. **VAGEN** (Wang & Zhang et al., NeurIPS
2025) — the multi-turn VLM-RL framework on which the ToS environment is
built — provides the only ready infrastructure for actually *training*
a world model on the ToS task; that direction is out of scope for our
3-day course project but is the natural Day-N follow-up.

## 5 · Where this project sits

```
                   passive       ─ active ──────────────────────────
                  reasoning      │ exploration                     │
       ┌──────────────────────┐  │ ┌─────────────────────────────┐ │
       │ bAbI / StepGame /    │  │ │ Theory of Space  (this work │ │
       │ SpartQA / SpatialVLM │  │ │  extends here)              │ │
       │ MMSI / VSI-Bench /   │  │ │                              │ │
       │ MindCube …           │  │ │  Belief probing, false-      │ │
       └──────────────────────┘  │ │  belief paradigm, drift      │ │
                                 │ │  diagnostics                 │ │
                                 │ └─────────────────────────────┘ │
                                 │                                 │
       ┌──────────────────────┐  │   ┌──────────────────────────┐  │
       │ Place / grid cells,  │──┼──▶│  Territorial belief       │  │
       │ MoHA, RoboMemory,    │  │   │  (this work introduces a  │  │
       │ Memory-Maze world    │  │   │  region-level unit + a    │  │
       │ models, R2I          │  │   │  confidence buffer)       │  │
       └──────────────────────┘  │   └──────────────────────────┘  │
                                 │                                 │
       ┌──────────────────────┐  │                                 │
       │ Animal territoriality│──┘                                 │
       │ (Burt, Powell &      │                                    │
       │  Mitchell), Voronoi  │                                    │
       │ coverage, MAGEC      │                                    │
       └──────────────────────┘                                    │
                                                                   │
                       ─────────────────────────────────────────────
```

Our contribution is therefore *positioned* as: a structurally
bio-inspired representation, evaluated through a recent active-exploration
probing benchmark, with the goal of mitigating two specific failure modes
(drift, inertia) that the benchmark itself diagnosed.

## 6 · Five gaps we (partially) close

These come directly from the analysis in
`multi_agent_territory_litreview.md` and are restated here for the
proposal:

1. **Region-level unit in foundation-model spatial reasoning.** No prior
   work imposes a territory layer between the LLM and the global cogmap.
2. **Familiarity as a structural property of belief, not a reward signal.**
3. **Stability vs revisability trade-off.** No prior work *measures*
   both jointly under a single representation; ToS metrics let us.
4. **Diagnostic methodology.** Substituting a structural baseline into
   a probing benchmark is itself a methodological pattern that, to our
   knowledge, has not been applied in this area.
5. *(Stretch / future)* **Multi-agent territorial handoff.** Out of scope
   for the 3-day project but an obvious extension that our representation
   is already shaped to support.

## 7 · References (proposal-grade subset)

* Banino, A. et al. *Vector-based navigation using grid-like representations*. **Nature** 557, 2018.
* Burda, Y. et al. *Exploration by Random Network Distillation*. **arXiv:1810.12894**, 2018.
* Burt, W. H. *Territoriality and home range concepts as applied to mammals*. **J. Mammalogy** 24, 1943.
* Chen, B. et al. *SpatialVLM*. **CVPR**, 2024.
* Cortés, J. et al. *Coverage control for mobile sensing networks*. **IEEE Trans. Robotics** 20(2), 2004.
* Hafner, D. et al. *DreamerV3*. **Nature** 632, 2023.
* Hafting, T. et al. *Microstructure of a spatial map in the entorhinal cortex*. **Nature** 436, 2005.
* Ma, W. et al. *3DSRBench*. arXiv:2412.07825, 2024.
* Mirzaee, R. et al. *SpartQA*. arXiv:2104.05832, 2021.
* O'Keefe, J. & Dostrovsky, J. *The hippocampus as a spatial map*. **Brain Research**, 1971.
* Pasukonis, J. & Hafner, D. *Evaluating Long-Term Memory in 3D Mazes*. **ICLR**, 2023.
* Pathak, D. et al. *Curiosity-driven Exploration by Self-supervised Prediction (ICM)*. **ICML**, 2017.
* Powell, R. & Mitchell, M. *What is a home range?* **J. Mammalogy** 93, 2012.
* Shi, Z. et al. *StepGame*. **AAAI**, 2022.
* Stensola, H. et al. *The entorhinal grid map is discretized*. **Nature** 492, 2012.
* Wang, K., Zhang, P., Wang, Z. et al. *VAGEN*. **NeurIPS**, 2025.
* Weston, J. et al. *Towards AI-complete Question Answering (bAbI)*. arXiv:1502.05698, 2015.
* Wimmer, H. & Perner, J. *Beliefs about beliefs* (false-belief paradigm). **Cognition** 13, 1983.
* Yang, J. et al. *Thinking in space (VSI-Bench)*. **CVPR**, 2025.
* Yang, S. et al. *MMSI-Bench*. arXiv:2505.23764, 2025.
* Yin, B. et al. *Spatial mental modeling from limited views (MindCube)*. arXiv:2506.21458, 2025.
* Zhang, P., Huang, Z., Wang, Y. et al. *Theory of Space*. **ICLR**, 2026.
