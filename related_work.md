# Related Work and Implementation Strategy

This note organizes the most relevant prior work for a territory-centered
multi-agent embodied navigation project. The emphasis is not on collecting as
many papers as possible, but on identifying which systems are close enough to
borrow from in code, architecture, and evaluation design.

## Reading Map

The most relevant literature falls into five buckets:

1. Multi-agent indoor semantic navigation
2. Multi-agent indoor benchmarks
3. Embodied collaboration with shared world models or beliefs
4. Heterogeneous embodied collaboration
5. Strong single-agent ObjectNav systems whose modules can be reused

Our central thesis is different from standard map-only or object-only
navigation: territory is treated as the intermediate organizational unit
between perception and action, and later between agents and collaboration.

## Comparison Table

| Work | Year | Task | Dataset / Platform | Main Mechanism | Code Status | Relation to Territory | Practical Value |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| PARTNR | 2025 | Embodied human-robot collaboration | Habitat / PARTNR benchmark | Planner + tools + skills + world graph | Public and mature | Not territory-centric, but already graph-organized | Best engineering reference for modular system layout |
| DR.WELL | 2025 | Embodied multi-agent collaboration | Custom benchmark setting | Symbolic world model + decentralized negotiation | No stable public code found | Very close in spirit: high-level organizational state | Best theory reference for coordination above action level |
| CoBel-World | 2025 | Embodied multi-agent collaboration | Custom benchmark setting | Shared belief world + intent inference + adaptive communication | No stable public code found | Strong match for territory-level belief sharing | Best reference for collaborative belief design |
| Heterogeneous Embodied Multi-Agent Collaboration | 2023 | Heterogeneous household collaboration | ProcTHOR | Hierarchical control + handshake communication | Project page available | Territory naturally matches capability-aware region assignment | Best reference for heterogeneous territorial division |
| MAIN | 2021 | Cooperative indoor navigation benchmark | Habitat-style benchmark | Communication-constrained collaborative exploration | Benchmark resources available | Shows why region-level communication matters | Good benchmark and problem-definition reference |
| MAP-THOR | 2024 | Long-horizon multi-agent planning | AI2-THOR | Collaborative long-horizon planning benchmark | Benchmark-oriented | Better for long tasks than near-term ObjectNav | Future benchmark candidate, not current base |
| Multi-Agent Embodied Visual Semantic Navigation with Scene Prior Knowledge | 2021 | Multi-agent semantic navigation | Indoor semantic navigation setting | Semantic map + scene prior + hierarchy | Code not clearly maintained | Early neighbor of territory idea | Good conceptual precursor, weak engineering base |
| CAMON | 2024 | Multi-object navigation with multiple agents | Embodied multi-object setting | LLM-based conversations + dynamic leader | Code status unclear | Closer to dialogue coordination than territory structure | Borrow communication ideas only |
| SEEK-Multi | 2026 | Collaborative semantic search | Project benchmark | Distributed scene graph + belief fusion + auction allocation | Project assets available, engineering maturity unclear | Very close application-wise, but not behavior-theoretic | Good design reference for allocation and fusion |
| WMNav | 2025 | Single-agent zero-shot ObjectNav | HM3D / MP3D | World model + memory + two-stage action proposal | Public | Single-agent only, but modules map well to our stack | Best short-term code reference for current Habitat path |

## What Existing Work Is Missing

The current literature already covers many pieces:

- semantic priors
- shared maps or scene graphs
- adaptive communication
- symbolic or belief-based collaboration
- heterogeneous role assignment

What is still under-specified is a single spatial-behavioral unit that can
simultaneously support:

- search organization
- handoff
- occupancy and access control
- boundary crossing decisions
- region-level memory and recovery

That is the gap territory can fill.

## What To Borrow From Each Work

### PARTNR

Borrow:

- clear module boundaries
- world graph abstraction
- planner / skill decomposition

Do not borrow directly:

- full benchmark assumptions
- human-robot specific planning stack

### DR.WELL

Borrow:

- high-level negotiation framing
- symbolic coordination logic
- partial-observability-aware collaboration

Do not borrow directly:

- exact symbolic formalism unless it maps cleanly to territories

### CoBel-World

Borrow:

- collaborative belief update
- intent-aware communication
- adaptive message scheduling

Do not borrow directly:

- agent-intent machinery that assumes language-heavy tasks

### Heterogeneous Embodied Multi-Agent Collaboration

Borrow:

- heterogeneous task assignment
- handoff protocols
- capability-aware decomposition

Do not borrow directly:

- task-specific tidying interfaces unless we later expand beyond navigation

### SEEK-Multi

Borrow:

- belief fusion
- auction-style task allocation
- communication efficiency ideas

Do not borrow directly:

- exact engineering assumptions until the implementation is inspected

### WMNav

Borrow:

- memory organization
- perception-to-proposal pipeline
- action filtering and high-level proposal separation

Do not borrow directly:

- single-agent assumptions as the final architecture

## Implementation Recommendation

Do not fully reproduce any one prior system before adapting it.

That route is too expensive because:

- PARTNR is much heavier than needed for immediate Habitat experiments
- MAIN is older and benchmark-oriented rather than architecture-oriented
- MAP-THOR lives in a different ecosystem
- DR.WELL and CoBel-World are stronger as conceptual references than code bases

Instead, the recommended path is:

1. Keep the current Habitat evaluation path intact
2. Rebuild the policy stack into explicit modules
3. Borrow module boundaries from PARTNR and WMNav
4. Borrow belief-sharing ideas from CoBel-World
5. Borrow allocation and coordination ideas from SEEK-Multi and DR.WELL

In short:

- reuse the current Habitat integration
- refactor our own architecture
- selectively reproduce mechanisms, not whole papers

## Proposed Territory-Centered Framework

### Single-Agent v2

Core modules:

- `perception.py`
- `belief_memory.py`
- `territorial_graph.py`
- `subgoal_planner.py`
- `local_controller.py`
- `stop_verifier.py`

Conceptual flow:

1. Perception extracts local geometry, salience, and boundary evidence.
2. Belief memory tracks likely territories, novelty, revisits, and target priors.
3. Territorial graph records region transitions and doorway-level links.
4. Subgoal planner chooses whether to stay, search, cross, or recover.
5. Local controller executes the subgoal through low-level actions.
6. Stop verifier decides when target evidence is strong enough to terminate.

### Multi-Agent v1

Add these layers on top:

- territory assignment
- territory occupancy tracking
- boundary negotiation
- territory-level communication
- region-level recovery and handoff

Communication should prioritize compact messages such as:

- territory searched / unsearched
- boundary blocked / free
- target belief raised / lowered
- teammate occupying region
- handoff requested

## Engineering Decision

The recommended engineering strategy is:

- do not rewrite Habitat from scratch
- do not fully reproduce PARTNR or MAIN
- refactor the current project into a modular architecture
- keep the current evaluator as the stable benchmark entrypoint

This gives the fastest path to:

- cleaner ablations
- easier migration to multi-agent settings
- clearer paper claims
- lower engineering risk

## Near-Term Milestones

1. Split the current monolithic policy into reusable modules
2. Validate the single-agent territorial stack on HM3D
3. Add explicit subgoal planning and stop verification
4. Introduce a shared territory belief for two simulated agents
5. Move to a benchmarked multi-agent setting once the abstraction is stable

## Multi-Agent Territorial Awareness — Second-Pass Review

See `multi_agent_territory_litreview.md` for the full survey (40+ works, 7
buckets). This section captures only what matters for planning.

### Where the field is (one-liners)

- **Multi-agent coverage / patrolling** — Voronoi-based (Cortés, Bullo) and
  learning-based (MAGEC, VDPPO patrol routing). All treat regions as either
  pre-assigned or emergent from policy, never as a first-class learnable unit.
- **Familiarity / visitation** — RND, ICM, count-based exploration. Used as
  exploration *bonus*, not as a *structural property* of space.
- **Shared world models** — GAWM, multimodal MARL world models, RoboMemory.
  Fuse observations into a joint latent, but no territorial ownership semantics.
- **Biological / cognitive** — place cells, grid cells (Banino et al.),
  MoHA (motivational cognitive maps). Closest to dual-source territory is MoHA,
  which modulates maps by motivation — replace motivation with familiarity and
  you have something close to our framing.

### Five specific gaps our dual-source framing hits

1. **Territory as explicit latent variable in world models** — nobody factorizes
   the latent into region + familiarity + ownership.
2. **Visitation density as first-class spatial unit** — visitation only appears
   as exploration reward, never as territory structure.
3. **Ownership transfer by familiarity** — multi-agent partitioning either
   pre-assigns or reallocates by proximity. None reallocates by familiarity.
4. **Familiarity-modulated subgoals in HRL** — HRL picks room / door subgoals
   but weights them uniformly. Familiarity should make recent territory
   cheaper to return to.
5. **Cognitive maps with ownership** — place/grid-cell models encode position
   but not "whose territory this is".

### Three closest-neighbor papers to read first

- **MoHA (Motivational Cognitive Maps, bioRxiv 2025)** — motivation-modulated
  maps; swap motivation for familiarity and we are adjacent.
- **RoboMemory (arXiv 2508.01415, 2024)** — multi-memory architecture with
  dynamic spatial KG; adding ownership is a small extension.
- **Memory Maze (Pasukonis & Hafner, ICLR 2023) + R2I (Chandar, ICLR 2024)** —
  the canonical benchmark; R2I is current SOTA via episodic memory retrieval
  and is the baseline to beat with territorial structure.

### Caveat

These summaries were generated by a research agent and not all papers have
been manually verified. Before citing in any paper draft, Google each name
(especially MoHA, ESWM, RoboMemory, MAGEC, GAWM) and check arXiv IDs. Flag
any that don't resolve.
