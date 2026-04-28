# Territory-Aware World Models for Multi-Agent Embodied Navigation: Literature Review

## Scope Note

This review covers recent work (2019–2025) on spatial partitioning, familiarity-based memory, multi-agent belief fusion, and biologically-inspired territoriality in embodied AI. We focus on papers not already covered in the existing related_work.md (PARTNR, DR.WELL, CoBel-World, SEEK-Multi, MAIN, MAP-THOR, CAMON, WMNav, etc.), with emphasis on works that treat spatial regions as learnable or dynamically-assigned units. We exclude pure visual navigation (without spatial abstraction), pure planning/path-finding without learning, and robotics works that do not connect to embodied RL. The review aims to identify gaps where dual-source territory (physical boundaries + exploration familiarity) offers a novel framing.

---

## Bucket 1: Multi-Agent Spatial Partitioning, Coverage, and Patrolling

### Classical and Learning-Based Approaches

The foundations of multi-robot coverage and partitioning rest on **Voronoi-based methods** (Cortés et al. 2004, 2005; Bullo et al.). These works assume a priori knowledge of environment density and assign robots to non-overlapping Voronoi cells, minimizing coverage time. Key references in the distributed control literature include:

- **MSP algorithm** (Multi-robot patrolling via balanced graph partitioning): Dynamic territory allocation using graph cuts, without requiring pre-known maps. Agents assume responsibility for patrolling their assigned subgraph.
- **Lee et al. (2016)** on structured triangulation for coverage and Voronoi partitions in multi-robot systems (published in *International Journal of Robotics Research*).
- **Applications of Voronoi Diagrams in Multi-Robot Coverage** (MDPI 2024): Recent survey of generalized Voronoi approaches for multi-robot region coverage, including time-varying target handling.

### Recent Learning-Based Partitioning

More recent work integrates learned policies with spatial partitioning:

- **Cooperative Patrol Routing** (arXiv:2501.08020): Uses value decomposition PPO (VDPPO) in decentralized POMDPs to learn unpredictable patrol routes in urban graphs. Agents learn distributed partitioning as an emergent behavior rather than explicit assignment.
- **Adaptive Partitioning for Coordinated Multi-Agent Perimeter Defense** (IROS 2020): Combines importance-aware territorial partitioning with adaptive boundary expansion via lightweight shared-memory overlays (CTAP algorithm).
- **Multi-SLAM with Lightweight Predictive Frontier Exploration** (LPFE): Graph-based collaborative SLAM enables robots to explore unknown environments while building shared occupancy representations. Exploration is coordinated via deterministic frontier allocation without explicit communication of subgoals.

### Online vs. Pre-Known Maps

**Pre-known approaches:** Voronoi-based and graph-partitioning methods assume static or semi-static environments with known topology.

**Online/unknown-environment methods:** Frontier-based exploration (LPFE, decentralized multi-robot SLAM) and emergent partitioning (cooperative patrolling via MARL) allocate and reallocate territory as the environment is discovered. These are closer to the dual-source framing, as territory emerges from both structural constraints (doorways, walls in discovered frontiers) and cumulative exploration.

---

## Bucket 2: Familiarity, Novelty, and Visitation-Based Memory in Embodied Agents

### Count-Based and Prediction-Error Exploration

The simplest formulation: count how many times a state has been visited and assign an intrinsic bonus to rare states. This foundational idea underlies several modern approaches:

- **Random Network Distillation (RND)** (Burda et al., arXiv:1810.12894): Uses prediction error of a fixed random network as an exploration bonus, avoiding the "noise TV" problem. Demonstrated state-of-the-art on Montezuma's Revenge and sparse-reward environments without demonstrations.
- **Intrinsic Curiosity Module (ICM)**: Prediction error of a learned model as intrinsic reward, foundational for self-supervised exploration.
- **Exploration surveys** (Lilianweng, Medium 2020; GeeksforGeeks summaries): Frame intrinsic motivation across information gain, learning progress, and state novelty.

### Episodic Memory with Spatial Structure

Recent work explicitly uses episodic (experience-centric) memory to build spatial representations:

- **RoboMemory** (arXiv:2508.01415): A brain-inspired multi-memory framework integrating Spatial, Temporal, Episodic, and Semantic memory in a parallelized architecture. Key innovation: dynamic spatial knowledge graph for scalable, consistent memory updates tied to agent poses over time.
- **ESWM (Episodic Spatial World Model)**: Constructs spatial models from sparse, disjoint episodic memories without additional training. Enables near-optimal exploration and navigation in novel environments by reconstructing space from past episodes.
- **AriGraph** (IJCAI 2025): Combines semantic knowledge graphs with episodic memory (episodic vertices and edges), enhancing text-based agent performance. Demonstrates that episodic grounding of spatial relationships improves generalization.
- **ESceme**: Vision-and-language navigation with episodic scene memory.
- **Learning through Experience** (OpenReview): Graph-based episodic memory representation enabling incremental storage and retrieval of spatial experiences.

### Place Cells and Grid Cells in Neural Agents

Inspired by neuroscience, several recent works show that spatial structures emerge in trained agents:

- **DeepMind "Vector-based navigation using grid-like representations"** (Nature 2018, Banino et al., with code on GitHub): Trained recurrent networks on virtual navigation tasks using velocity signals alone. Grid-like hexagonal firing patterns **spontaneously emerged**, mirroring biological grid cells. Striking convergence suggests grid cells provide efficient spatial codes even in artificial agents.
- **Place Cells Refining Grid Cells** (Nature Scientific Reports 2022): Place cells dynamically refine grid cell activities to reduce path integration error, demonstrating a division of labor in neural spatial coding.
- **Bio-Inspired Grid-Cell Navigation in Robotics** (PMC 2024 survey): Reviews how place and grid cell models inform robotic path integration and embodied navigation.
- **Grid Cells Are Ubiquitous in Neural Networks** (arXiv:2003.03482): Analysis showing grid-like responses appear in diverse architectures, not just in specialized spatial modules.

### Visitation Heatmaps and Occupancy Distribution

- **Embodied 3D Occupancy Prediction** (EmbodiedOcc, arXiv:2412.04380): Agents maintain progressively refined occupancy maps through embodied exploration, with spatial memory enabling obstacle avoidance and scene understanding.
- **GSMem (3D Gaussian Splatting as Persistent Spatial Memory)** (arXiv:2603.19137): Uses dense radiance fields as persistent memory, allowing agents to re-visit and render previously explored regions, encoding familiarity through photorealistic reconstruction.

### Gap: Visitation as First-Class Spatial Unit

**Critical observation:** Most work treats visitation counts or occupancy as a *signal* (for bonus rewards or memory content), not as a *structural unit* defining spatial regions. The dual-source territory idea—where territory emerges from both walls (physical partition) and patrol density (exploration familiarity)—remains largely unexplored in the embodied RL literature.

---

## Bucket 3: Multi-Agent Shared Maps, Belief Fusion, and Territorial Handoff

### Shared World Models and Belief Fusion

- **From Pixels to Cooperation: Multimodal World Models** (arXiv:2511.01310): Learns a unified, predictive model by fusing multimodal, partially-observable information from all agents using attention-based mechanisms. Agents train cooperative policies in the latent space of this shared world model, decoupling representation learning from policy learning.
- **GAWM: Global-Aware World Model for MARL** (arXiv:2501.10116): Latent variable world model ensuring global consistency and stability. Significantly improves global representation and sample efficiency for multi-agent control.
- **Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective** (arXiv:2505.20922, MADIFF framework): Treats diffusion models as generative priors for jointly modeling agent dynamics, achieving state-of-the-art on multi-agent control benchmarks.
- **Theory and Practice of Multi-Agent World Models** (U. Maryland): Theoretical framework for understanding when and how shared world models improve MARL performance.

### Decentralized SLAM and Exploration Coordination

- **Multi-Robot Autonomous Exploration with Localization Uncertainty** (arXiv:2403.04021): Asynchronous EM-based decentralized exploration tightly coupled with factor-graph SLAM. Handles communication-restricted scenarios.
- **Cooperative Frontier-Based Exploration** (multiple works): Agents coordinate by discovering and claiming frontier points (boundaries between explored/unexplored space). Lighter communication overhead than centralized belief fusion.
- **Multi-Robot Collaborative SLAM (Multi-SLAM) with LPFE** (IEEE 2024): Distributed lightweight predictive frontier exploration; robots anticipate each other's actions without explicit communication, using a deterministic heuristic on a shared global map.
- **Efficient Frontier Management for Collaborative Active SLAM** (arXiv:2310.01967): Manages the frontier candidate set to reduce computational overhead while maintaining exploration efficiency.

### Task Allocation by Spatial Proximity

- **Multi-Agent RL Task Allocation (MDPI 2025)**: Surveys decentralized task allocation; several works incorporate **Distance-to-Task-Location + Capability Match (DTLCM)** mechanisms. UAVs are assigned to tasks based on both spatial proximity and computational capacity.
- **Multi-Agent RL for UAV Edge Computing** (PMC 2024): Proximal task offloading reduces latency by ~12% when agents preferentially take nearby tasks.
- **Scalable Localized Policies for Multi-Agent Networks** (MLPR 2020): Agents learn policies that depend only on local neighborhoods, enabling scaling without centralized coordination. LOMAQ decomposes the problem into local sub-problems.
- **On the Near-Optimality of Local Policies** (OpenReview): Theoretical analysis of when localized policies achieve global optimality in cooperative MARL.

### Occupancy Maps and Belief Representation

- **Uncertainty-Aware Occupancy Map Prediction** (ResearchGate): Generative networks predict occupancy beyond the field of view, with variance estimates for risk-aware planning.
- **Occupancy Grid Mapping Surveys** (MDPI, IEEE): Fundamental tools for multi-robot mapping. Cells encode occupancy probability; uncertainty quantification enables active perception.
- **Probabilistic Occupancy Grids for Bayesian Exploration** (Robotics and Autonomous Systems 2019): Robots coordinate by maximizing information gain over shared occupancy models.

### Territorial Handoff (Emerging Gap)

None of the reviewed papers explicitly model **ownership transfer** or **responsibility handoff** of regions between agents. This is a key gap: in real animal systems (e.g., canines, primates), territories can shift or be defended/abandoned. In MARL, most work assumes static or rigid partitioning, or emergent partitioning via frontier discovery. Dual-source territory with dynamic handoff (based on familiarity and patrol density) is largely unstudied.

---

## Bucket 4: Biological and Cognitive-Science Inspiration for AI Territoriality

### Animal Territoriality and Home Range

- **Reinforcement Learning from Wild Animal Videos** (arXiv:2412.04273): Extracts behavioral skills (walking, jumping, stillness) from animal videos and grounds them in robot policies. Shows that animal behavior provides inductive biases for RL.
- **AnimalEnvNet: Deep RL for Animal Agent Construction** (MDPI 2024): Fuses historical trajectory data and remote sensing to simulate animal agents, demonstrating RL's applicability to movement and territorial behavior modeling.
- **Home Range Estimation via Active Learning** (MDPI Geoinformation 2023): Classical spatial ecology formulation: home range is the area where an animal acquires resources, mates, reproduces. Estimating home ranges from trajectory data is a canonical problem.
- **Machine Learning for Inferring Animal Behavior from Location/Movement Data** (ScienceDirect survey): Comprehensive review of how ML infers territoriality, foraging, and dispersal from GPS tracks.
- **Probabilistic Generative Modeling of Animal Behavior** (Neural Networks 2021): RL and unsupervised learning extract intrinsic features of animal behavior, showing parallels to intrinsic motivation in RL agents.

### Hippocampal and Entorhinal Inspiration

- **Brain-Inspired Hippocampal Spatial Cognition** (Brain Sciences 2022): Memory-replay mechanisms in the hippocampus enable learning of place and grid cells, informing neural architecture design.
- **Learning Dynamic Cognitive Maps with Autonomous Navigation** (Frontiers 2024): Agents build and update cognitive maps dynamically, mirroring hippocampal place cells and entorhinal grid cells.
- **Motivational Cognitive Maps** (bioRxiv 2025): MoHA framework integrates interoceptive (motivational) and exteroceptive (visual) information to generate motivationally-modulated cognitive maps; animals and agents prioritize familiar regions with high resource value.
- **Automated Construction of Cognitive Maps via Visual Predictive Coding** (Nature Machine Intelligence 2024): Agents construct cognitive maps through predictive coding, bridging neuroscience and embodied AI.
- **Orthogonalized State Machine from Hippocampal Learning** (bioRxiv 2023): Learning produces hippocampal-like representations; state spaces become orthogonalized, enabling efficient generalization and transfer.

### Key Link: Familiarity as Motivational Modulation

The biology literature emphasizes that **animal territoriality is tied to resource-dependent motivation**: animals defend or maintain territories where they have high reward expectancy (food, mates, safety). A dual-source model where familiarity (from visitation) modulates territory strength or attractiveness echoes this biological insight. Current RL models primarily use familiarity for *exploration bonus* (novelty-seeking); using it as a *territory ownership signal* is underexplored.

---

## Bucket 5: Memory Maze and World Models for Long-Horizon Navigation

### The Memory Maze Benchmark

- **Evaluating Long-Term Memory in 3D Mazes** (Pasukonis, Hafner, Lillicrap; ICLR 2023): A 3D maze environment with random layouts where agents must remember object locations, wall configurations, and their own position over extended episodes. The agent needs:
  - **Spatial memory** (wall layout, topology),
  - **Episodic memory** (where objects were),
  - **Egocentric localization** (where am I?).
  
This benchmark is exactly suited to test world models that encode territory (regions with known structure) vs. explicit waypoints.

### World Models for Long-Horizon Tasks

- **Mastering Memory Tasks with World Models** (ICLR 2023, Pasukonis et al.): Agents trained with learned world models (latent dynamics) dramatically outperform model-free methods on Memory Maze, demonstrating the critical importance of predictive models for long-term memory tasks.
- **Recall to Imagine (R2I)** (ICLR 2024): Builds on world models with episodic memory retrieval. R2I retrieves past episodes and uses them to improve planning, achieving superhuman performance on Memory Maze.

### Relevance to Territory-Aware Models

Memory Maze naturally admits a territorial decomposition: the agent could learn to encode regions (rooms, corridors, open areas) separately from object memories, and track familiarity per region. Current world models treat the entire latent space as a monolithic representation. A territory-aware world model might factorize the latent space into:
- **Territory embeddings** (familiar regions with known structure),
- **Novelty signals** (unexplored or weakly-explored areas),
- **Episodic anchors** (specific remembered events within territories).

This is a clear gap: no published work explicitly uses *territory as a latent variable* in world models for embodied navigation.

---

## Bucket 6: Graph Neural Networks, Locality, and Hierarchical RL

### Graph-Based Multi-Agent Coordination

- **Graph Neural Network-based MARL for Resilient Coordination** (arXiv:2403.13093): GNN-based approach (MAGEC, using MAPPO and inductive GNNs) for multi-agent patrolling, vehicle routing, and swarm navigation. Handles agent attrition and communication uncertainty.
- **Graph Attention Networks for Multi-Agent Path Finding** (PLOS One 2024): Fuses temporal and spatial dimensions via GAT for dynamic trajectory planning. MACNS integrates GNN and deep RL for collaborative navigation in dynamic environments.
- **G-Designer: Architecting Multi-Agent Communication Topologies via GNNs** (arXiv:2410.11782): Learns communication graphs for MARL teams, determining which agents should exchange information based on task structure.

### Locality-Aware and Hierarchical Policies

- **Locality Matters: Scalable Value Decomposition** (AAAI 2021): LOMAQ decomposes value functions by locality; each agent's policy depends only on nearby agents' observations. Scales better than global-attention MARL.
- **Path Planning via MARL in Dynamic Environments** (arXiv:2511.15284): Hierarchical, region-aware RL framework where the environment is decomposed into regions and local agents adapt to changes. Enables efficient retrain when a local region changes.
- **Scalable RL of Localized Policies for Multi-Agent Networks** (MLPR 2020): Distributed actor-critic where agents learn policies using only local state-action spaces.
- **Hierarchical Reinforcement Learning in Spatial Navigation** (arXiv:2504.18794): HRL with multi-goal navigation; subgoals often correspond to doorways or room boundaries—natural spatial abstractions.
- **Hierarchical Multi-Agent Skill Discovery** (NeurIPS 2023): Learn a hierarchy of skills; middle level often naturally corresponds to spatial subtasks (e.g., "reach the east wing").

### Key Observation: Locality as Implicit Territory

These works show that locality-based decomposition (neighbors, regions) emerges naturally as a solution to scalability. However, they do not explicitly frame locality as "territory with ownership" or "familiarity-modulated regions." The connection is there, but unnamed.

---

## Bucket 7: Intrinsic Motivation and Exploration Strategies (Revisited)

### Broader Coverage of Intrinsic Motivation

- **Information-Theoretic Perspective on Intrinsic Motivation** (arXiv:2209.08890): Unifies prediction-error, empowerment, and mutual information approaches under an information-theoretic framework.
- **LLM-Driven Intrinsic Motivation for Sparse Reward RL** (arXiv:2508.18420): Uses language models to generate semantic curiosity bonuses, beyond pixel-level novelty.
- **Fostering Intrinsic Motivation with Pretrained Foundation Models** (arXiv:2410.07404): Foundation models (e.g., CLIP, language models) provide rich intrinsic signals by semantic relevance.
- **Surprise-Adaptive Intrinsic Motivation** (arXiv:2405.17243): Adapts surprise (divergence between prediction and reality) dynamically, rather than using fixed novelty metrics.

### Visitation Counts in Multi-Agent Settings

The literature on count-based exploration predominantly addresses single-agent settings. Multi-agent visitation-based bonuses remain underexplored: should exploration bonuses be agent-specific (based on each agent's personal visit counts) or global (based on team coverage)? Should agents actively avoid over-explored regions (competitive) or share exploration benefits (cooperative)? These questions are largely unanswered and central to a dual-source territory framing.

---

## Gap Analysis: Where Dual-Source Territory Differs

Based on the comprehensive review, here are **5 specific gaps** where the dual-source (physical + familiarity) territory framing is novel:

### Gap 1: Territory as an Explicit Latent Variable in World Models

**Current state:** World models (GAWM, Multimodal WM, Diffusion-inspired) treat the environment as a monolithic latent space or as a collection of agent-centric observations. Spatial structure (room topology) and familiarity (visitation frequency) are implicit.

**Dual-source difference:** A territory-aware world model would factorize the latent space into:
- Region embeddings (learned from wall/door configurations),
- Familiarity scores (based on cumulative visitation/patrol density),
- Ownership/responsibility flags (which agent is responsible?).

**Closest neighbors:** ESWM (episodic spatial worlds), RoboMemory (spatial knowledge graphs), but neither explicitly ties territory to ownership or dual-source framing.

### Gap 2: Visitation Density as a First-Class Spatial Unit (Not Just an Exploration Signal)

**Current state:** Visitation counts, heatmaps, and density distributions are used to compute exploration bonuses (RND, ICM) or for post-hoc analysis (animal trajectory heatmaps).

**Dual-source difference:** Familiarity (measured as visitation density, dwell time per region) directly shapes the spatial partition, not merely as an extrinsic reward, but as an intrinsic property of the region itself. A region becomes "more territory" the more it is patrolled.

**Closest neighbors:** Biological territoriality papers (home-range estimation, animal behavior modeling), but these do not bridge to embodied RL.

### Gap 3: Territory Ownership Transfer Based on Familiarity

**Current state:** Multi-agent partitioning (Voronoi, graph-based, frontier-based) either pre-assigns regions or allocates based on proximity/capability. Once assigned, regions are fixed or only reassigned on task failure.

**Dual-source difference:** Territories could dynamically shift hands based on relative familiarity: if agent A has patrolled region R more than agent B, region R becomes "stronger" for agent A, and handoff to agent B requires overcoming that familiarity bias.

**Closest neighbors:** Cooperative patrolling (arXiv:2501.08020) uses emergent learning but not explicit familiarity-based ownership; decentralized SLAM + frontier exploration coordinate territory without ownership semantics.

### Gap 4: Hierarchical RL with Explicitly Familiarity-Modulated Subgoals

**Current state:** Hierarchical RL methods (HRL in spatial nav, skill discovery) learn subgoals that often correspond to spatial waypoints or rooms, but the connection between subgoal "importance" and familiarity is unstated.

**Dual-source difference:** Subgoals (corridors, entry points) could be weighted by familiarity: highly-explored subgoals are less important for novelty-driven agents, but remain important for task-completion agents. A territory-aware hierarchy would balance task reward with familiarity-based urgency.

**Closest neighbors:** Hierarchical Multi-Agent Skill Discovery (NeurIPS 2023), HRL in Spatial Navigation (arXiv:2504.18794), but neither models familiarity-modulated goal priority.

### Gap 5: Cognitive Maps with Territorial Ownership

**Current state:** Cognitive maps (inspired by hippocampus/entorhinal cortex) encode position, topology, and object locations. They do not encode territorial ownership or degrees of familiarity.

**Dual-source difference:** A territorial cognitive map would explicitly represent:
- Which regions agent i has traversed,
- Relative familiarity (i's history vs. teammates' histories),
- Ownership probability or responsibility probability per region.

This would require extending place-cell and grid-cell models to include a "territorial dimension."

**Closest neighbors:** MoHA (Motivational Cognitive Maps), Automated Cognitive Map Construction (Nature MI 2024), but these focus on motivation for resources, not territorial ownership.

---

## Comparison Table

| Work | Year | Task | Mechanism | Code | Relation to Territory | Value |
|------|------|------|-----------|------|----------------------|-------|
| Voronoi Coverage (Cortés, Bullo, et al.) | 2004–2005 | Multi-robot coverage | Partition space into Voronoi cells; Lloyd's algorithm minimizes coverage time | Public (classical) | **Physical partition only**; assumes pre-known density. No familiarity. | Foundational; widely cited. Baseline for pre-known environments. |
| Multi-SLAM with LPFE | 2024 | Collaborative exploration | Graph-based SLAM + frontier exploration; distributed allocation via lightweight heuristic | Yes (GitHub) | **Hybrid**; combines structural frontiers (known map partitions) with online allocation. Lightweight version of belief fusion. | Efficient decentralized approach; scalable to 3+ robots. |
| RND (Burda et al.) | 2018 | Sparse-reward exploration | Fixed random network; prediction error as exploration bonus | Yes | **Familiarity signal only**; counts novelty, not ownership. Single-agent primarily. | Demonstrated superhuman Montezuma's Revenge without demos. Influential method. |
| DeepMind Grid Cells (Banino et al.) | 2018 | Virtual navigation | Recurrent network learns from velocity signals; grid-like patterns emerge | Yes (GitHub) | **Implicit spatial structure**; grids encode space, not territory. No multi-agent extension. | Striking biological convergence. Foundational for bio-inspired spatial RL. |
| RoboMemory | 2024 | Long-horizon embodied tasks | Brain-inspired multi-memory (spatial, temporal, episodic, semantic); dynamic spatial knowledge graph | Unreleased (preprint) | **Spatial memory only**; tracks pose and episodic content, but no explicit ownership or familiarity-based partitioning. | Comprehensive memory architecture; brain-inspired design. Promising for complex navigation. |
| ESWM (Episodic Spatial World Model) | 2025 | Novel environment exploration | Constructs spatial models from sparse episodic memories without re-training | Likely (recent) | **Episodic + spatial fusion**; reconstructs space from visits, but single-agent and no territorial ownership. | Near-optimal exploration; generalization without retraining. |
| From Pixels to Cooperation (Multimodal WM) | 2025 | Multi-agent control | Attention-based fusion of multimodal observations into shared latent world model | Likely (recent) | **Shared world model only**; no explicit spatial partitioning or territory concepts. | Fast, sample-efficient MARL in latent space. |
| GAWM (Global-Aware World Model) | 2025 | Multi-agent control | Latent variable world model; ensures global consistency | Unreleased (preprint) | **Global consistency, not territoriality**; stabilizes shared models but no territorial semantics. | Strong MARL performance; state-of-the-art on benchmarks. |
| GNN-based MARL Coordination (MAGEC) | 2024 | Multi-robot patrolling | GNN + MAPPO; inductive approach for robustness to attrition and communication loss | Yes (likely GitHub) | **Neighborhood-based locality only**; no territorial ownership or familiarity weighting. | Handles agent attrition and partial observability. Scalable. |
| Locality Matters (LOMAQ) | 2021 | Multi-agent cooperative control | Value decomposition using local neighborhoods; each agent's policy is localized | Yes (likely) | **Locality as scalability measure, not territory**; implicitly decomposes space but does not name regions or ownership. | Significant scaling improvement; fundamental for large teams. |
| Cooperative Patrol Routing (arXiv:2501.08020) | 2025 | Urban patrol (graph-based) | VDPPO in decentralized POMDP; emergent partitioning via learned policies | Likely (recent) | **Emergent partitioning**; no explicit territory mechanism. Agents learn who patrols where without territorial semantics. | Reduced idleness; unpredictable routes. Urban/crime-reduction application. |
| Hierarchical RL in Spatial Nav (arXiv:2504.18794) | 2025 | Multi-goal navigation | HRL with subgoals corresponding to spatial landmarks/rooms | Likely (recent) | **Implicit rooms as subgoals**; no familiarity or ownership tracking. Subgoals are static. | Multi-goal scalability. Natural spatial decomposition. |
| AriGraph (Episodic + Semantic Knowledge Graph) | 2025 | Text-based agent navigation | Knowledge graph extended with episodic memory vertices; improve navigation and generalization | Likely (recent) | **Episodic grounding of semantic space**; no physical-world grounding or territorial ownership. | Strong performance on text-based games; shows value of episodic grounding. |
| MoHA (Motivational Cognitive Maps) | 2025 | Resource-dependent navigation | Hippocampus-inspired; integrates motivational and visual signals to modulate cognitive maps | Unreleased (bioRxiv preprint) | **Motivational modulation of spatial maps**; closest to dual-source idea, but motivation is task-reward, not familiarity-based territory. | Novel bio-inspired integration. Potential bridge to territoriality. |
| Memory Maze Benchmark (Pasukonis et al.) | 2023 | Long-horizon 3D maze navigation | Randomized mazes; evaluate long-term memory via object/wall/pose recall | Yes (GitHub jurgisp/memory-maze) | **Perfect test bed for territorial decomposition**, but benchmark itself does not require or encourage territorial framing. | Canonical long-horizon memory task; used for world-model validation. |
| R2I (Recall to Imagine) | 2024 | Memory-intensive embodied tasks | World model + episodic memory retrieval; retrieve and re-imagine past episodes for planning | Yes (GitHub chandar-lab/Recall2Imagine) | **Episodic-based planning**; does not use territorial structure, but shows power of episodic memory for long-horizon tasks. | Superhuman on Memory Maze; state-of-the-art world-model method. |

---

## Summary and Recommendations

### 3 Most Important Findings

1. **No Explicit Territorial Ownership in MARL:** Despite extensive work on spatial partitioning (Voronoi), multi-agent coordination (GNNs, locality-aware policies), and multi-agent world models, no published work explicitly models territory as a first-class spatial unit with dual-source (physical + familiarity) definition and ownership semantics. This is a clear gap.

2. **Familiarity is Signaling, Not Structure:** Visitation frequency is used as an *exploration signal* (RND, ICM, count-based bonuses) or for *post-hoc analysis* (animal trajectories, heatmaps). It is not yet treated as a *structural property* of regions that affects partitioning, ownership transfer, or goal prioritization. Bridging this gap could fundamentally change how agents decompose space.

3. **Cognitive Maps and HRL Provide Natural Foundations:** Recent work on hierarchical RL (subgoals as rooms/corridors), cognitive maps (place cells, grid cells), and episodic memory (ESWM, RoboMemory) shows that agents naturally learn spatial abstractions. A dual-source territorial framework could unify these by making territory an explicit learned unit that combines structural and familiarity dimensions.

### 3 Closest-Neighbor Papers to Read First

1. **MoHA (Motivational Cognitive Maps)** (bioRxiv 2025): The closest published work to dual-source territory. Shows how motivation modulates spatial maps; the natural next step is to replace task motivation with familiarity-based motivation and explore territorial emergence.

2. **RoboMemory** (arXiv:2508.01415): Comprehensive multi-memory architecture with spatial knowledge graphs. Provides the technical foundation for tracking spatial structure + episodic content; extending with territorial ownership is straightforward.

3. **Memory Maze** (ICLR 2023) + **R2I** (ICLR 2024): Perfect testbed for territory-aware world models. The 3D randomized mazes naturally decompose into regions; agents could learn territorial representations (familiar corridors, unexplored rooms) and use them for planning. Starting single-agent here before multi-agent extension is sound.

---

## Conclusion

The literature reveals a mature landscape of multi-agent spatial reasoning, familiarity-driven exploration, and brain-inspired cognitive mapping. However, the *combination* of physical and familiarity-based territory—as a learned, dynamically-assigned spatial unit with explicit ownership semantics—remains largely unexplored. This dual-source framing offers a novel angle for embodied multi-agent RL, particularly when combined with world models (for scalability) and episodic memory (for long-horizon reasoning). The Memory Maze benchmark and recent work on hierarchical RL provide both the task environment and architectural foundations to pursue this direction.

