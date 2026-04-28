# Habitat ObjectNav Migration

This folder adapts the territorial / neural-circuit navigation prototype to a
standard Habitat ObjectNav style benchmark.

## Expected prerequisites

- `habitat-lab`
- `habitat-baselines`
- `habitat-sim`
- HM3D scenes
- HM3D ObjectNav episodes

## Main files

- `policy.py`: top-level orchestrator for the territorial navigation stack
- `perception.py`: territorial perception and signature extraction
- `belief_memory.py`: territory prototypes, revisit tracking, and progress memory
- `territorial_graph.py`: region-level context interface
- `subgoal_planner.py`: high-level action scoring / future doorway planning hook
- `local_controller.py`: low-level corrective controller
- `stop_verifier.py`: target stop decision logic
- `eval_habitat_objectnav.py`: evaluation entrypoint
- `config_template.md`: expected dataset and config paths

## Intended workflow

1. Install Habitat dependencies and datasets.
2. Point the evaluator to a Habitat ObjectNav config.
3. Run evaluation for:
   - a simple flat baseline
   - the territorial baseline
   - the neural-circuit policy

This stack is still lightweight, but it is now intentionally split into
territorial modules so later upgrades can target:

- explicit territory graphs
- goal-conditioned belief updates
- multi-agent territory assignment
- boundary negotiation and handoff
