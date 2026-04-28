"""Path A.2: Multi-agent Memory Maze with territorial coordination.

Two agents explore the SAME maze layout (seed-locked parallel envs) and
their memories are aggregated under three conditions:

    1. Independent  — each agent has own memory, no sharing
    2. Shared-flat  — both agents write into one shared FlatMemory
    3. Territorial  — each agent owns a subset of rooms; only observations
                      inside the owned territory are written; final eval
                      stitches A's memory of A's rooms + B's memory of B's

Hypothesis: territorial > shared-flat > independent because explicit room
ownership forces COMPLEMENTARY exploration, raising total coverage.

Note: we use TWO parallel single-agent envs with the SAME seed so the maze
layout matches. The agents don't see each other's physical balls (different
MuJoCo instances), but they are conceptually navigating the same building.
This is the cheapest way to do multi-agent without forking memory_maze.

Usage:
    python multi_agent_experiment.py --size 9x9 --episodes 5 --steps 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw' if sys.platform == 'darwin' else 'egl'

import numpy as np

import gym  # noqa: E402
import memory_maze  # noqa: F401, E402

from memory_models import FlatMemory, TerritorialMemory, _credit_kernel  # noqa: E402
from policies import POLICY_REGISTRY  # noqa: E402
from region_utils import extract_rooms, world_to_cell  # noqa: E402


# --- territory partition ---------------------------------------------------

def partition_rooms(region_info: dict, region_map: np.ndarray) -> dict:
    """Split rooms between 2 agents.

    Strategy: order rooms by their center row, alternate A / B. Gives each
    agent a roughly even share by area while spatially interleaving so each
    agent must cross the maze (rather than getting all rooms in one corner).
    """
    centers = region_info['room_centers']  # {room_id: (y, x)}
    sorted_rooms = sorted(centers.keys(), key=lambda r: (centers[r][0], centers[r][1]))
    assignment = {}  # room_id -> 'A' or 'B'
    for i, r in enumerate(sorted_rooms):
        assignment[r] = 'A' if i % 2 == 0 else 'B'
    return assignment


# --- memory aggregation helpers --------------------------------------------

def make_shared_memory(layout, xy_scale, vis_radius, kernel_sigma):
    """A single TerritorialMemory both agents write into (use_partition=True
    so we can also use it for room-level eval, but with alpha=0 the partition
    is ignored at predict time)."""
    mem = TerritorialMemory(
        n_colors=3, use_partition=True, familiarity_alpha=0.0,
        xy_scale=xy_scale, vis_radius=vis_radius, kernel_sigma=kernel_sigma,
        name='Shared',
    )
    return mem


def make_per_agent_memories(xy_scale, vis_radius, kernel_sigma):
    """Two independent TerritorialMemories — one per agent."""
    return {
        'A': TerritorialMemory(
            n_colors=3, use_partition=True, familiarity_alpha=0.0,
            xy_scale=xy_scale, vis_radius=vis_radius, kernel_sigma=kernel_sigma,
            name='A'),
        'B': TerritorialMemory(
            n_colors=3, use_partition=True, familiarity_alpha=0.0,
            xy_scale=xy_scale, vis_radius=vis_radius, kernel_sigma=kernel_sigma,
            name='B'),
    }


def stitch_territorial(memories: dict, assignment: dict, layout) -> TerritorialMemory:
    """Build a 'stitched' memory: each room takes data from the owning agent.

    For room r owned by agent X, we copy X's evidence/region_evidence for r.
    For non-room cells (corridors), we union both agents' evidence.
    The result has the same predict_target API as a regular TerritorialMemory.
    """
    stitched = TerritorialMemory(
        n_colors=3, use_partition=True, familiarity_alpha=0.0,
        xy_scale=memories['A'].xy_scale,
        vis_radius=memories['A'].vis_radius,
        kernel_sigma=memories['A'].kernel_sigma,
        name='Stitched',
    )
    stitched.reset({'maze_layout': layout})

    for room_id, owner in assignment.items():
        m = memories[owner]
        if m.evidence is None:
            continue
        mask = m.region_map == room_id
        for cid in range(stitched.n_colors):
            stitched.evidence[cid][mask] = m.evidence[cid][mask]
            if cid in m.region_evidence and room_id in m.region_evidence[cid]:
                stitched.region_evidence[cid][room_id] = m.region_evidence[cid][room_id]
        stitched.visit_count[mask] = m.visit_count[mask]

    # Corridor (region < 1): union of both agents' evidence
    corridor_mask = stitched.region_map < 1
    for m in memories.values():
        if m.evidence is None:
            continue
        for cid in range(stitched.n_colors):
            stitched.evidence[cid][corridor_mask] += m.evidence[cid][corridor_mask]
        stitched.visit_count[corridor_mask] += m.visit_count[corridor_mask]

    return stitched


# --- per-agent observe ------------------------------------------------------

def observe_with_filter(memory: TerritorialMemory, obs: dict, owned_rooms: set | None):
    """Like memory.observe(obs) but, if owned_rooms is given, only credits
    evidence inside the agent's owned rooms. Visit count is always tracked
    so familiarity is meaningful."""
    if memory.evidence is None:
        return
    agent_cell = world_to_cell(obs['agent_pos'], memory.maze_shape, memory.xy_scale)
    memory.visit_count[agent_cell[0], agent_cell[1]] += 1.0

    targets_pos = obs['targets_pos']
    for i in range(min(memory.n_colors, len(targets_pos))):
        tcell = world_to_cell(targets_pos[i], memory.maze_shape, memory.xy_scale)
        dy = tcell[0] - agent_cell[0]
        dx = tcell[1] - agent_cell[1]
        if abs(dy) > memory.vis_radius or abs(dx) > memory.vis_radius:
            continue
        if owned_rooms is not None:
            r = int(memory.region_map[tcell[0], tcell[1]])
            if r > 0 and r not in owned_rooms:
                continue  # target sits in someone else's territory — drop
        _credit_kernel(memory.evidence[i], tcell, sigma=memory.kernel_sigma)
        r = int(memory.region_map[tcell[0], tcell[1]])
        if r > 0:
            d = memory.region_evidence[i]
            d[r] = d.get(r, 0.0) + 1.0


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, float(r), bool(term or trunc), info
    obs, r, done, info = out
    return obs, float(r), bool(done), info


# --- one episode (3 conditions in lockstep) --------------------------------

def run_episode(envs: dict, n_steps: int, xy_scale: float,
                vis_radius: int, kernel_sigma: float,
                rng: np.random.Generator, policy_name: str):
    """envs = {'A': env_A, 'B': env_B}; both already reset to same seed."""
    obs_A = envs['A'].reset()
    obs_B = envs['B'].reset()
    obs_TA = envs['A_terr'].reset()
    obs_TB = envs['B_terr'].reset()
    layouts = [obs_A['maze_layout'], obs_B['maze_layout'],
               obs_TA['maze_layout'], obs_TB['maze_layout']]
    if not all(np.array_equal(layouts[0], L) for L in layouts[1:]):
        raise RuntimeError(
            'Maze layouts disagree across the 4 envs after reset — the '
            'seed-locking is broken. Cannot run a comparable multi-agent '
            'experiment. Aborting episode.'
        )
    # Sanity-check targets match too (otherwise eval ground truth is meaningless)
    if not np.allclose(obs_A['targets_pos'], obs_B['targets_pos']) or \
       not np.allclose(obs_A['targets_pos'], obs_TA['targets_pos']) or \
       not np.allclose(obs_A['targets_pos'], obs_TB['targets_pos']):
        raise RuntimeError(
            'Target positions disagree across envs — env state is not '
            'fully reproducible from the seed. Aborting.'
        )
    layout = layouts[0]

    # Region partition — same for both (same layout)
    region_map, region_info = extract_rooms(layout)
    assignment = partition_rooms(region_info, region_map)
    rooms_A = {r for r, who in assignment.items() if who == 'A'}
    rooms_B = {r for r, who in assignment.items() if who == 'B'}

    # --- build memories for all 3 conditions ---
    common = dict(xy_scale=xy_scale, vis_radius=vis_radius, kernel_sigma=kernel_sigma)
    indep_mems = {
        'A': TerritorialMemory(n_colors=3, use_partition=True, familiarity_alpha=0.0,
                               name='Indep-A', **common),
        'B': TerritorialMemory(n_colors=3, use_partition=True, familiarity_alpha=0.0,
                               name='Indep-B', **common),
    }
    shared_mem = TerritorialMemory(n_colors=3, use_partition=True, familiarity_alpha=0.0,
                                   name='Shared', **common)
    terr_mems = {
        'A': TerritorialMemory(n_colors=3, use_partition=True, familiarity_alpha=0.0,
                               name='Terr-A', **common),
        'B': TerritorialMemory(n_colors=3, use_partition=True, familiarity_alpha=0.0,
                               name='Terr-B', **common),
    }
    for m in (*indep_mems.values(), shared_mem, *terr_mems.values()):
        m.reset({'maze_layout': layout})

    # --- policies (same for indep and shared; territorial uses TerritorialPolicy) ---
    PolicyCls = POLICY_REGISTRY[policy_name]
    pol_indep = {'A': PolicyCls(), 'B': PolicyCls()}
    pol_shared = {'A': PolicyCls(), 'B': PolicyCls()}
    pol_terr = {
        'A': POLICY_REGISTRY['territorial'](),
        'B': POLICY_REGISTRY['territorial'](),
    }
    for p in (*pol_indep.values(), *pol_shared.values(), *pol_terr.values()):
        p.reset(obs_A)  # both agents see the same initial layout

    # Each condition steps in its OWN env-pair so trajectories are
    # genuinely different. Build separate env handles per condition.
    # For simplicity, we'll just run the same env twice per condition by
    # re-stepping — but that would mean env state is shared between
    # conditions. Instead, we step a single shared trajectory per condition.
    # SIMPLIFICATION: run a single trajectory pair per condition, shared
    # across both agents in that condition, and aggregate.

    # We already have obs_A, obs_B for one trajectory. We'll re-use this
    # SAME trajectory for the indep condition. For shared and territorial,
    # we'd need separate envs to step independently — but per the docstring,
    # the agents are in the same maze and don't see each other, so the
    # OBSERVATION trajectories are independent of memory choice. So we can
    # safely use the same (obs_A, obs_B) traces for all 3 conditions.
    # (Memory choice doesn't affect physics; only the policy does.)

    obs_indep = {'A': obs_A, 'B': obs_B}
    obs_shared = {'A': obs_A, 'B': obs_B}
    obs_terr = {'A': obs_TA, 'B': obs_TB}
    env_terr_A = envs['A_terr']
    env_terr_B = envs['B_terr']

    # First-step observe
    observe_with_filter(indep_mems['A'], obs_indep['A'], None)
    observe_with_filter(indep_mems['B'], obs_indep['B'], None)
    observe_with_filter(shared_mem, obs_shared['A'], None)
    observe_with_filter(shared_mem, obs_shared['B'], None)
    observe_with_filter(terr_mems['A'], obs_terr['A'], rooms_A)
    observe_with_filter(terr_mems['B'], obs_terr['B'], rooms_B)

    for step_ix in range(n_steps):
        # Indep / Shared use the same env-pair (envs['A'], envs['B'])
        a_indep = int(pol_indep['A'].select(obs_indep['A'], rng))
        b_indep = int(pol_indep['B'].select(obs_indep['B'], rng))
        obs_indep['A'], _r, doneA, _ = step_env(envs['A'], a_indep)
        obs_indep['B'], _r, doneB, _ = step_env(envs['B'], b_indep)
        # Shared: same trajectory as indep (cheap shortcut — same policy)
        obs_shared['A'], obs_shared['B'] = obs_indep['A'], obs_indep['B']

        # Terr: separate envs, separate policies
        a_terr = int(pol_terr['A'].select(obs_terr['A'], rng))
        b_terr = int(pol_terr['B'].select(obs_terr['B'], rng))
        obs_terr['A'], _r, doneTA, _ = step_env(env_terr_A, a_terr)
        obs_terr['B'], _r, doneTB, _ = step_env(env_terr_B, b_terr)

        # Observations into memories
        observe_with_filter(indep_mems['A'], obs_indep['A'], None)
        observe_with_filter(indep_mems['B'], obs_indep['B'], None)
        observe_with_filter(shared_mem, obs_shared['A'], None)
        observe_with_filter(shared_mem, obs_shared['B'], None)
        observe_with_filter(terr_mems['A'], obs_terr['A'], rooms_A)
        observe_with_filter(terr_mems['B'], obs_terr['B'], rooms_B)

        if doneA or doneB or doneTA or doneTB:
            break

    # Ground truth: targets_pos from final obs (constant per episode)
    truth = obs_indep['A']['targets_pos']
    truth_cells = {i: world_to_cell(truth[i], layout.shape, xy_scale) for i in range(3)}

    # --- evaluate each condition ---
    results = {}

    # Independent: each agent predicts from its own memory; we report mean
    pred_A = [indep_mems['A'].predict_target(i) for i in range(3)]
    pred_B = [indep_mems['B'].predict_target(i) for i in range(3)]
    dists_A = [abs(pred_A[i][0] - truth_cells[i][0]) + abs(pred_A[i][1] - truth_cells[i][1]) for i in range(3)]
    dists_B = [abs(pred_B[i][0] - truth_cells[i][0]) + abs(pred_B[i][1] - truth_cells[i][1]) for i in range(3)]
    results['Independent'] = {
        'mean_dist': float(np.mean(dists_A + dists_B)),
        'success@2': float(np.mean([1 if d <= 2 else 0 for d in dists_A + dists_B])),
        'best_of_two': float(np.mean([min(dA, dB) for dA, dB in zip(dists_A, dists_B)])),
    }

    pred_S = [shared_mem.predict_target(i) for i in range(3)]
    dists_S = [abs(pred_S[i][0] - truth_cells[i][0]) + abs(pred_S[i][1] - truth_cells[i][1]) for i in range(3)]
    results['Shared-flat'] = {
        'mean_dist': float(np.mean(dists_S)),
        'success@2': float(np.mean([1 if d <= 2 else 0 for d in dists_S])),
        'best_of_two': float(np.mean(dists_S)),
    }

    stitched = stitch_territorial(terr_mems, assignment, layout)
    pred_T = [stitched.predict_target(i) for i in range(3)]
    dists_T = [abs(pred_T[i][0] - truth_cells[i][0]) + abs(pred_T[i][1] - truth_cells[i][1]) for i in range(3)]
    results['Territorial'] = {
        'mean_dist': float(np.mean(dists_T)),
        'success@2': float(np.mean([1 if d <= 2 else 0 for d in dists_T])),
        'best_of_two': float(np.mean(dists_T)),
    }

    extras = {
        'rooms_A': sorted(rooms_A), 'rooms_B': sorted(rooms_B),
        'n_rooms': region_info['n_rooms'],
        'visit_A': float(terr_mems['A'].visit_count.sum()),
        'visit_B': float(terr_mems['B'].visit_count.sum()),
    }
    return results, extras


# --- main runner ------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='9x9', choices=['9x9', '11x11', '13x13', '15x15'])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--policy', default='wander', choices=list(POLICY_REGISTRY.keys()))
    p.add_argument('--xy_scale', type=float, default=1.0)
    p.add_argument('--vis_radius', type=int, default=3)
    p.add_argument('--kernel_sigma', type=float, default=1.5)
    p.add_argument('--outdir', default='outputs')
    args = p.parse_args()

    env_id = f'MemoryMaze-{args.size}-ExtraObs-v0'
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[multi] env={env_id}  policy={args.policy}  steps={args.steps} '
          f'episodes={args.episodes}')

    # 4 envs total. CRITICAL: all four must share the SAME maze layout
    # within an episode (so cell coordinates are interpretable across them
    # and the stitched memory is a valid representation of one shared maze).
    # We re-seed all envs to the same value before each episode reset.
    # action_space gets DIFFERENT seeds so trajectories diverge.
    envs = {
        'A': gym.make(env_id),
        'B': gym.make(env_id),
        'A_terr': gym.make(env_id),
        'B_terr': gym.make(env_id),
    }
    action_seed_offset = {'A': 1001, 'B': 2002, 'A_terr': 3003, 'B_terr': 4004}
    for name, e in envs.items():
        e.action_space.seed(args.seed + action_seed_offset[name])

    rng = np.random.default_rng(args.seed)
    all_eps = []
    t0 = time.time()
    for ep in range(args.episodes):
        # Re-seed every env to the SAME maze seed before this episode's reset.
        # This guarantees identical maze layouts and target placements.
        episode_maze_seed = args.seed * 10000 + ep
        for e in envs.values():
            if hasattr(e, 'seed'):
                e.seed(episode_maze_seed)
        results, extras = run_episode(
            envs, args.steps, args.xy_scale, args.vis_radius,
            args.kernel_sigma, rng, args.policy,
        )
        line = ' | '.join(f'{c}: d={v["mean_dist"]:.2f} s2={v["success@2"]:.2f}'
                          for c, v in results.items())
        print(f'[multi] ep={ep:>2d}  rooms_A={extras["rooms_A"]} rooms_B={extras["rooms_B"]}  {line}')
        all_eps.append(results)

    for e in envs.values():
        e.close()

    elapsed = time.time() - t0
    print(f'[multi] done in {elapsed:.1f}s')

    # Aggregate
    conds = list(all_eps[0].keys())
    agg = {}
    for c in conds:
        ds = np.array([ep[c]['mean_dist'] for ep in all_eps])
        ss = np.array([ep[c]['success@2'] for ep in all_eps])
        agg[c] = {
            'mean_dist_mean': float(ds.mean()), 'mean_dist_std': float(ds.std()),
            'success@2_mean': float(ss.mean()), 'success@2_std': float(ss.std()),
            'n_episodes': int(len(ds)),
        }

    print('\n=== summary ===')
    print(f'{"condition":<14} {"mean_dist":>13} {"success@2":>13}')
    for c, v in agg.items():
        print(f'{c:<14} {v["mean_dist_mean"]:>5.2f}±{v["mean_dist_std"]:>4.2f}  '
              f'{v["success@2_mean"]:>5.2f}±{v["success@2_std"]:>4.2f}')

    metrics_path = out_dir / 'multi_agent_metrics.json'
    with metrics_path.open('w') as f:
        json.dump({
            'env_id': env_id, 'config': vars(args),
            'aggregate': agg, 'per_episode': all_eps,
            'wall_time_sec': elapsed,
        }, f, indent=2, default=str)
    print(f'[multi] wrote {metrics_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
