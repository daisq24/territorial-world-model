"""Memory Maze sanity probe + xy_scale calibration.

Run this first to confirm:

    1. memory-maze is installed and the env can be created
    2. ExtraObs wrapper exposes the keys we expect
    3. agent_pos can be converted to maze cells (calibrates xy_scale)
    4. rendering works (saves probe_output.png)

Usage:
    python env_probe.py [--size 9x9] [--steps 200] [--seed 0] [--top]

Exit codes:
    0  = everything works
    1  = import failed
    2  = env creation / reset failed (likely MUJOCO_GL issue)
    3  = obs schema mismatch (Memory Maze API changed)
    4  = agent_pos calibration looks wrong
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def _set_mujoco_gl_default():
    if 'MUJOCO_GL' in os.environ:
        return os.environ['MUJOCO_GL']
    # macOS → glfw (no EGL), Linux → egl (headless GPU)
    os.environ['MUJOCO_GL'] = 'glfw' if sys.platform == 'darwin' else 'egl'
    return os.environ['MUJOCO_GL']


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='9x9', choices=['9x9', '11x11', '13x13', '15x15'])
    p.add_argument('--steps', type=int, default=200,
                   help='Random steps used for xy_scale calibration')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--top', action='store_true',
                   help='Use the -Top top-down rendering variant')
    p.add_argument('--out', default='outputs/probe_output.png')
    args = p.parse_args()

    mujoco_gl = _set_mujoco_gl_default()
    print(f'[probe] MUJOCO_GL={mujoco_gl}')

    # --- imports ---
    try:
        import gym  # noqa: F401  memory-maze uses OpenAI gym, NOT gymnasium
        import numpy as np
        import memory_maze  # noqa: F401  triggers env registration
    except ImportError as e:
        print(f'[probe] FAIL: import error — {e}')
        print('[probe] Try: pip install memory-maze gym numpy pillow matplotlib scipy')
        return 1

    # --- env creation ---
    suffix = '-Top' if args.top else ''
    env_id = f'memory_maze:MemoryMaze-{args.size}-ExtraObs{suffix}-v0'
    print(f'[probe] creating {env_id} ...')
    try:
        env = gym.make(env_id)
        env.action_space.seed(args.seed)
        if hasattr(env, 'seed'):
            env.seed(args.seed)
        obs = env.reset()
    except Exception as e:
        print(f'[probe] FAIL: env init crashed — {e}')
        traceback.print_exc()
        if 'EGL' in str(e).upper() or 'GLFW' in str(e).upper() or 'MUJOCO' in str(e).upper():
            print('[probe] HINT: try MUJOCO_GL=glfw (mac/desktop) or MUJOCO_GL=osmesa (CPU headless)')
        return 2

    # --- obs schema ---
    print('\n[probe] obs keys:')
    expected = {'image', 'agent_pos', 'maze_layout',
                'target_color', 'target_pos', 'targets_pos'}
    if not isinstance(obs, dict):
        print('[probe] FAIL: obs is not a dict — wrong env variant?')
        return 3
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            print(f'  {k:<14} shape={v.shape}  dtype={v.dtype}')
        else:
            print(f'  {k:<14} {type(v).__name__}={v}')
    missing = expected - set(obs.keys())
    if missing:
        print(f'[probe] WARNING: missing expected keys {missing}')

    layout = obs['maze_layout']
    print(f'\n[probe] action_space: {env.action_space}')
    print(f'[probe] maze_layout shape={layout.shape}, free cells={int(layout.sum())}/{layout.size}')
    print(f'[probe] agent_pos initial: {obs["agent_pos"]}')
    if 'targets_pos' in obs:
        print(f'[probe] targets_pos initial:\n{obs["targets_pos"]}')

    # --- xy_scale calibration ---
    # agent_pos is in MuJoCo world coordinates. We try a few candidate scales
    # and pick the one where the rounded cell is most often walkable.
    from region_utils import world_to_cell

    candidate_scales = [1.0, 2.0, 0.5, 4.0, 0.25]
    walkable_hits = {s: 0 for s in candidate_scales}
    visited_cells = {s: set() for s in candidate_scales}
    n_steps = 0

    print(f'\n[probe] running {args.steps}-step random rollout to calibrate xy_scale ...')
    total_reward = 0.0
    for _ in range(args.steps):
        a = env.action_space.sample()
        step_out = env.step(a)
        if len(step_out) == 5:
            obs, reward, term, trunc, _info = step_out
            done = term or trunc
        else:
            obs, reward, done, _info = step_out
        total_reward += float(reward)
        n_steps += 1
        for s in candidate_scales:
            cell = world_to_cell(obs['agent_pos'], layout.shape, s)
            visited_cells[s].add(cell)
            if layout[cell[0], cell[1]] == 1:
                walkable_hits[s] += 1
        if done:
            obs = env.reset()
            layout = obs['maze_layout']

    # --- pick best scale ---
    # The right scale must satisfy ALL of:
    #   (a) walkable% high (rounded cells land on walkable maze cells)
    #   (b) cells are NOT all clamped to the maze boundary (otherwise
    #       agent_pos / scale > maze size and clamp gives a false positive)
    #   (c) prefer the scale whose implied cell range matches the maze size,
    #       i.e. max(agent_pos)/scale is close to maze_shape - 1.
    H, W = layout.shape
    print('\n[probe] xy_scale calibration:')
    print(f'  (maze is {H}x{W}; want max(agent_pos)/scale ≈ {H-1})')
    print('  scale  walkable%  unique  boundary%  range_fit')
    best_scale = None
    best_score = -1.0

    # Recompute max abs of agent_pos seen — needed to detect overflow into clamp.
    # We don't have the full trajectory; approximate from visited_cells * scale.
    for s in candidate_scales:
        pct = 100.0 * walkable_hits[s] / max(1, n_steps)
        cells = visited_cells[s]
        n_unique = len(cells)
        # boundary fraction: how many visited cells are on the maze edge
        n_boundary = sum(1 for (i, j) in cells
                         if i == 0 or i == H - 1 or j == 0 or j == W - 1)
        boundary_pct = 100.0 * n_boundary / max(1, n_unique)
        # max cell index visited
        max_i = max((i for i, j in cells), default=0)
        max_j = max((j for i, j in cells), default=0)
        range_fit = (max_i + max_j) / max(1, (H - 1) + (W - 1))  # ≤1 normally
        # Penalize scales where >80% of cells are on the boundary (overflow signal).
        valid = boundary_pct < 80.0
        # Composite: walkable% (main) + range_fit bonus + variety, hard veto on overflow.
        composite = (pct if valid else 0.0) + 30.0 * min(range_fit, 1.0) + 1.0 * min(n_unique, 10)
        if composite > best_score:
            best_score = composite
            best_scale = s
        flag = ' (overflow!)' if not valid else ''
        print(f'  {s:<6} {pct:>7.1f}%  {n_unique:>4d}  {boundary_pct:>7.1f}%  {range_fit:>5.2f}{flag}')

    print(f'  best  → xy_scale = {best_scale}')
    walkable_pct_best = 100.0 * walkable_hits[best_scale] / max(1, n_steps)

    print(f'\n[probe] random rollout: total_reward={total_reward:.1f} (target hits)')

    # --- save first frame ---
    try:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from region_utils import extract_rooms

        region_map, info = extract_rooms(layout)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(obs['image'])
        axes[0].set_title('first-person')
        axes[0].axis('off')
        axes[1].imshow(layout, cmap='gray_r')
        axes[1].set_title(f'maze_layout {layout.shape}')
        axes[1].axis('off')
        axes[2].imshow(region_map, cmap='tab20')
        axes[2].set_title(f'rooms detected: {info["n_rooms"]}')
        axes[2].axis('off')
        plt.suptitle(f'env_probe: {env_id}  xy_scale={best_scale}  walkable={walkable_pct_best:.0f}%')
        plt.tight_layout()
        plt.savefig(args.out, dpi=100, bbox_inches='tight')
        print(f'[probe] wrote {args.out}')
    except Exception as e:
        print(f'[probe] WARNING: figure save failed — {e}')

    env.close()

    if walkable_pct_best < 50:
        print('\n[probe] FAIL: best xy_scale only hits walkable cells {:.0f}% — '
              'world_to_cell needs work (likely needs an offset).'.format(walkable_pct_best))
        return 4

    print(f'\n[probe] OK ✓  use xy_scale={best_scale} in experiment.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
