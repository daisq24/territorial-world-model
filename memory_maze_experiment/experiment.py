"""Path A: FlatMemory vs TerritorialMemory on Memory Maze.

Offline probing evaluation:
    - Run a fixed exploration policy for K steps.
    - During the rollout, every memory model observes obs.
    - At evaluation time, query each memory: "where is the target with color C?"
      Compare prediction (a maze cell) to ground-truth target_pos.

Memories compared (via --conditions, default = all four):

    Flat                       baseline: per-cell color counts
    Territorial(partition)     rooms only, no familiarity weighting
    Territorial(familiarity)   visit-weighted flat, no rooms
    Territorial(dual)          rooms × familiarity — the full proposal

Usage:
    # quick smoke test
    python experiment.py --size 9x9 --episodes 3 --steps 200 --seeds 1
    # full main experiment
    python experiment.py --size 9x9 --episodes 20 --seeds 5 --steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Set MUJOCO_GL before any mujoco-touching import.
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw' if sys.platform == 'darwin' else 'egl'

import gym  # noqa: E402  -- memory-maze uses OpenAI gym, NOT gymnasium
import memory_maze  # noqa: F401, E402

from memory_models import FlatMemory, TerritorialMemory  # noqa: E402
from policies import POLICY_REGISTRY  # noqa: E402
from region_utils import world_to_cell  # noqa: E402


# --- ground-truth probing ---------------------------------------------------

class GroundTruth:
    """Ground truth via TARGETS_AS_ROW_INDEX:

    `targets_pos: (3, 2)` is constant for the whole episode and contains the
    world positions of all 3 targets. We use row index (0, 1, 2) as the
    canonical color id. This bypasses the "agent must reach a target before
    target_color changes" issue in Memory Maze and gives us 3 ground-truth
    (id, position) pairs every episode regardless of agent performance.
    """

    def __init__(self, n_colors: int = 3):
        self.n_colors = n_colors
        self.pos: dict[int, tuple[float, float]] = {}

    def set_from_targets(self, targets_pos):
        for i in range(min(self.n_colors, len(targets_pos))):
            self.pos[i] = (float(targets_pos[i][0]), float(targets_pos[i][1]))


# --- memory factory ---------------------------------------------------------

def build_memories(condition_names: list[str], xy_scale: float,
                   miss_rate: float = 0.0, false_positive_rate: float = 0.0,
                   kernel_sigma: float = 1.0, vis_radius: int = 3,
                   noise_seed: int = 0) -> dict:
    """Build a dict of {name: memory} for the requested ablation conditions.

    All conditions share the same observation noise (miss + false_positive),
    so the comparison is fair: only the prediction logic differs between them.
    Use the same noise_seed across builds to keep noise sequences aligned.
    """
    common = dict(
        xy_scale=xy_scale, vis_radius=vis_radius, kernel_sigma=kernel_sigma,
        miss_rate=miss_rate, false_positive_rate=false_positive_rate,
        noise_seed=noise_seed,
    )
    out = {}
    for name in condition_names:
        if name == 'flat':
            out['Flat'] = FlatMemory(n_colors=3, **common)
        elif name == 'partition':
            out['Territorial(partition)'] = TerritorialMemory(
                n_colors=3, use_partition=True, familiarity_alpha=0.0,
                name='Territorial(partition)', **common)
        elif name == 'familiarity':
            out['Territorial(familiarity)'] = TerritorialMemory(
                n_colors=3, use_partition=False, familiarity_alpha=0.5,
                name='Territorial(familiarity)', **common)
        elif name == 'dual':
            out['Territorial(dual)'] = TerritorialMemory(
                n_colors=3, use_partition=True, familiarity_alpha=0.5,
                name='Territorial(dual)', **common)
        else:
            raise ValueError(f'unknown condition: {name}')
    return out


# --- one episode ------------------------------------------------------------

def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, float(r), bool(term or trunc), info
    obs, r, done, info = out
    return obs, float(r), bool(done), info


def run_episode(env, memories: dict, policy, n_steps: int,
                rng: np.random.Generator, xy_scale: float,
                record_trajectory: bool = False, record_frames: bool = False):
    obs = env.reset()
    layout = obs['maze_layout']
    truth = GroundTruth(n_colors=3)
    truth.set_from_targets(obs['targets_pos'])

    for m in memories.values():
        m.reset(obs)
        m.observe(obs)
    policy.reset(obs)

    trajectory = [world_to_cell(obs['agent_pos'], layout.shape, xy_scale)] if record_trajectory else None
    frames = [obs['image']] if (record_frames and 'image' in obs) else None

    for step_ix in range(n_steps):
        action = int(policy.select(obs, rng))
        obs, r, done, _info = _step_env(env, action)
        for m in memories.values():
            m.observe(obs)
        # targets_pos is constant per episode, no need to update truth
        if trajectory is not None:
            trajectory.append(world_to_cell(obs['agent_pos'], layout.shape, xy_scale))
        if frames is not None and step_ix % 3 == 0 and 'image' in obs:
            frames.append(obs['image'])
        if done:
            break

    # --- evaluation ---
    truth_cells: dict[int, tuple[int, int]] = {
        cid: world_to_cell(np.asarray(pos), layout.shape, xy_scale)
        for cid, pos in truth.pos.items()
    }

    per_condition = {}
    for name, m in memories.items():
        dists, succ2, succ4 = [], [], []
        per_color = []
        for cid in sorted(truth_cells.keys()):
            true_cell = truth_cells[cid]
            # color_index passes through ints; we use row index as the id
            pred = m.predict_target(cid)
            d = abs(pred[0] - true_cell[0]) + abs(pred[1] - true_cell[1])
            dists.append(d)
            succ2.append(1 if d <= 2 else 0)
            succ4.append(1 if d <= 4 else 0)
            per_color.append({'color_id': cid, 'pred': list(pred),
                              'truth': list(true_cell), 'manhattan_dist': int(d)})
        per_condition[name] = {
            'mean_dist': float(np.mean(dists)) if dists else float('nan'),
            'success@2': float(np.mean(succ2)) if succ2 else float('nan'),
            'success@4': float(np.mean(succ4)) if succ4 else float('nan'),
            'per_color': per_color,
        }

    extra = {
        'n_colors_seen': len(truth_cells),
        'final_layout': layout,
    }
    # Pull region_map / visit_count from first available territorial memory
    for m in memories.values():
        if isinstance(m, TerritorialMemory):
            extra['region_map'] = m.region_map
            extra['visit_count'] = m.visit_count.copy() if m.visit_count is not None else None
            extra['_mem_for_viz'] = m
            break
    if record_trajectory:
        extra['_trajectory'] = np.array(trajectory)
        extra['_final_targets_pos'] = obs['targets_pos'] if 'targets_pos' in obs else None
        extra['_maze_layout'] = layout
    if record_frames:
        extra['_frames'] = frames
    return per_condition, extra


# --- aggregation + figures --------------------------------------------------

def aggregate(per_episode: list[dict]) -> dict:
    if not per_episode:
        return {}
    conds = list(per_episode[0].keys())
    agg = {}
    for c in conds:
        ds = np.array([ep[c]['mean_dist'] for ep in per_episode if not np.isnan(ep[c]['mean_dist'])])
        s2 = np.array([ep[c]['success@2'] for ep in per_episode if not np.isnan(ep[c]['success@2'])])
        s4 = np.array([ep[c]['success@4'] for ep in per_episode if not np.isnan(ep[c]['success@4'])])
        agg[c] = {
            'mean_dist_mean': float(ds.mean()) if len(ds) else float('nan'),
            'mean_dist_std':  float(ds.std()) if len(ds) else float('nan'),
            'success@2_mean': float(s2.mean()) if len(s2) else float('nan'),
            'success@2_std':  float(s2.std()) if len(s2) else float('nan'),
            'success@4_mean': float(s4.mean()) if len(s4) else float('nan'),
            'success@4_std':  float(s4.std()) if len(s4) else float('nan'),
            'n_episodes': int(len(ds)),
        }
    return agg


def save_summary_figure(out_dir: Path, agg: dict, env_id: str, policy_name: str):
    import matplotlib.pyplot as plt
    conds = list(agg.keys())
    means = [agg[c]['success@2_mean'] for c in conds]
    stds = [agg[c]['success@2_std'] for c in conds]
    palette = ['#888888', '#3a86ff', '#f2c14e', '#e63946']
    colors = [palette[i % len(palette)] for i in range(len(conds))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(conds, means, yerr=stds, capsize=5, color=colors)
    axes[0].set_ylabel('success @ d≤2 cells')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title('Target localization success')
    for b, v in zip(bars, means):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    means_d = [agg[c]['mean_dist_mean'] for c in conds]
    stds_d = [agg[c]['mean_dist_std'] for c in conds]
    axes[1].bar(conds, means_d, yerr=stds_d, capsize=5, color=colors)
    axes[1].set_ylabel('Manhattan distance (lower better)')
    axes[1].set_title('Mean distance to true target')
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
    plt.suptitle(f'{env_id}  |  policy={policy_name}')
    plt.tight_layout()
    path = out_dir / 'comparison.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def save_episode_figure(out_dir: Path, ep_global: int, extra: dict):
    if 'region_map' not in extra or 'visit_count' not in extra:
        return None
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(extra['final_layout'], cmap='gray_r')
    axes[0].set_title('maze_layout')
    axes[0].axis('off')
    axes[1].imshow(extra['region_map'], cmap='tab20')
    axes[1].set_title('rooms (TerritorialMemory)')
    axes[1].axis('off')
    if extra.get('visit_count') is not None:
        axes[2].imshow(extra['visit_count'], cmap='hot')
        axes[2].set_title('visit_count (familiarity)')
    axes[2].axis('off')
    plt.suptitle(f'episode {ep_global}: {extra["n_colors_seen"]} colors observed')
    plt.tight_layout()
    path = out_dir / f'familiarity_map_ep{ep_global:02d}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path


def save_per_episode_viz(out_dir: Path, ep_global: int, extra: dict):
    """Optional: rich per-episode visualization via visualize.py module."""
    try:
        from visualize import plot_analysis, save_gif  # noqa: F401
    except ImportError:
        return None
    if '_mem_for_viz' not in extra or '_trajectory' not in extra:
        return None
    try:
        from visualize import plot_analysis, save_gif
        m = extra['_mem_for_viz']
        data = {
            'maze_layout': extra['_maze_layout'],
            'region_map': m.region_map,
            'region_info': m.region_info,
            'trajectory': extra['_trajectory'],
            'visit_count': m.visit_count,
            'evidence_territorial': m.evidence,
            'evidence_flat': None,
            'targets_pos': extra.get('_final_targets_pos'),
            'memory_territorial': m,
            'memory_flat': None,
            'frames': extra.get('_frames', []),
        }
        plot_analysis(data, out_dir / f'viz_ep{ep_global:02d}_analysis.png')
        if extra.get('_frames'):
            save_gif(extra['_frames'], out_dir / f'viz_ep{ep_global:02d}_episode.gif', fps=10)
    except Exception as e:
        print(f'(viz skipped for ep {ep_global}: {e})')


# --- main runner ------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='9x9', choices=['9x9', '11x11', '13x13', '15x15'])
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--policy', default='wander', choices=list(POLICY_REGISTRY.keys()))
    p.add_argument('--xy_scale', type=float, default=1.0,
                   help='From env_probe.py calibration')
    p.add_argument('--miss_rate', type=float, default=0.3,
                   help='Per-step prob of missing a target observation')
    p.add_argument('--fp_rate', type=float, default=0.05,
                   help='Per-step prob of crediting a random cell as target')
    p.add_argument('--kernel_sigma', type=float, default=1.5)
    p.add_argument('--vis_radius', type=int, default=3)
    p.add_argument('--conditions', default='flat,partition,familiarity,dual')
    p.add_argument('--top', action='store_true', help='Use -Top env variant (better for GIFs)')
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--save_figs', type=int, default=3,
                   help='Save light per-episode figure for first N episodes')
    p.add_argument('--viz', type=int, default=0,
                   help='Save rich visualize.py analysis+GIF for first N episodes')
    args = p.parse_args()

    suffix = '-Top' if (args.top or args.viz > 0) else ''
    env_id = f'MemoryMaze-{args.size}-ExtraObs{suffix}-v0'
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_names = [s.strip() for s in args.conditions.split(',')]

    print(f'[exp] env={env_id}  policy={args.policy}  steps={args.steps} '
          f'episodes={args.episodes} seeds={args.seeds} xy_scale={args.xy_scale}')
    print(f'[exp] conditions={condition_names}')

    PolicyCls = POLICY_REGISTRY[args.policy]

    all_episode_metrics: list[dict] = []
    fig_count = 0
    viz_count = 0
    t0 = time.time()

    for seed_ix in range(args.seeds):
        env = gym.make(env_id)
        env.action_space.seed(seed_ix)
        if hasattr(env, 'seed'):
            env.seed(seed_ix)
        rng = np.random.default_rng(seed_ix)

        for ep in range(args.episodes):
            # Use a per-episode noise seed so all 4 conditions see the SAME
            # noise sequence — this is critical for fair comparison.
            ep_noise_seed = seed_ix * 100_000 + ep
            memories = build_memories(
                condition_names, args.xy_scale,
                miss_rate=args.miss_rate, false_positive_rate=args.fp_rate,
                kernel_sigma=args.kernel_sigma, vis_radius=args.vis_radius,
                noise_seed=ep_noise_seed,
            )
            policy = PolicyCls()
            ep_global = seed_ix * args.episodes + ep
            do_viz = viz_count < args.viz
            per_cond, extra = run_episode(
                env, memories, policy, args.steps, rng, args.xy_scale,
                record_trajectory=do_viz, record_frames=do_viz,
            )
            all_episode_metrics.append(per_cond)

            line = ' | '.join(
                f'{c}: d={per_cond[c]["mean_dist"]:.2f} s2={per_cond[c]["success@2"]:.2f}'
                for c in per_cond
            )
            print(f'[exp] seed={seed_ix} ep={ep:>3d}  colors={extra["n_colors_seen"]}  {line}')

            if fig_count < args.save_figs:
                save_episode_figure(out_dir, ep_global, extra)
                fig_count += 1
            if do_viz:
                save_per_episode_viz(out_dir, ep_global, extra)
                viz_count += 1

        env.close()

    elapsed = time.time() - t0
    print(f'[exp] done in {elapsed:.1f}s ({elapsed/max(1,len(all_episode_metrics)):.2f}s/ep)')

    agg = aggregate(all_episode_metrics)
    metrics_path = out_dir / 'metrics.json'
    with metrics_path.open('w') as f:
        json.dump({
            'env_id': env_id,
            'config': vars(args),
            'aggregate': agg,
            'per_episode': all_episode_metrics,
            'wall_time_sec': elapsed,
        }, f, indent=2, default=str)
    print(f'[exp] wrote {metrics_path}')

    fig_path = save_summary_figure(out_dir, agg, env_id, args.policy)
    print(f'[exp] wrote {fig_path}')

    print('\n=== summary ===')
    print(f'{"condition":<28} {"mean_dist":>11} {"success@2":>11}')
    for c, v in agg.items():
        print(f'{c:<28} {v["mean_dist_mean"]:>5.2f}±{v["mean_dist_std"]:>4.2f}  '
              f'{v["success@2_mean"]:>5.2f}±{v["success@2_std"]:>4.2f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
