"""Visualizations for the Memory Maze territorial experiment.

Two modes:

1. Run a single episode with full visualization:
       python visualize.py --size 9x9 --steps 300 --seed 0 --gif
   Saves:
       outputs/viz_analysis.png  — 6-panel analysis figure
       outputs/viz_episode.gif   — top-down recording of the episode

2. Re-plot from an existing metrics.json:
       python visualize.py --from-metrics outputs/metrics.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw' if sys.platform == 'darwin' else 'egl'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import gym  # noqa: E402
import memory_maze  # noqa: F401, E402

from memory_models import FlatMemory, TerritorialMemory  # noqa: E402
from region_utils import extract_rooms, world_to_cell  # noqa: E402


TARGET_RGB = [
    (170, 38, 30),   # red
    (99, 170, 88),   # green
    (39, 140, 217),  # blue
    (93, 105, 199),  # purple
    (220, 193, 59),  # yellow
    (220, 128, 107), # salmon
]


def run_one_episode(env_id: str, steps: int, seed: int, record_gif: bool):
    """Run one episode, track trajectory, and (optionally) record frames."""
    env = gym.make(env_id)
    obs = env.reset()

    maze_shape = obs['maze_layout'].shape
    territorial = TerritorialMemory()
    flat = FlatMemory()
    territorial.reset(obs)
    flat.reset(obs)
    territorial.observe(obs)
    flat.observe(obs)

    # Track trajectory and optionally frames
    trajectory = [world_to_cell(obs['agent_pos'], maze_shape)]
    frames = []
    if record_gif and 'image' in obs:
        frames.append(obs['image'])

    rng = np.random.default_rng(seed)
    for step in range(steps):
        action = int(rng.integers(0, env.action_space.n))
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        territorial.observe(obs)
        flat.observe(obs)
        trajectory.append(world_to_cell(obs['agent_pos'], maze_shape))
        if record_gif and 'image' in obs and step % 3 == 0:  # subsample frames for smaller GIF
            frames.append(obs['image'])

        if done:
            break

    # Final state for evaluation
    final_targets_pos = obs['targets_pos']
    initial_maze = territorial.region_map  # already computed at reset
    env.close()

    return {
        'maze_layout': obs['maze_layout'],
        'region_map': initial_maze,
        'region_info': territorial.region_info,
        'trajectory': np.array(trajectory),
        'visit_count': territorial.visit_count,
        'evidence_territorial': territorial.evidence,
        'evidence_flat': flat.counts,
        'targets_pos': final_targets_pos,
        'memory_territorial': territorial,
        'memory_flat': flat,
        'frames': frames,
    }


def plot_analysis(data: dict, out_path: Path):
    """6-panel figure: maze, rooms, trajectory, familiarity, predictions."""
    maze = data['maze_layout']
    region = data['region_map']
    traj = data['trajectory']
    visits = data['visit_count']
    targets_pos = data['targets_pos']
    maze_shape = maze.shape

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle('Memory Maze — single-episode territorial analysis', fontsize=13)

    # --- Panel 1: raw maze layout ---
    ax = axes[0, 0]
    ax.imshow(maze, cmap='gray', interpolation='nearest')
    ax.set_title('maze_layout  (1 = walkable, 0 = wall)')
    for tcell in [world_to_cell(t, maze_shape) for t in targets_pos]:
        ax.scatter(tcell[1], tcell[0], marker='*', s=150, c='gold', edgecolors='k', linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 2: physical partition (rooms) ---
    ax = axes[0, 1]
    cmap = plt.cm.tab10
    disp = np.zeros((*maze_shape, 3))
    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            r = int(region[i, j])
            if r == 0:
                disp[i, j] = (0.15, 0.15, 0.15)  # wall
            elif r == -1:
                disp[i, j] = (0.7, 0.7, 0.7)     # corridor
            else:
                disp[i, j] = cmap((r - 1) % 10)[:3]
    ax.imshow(disp, interpolation='nearest')
    n_rooms = data['region_info']['n_rooms']
    ax.set_title(f'physical partition  ({n_rooms} rooms + corridors)')
    for r, c in data['region_info']['room_centers'].items():
        ax.text(c[1], c[0], str(r), color='w', ha='center', va='center',
                fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 3: trajectory overlaid on maze ---
    ax = axes[0, 2]
    ax.imshow(maze, cmap='gray', interpolation='nearest', alpha=0.5)
    ax.plot(traj[:, 1], traj[:, 0], '-', color='#2c7bb6', linewidth=1.0, alpha=0.7)
    ax.scatter(traj[0, 1], traj[0, 0], marker='o', s=80, c='lime',
               edgecolors='k', linewidths=1, label='start', zorder=3)
    ax.scatter(traj[-1, 1], traj[-1, 0], marker='s', s=80, c='orange',
               edgecolors='k', linewidths=1, label='end', zorder=3)
    ax.set_title(f'trajectory  ({len(traj)} steps)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 4: familiarity heatmap (dynamic territory) ---
    ax = axes[1, 0]
    fam_display = np.where(maze > 0, visits, np.nan)
    im = ax.imshow(fam_display, cmap='viridis', interpolation='nearest')
    ax.set_title('familiarity  (visit count per cell)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 5: memory predictions ---
    ax = axes[1, 1]
    ax.imshow(maze, cmap='gray', interpolation='nearest', alpha=0.3)
    n_colors_to_show = len(targets_pos)
    flat_mem = data.get('memory_flat')
    show_flat = flat_mem is not None and getattr(flat_mem, 'counts', None) is not None
    for c_ix in range(n_colors_to_show):
        truth = world_to_cell(targets_pos[c_ix], maze_shape)
        pred_t = data['memory_territorial'].predict_target(c_ix)
        rgb = [v / 255.0 for v in TARGET_RGB[c_ix]]
        ax.scatter(truth[1], truth[0], marker='*', s=200, c=[rgb],
                   edgecolors='k', linewidths=0.8, label=f'truth {c_ix}' if c_ix == 0 else None)
        ax.scatter(pred_t[1], pred_t[0], marker='o', s=100, facecolors='none',
                   edgecolors=rgb, linewidths=2.0,
                   label='terr pred' if c_ix == 0 else None)
        if show_flat:
            pred_f = flat_mem.predict_target(c_ix)
            ax.scatter(pred_f[1], pred_f[0], marker='x', s=100, c=[rgb], linewidths=2,
                       label='flat pred' if c_ix == 0 else None)
    title = 'predictions  (★ truth, ○ territorial' + (', × flat)' if show_flat else ')')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 6: familiarity per room (bar chart) ---
    ax = axes[1, 2]
    fam = data['memory_territorial'].familiarity_per_region()
    if fam:
        rooms = sorted(fam.keys())
        values = [fam[r] for r in rooms]
        colors = [cmap((r - 1) % 10) for r in rooms]
        ax.bar([str(r) for r in rooms], values, color=colors, edgecolor='k', linewidth=0.6)
        ax.set_xlabel('room id')
        ax.set_ylabel('share of visits')
        ax.set_title('familiarity per room')
        ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    else:
        ax.text(0.5, 0.5, '(no rooms detected)', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('familiarity per room')

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f'Saved {out_path}')


def save_gif(frames: list, out_path: Path, fps: int = 10):
    if not frames:
        print('(no frames recorded; re-run with --gif)')
        return
    imageio.mimsave(out_path, frames, fps=fps)
    print(f'Saved {out_path}  ({len(frames)} frames, {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='9x9', choices=['9x9', '11x11', '13x13', '15x15'])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gif', action='store_true', help='also record a top-down GIF')
    parser.add_argument('--outdir', default='outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The -Top- variant gives a top-down camera — nicer for a GIF.
    env_id = f'MemoryMaze-{args.size}-ExtraObs-Top-v0'
    print(f'Running one episode of {env_id}  (steps={args.steps}, seed={args.seed})')

    data = run_one_episode(env_id, args.steps, args.seed, record_gif=args.gif)

    plot_analysis(data, outdir / 'viz_analysis.png')
    if args.gif:
        save_gif(data['frames'], outdir / 'viz_episode.gif', fps=10)

    print('\nDone. Open the outputs folder to see the figures.')


if __name__ == '__main__':
    main()
