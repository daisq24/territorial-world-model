"""Visual demo of the household ToM experiment — for class / paper figures.

Generates two artifacts under outputs/:

    demo_static.png       3×4 grid:
                            rows = {flat, territorial, territorial_tom}
                            cols = {init, after-disturbance, mid-episode, final}
    demo_animation.gif    side-by-side animation of all 3 methods on the
                          same seed, ~5 fps.

The visualization makes the contrast between the 3 methods immediately legible:

    walls                  dark gray
    agent-A territory      light blue tint
    agent-B territory      light green tint
    corridors / unowned    white
    object @ canonical home   green ✓
    object @ intentional alt   gold ★ (partner's deliberate rearrangement)
    object @ accidental random red ✗ (what the fixer should fix)
    agent A                red circle, label 'A'
    agent B                blue circle, label 'B'

Run:
    python visualize_demo.py --seed 7 --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agents import FIXER_REGISTRY  # noqa: E402
from household_env import HouseholdEnv, ACTIONS  # noqa: E402


# --- recording -------------------------------------------------------------

def _snapshot(env: HouseholdEnv, obs: dict, fixer_action: int | None) -> dict:
    """Capture everything we need to render this frame."""
    s = env.state
    return {
        'step': s.step,
        'pos_A': s.pos['A'],
        'pos_B': s.pos['B'],
        'inv_A': s.inventory['A'],
        'inv_B': s.inventory['B'],
        'objects': {oid: {'cell': o.cell, 'home': o.home,
                          'alt_good': o.alt_good, 'holder': o.holder}
                    for oid, o in s.objects.items()},
        'intentional': dict(s.intentional_displacements),
        'fixer_action': ACTIONS[int(fixer_action)] if fixer_action is not None else None,
    }


def run_and_record(env: HouseholdEnv, fixer, max_steps: int) -> list[dict]:
    obs = env.reset()
    fixer.reset(obs)
    frames = [_snapshot(env, obs, None)]
    for _ in range(max_steps):
        a_B = fixer.select(obs)
        obs, _info = env.step({'A': None, 'B': a_B})
        frames.append(_snapshot(env, obs, a_B))
    return frames


# --- rendering -------------------------------------------------------------

# Object status classification helper
def _classify(obj: dict, intentional: dict) -> str:
    cell = tuple(obj['cell'])
    home = tuple(obj['home'])
    alt = tuple(obj['alt_good'])
    if cell == home:
        return 'home'
    if cell == alt and intentional.get(obj.get('obj_id', -1) if 'obj_id' in obj else -1, False):
        return 'alt_intentional'
    return 'wrong'


def _render_grid(ax, layout: np.ndarray, region_map: np.ndarray,
                 territory_A: set[int], territory_B: set[int]):
    H, W = layout.shape
    # background: white
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # invert y so row 0 is at top
    ax.set_aspect('equal')
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, color='#e0e0e0', linewidth=0.5)
    ax.set_axisbelow(True)

    for i in range(H):
        for j in range(W):
            if layout[i, j] == 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor='#3a3a3a', edgecolor='none'))
            else:
                r = int(region_map[i, j])
                if r in territory_A:
                    color = '#dceaf7'   # light blue
                elif r in territory_B:
                    color = '#d9eed3'   # light green
                else:
                    color = '#ffffff'
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, edgecolor='none'))


def _render_objects(ax, frame: dict, homes: dict):
    intent = frame['intentional']
    for oid, obj in frame['objects'].items():
        cell = tuple(obj['cell'])
        home = tuple(obj['home'])
        alt = tuple(obj['alt_good'])
        is_intentional = bool(intent.get(oid, False)) and cell == alt
        if obj['holder'] is not None:
            # Object is being carried; draw at agent's pos with a small offset later
            continue
        # Status colors
        if cell == home:
            color, marker, lw = '#2a9d3f', 'o', 0
            edge = 'white'
        elif is_intentional:
            color, marker, lw = '#f2a900', '*', 0
            edge = '#9c6800'
        else:
            color, marker, lw = '#d62828', 'X', 1.5
            edge = '#7a1010'
        ax.scatter(cell[1], cell[0], s=240, c=color, marker=marker,
                   edgecolors=edge, linewidths=lw, zorder=4)
        ax.text(cell[1] + 0.32, cell[0] + 0.32, str(oid),
                fontsize=7, color='black', zorder=5)


def _render_agents(ax, frame: dict):
    pa = frame['pos_A']
    pb = frame['pos_B']
    # A: red
    ax.add_patch(Circle((pa[1], pa[0]), 0.32, facecolor='#d62828',
                        edgecolor='black', linewidth=1.2, zorder=6))
    ax.text(pa[1], pa[0], 'A', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', zorder=7)
    # B: blue
    ax.add_patch(Circle((pb[1], pb[0]), 0.32, facecolor='#2c7bb6',
                        edgecolor='black', linewidth=1.2, zorder=6))
    ax.text(pb[1], pb[0], 'B', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', zorder=7)
    # Inventory label if carrying
    if frame['inv_B'] is not None:
        ax.text(pb[1], pb[0] + 0.55, f'[{frame["inv_B"]}]', ha='center',
                fontsize=7, color='#2c7bb6', fontweight='bold', zorder=7)
    if frame['inv_A'] is not None:
        ax.text(pa[1], pa[0] + 0.55, f'[{frame["inv_A"]}]', ha='center',
                fontsize=7, color='#d62828', fontweight='bold', zorder=7)


def render_frame(ax, frame: dict, layout: np.ndarray, region_map: np.ndarray,
                 territory_A: set[int], territory_B: set[int],
                 homes: dict, title: str = ''):
    ax.cla()
    _render_grid(ax, layout, region_map, territory_A, territory_B)
    _render_objects(ax, frame, homes)
    _render_agents(ax, frame)
    ax.set_title(title, fontsize=10)


def _legend_patches():
    return [
        Patch(facecolor='#dceaf7', edgecolor='gray', label="A's territory"),
        Patch(facecolor='#d9eed3', edgecolor='gray', label="B's territory"),
        Patch(facecolor='#3a3a3a', label='wall'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2a9d3f',
                   markersize=10, label='at canonical home'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#f2a900',
                   markersize=14, label='at intentional alt'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#d62828',
                   markersize=11, label='accidentally placed'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828',
                   markersize=12, markeredgecolor='black', label='partner A'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6',
                   markersize=12, markeredgecolor='black', label='fixer B'),
    ]


# --- the two artifacts -----------------------------------------------------

def make_static(trajectories: dict, env_layout, env_region_map,
                terr_A, terr_B, homes, save_path: Path,
                key_steps: list[int] | None = None):
    if key_steps is None:
        T = len(next(iter(trajectories.values())))
        key_steps = [0, 35, T // 2, T - 1]

    methods = list(trajectories.keys())
    fig, axes = plt.subplots(len(methods), len(key_steps),
                             figsize=(3.6 * len(key_steps), 3.6 * len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)

    step_titles = ['init', 'after disturbances', 'mid-episode', 'final']

    for r, m in enumerate(methods):
        for c, t in enumerate(key_steps):
            t = min(t, len(trajectories[m]) - 1)
            ax = axes[r, c]
            title = f'{m}\nstep {t} ({step_titles[c] if c < len(step_titles) else ""})' \
                    if r == 0 else f'step {t}'
            render_frame(ax, trajectories[m][t], env_layout, env_region_map,
                         terr_A, terr_B, homes, title=title)

        # Method label on the left
        axes[r, 0].set_ylabel(m, fontsize=12, fontweight='bold')

    # legend (one combined, on top)
    fig.legend(handles=_legend_patches(), loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    plt.suptitle('Household ToM Experiment — three fixer policies on identical episode',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[viz] wrote {save_path}')


def make_animation(trajectories: dict, env_layout, env_region_map,
                   terr_A, terr_B, homes, save_path: Path, fps: int = 5):
    methods = list(trajectories.keys())
    T = max(len(traj) for traj in trajectories.values())

    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    def update(t):
        for ax, m in zip(axes, methods):
            t_clamped = min(t, len(trajectories[m]) - 1)
            render_frame(ax, trajectories[m][t_clamped], env_layout, env_region_map,
                         terr_A, terr_B, homes, title=f'{m}\nstep {t_clamped}')
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=200, blit=False)
    fig.legend(handles=_legend_patches(), loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    plt.suptitle('Household ToM — animated comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f'[viz] wrote {save_path}')


# --- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--no_animation', action='store_true',
                   help='skip the animated GIF (only output static figure)')
    args = p.parse_args()

    out_dir = _HERE / args.outdir
    out_dir.mkdir(exist_ok=True, parents=True)

    methods = ['flat', 'territorial', 'territorial_tom']
    trajectories = {}

    print(f'[viz] running 3 methods on seed={args.seed} ...')
    layout = None
    region_map = None
    terr_A = terr_B = None
    homes = None
    for cname in methods:
        FixerCls = FIXER_REGISTRY[cname]
        fixer = FixerCls(agent_id='B', seed=args.seed)
        env = HouseholdEnv(seed=args.seed, max_steps=args.steps,
                           partner_schedule=[30, 80, 130])
        traj = run_and_record(env, fixer, args.steps)
        trajectories[cname] = traj
        if layout is None:
            layout = env.layout
            region_map = env.region_map
            # Compute territory split (matches TerritorialReverter logic)
            rooms = sorted(env.region_info['room_centers'].keys())
            terr_A = {r for i, r in enumerate(rooms) if i % 2 == 0}
            terr_B = {r for i, r in enumerate(rooms) if i % 2 == 1}
            homes = traj[0]['objects']  # for legend reference

    # Add obj_id to each obj dict for _classify (it iterates without obj_id)
    for traj in trajectories.values():
        for f in traj:
            for oid, o in f['objects'].items():
                o['obj_id'] = oid

    static_path = out_dir / 'demo_static.png'
    make_static(trajectories, layout, region_map, terr_A, terr_B, homes, static_path)

    if not args.no_animation:
        anim_path = out_dir / 'demo_animation.gif'
        make_animation(trajectories, layout, region_map, terr_A, terr_B, homes, anim_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
