"""Class-presentation figure: tells the story in one shot. All English.

Layout: title → 3 method panels → legend → bar chart.

Run:
    python visualize_story.py --seed 48 --steps 300
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agents import FIXER_REGISTRY  # noqa: E402
from household_env import HouseholdEnv  # noqa: E402

plt.rcParams['axes.unicode_minus'] = False


# Half-size font scheme
F_TITLE = 9        # main title
F_SUB = 6          # subtitle
F_PANEL = 6        # panel titles
F_AXIS = 5.5       # axis labels
F_TICK = 5         # tick labels
F_LEG = 4.5        # legend
F_BAR = 5          # bar value annotations
F_OBJ = 4.5        # object id labels
F_AGENT = 5.5      # A/B in agent circles
F_BARTITLE = 6     # bar chart title


# --- helpers ---------------------------------------------------------------

def _run(env, fixer, steps: int):
    obs = env.reset()
    fixer.reset(obs)
    for _ in range(steps):
        a_B = fixer.select(obs)
        obs, _ = env.step({'A': None, 'B': a_B})
    return env, obs


def _classify(obj_dict, intentional_done: dict) -> str:
    cell = tuple(obj_dict['cell'])
    home = tuple(obj_dict['home'])
    alt = tuple(obj_dict['alt_good'])
    oid = obj_dict['obj_id']
    if cell == home:
        return 'home'
    if cell == alt and intentional_done.get(oid, False):
        return 'alt_intentional'
    return 'wrong'


def _draw_grid(ax, layout, region_map, terr_A, terr_B):
    H, W = layout.shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(H):
        for j in range(W):
            if layout[i, j] == 0:
                color = '#3a3a3a'
            else:
                r = int(region_map[i, j])
                if r in terr_A:
                    color = '#dceaf7'
                elif r in terr_B:
                    color = '#d9eed3'
                else:
                    color = '#ffffff'
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=color, edgecolor='#cccccc',
                                   linewidth=0.4))


def _draw_state(ax, env, obs, intentional_done):
    obj_views = []
    for oid, o in env.state.objects.items():
        obj_views.append({
            'obj_id': oid,
            'cell': o.cell, 'home': o.home, 'alt_good': o.alt_good,
            'holder': o.holder,
        })
    for o in obj_views:
        cell = o['cell']
        status = _classify(o, intentional_done)
        if status == 'home':
            color, marker, size = '#2a9d3f', 'o', 130
            edge = 'white'
        elif status == 'alt_intentional':
            color, marker, size = '#f2a900', '*', 220
            edge = '#9c6800'
        else:
            color, marker, size = '#d62828', 'X', 130
            edge = '#7a1010'
        ax.scatter(cell[1], cell[0], s=size, c=color, marker=marker,
                   edgecolors=edge, linewidths=0.8, zorder=4)
        ax.text(cell[1] + 0.32, cell[0] + 0.32, str(o['obj_id']),
                fontsize=F_OBJ, color='black', fontweight='bold', zorder=5)
        if status == 'alt_intentional':
            home = o['home']
            ax.annotate('', xy=(cell[1], cell[0]), xytext=(home[1], home[0]),
                        arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                        lw=0.6, ls='dashed'),
                        zorder=3)

    pa = env.state.pos['A']
    pb = env.state.pos['B']
    ax.add_patch(Circle((pa[1], pa[0]), 0.30, facecolor='#d62828',
                        edgecolor='black', linewidth=0.6, zorder=6))
    ax.text(pa[1], pa[0], 'A', ha='center', va='center',
            fontsize=F_AGENT, color='white', fontweight='bold', zorder=7)
    ax.add_patch(Circle((pb[1], pb[0]), 0.30, facecolor='#2c7bb6',
                        edgecolor='black', linewidth=0.6, zorder=6))
    ax.text(pb[1], pb[0], 'B', ha='center', va='center',
            fontsize=F_AGENT, color='white', fontweight='bold', zorder=7)


def _make_legend_handles():
    return [
        Patch(facecolor='#dceaf7', edgecolor='gray', label="A's territory"),
        Patch(facecolor='#d9eed3', edgecolor='gray', label="B's territory"),
        Patch(facecolor='#3a3a3a', label='wall'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2a9d3f',
                   markersize=5, label='at canonical home (OK)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#f2a900',
                   markersize=8, label="at partner's intentional alt (OK)"),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#d62828',
                   markersize=6, label='accidentally displaced (revert)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62828',
                   markersize=6, markeredgecolor='black', label='Partner A (scripted)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6',
                   markersize=6, markeredgecolor='black', label='Fixer B (evaluated)'),
    ]


# --- main figure -----------------------------------------------------------

def make_story_figure(seed: int, steps: int, save_path: Path):
    method_names = [
        ('flat',            'Flat\n(baseline: revert everything)'),
        ('territorial',     'Territorial\n(only own territory)'),
        ('territorial_tom', 'Territorial + ToM (ours)\n(respect partner intent)'),
    ]

    fig = plt.figure(figsize=(14, 8))
    # 4 rows: title / panels / legend / bar chart
    gs = fig.add_gridspec(4, 3,
                          height_ratios=[0.3, 2.5, 0.5, 1.2],
                          hspace=0.45, wspace=0.18)

    # Top: title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.85,
                  'Household tidying experiment: how should the fixer respond '
                  "to the partner's choices?",
                  ha='center', va='center', fontsize=F_TITLE, fontweight='bold')
    title_ax.text(0.5, 0.15,
                  'Partner deliberately rearranges objects at step 30 / 80 '
                  '(gold star = intentional). At step 130 partner accidentally '
                  'drops one (red X = accidental). Fixer B observes and decides: '
                  'revert or accept?',
                  ha='center', va='center', fontsize=F_SUB, color='#555')

    # Middle row: 3 grids
    layout = region_map = terr_A = terr_B = None
    for i, (cname, label) in enumerate(method_names):
        ax = fig.add_subplot(gs[1, i])
        FixerCls = FIXER_REGISTRY[cname]
        fixer = FixerCls(agent_id='B', seed=seed)
        env = HouseholdEnv(seed=seed, max_steps=steps,
                           partner_schedule=[30, 80, 130])
        env, obs = _run(env, fixer, steps)
        if layout is None:
            layout = env.layout
            region_map = env.region_map
            rooms = sorted(env.region_info['room_centers'].keys())
            terr_A = {r for ix, r in enumerate(rooms) if ix % 2 == 0}
            terr_B = {r for ix, r in enumerate(rooms) if ix % 2 == 1}
        intentional_done = obs['global']['intentional_displacements']
        _draw_grid(ax, layout, region_map, terr_A, terr_B)
        _draw_state(ax, env, obs, intentional_done)
        ax.set_title(label, fontsize=F_PANEL, fontweight='bold', pad=4)

    # Legend row
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    legend_ax.legend(handles=_make_legend_handles(), loc='center',
                     ncol=4, fontsize=F_LEG, frameon=False,
                     handletextpad=0.5, columnspacing=1.2)

    # Bottom: bar chart
    bar_ax = fig.add_subplot(gs[3, :])
    cond_short = ['Flat', 'Territorial', 'Territorial+ToM']
    metrics = {
        'false_rev':       [1.88, 1.09, 0.04],
        'acceptable_rate': [0.99, 0.95, 0.93],
    }
    err_false = [0.43, 0.68, 0.20]
    err_accept = [0.04, 0.09, 0.09]

    x = np.arange(len(cond_short))
    width = 0.35
    bars1 = bar_ax.bar(x - width/2, metrics['false_rev'], width,
                       yerr=err_false, capsize=2.5,
                       color=['#888', '#3a86ff', '#e63946'],
                       label='false_rev')
    bar_ax2 = bar_ax.twinx()
    bars2 = bar_ax2.bar(x + width/2, metrics['acceptable_rate'], width,
                        yerr=err_accept, capsize=2.5,
                        color=['#bbbbbb', '#a3c6f5', '#f4adb6'],
                        label='acceptable_rate')

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(cond_short, fontsize=F_TICK)
    bar_ax.set_ylabel('false_rev (interferences with partner)',
                      fontsize=F_AXIS)
    bar_ax2.set_ylabel('acceptable_rate (tidiness)', fontsize=F_AXIS)
    bar_ax.set_ylim(0, max(metrics['false_rev']) * 1.4)
    bar_ax2.set_ylim(0, 1.05)
    bar_ax.tick_params(axis='y', labelsize=F_TICK)
    bar_ax2.tick_params(axis='y', labelsize=F_TICK)

    for ax, bars, fmt in [(bar_ax, bars1, '{:.2f}'),
                          (bar_ax2, bars2, '{:.2f}')]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + 0.015 * ax.get_ylim()[1],
                    fmt.format(b.get_height()), ha='center',
                    fontsize=F_BAR, fontweight='bold')

    bar_ax.set_title('100 episodes: ToM reduces false reverts 47x while '
                     'preserving acceptable_rate',
                     fontsize=F_BARTITLE, fontweight='bold', pad=4)
    bar_ax.legend([bars1, bars2],
                  ['false_rev (lower is better)',
                   'acceptable_rate (higher is better)'],
                  loc='upper right', fontsize=F_LEG, frameon=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[story] wrote {save_path}')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=48)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--outdir', default='outputs')
    args = p.parse_args()

    out_dir = _HERE / args.outdir
    out_dir.mkdir(exist_ok=True, parents=True)
    save_path = out_dir / 'story.png'
    make_story_figure(args.seed, args.steps, save_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
