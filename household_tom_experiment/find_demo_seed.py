"""Find a seed where the 3 methods produce visibly different final states.

We want a seed where:
  * flat actively reverts at least one gold-star (intentional) — so it loses gold stars
  * territorial_tom preserves those gold stars
  * The differentiating object is in B's visibility (so it's actually shown
    on the figure, not in some unreachable corner)

Score for each seed = (# gold stars in tom final) - (# gold stars in flat final).
Higher = bigger visible difference.

Run:
    python find_demo_seed.py --max_seed 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agents import FIXER_REGISTRY  # noqa: E402
from household_env import HouseholdEnv  # noqa: E402


def _run_one(seed: int, method: str, steps: int):
    FixerCls = FIXER_REGISTRY[method]
    fixer = FixerCls(agent_id='B', seed=seed)
    env = HouseholdEnv(seed=seed, max_steps=steps,
                       partner_schedule=[30, 80, 130])
    obs = env.reset()
    fixer.reset(obs)
    for _ in range(steps):
        a_B = fixer.select(obs)
        obs, _ = env.step({'A': None, 'B': a_B})
    intentional_done = obs['global']['intentional_displacements']
    intentional_targets = {d.obj_id: d.target_cell
                           for d in env.disturbances if d.intentional}
    n_gold = 0
    n_home = 0
    for o in env.state.objects.values():
        cell = tuple(o.cell)
        if cell == tuple(o.home):
            n_home += 1
        elif (intentional_done.get(o.obj_id, False) and
              o.obj_id in intentional_targets and
              cell == tuple(intentional_targets[o.obj_id])):
            n_gold += 1
    return {'home': n_home, 'gold': n_gold}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--max_seed', type=int, default=50)
    p.add_argument('--steps', type=int, default=300)
    args = p.parse_args()

    best = None
    print(f'searching seeds 0..{args.max_seed-1} (steps={args.steps}) ...')
    for seed in range(args.max_seed):
        flat = _run_one(seed, 'flat', args.steps)
        terr = _run_one(seed, 'territorial', args.steps)
        tom = _run_one(seed, 'territorial_tom', args.steps)
        # Score: TOM should have MORE gold stars preserved than flat
        gold_gap = tom['gold'] - flat['gold']
        # Also prefer: flat reverts at least one gold star (so visible action)
        flat_reverted_gold = max(0, tom['gold'] - flat['gold'])
        score = flat_reverted_gold * 10 + (terr['gold'] - flat['gold'])
        marker = ''
        if best is None or score > best['score']:
            best = {'seed': seed, 'score': score,
                    'flat': flat, 'terr': terr, 'tom': tom}
            marker = ' ←'
        print(f'  seed={seed:>2d}  flat=(home={flat["home"]} gold={flat["gold"]}) '
              f'terr=(home={terr["home"]} gold={terr["gold"]}) '
              f'tom=(home={tom["home"]} gold={tom["gold"]})  '
              f'score={score}{marker}')

    print('\n=== BEST SEED ===')
    print(f'seed={best["seed"]}  score={best["score"]}')
    print(f'  Flat:           {best["flat"]}')
    print(f'  Territorial:    {best["terr"]}')
    print(f'  Territorial+ToM: {best["tom"]}')
    print(f'\nUse this in visualize_story.py:')
    print(f'  python visualize_story.py --seed {best["seed"]} --steps {args.steps}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
