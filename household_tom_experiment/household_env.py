"""HouseholdEnv: 2D grid world for testing territorial + ToM coordination.

The setup matches the user's thesis: two agents share a multi-room house.
Objects have canonical "home" cells where they belong. A scripted partner
agent (A) periodically moves objects — sometimes intentionally (to a known
"alternative-good" cell, simulating a deliberate re-arrangement) and
sometimes accidentally (to a random cell). The fixer agent (B) must
decide, when it finds an object out of place, whether to:

  * revert the object to its canonical home, or
  * accept the displacement (because the partner *meant* to move it).

The minimum signal needed for B to make this decision is: did the partner
recently visit this cell? If yes, the displacement was probably intentional.
This is the simplest possible Theory-of-Mind proxy.

Layout (8x8):
    walls form 4 rooms with doorway corridors. Specifically:
      - vertical wall at column 4 (with door at row 4)
      - horizontal wall at row 4 (with door at column 2 and column 6)
    → 4 quadrant rooms + corridor doorways.

API:
    env = HouseholdEnv(seed=0)
    obs = env.reset()                # → {'A': dict, 'B': dict, 'global': dict}
    obs, info = env.step({'A': aA, 'B': aB})

    Each per-agent obs dict contains:
      'pos':              (i, j) own cell
      'partner_pos':      (i, j) or None if not in own visibility
      'visible_objects':  list of (obj_id, (i, j)) within visibility
      'partner_seen':     bool — was partner visible this step
      'inventory':        obj_id or None
      'maze_layout':      (H, W) — fully known to both agents
      'region_map':       (H, W) — fully known to both agents
      'canonical_homes':  {obj_id: (i, j)} — fully known to both agents
      'step':             int

Actions (Discrete(7)):
    0: stay
    1: up      (decrease row)
    2: down    (increase row)
    3: left    (decrease col)
    4: right   (increase col)
    5: pickup  (the object at own cell, if any and inventory is empty)
    6: place   (own inventory at own cell, if cell has no other object)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Allow running as a script (relative to repo) and as a module.
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent / 'memory_maze_experiment'
if str(_MM) not in sys.path:
    sys.path.insert(0, str(_MM))

from region_utils import extract_rooms  # noqa: E402


# --- layout ----------------------------------------------------------------

def make_default_layout(H: int = 8, W: int = 8) -> np.ndarray:
    """Build an 8x8 layout with 4 rooms separated by walls + doorways.

    1 = walkable, 0 = wall.
    Wall structure (4 doors total — every room is reachable from every other):
      - column 3 is a wall, with doors at row 1 (top) and row 5 (bottom)
      - row 4 is a wall,    with doors at col 1 (left) and col 5 (right)
    """
    layout = np.ones((H, W), dtype=np.uint8)
    # Vertical wall at col 3 with two doors
    layout[:, 3] = 0
    layout[1, 3] = 1
    layout[5, 3] = 1
    # Horizontal wall at row 4 with two doors (apply AFTER col-3 ops)
    layout[4, :] = 0
    layout[4, 1] = 1
    layout[4, 5] = 1
    return layout


# --- env state -------------------------------------------------------------

@dataclass
class ObjectState:
    obj_id: int
    cell: tuple[int, int]
    home: tuple[int, int]              # canonical home
    alt_good: tuple[int, int]           # the "intentional alternative" home
    holder: Optional[str] = None        # 'A' / 'B' / None


@dataclass
class Disturbance:
    """A scheduled action by the scripted partner."""
    step: int
    obj_id: int
    target_cell: tuple[int, int]
    intentional: bool   # ground truth — used for metric only, not visible to fixer


@dataclass
class HouseholdState:
    layout: np.ndarray
    region_map: np.ndarray
    region_info: dict
    objects: dict[int, ObjectState]
    pos: dict[str, tuple[int, int]]          # 'A': cell, 'B': cell
    inventory: dict[str, Optional[int]]      # 'A': obj_id or None
    step: int = 0
    canonical_homes: dict[int, tuple[int, int]] = field(default_factory=dict)
    intentional_displacements: dict[int, bool] = field(default_factory=dict)
    # ↑ obj_id → True if last partner-disturbance of this object was intentional
    completed_disturbances: set = field(default_factory=set)
    # ↑ ids of disturbances partner has already executed once. Once executed,
    #   partner does NOT redo even if the fixer reverts the object — this
    #   stops the partner↔fixer infinite-loop bug.


# --- the environment -------------------------------------------------------

ACTIONS = ['stay', 'up', 'down', 'left', 'right', 'pickup', 'place']

_DELTA = {
    'stay':  (0, 0),
    'up':    (-1, 0),
    'down':  ( 1, 0),
    'left':  (0, -1),
    'right': (0,  1),
}


class HouseholdEnv:
    """Multi-agent 2D household env for territorial+ToM experiments."""

    def __init__(
        self,
        layout: Optional[np.ndarray] = None,
        n_objects: int = 5,
        max_steps: int = 200,
        visibility: int = 3,
        partner_schedule: Optional[list[int]] = None,
        seed: int = 0,
    ):
        self.layout = layout if layout is not None else make_default_layout()
        self.H, self.W = self.layout.shape
        self.n_objects = n_objects
        self.max_steps = max_steps
        self.visibility = visibility
        self.partner_schedule = partner_schedule or [30, 80, 130]
        self.rng = np.random.RandomState(seed)
        self.region_map, self.region_info = extract_rooms(self.layout)
        self.state: Optional[HouseholdState] = None
        self.disturbances: list[Disturbance] = []  # set per episode
        self._partner_path: list[tuple[int, int]] = []  # path to current target

    # ----- reset ---------------------------------------------------------

    def reset(self) -> dict:
        # Pick canonical home cells: one per room, plus extras placed in random walkable cells
        room_centers = self.region_info['room_centers']
        rooms_by_size = sorted(self.region_info['room_sizes'].items(),
                               key=lambda kv: -kv[1])
        # Use up to n_objects rooms; if more objects than rooms, repeat.
        homes = {}
        for i in range(self.n_objects):
            r_id = rooms_by_size[i % len(rooms_by_size)][0]
            ys, xs = np.where(self.region_map == r_id)
            # Pick a stable cell within the room — center-ish
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            # If multiple objects share a room, jitter
            offset = i // len(rooms_by_size)
            cell = (max(0, min(self.H - 1, cy + offset)),
                    max(0, min(self.W - 1, cx)))
            if self.layout[cell] == 0:
                # find nearest walkable
                cell = self._nearest_walkable(cell)
            homes[i] = cell

        # Build object states. Each object's alt_good is a different walkable
        # cell in the SAME room — represents a "rearranged but still good" spot.
        objects = {}
        for obj_id, home in homes.items():
            r_id = int(self.region_map[home])
            ys, xs = np.where((self.region_map == r_id) &
                              (np.arange(self.H * self.W).reshape(self.H, self.W) //
                               self.W != home[0] * self.W + home[1]))
            # Simpler: pick another random walkable in same room ≠ home
            same_room = list(zip(*np.where(self.region_map == r_id)))
            same_room = [c for c in same_room if c != home and self.layout[c] == 1]
            alt_good = same_room[self.rng.randint(len(same_room))] if same_room else home
            objects[obj_id] = ObjectState(obj_id=obj_id, cell=home, home=home,
                                          alt_good=alt_good)

        # Schedule partner disturbances. Out of 3 disturbances, the first 2
        # are "intentional" (move to alt_good); last one is "accidental"
        # (move to a random walkable cell, preferably in a different room).
        self.disturbances = []
        chosen = self.rng.choice(self.n_objects, size=min(3, self.n_objects),
                                 replace=False).tolist()
        for k, step in enumerate(self.partner_schedule):
            obj_id = chosen[k % len(chosen)]
            if k < 2:  # intentional
                tgt = objects[obj_id].alt_good
                intentional = True
            else:  # accidental: random walkable cell
                walk = list(zip(*np.where(self.layout == 1)))
                tgt = walk[self.rng.randint(len(walk))]
                intentional = False
            self.disturbances.append(Disturbance(
                step=step, obj_id=obj_id, target_cell=tgt, intentional=intentional))

        # Agent positions: A in room with most disturbances; B in opposite corner.
        all_walk = list(zip(*np.where(self.layout == 1)))
        # Heuristic spawn: A near (1,1), B near (H-2, W-2).
        pos_A = self._nearest_walkable((1, 1))
        pos_B = self._nearest_walkable((self.H - 2, self.W - 2))

        self.state = HouseholdState(
            layout=self.layout,
            region_map=self.region_map,
            region_info=self.region_info,
            objects=objects,
            pos={'A': pos_A, 'B': pos_B},
            inventory={'A': None, 'B': None},
            step=0,
            canonical_homes=homes,
            intentional_displacements={},
        )
        self._partner_path = []
        return self._make_obs()

    # ----- step ----------------------------------------------------------

    def step(self, actions: dict) -> tuple[dict, dict]:
        """actions = {'A': int_action, 'B': int_action}"""
        assert self.state is not None
        # If A's action is None (test mode), we run scripted partner instead
        a_A = actions.get('A')
        a_B = actions.get('B')
        if a_A is None:
            a_A = self._scripted_partner_action()
        self._apply_action('A', a_A)
        self._apply_action('B', a_B)
        self.state.step += 1
        info = {
            'disturbance_log': self._latest_disturbance_info(),
            'objects': {k: dataclasses.asdict(v) for k, v in self.state.objects.items()},
        }
        return self._make_obs(), info

    # ----- helpers -------------------------------------------------------

    def _apply_action(self, agent: str, action: int):
        name = ACTIONS[int(action)]
        s = self.state
        cur = s.pos[agent]
        if name in ('up', 'down', 'left', 'right'):
            di, dj = _DELTA[name]
            nc = (cur[0] + di, cur[1] + dj)
            if self._is_walkable(nc):
                s.pos[agent] = nc
        elif name == 'pickup':
            if s.inventory[agent] is None:
                # Pick up first object at this cell that nobody is holding
                for obj in s.objects.values():
                    if obj.cell == cur and obj.holder is None:
                        obj.holder = agent
                        s.inventory[agent] = obj.obj_id
                        break
        elif name == 'place':
            if s.inventory[agent] is not None:
                obj_id = s.inventory[agent]
                obj = s.objects[obj_id]
                # Cell must not have another object
                if not any(o.cell == cur and o.obj_id != obj_id for o in s.objects.values()):
                    obj.cell = cur
                    obj.holder = None
                    s.inventory[agent] = None
        # 'stay' or invalid → no-op

    def _scripted_partner_action(self) -> int:
        """Plan a path step toward the next pending disturbance.

        A disturbance is "pending" only until partner has executed it ONCE.
        After that, partner moves on regardless of what the fixer does to
        the object. This prevents partner↔fixer infinite redo loops.
        """
        s = self.state
        pending = [(idx, d) for idx, d in enumerate(self.disturbances)
                   if d.step <= s.step + 5 and idx not in s.completed_disturbances]
        if not pending:
            return 0
        d_idx, d = pending[0]
        cur = s.pos['A']
        obj = s.objects[d.obj_id]
        if s.inventory['A'] is None and obj.holder is None:
            if cur == obj.cell:
                return ACTIONS.index('pickup')
            return self._step_toward(cur, obj.cell)
        elif s.inventory['A'] == d.obj_id:
            if cur == d.target_cell:
                s.intentional_displacements[d.obj_id] = d.intentional
                s.completed_disturbances.add(d_idx)  # ← one-shot, no redo
                return ACTIONS.index('place')
            return self._step_toward(cur, d.target_cell)
        else:
            return 0

    def _step_toward(self, cur: tuple[int, int], goal: tuple[int, int]) -> int:
        """Pick a single move action approaching `goal` along walkable cells.
        BFS for the next step."""
        if cur == goal:
            return 0
        # Cheap BFS
        from collections import deque
        H, W = self.layout.shape
        visited = {cur}
        q = deque([(cur, [])])
        while q:
            pos, path = q.popleft()
            for name in ('up', 'down', 'left', 'right'):
                di, dj = _DELTA[name]
                nc = (pos[0] + di, pos[1] + dj)
                if nc in visited or not (0 <= nc[0] < H and 0 <= nc[1] < W):
                    continue
                if self.layout[nc] == 0:
                    continue
                visited.add(nc)
                new_path = path + [name]
                if nc == goal:
                    return ACTIONS.index(new_path[0])
                q.append((nc, new_path))
        return 0  # unreachable → stay

    def _disturbance_done(self, d: Disturbance) -> bool:
        s = self.state
        obj = s.objects[d.obj_id]
        return obj.cell == d.target_cell and obj.holder is None and \
               d.obj_id in s.intentional_displacements

    def _latest_disturbance_info(self) -> Optional[dict]:
        s = self.state
        for d in self.disturbances:
            if d.step == s.step - 1:
                return {'obj_id': d.obj_id, 'target': d.target_cell,
                        'intentional': d.intentional}
        return None

    def _is_walkable(self, c: tuple[int, int]) -> bool:
        i, j = c
        if not (0 <= i < self.H and 0 <= j < self.W):
            return False
        return bool(self.layout[i, j] == 1)

    def _nearest_walkable(self, target: tuple[int, int]) -> tuple[int, int]:
        if self._is_walkable(target):
            return target
        # Spiral search
        from collections import deque
        H, W = self.layout.shape
        q = deque([target])
        visited = {target}
        while q:
            pos = q.popleft()
            for name in ('up', 'down', 'left', 'right'):
                di, dj = _DELTA[name]
                nc = (pos[0] + di, pos[1] + dj)
                if nc in visited or not (0 <= nc[0] < H and 0 <= nc[1] < W):
                    continue
                visited.add(nc)
                if self.layout[nc] == 1:
                    return nc
                q.append(nc)
        return target

    def _visible_cells(self, agent: str) -> set:
        cur = self.state.pos[agent]
        cells = set()
        for di in range(-self.visibility, self.visibility + 1):
            for dj in range(-self.visibility, self.visibility + 1):
                nc = (cur[0] + di, cur[1] + dj)
                if 0 <= nc[0] < self.H and 0 <= nc[1] < self.W:
                    cells.add(nc)
        return cells

    def _make_obs(self) -> dict:
        s = self.state

        def per_agent(agent: str) -> dict:
            visible = self._visible_cells(agent)
            visible_objects = [(o.obj_id, o.cell) for o in s.objects.values()
                               if o.cell in visible]
            partner = 'B' if agent == 'A' else 'A'
            partner_pos = s.pos[partner] if s.pos[partner] in visible else None
            return {
                'pos': s.pos[agent],
                'partner_pos': partner_pos,
                'partner_seen': partner_pos is not None,
                'visible_objects': visible_objects,
                'visible_cells': visible,
                'inventory': s.inventory[agent],
                'maze_layout': self.layout,
                'region_map': self.region_map,
                'region_info': self.region_info,
                'canonical_homes': dict(s.canonical_homes),
                'step': s.step,
            }

        return {
            'A': per_agent('A'),
            'B': per_agent('B'),
            'global': {
                'objects': {oid: dataclasses.asdict(o) for oid, o in s.objects.items()},
                'step': s.step,
                'intentional_displacements': dict(s.intentional_displacements),
            },
        }


if __name__ == '__main__':
    # Quick sanity check
    env = HouseholdEnv(seed=0)
    obs = env.reset()
    print('layout:')
    print(env.layout)
    print(f'\nrooms detected: {env.region_info["n_rooms"]}')
    print(f'object homes: {obs["global"]["objects"]}')
    print(f'agent A pos: {obs["A"]["pos"]}, B pos: {obs["B"]["pos"]}')
    print(f'disturbances: {[(d.step, d.obj_id, d.target_cell, d.intentional) for d in env.disturbances]}')

    # Run partner alone (B stays)
    for t in range(50):
        obs, info = env.step({'A': None, 'B': 0})
        if info['disturbance_log']:
            print(f'step {t}: disturbance {info["disturbance_log"]}')
