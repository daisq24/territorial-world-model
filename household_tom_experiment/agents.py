"""Three fixer agents for the household env.

All three share the same exploration + carry-and-place mechanics. They
differ ONLY in the revert-decision rule:

    FlatReverter        — revert any out-of-place object, anywhere.
    TerritorialReverter — revert only objects inside own territory; ignore
                          out-of-place objects outside.
    TerritorialToMReverter — within own territory, when an out-of-place
                          object is found, query the partner-visit map.
                          If partner has visited this cell recently → infer
                          the displacement was intentional → ACCEPT (don't
                          revert). Otherwise → revert.

Common behavior:
    - Each agent maintains:
        own_visit_count[H, W]
        partner_visit_count[H, W]   (incremented when partner is observed)
        belief_object_cell[obj_id] = (i, j) or None
        last_seen_step[obj_id] = int
    - Each step the agent:
        1. updates own_visit / partner_visit
        2. updates belief_object_cell from visible_objects
        3. if carrying nothing:
             a. find a "high-priority" out-of-place object to revert
                (filtered by territory / ToM rules)
             b. if found and within reach → walk to it, then pickup
             c. else → wander (least-visited neighbor or random)
        4. if carrying something:
             walk toward its canonical home, then place

Action space matches HouseholdEnv:
    0 stay | 1 up | 2 down | 3 left | 4 right | 5 pickup | 6 place
"""

from __future__ import annotations

import numpy as np

ACT_STAY, ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_PICKUP, ACT_PLACE = range(7)

_NEIGHBORS = {
    ACT_UP:    (-1, 0),
    ACT_DOWN:  ( 1, 0),
    ACT_LEFT:  (0, -1),
    ACT_RIGHT: (0,  1),
}


def _step_toward(layout, cur, goal) -> int:
    """BFS one step toward goal along walkable cells. Returns an action."""
    if cur == goal:
        return ACT_STAY
    from collections import deque
    H, W = layout.shape
    visited = {cur}
    q = deque([(cur, [])])
    while q:
        pos, path = q.popleft()
        for act, (di, dj) in _NEIGHBORS.items():
            nc = (pos[0] + di, pos[1] + dj)
            if nc in visited or not (0 <= nc[0] < H and 0 <= nc[1] < W):
                continue
            if layout[nc] == 0:
                continue
            visited.add(nc)
            new_path = path + [act]
            if nc == goal:
                return new_path[0]
            q.append((nc, new_path))
    return ACT_STAY


def _least_visited_neighbor(layout, visit_count, cur, rng) -> int:
    """Pick the action toward the least-visited walkable neighbor."""
    H, W = layout.shape
    best, best_score = None, float('inf')
    candidates = list(_NEIGHBORS.items())
    rng.shuffle(candidates)  # tiebreak randomly
    for act, (di, dj) in candidates:
        ni, nj = cur[0] + di, cur[1] + dj
        if not (0 <= ni < H and 0 <= nj < W):
            continue
        if layout[ni, nj] == 0:
            continue
        score = float(visit_count[ni, nj])
        if score < best_score:
            best_score = score
            best = act
    return best if best is not None else ACT_STAY


# --- common base -----------------------------------------------------------

class BaseFixer:
    """Shared mechanics: exploration, belief tracking, action selection.

    Subclasses override `_should_revert(obj_id, observed_cell, ctx)`.
    """

    name = 'BaseFixer'

    def __init__(self, agent_id: str = 'B', seed: int = 0,
                 partner_visit_decay: float = 0.99,
                 partner_visit_threshold: float = 0.5):
        self.agent_id = agent_id
        self.rng = np.random.RandomState(seed)
        self.partner_visit_decay = partner_visit_decay
        self.partner_visit_threshold = partner_visit_threshold
        self.own_visit = None
        self.partner_visit = None  # exponentially decaying counter
        self.belief_object_cell: dict[int, tuple[int, int]] = {}
        self.last_seen_step: dict[int, int] = {}
        self.target_obj: int | None = None  # currently being reverted
        self.target_phase: str = 'idle'  # 'goto_obj', 'goto_home', 'idle'
        self.maze_shape = None

    def reset(self, obs: dict):
        layout = obs[self.agent_id]['maze_layout']
        self.maze_shape = layout.shape
        self.own_visit = np.zeros(self.maze_shape, dtype=np.float32)
        self.partner_visit = np.zeros(self.maze_shape, dtype=np.float32)
        self.belief_object_cell = {}
        self.last_seen_step = {}
        self.target_obj = None
        self.target_phase = 'idle'

    def _territory_rooms(self, region_info, region_map) -> set[int]:
        """Default: own everything. Overridden by Territorial variants."""
        return set(region_info['room_centers'].keys())

    def _should_revert(self, obj_id: int, observed_cell, ctx: dict) -> bool:
        """Subclass hook. ctx contains region_map, partner_visit, threshold."""
        return True  # FlatReverter default

    # ----- main step -----

    def select(self, obs: dict) -> int:
        my = obs[self.agent_id]
        cur = my['pos']
        layout = my['maze_layout']
        region_map = my['region_map']
        region_info = my['region_info']
        homes = my['canonical_homes']
        step = my['step']

        # 1. Decay + bookkeeping
        self.partner_visit *= self.partner_visit_decay
        self.own_visit[cur] += 1.0
        if my['partner_seen'] and my['partner_pos'] is not None:
            self.partner_visit[my['partner_pos']] += 1.0
            # Also bump cells the partner is "near" — coarse model
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    pi, pj = my['partner_pos'][0] + di, my['partner_pos'][1] + dj
                    if 0 <= pi < self.maze_shape[0] and 0 <= pj < self.maze_shape[1]:
                        self.partner_visit[pi, pj] += 0.3

        # 2. Update belief from visible objects
        for obj_id, cell in my['visible_objects']:
            self.belief_object_cell[obj_id] = cell
            self.last_seen_step[obj_id] = step

        territory = self._territory_rooms(region_info, region_map)

        # 3. Carrying something? → walk to its home, place when there.
        if my['inventory'] is not None:
            obj_id = my['inventory']
            home = homes[obj_id]
            if cur == home:
                # Refresh belief
                self.belief_object_cell[obj_id] = home
                self.target_obj = None
                self.target_phase = 'idle'
                return ACT_PLACE
            return _step_toward(layout, cur, home)

        # 4. Find an out-of-place object to revert (under method's rule).
        if self.target_obj is None or self.target_obj not in self.belief_object_cell:
            self.target_obj = self._select_target(homes, region_map, territory)

        if self.target_obj is not None and self.target_obj in self.belief_object_cell:
            obj_cell = self.belief_object_cell[self.target_obj]
            if cur == obj_cell:
                return ACT_PICKUP
            return _step_toward(layout, cur, obj_cell)

        # 5. Else explore (least-visited neighbor with tiny random noise).
        return _least_visited_neighbor(layout, self.own_visit, cur, self.rng)

    def _select_target(self, homes, region_map, territory) -> int | None:
        """Pick the highest-priority out-of-place object the method allows."""
        ctx = {
            'region_map': region_map,
            'partner_visit': self.partner_visit,
            'threshold': self.partner_visit_threshold,
            'homes': homes,
        }
        candidates = []
        for obj_id, cell in self.belief_object_cell.items():
            home = homes.get(obj_id)
            if home is None or cell == home:
                continue  # at home → not a target
            r = int(region_map[cell[0], cell[1]])
            if r > 0 and r not in territory:
                continue  # outside our territory (only matters for Territorial variants)
            if not self._should_revert(obj_id, cell, ctx):
                continue  # ToM said: probably intentional, accept
            candidates.append((obj_id, cell))
        if not candidates:
            return None
        # Closest to current pos wins (cheap heuristic)
        return candidates[0][0]


# --- three concrete fixers -------------------------------------------------

class FlatReverter(BaseFixer):
    name = 'FlatReverter'

    def _territory_rooms(self, region_info, region_map):
        return set(region_info['room_centers'].keys())  # owns everything

    def _should_revert(self, obj_id, observed_cell, ctx):
        return True  # always revert


class TerritorialReverter(BaseFixer):
    name = 'TerritorialReverter'

    def __init__(self, agent_id='B', owned_rooms: set | None = None, **kw):
        super().__init__(agent_id=agent_id, **kw)
        self.owned_rooms_override = owned_rooms

    def _territory_rooms(self, region_info, region_map):
        if self.owned_rooms_override is not None:
            return self.owned_rooms_override
        # Default split: rooms with even ID → A, odd → B (matches partner)
        rooms = sorted(region_info['room_centers'].keys())
        if self.agent_id == 'A':
            return {r for i, r in enumerate(rooms) if i % 2 == 0}
        else:
            return {r for i, r in enumerate(rooms) if i % 2 == 1}

    def _should_revert(self, obj_id, observed_cell, ctx):
        return True  # within territory → always revert; outside is filtered upstream


class TerritorialToMRoomOnly(TerritorialReverter):
    """Ablation: ToM uses ONLY the same-room signal (no partner_visit).

    Accept (don't revert) IFF the displaced object is in the same room as its
    canonical home. Tests whether room-level structure alone is enough.
    Expected weakness: an *accidental* drop that lands in the home room
    will be wrongly accepted (no partner-presence check).
    """
    name = 'TerritorialToMRoomOnly'

    def _should_revert(self, obj_id, observed_cell, ctx):
        region_map = ctx['region_map']
        homes = ctx['homes']
        home_cell = homes.get(obj_id)
        if home_cell is None:
            return True
        obs_room = int(region_map[observed_cell[0], observed_cell[1]])
        home_room = int(region_map[home_cell[0], home_cell[1]])
        same_room = (obs_room == home_room and obs_room > 0)
        if same_room:
            return False
        return True


class TerritorialToMVisitOnly(TerritorialReverter):
    """Ablation: ToM uses ONLY the partner_visit signal (no same-room rule).

    Accept (don't revert) IFF partner has been observed near the displaced
    cell recently. Tests whether partner-presence alone is enough.
    Expected weakness: an *accidental* drop where partner was passing through
    will be wrongly accepted (no spatial-coherence check).
    """
    name = 'TerritorialToMVisitOnly'

    def _should_revert(self, obj_id, observed_cell, ctx):
        partner_vis = float(ctx['partner_visit'][observed_cell[0], observed_cell[1]])
        if partner_vis > ctx['threshold']:
            return False
        return True


class TerritorialToMReverter(TerritorialReverter):
    """The user's actual proposed method: territorial × Theory-of-Mind.

    ToM rule (two-signal):
      Accept (don't revert) IFF
        (1) the object is in the SAME room as its canonical home
            — partner's intentional rearrangements stay in the home room
              (an "alt_good" cell within the room), whereas accidents land
              in an arbitrary room;
        (2) AND partner has been observed near this cell recently
            (so we know partner is the one who placed it, not random env).

    Either signal alone is too weak:
      - "same room" alone would let an obj that fell off-shelf into an alt
        cell get accepted (no partner involvement).
      - "partner visited" alone would let an obj partner *accidentally* dropped
        in the wrong room get accepted (which is what we observed in v1).

    The conjunction matches the user's stated thesis: "if partner moved it
    to a place that makes sense AND it was partner that did it, defer."
    """
    name = 'TerritorialToMReverter'

    def _should_revert(self, obj_id, observed_cell, ctx):
        region_map = ctx['region_map']
        homes = ctx['homes']
        home_cell = homes.get(obj_id)
        if home_cell is None:
            return True
        obs_room = int(region_map[observed_cell[0], observed_cell[1]])
        home_room = int(region_map[home_cell[0], home_cell[1]])
        same_room = (obs_room == home_room and obs_room > 0)
        partner_vis = float(ctx['partner_visit'][observed_cell[0], observed_cell[1]])
        partner_was_here = partner_vis > ctx['threshold']
        if same_room and partner_was_here:
            return False  # likely intentional rearrangement → accept
        return True       # different room OR no partner activity → revert


FIXER_REGISTRY = {
    'flat': FlatReverter,
    'territorial': TerritorialReverter,
    'tom_room': TerritorialToMRoomOnly,    # ablation: same-room signal alone
    'tom_visit': TerritorialToMVisitOnly,  # ablation: partner-visit signal alone
    'territorial_tom': TerritorialToMReverter,  # full: both signals (ours)
}
