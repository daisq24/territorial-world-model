"""Exploration policies for Memory Maze.

Four levels, from dumb to smart:

- RandomPolicy: uniform over 6 discrete actions. Default baseline.
- WanderPolicy: random-turn when stuck, forward-bias otherwise.
  Solves the "ball pinned in a corner" problem with random.
- NoveltyPolicy: steer toward least-visited neighboring cell.
  Drives cell-level coverage.
- TerritorialPolicy: prefer the least-visited ROOM; within that room,
  use novelty-style steering. This is the one where territory actually
  drives exploration — the point of the whole project.

All share the same interface:
    policy.reset(initial_obs)
    action = policy.select(obs, rng)  -> int in [0, 6)

Memory Maze actions:
    0 noop | 1 forward | 2 left | 3 right | 4 fwd+left | 5 fwd+right
"""

from __future__ import annotations

import numpy as np

from region_utils import extract_rooms, world_to_cell


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _action_toward(from_cell, to_cell, agent_dir, rng) -> int:
    """Pick an action that moves from `from_cell` toward `to_cell`,
    given the agent's current facing direction (2D unit vector).

    Conventions: maze is indexed [row=i, col=j]. agent_pos is (x, y) in
    world coords, where x maps to column j and y maps to row i. So a
    cell offset (di, dj) corresponds to a world direction (dj, di).
    """
    dy = to_cell[0] - from_cell[0]
    dx = to_cell[1] - from_cell[1]
    target_dir = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(target_dir))
    if norm < 1e-6:
        # already there — just wiggle
        return int(rng.choice([1, 2, 3]))
    target_dir /= norm

    ad = np.asarray(agent_dir, dtype=np.float64)
    adn = np.linalg.norm(ad)
    if adn < 1e-6:
        return 1
    ad = ad / adn

    cos_angle = float(np.dot(ad, target_dir))
    # cross-product z-component: positive = target is on the LEFT of agent_dir
    cross = float(ad[0] * target_dir[1] - ad[1] * target_dir[0])

    if cos_angle > 0.7:        # roughly facing it
        return 1  # forward
    elif cos_angle > 0.0:      # off to the side — forward + slight turn
        return 4 if cross > 0 else 5
    else:                       # pointing wrong way — pure rotate
        return 2 if cross > 0 else 3


def _stuck_check(self, pos) -> bool:
    """Helper shared across policies: did we fail to move since last step?"""
    moved = self.last_pos is not None and float(np.linalg.norm(pos - self.last_pos)) > 0.1
    if self.last_pos is not None and not moved:
        self.stuck_count += 1
    else:
        self.stuck_count = 0
    self.last_pos = pos.copy()
    return self.stuck_count > 3


# ------------------------------------------------------------
# Policies
# ------------------------------------------------------------

class RandomPolicy:
    name = 'Random'

    def reset(self, obs):
        pass

    def select(self, obs, rng):
        return int(rng.integers(0, 6))


class WanderPolicy:
    """Break out of corners by detecting stuck-ness, otherwise bias forward."""
    name = 'Wander'

    def reset(self, obs):
        self.last_pos = None
        self.stuck_count = 0

    def select(self, obs, rng):
        if _stuck_check(self, obs['agent_pos']):
            self.stuck_count = 0
            return int(rng.choice([2, 3]))  # rotate out
        # Heavy forward bias; occasional turn to prevent infinite loops
        return int(rng.choice([1, 1, 1, 1, 4, 5, 2, 3]))


class NoveltyPolicy:
    """Steer toward the least-visited walkable neighbor cell."""
    name = 'Novelty'

    def reset(self, obs):
        self.visit_count = np.zeros(obs['maze_layout'].shape, dtype=np.float32)
        self.last_pos = None
        self.stuck_count = 0

    def select(self, obs, rng):
        maze = obs['maze_layout']
        cell = world_to_cell(obs['agent_pos'], maze.shape)
        self.visit_count[cell] += 1.0

        if _stuck_check(self, obs['agent_pos']):
            self.stuck_count = 0
            return int(rng.choice([2, 3]))

        # Search 8 neighbors for the least-visited walkable one
        best_cell = None
        best_score = np.inf
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = cell[0] + di, cell[1] + dj
            if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj] > 0:
                score = self.visit_count[ni, nj] + rng.random() * 0.01  # tiebreak
                if score < best_score:
                    best_score = score
                    best_cell = (ni, nj)
        if best_cell is None:
            return 1
        return _action_toward(cell, best_cell, obs['agent_dir'], rng)


class TerritorialPolicy:
    """Prefer the least-visited ROOM. Within the current room, use novelty.

    This is where territory drives *action*: the agent actively seeks out
    under-explored territories, not just under-explored cells.
    """
    name = 'Territorial'

    def reset(self, obs):
        self.region_map, self.region_info = extract_rooms(obs['maze_layout'])
        self.room_visits = {r: 0 for r in self.region_info['room_centers']}
        self.visit_count = np.zeros(obs['maze_layout'].shape, dtype=np.float32)
        self.last_pos = None
        self.stuck_count = 0

    def select(self, obs, rng):
        maze = obs['maze_layout']
        cell = world_to_cell(obs['agent_pos'], maze.shape)
        self.visit_count[cell] += 1.0
        cur_region = int(self.region_map[cell])
        if cur_region > 0:
            self.room_visits[cur_region] += 1

        if _stuck_check(self, obs['agent_pos']):
            self.stuck_count = 0
            return int(rng.choice([2, 3]))

        # If we have unvisited rooms, target the least-visited one
        if self.room_visits:
            target_room = min(self.room_visits, key=self.room_visits.get)
        else:
            target_room = None

        in_target_room = (cur_region == target_room and target_room is not None)

        if in_target_room or target_room is None:
            # Wander inside current room (novelty-style on cells)
            best_cell = None
            best_score = np.inf
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = cell[0] + di, cell[1] + dj
                if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1] and maze[ni, nj] > 0:
                    score = self.visit_count[ni, nj] + rng.random() * 0.01
                    if score < best_score:
                        best_score = score
                        best_cell = (ni, nj)
            if best_cell is None:
                return 1
            return _action_toward(cell, best_cell, obs['agent_dir'], rng)

        # Otherwise head toward the target room's center
        target_center = self.region_info['room_centers'][target_room]
        target_cell = (int(round(target_center[0])), int(round(target_center[1])))
        return _action_toward(cell, target_cell, obs['agent_dir'], rng)


POLICY_REGISTRY = {
    'random': RandomPolicy,
    'wander': WanderPolicy,
    'novelty': NoveltyPolicy,
    'territorial': TerritorialPolicy,
}
