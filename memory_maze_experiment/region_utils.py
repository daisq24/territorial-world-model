"""Utilities for extracting physical territory structure from Memory Maze layouts.

The `maze_layout` observation is a 2D grid where 1 = walkable, 0 = wall.
Memory Maze is generated with rooms connected by corridors (DMLab-style).

We extract two things:
    - `rooms`: connected components of walkable cells that are "wide" (not corridors).
      A cell belongs to a room if its 3x3 walkable neighborhood is fully walkable.
    - `corridors`: walkable cells that are not rooms.
    - `region_map`: a 2D array of ints where each walkable cell is labeled with
      a region id (>=1 for rooms, 0 for walls, -1 for corridors). This is the
      "physical partition" signal for TerritorialMemory.

This is heuristic — Memory Maze's maze generator doesn't expose its internal
room list through the public API, so we reconstruct rooms from the occupancy
grid. In practice this lines up with the perceived rooms because the generator
places "rooms" as rectangular open regions and connects them with 1-cell-wide
corridors.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def extract_rooms(maze_layout: np.ndarray) -> tuple[np.ndarray, dict]:
    """Extract rooms from a Memory Maze layout.

    Args:
        maze_layout: 2D uint8 array, 1 = walkable, 0 = wall.

    Returns:
        region_map: 2D int array with the same shape as maze_layout.
            0   = wall
           -1   = corridor (walkable but narrow)
            >=1 = room id
        info: dict with keys {n_rooms, room_sizes, room_centers}.
    """
    walkable = (maze_layout > 0).astype(np.uint8)
    H, W = walkable.shape

    # A "room cell" is a walkable cell whose full 3x3 neighborhood is walkable.
    # This excludes cells in 1-wide corridors and on boundaries against walls.
    room_mask = np.zeros_like(walkable, dtype=np.uint8)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if walkable[i, j]:
                patch = walkable[i - 1:i + 2, j - 1:j + 2]
                if patch.sum() == 9:
                    room_mask[i, j] = 1

    # Grow the room mask by one step: any walkable cell adjacent to a room cell
    # also belongs to that room. This captures the rim of the room that is
    # walkable but fails the 3x3 test because it borders a wall.
    grown = room_mask.copy()
    for _ in range(1):
        padded = np.pad(grown, 1, mode='constant')
        neighbors = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        grown = np.where((walkable == 1) & (neighbors > 0), 1, grown).astype(np.uint8)

    # Label connected components of grown room cells.
    labels, n_rooms = label(grown, structure=np.ones((3, 3), dtype=int))

    region_map = np.where(walkable == 0, 0, -1).astype(np.int32)  # walls=0, corridor=-1
    region_map = np.where(labels > 0, labels.astype(np.int32), region_map)

    room_sizes = {int(r): int((labels == r).sum()) for r in range(1, n_rooms + 1)}
    # Merge any tiny "rooms" (< 4 cells) back into corridor — generator artifacts.
    for r, size in list(room_sizes.items()):
        if size < 4:
            region_map = np.where(region_map == r, -1, region_map)
            del room_sizes[r]

    # Reindex rooms to 1..K contiguously.
    unique_rooms = sorted(room_sizes.keys())
    remap = {old: new for new, old in enumerate(unique_rooms, start=1)}
    new_region_map = region_map.copy()
    for old, new in remap.items():
        new_region_map = np.where(region_map == old, new, new_region_map)

    # Recompute room centers from the reindexed map
    final_rooms = sorted(set(new_region_map[new_region_map > 0].tolist()))
    room_centers = {}
    room_sizes_out = {}
    for r in final_rooms:
        ys, xs = np.where(new_region_map == r)
        room_centers[int(r)] = (float(ys.mean()), float(xs.mean()))
        room_sizes_out[int(r)] = int(len(ys))

    info = {
        'n_rooms': len(final_rooms),
        'room_sizes': room_sizes_out,
        'room_centers': room_centers,
    }
    return new_region_map, info


def cell_to_region(region_map: np.ndarray, cell: tuple[int, int]) -> int:
    """Return the region id for a grid cell. 0 = wall, -1 = corridor, >=1 = room."""
    i, j = int(cell[0]), int(cell[1])
    if 0 <= i < region_map.shape[0] and 0 <= j < region_map.shape[1]:
        return int(region_map[i, j])
    return 0


def world_to_cell(pos_xy: np.ndarray, maze_shape: tuple[int, int], xy_scale: float = 1.0) -> tuple[int, int]:
    """Convert Memory Maze world coordinates to a cell in the maze_layout grid.

    Empirically (Memory Maze v1.0.3 with ExtraObs wrapper): agent_pos is already
    expressed in cell-center units, e.g. (2.5, 2.5) is the centre of cell (2, 2).
    So the right thing is just `floor(x)` (NOT round, which has banker's-rounding
    issues at .5 values that would push the agent across cell boundaries).

    `xy_scale` divides the input coords first, for installations / variants where
    the wrapper hasn't already done the normalization. Default 1.0 is correct
    for the standard ExtraObs envs verified by env_probe.py.
    """
    x = float(pos_xy[0]) / xy_scale
    y = float(pos_xy[1]) / xy_scale
    # floor() avoids Python's banker's-rounding (round(2.5)==2 but round(3.5)==4)
    # which makes agent jump cells on .5 boundaries.
    j = int(np.floor(x))
    i = int(np.floor(y))
    H, W = maze_shape
    i = max(0, min(H - 1, i))
    j = max(0, min(W - 1, j))
    return i, j
