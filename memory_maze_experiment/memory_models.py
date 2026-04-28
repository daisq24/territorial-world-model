"""FlatMemory vs TerritorialMemory for Memory Maze.

Both memories have the same interface:

    memory.reset(initial_obs)
    memory.observe(obs)              # called every step
    memory.predict_target(color_id)  # returns a (y, x) cell prediction
    memory.color_index(rgb_or_idx)   # 0..2 id (also accepts int passthrough)

Color-id convention (TARGETS_AS_ROW_INDEX):
    We use the row index of `targets_pos` (0, 1, 2) as the canonical color id.
    Memory Maze exposes ALL 3 target world positions at every step in
    `targets_pos: (3, 2)`, but only `target_color` (the *currently asked*
    one) tells us a color RGB. In offline probing we don't actually need
    real RGBs — we just need a stable id per target. Row index is stable
    for the entire episode and avoids the "agent never reaches a target,
    so target_color never changes, so we only ever see 1 color" problem.

Observation model (NOISY_KERNEL):
    When the agent comes within `vis_radius` cells of target i, we credit
    evidence at the target's cell PLUS its neighbors with gaussian decay.
    This models the realistic "I saw something around there but I'm not
    sure exactly which cell" signal — without this, evidence concentrates
    on a single cell and all 4 conditions return identical predictions.

FlatMemory keeps per-cell counts of observed target colors.
TerritorialMemory adds:
    - a physical partition (rooms) extracted from maze_layout
    - a familiarity score per (cell, region) that updates online
    - prediction is hierarchical: first predict the region, then the cell

The dual-source territory in TerritorialMemory is the fusion of:
    (a) physical room id for each cell (static, from maze_layout)
    (b) familiarity weight per cell (dynamic, from visitation)

This is the key knob the experiment tests.
"""

from __future__ import annotations

import numpy as np

from region_utils import extract_rooms, world_to_cell


# --- shared color-indexing helper -------------------------------------------

class ColorIndexer:
    """Map RGB float tuples to a canonical 0..n-1 index.

    Memory Maze's `target_color` is a (3,) float RGB vector. We want to use
    integer ids 0..n-1 inside the memory tensors. This class registers each
    distinct RGB seen and returns its index. Comparisons use a small
    tolerance because float quantization can perturb the bytes.
    """

    def __init__(self, n_max: int = 3, tol: float = 1e-3):
        self.n_max = n_max
        self.tol = tol
        self.palette: list[np.ndarray] = []  # list of (3,) arrays

    def reset(self):
        self.palette = []

    def index(self, rgb) -> int:
        rgb = np.asarray(rgb, dtype=np.float64).reshape(-1)[:3]
        for i, p in enumerate(self.palette):
            if np.max(np.abs(p - rgb)) < self.tol:
                return i
        if len(self.palette) >= self.n_max:
            # Unknown color and palette full — return -1 to signal skip.
            return -1
        self.palette.append(rgb)
        return len(self.palette) - 1

    def __len__(self) -> int:
        return len(self.palette)


# --- shared evidence kernel ------------------------------------------------

def _credit_kernel(evidence_slice: np.ndarray, tcell: tuple[int, int],
                   sigma: float = 1.0, weight: float = 1.0):
    """Add weighted evidence around `tcell` with gaussian decay.

    Without this, all observations of a target end up at exactly one cell
    and every memory's argmax returns that one cell — so all conditions
    converge to the same answer. The kernel models realistic perceptual
    uncertainty: the agent thinks the target is around `tcell`, with most
    mass on the actual cell and decreasing weight on neighbors.
    """
    H, W = evidence_slice.shape
    radius = int(np.ceil(2 * sigma))
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = tcell[0] + di, tcell[1] + dj
            if 0 <= ni < H and 0 <= nj < W:
                d2 = di * di + dj * dj
                w = weight * np.exp(-0.5 * d2 / (sigma * sigma + 1e-9))
                evidence_slice[ni, nj] += w


class FlatMemory:
    """Baseline: per-cell evidence map of observed target positions.

    Uses targets_pos[i] (i = 0/1/2) as the i-th color id. Whenever the agent
    is within `vis_radius` cells of target i, evidence is credited around
    target i's cell with a gaussian kernel.
    """

    name = 'FlatMemory'

    def __init__(self, n_colors: int = 3, xy_scale: float = 1.0,
                 vis_radius: int = 3, kernel_sigma: float = 1.0,
                 miss_rate: float = 0.0, false_positive_rate: float = 0.0,
                 noise_seed: int = 0):
        self.n_colors = n_colors
        self.xy_scale = xy_scale
        self.vis_radius = vis_radius
        self.kernel_sigma = kernel_sigma
        self.miss_rate = miss_rate
        self.false_positive_rate = false_positive_rate
        self._rng = np.random.RandomState(noise_seed)
        self.counts: np.ndarray | None = None
        self.maze_shape: tuple[int, int] | None = None
        self._walkable: np.ndarray | None = None  # (H, W) of valid cells for FP

    def reset(self, initial_obs: dict):
        layout = initial_obs['maze_layout']
        self.maze_shape = layout.shape
        self.counts = np.zeros((self.n_colors, *self.maze_shape), dtype=np.float32)
        self._walkable = np.argwhere(layout > 0)  # array of (i, j) tuples

    def color_index(self, rgb_or_idx) -> int:
        if isinstance(rgb_or_idx, (int, np.integer)):
            return int(rgb_or_idx)
        return -1

    def observe(self, obs: dict):
        assert self.counts is not None
        agent_cell = world_to_cell(obs['agent_pos'], self.maze_shape, self.xy_scale)
        targets_pos = obs['targets_pos']
        for i in range(min(self.n_colors, len(targets_pos))):
            tcell = world_to_cell(targets_pos[i], self.maze_shape, self.xy_scale)
            dy = tcell[0] - agent_cell[0]
            dx = tcell[1] - agent_cell[1]
            if abs(dy) <= self.vis_radius and abs(dx) <= self.vis_radius:
                # MISS: fail to credit even though target is in range
                if self._rng.rand() >= self.miss_rate:
                    _credit_kernel(self.counts[i], tcell, sigma=self.kernel_sigma)
            # FALSE POSITIVE: credit a random walkable cell as if we saw target i
            if self.false_positive_rate > 0 and self._rng.rand() < self.false_positive_rate:
                if len(self._walkable) > 0:
                    fp_idx = self._rng.randint(0, len(self._walkable))
                    fcell = tuple(self._walkable[fp_idx])
                    _credit_kernel(self.counts[i], fcell, sigma=self.kernel_sigma, weight=0.5)

    def predict_target(self, color_id: int) -> tuple[int, int]:
        assert self.counts is not None
        if not (0 <= color_id < self.n_colors):
            return (self.maze_shape[0] // 2, self.maze_shape[1] // 2)
        c = self.counts[color_id]
        if c.sum() == 0:
            return (self.maze_shape[0] // 2, self.maze_shape[1] // 2)
        flat = np.argmax(c)
        return (int(flat // c.shape[1]), int(flat % c.shape[1]))


class TerritorialMemory:
    """Dual-source territorial memory.

    Physical partition: rooms from maze_layout (static, computed at reset).
    Familiarity: visitation-based weights per cell and per region (dynamic).

    Prediction:
        1. For color c, aggregate target evidence per region.
        2. Pick the region with highest (evidence * familiarity^alpha).
        3. Inside that region, return the cell with highest evidence.

    The familiarity prior matters when two regions have equal raw evidence —
    the memory prefers the region the agent knows better (is more certain about).

    Ablation knobs:
        use_partition=False, familiarity_alpha=0.0 → behaves like FlatMemory
        use_partition=True,  familiarity_alpha=0.0 → partition-only
        use_partition=False, familiarity_alpha>0   → familiarity-only
        use_partition=True,  familiarity_alpha>0   → dual-source (full)
    """

    def __init__(
        self,
        n_colors: int = 3,
        familiarity_alpha: float = 0.5,
        use_partition: bool = True,
        xy_scale: float = 1.0,
        vis_radius: int = 3,
        kernel_sigma: float = 1.0,
        miss_rate: float = 0.0,
        false_positive_rate: float = 0.0,
        noise_seed: int = 0,
        name: str | None = None,
    ):
        self.n_colors = n_colors
        self.familiarity_alpha = familiarity_alpha
        self.use_partition = use_partition
        self.xy_scale = xy_scale
        self.vis_radius = vis_radius
        self.kernel_sigma = kernel_sigma
        self.miss_rate = miss_rate
        self.false_positive_rate = false_positive_rate
        self._rng = np.random.RandomState(noise_seed)
        # Auto-name based on knobs if no override given
        if name is None:
            if use_partition and familiarity_alpha > 0:
                name = 'Territorial(dual)'
            elif use_partition:
                name = 'Territorial(partition)'
            elif familiarity_alpha > 0:
                name = 'Territorial(familiarity)'
            else:
                name = 'Territorial(none)'
        self.name = name

        # Static
        self.maze_shape: tuple[int, int] | None = None
        self.region_map: np.ndarray | None = None
        self.region_info: dict | None = None

        # Dynamic
        self.visit_count: np.ndarray | None = None   # (H, W)
        self.evidence: np.ndarray | None = None      # (n_colors, H, W)
        self.region_evidence: dict | None = None     # {color: {region: float}}

    def reset(self, initial_obs: dict):
        layout = initial_obs['maze_layout']
        self.maze_shape = layout.shape
        self.region_map, self.region_info = extract_rooms(layout)
        self.visit_count = np.zeros(self.maze_shape, dtype=np.float32)
        self.evidence = np.zeros((self.n_colors, *self.maze_shape), dtype=np.float32)
        self.region_evidence = {c: {} for c in range(self.n_colors)}
        self._walkable = np.argwhere(layout > 0)

    def color_index(self, rgb_or_idx) -> int:
        if isinstance(rgb_or_idx, (int, np.integer)):
            return int(rgb_or_idx)
        return -1

    def observe(self, obs: dict):
        assert self.evidence is not None
        agent_cell = world_to_cell(obs['agent_pos'], self.maze_shape, self.xy_scale)
        self.visit_count[agent_cell[0], agent_cell[1]] += 1.0

        targets_pos = obs['targets_pos']
        for i in range(min(self.n_colors, len(targets_pos))):
            tcell = world_to_cell(targets_pos[i], self.maze_shape, self.xy_scale)
            dy = tcell[0] - agent_cell[0]
            dx = tcell[1] - agent_cell[1]
            if abs(dy) <= self.vis_radius and abs(dx) <= self.vis_radius:
                if self._rng.rand() >= self.miss_rate:
                    _credit_kernel(self.evidence[i], tcell, sigma=self.kernel_sigma)
                    r = int(self.region_map[tcell[0], tcell[1]])
                    if r > 0:
                        d = self.region_evidence[i]
                        d[r] = d.get(r, 0.0) + 1.0
            # FALSE POSITIVE: credit a random walkable cell with half-weight
            if self.false_positive_rate > 0 and self._rng.rand() < self.false_positive_rate:
                if len(self._walkable) > 0:
                    fp_idx = self._rng.randint(0, len(self._walkable))
                    fcell = tuple(self._walkable[fp_idx])
                    _credit_kernel(self.evidence[i], fcell, sigma=self.kernel_sigma, weight=0.5)
                    r = int(self.region_map[fcell[0], fcell[1]])
                    if r > 0:
                        d = self.region_evidence[i]
                        d[r] = d.get(r, 0.0) + 0.5

    def familiarity_per_region(self) -> dict[int, float]:
        """Dynamic familiarity per region = normalized visitation share."""
        assert self.visit_count is not None and self.region_map is not None
        total = float(self.visit_count.sum()) + 1e-6
        fam = {}
        for r in sorted(set(self.region_map[self.region_map > 0].tolist())):
            mask = self.region_map == r
            fam[int(r)] = float(self.visit_count[mask].sum()) / total
        return fam

    def familiarity_per_cell(self) -> np.ndarray:
        """Visit-share map (H, W) used in familiarity-only ablation."""
        assert self.visit_count is not None
        total = float(self.visit_count.sum()) + 1e-6
        return self.visit_count / total

    def predict_target(self, color_id: int) -> tuple[int, int]:
        assert self.evidence is not None
        if not (0 <= color_id < self.n_colors):
            return (self.maze_shape[0] // 2, self.maze_shape[1] // 2)

        if self.use_partition:
            fam = self.familiarity_per_region()
            region_ev = self.region_evidence[color_id] if self.region_evidence else {}
            if region_ev:
                # Additive familiarity boost: evidence dominates, familiarity
                # is a small tiebreaker. Avoids the failure mode where a low-
                # familiarity *correct* room loses to a high-familiarity wrong
                # room because of multiplicative crushing.
                scores = {
                    r: ev * (1.0 + self.familiarity_alpha * fam.get(r, 0.0))
                    for r, ev in region_ev.items()
                }
                best_region = max(scores, key=scores.get)
                mask = self.region_map == best_region
                c_in_region = np.where(mask, self.evidence[color_id], -1.0)
                flat = int(np.argmax(c_in_region))
                return (flat // c_in_region.shape[1], flat % c_in_region.shape[1])
            # No region-level evidence yet → fall through to flat argmax.

        # Familiarity-weighted flat (or pure flat when alpha=0).
        c = self.evidence[color_id].astype(np.float32)
        if c.sum() == 0:
            return (self.maze_shape[0] // 2, self.maze_shape[1] // 2)
        if self.familiarity_alpha > 0:
            # Additive boost, same logic as the partition path.
            fam_cell = self.familiarity_per_cell()
            # Normalize so max(fam_cell) = 1 → boost ranges from 0 to alpha.
            fam_norm = fam_cell / (fam_cell.max() + 1e-9)
            c = c * (1.0 + self.familiarity_alpha * fam_norm)
        flat = np.argmax(c)
        return (int(flat // c.shape[1]), int(flat % c.shape[1]))
