import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.config import PATCH_SIZE, PATCH_SIZE_COARSE, COARSE_STRIDE, DEM_STEP

# 8-connected grid actions: (row_delta, col_delta, move_distance)
ACTIONS = [
    (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    ( 0, -1, 1.0),                 ( 0, 1, 1.0),
    ( 1, -1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
]


class DronePathEnv(gym.Env):
    """
    Custom Gymnasium environment for drone path-planning on a DEM grid.

    Observation (flat float32 vector, length = 9 + patch_size² + patch_size_coarse²):
      [0:2]      current position, normalised by grid dims
      [2:4]      goal position, normalised by grid dims
      [4]        Euclidean distance to goal, normalised by grid diagonal
      [5:7]      direction to goal as (sin θ, cos θ)
      [7]        last action taken, normalised to [0, 1]
      [8]        accumulated step cost, normalised (clipped at 1)
      [9:9+P²]   flattened fine elevation patch (PATCH_SIZE × PATCH_SIZE), [0, 1]
      [9+P²:]    flattened coarse elevation patch (PATCH_SIZE_COARSE × PATCH_SIZE_COARSE, stride COARSE_STRIDE), [0, 1]

    Action space:
      Discrete(8) — the eight grid-compass directions.

    Episode termination:
      • Agent reaches goal cell        → terminated=True, large bonus
      • Agent steps out-of-bounds/NaN  → terminated=True, large penalty
      • Agent exceeds max_steps        → truncated=True
    """

    metadata = {}

    def __init__(self, elev_sub, start, goal, reward_fn, max_steps,
                 patch_size=PATCH_SIZE, endpoint_mode="fixed_single", endpoint_list=None):
        super().__init__()

        self.elev             = elev_sub.astype(np.float32)
        self._default_start   = start
        self._default_goal    = goal
        self.start            = start
        self.goal             = goal
        self.reward_fn        = reward_fn
        self.max_steps        = max_steps
        self.patch_size       = patch_size
        self._half            = patch_size // 2
        self._endpoint_mode   = endpoint_mode
        self._endpoint_list   = endpoint_list or []
        self._endpoint_idx    = 0

        self.rows, self.cols = self.elev.shape
        self._elev_min   = float(np.nanmin(self.elev))
        self._elev_max   = float(np.nanmax(self.elev))
        self._elev_range = self._elev_max - self._elev_min
        self._max_dist   = float(np.sqrt(self.rows ** 2 + self.cols ** 2))

        # Coarse patch parameters
        self.patch_size_coarse = PATCH_SIZE_COARSE
        self.coarse_stride     = COARSE_STRIDE
        self._half_coarse      = self.patch_size_coarse // 2

        # Pre-compute valid cells and a generous step budget for random mode
        self._valid_cells       = np.argwhere(~np.isnan(self.elev))
        self._random_max_steps  = int(np.sqrt(self.rows ** 2 + self.cols ** 2) * 3)

        obs_dim = 9 + patch_size * patch_size + self.patch_size_coarse * self.patch_size_coarse
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        # Episode state (reset each episode)
        self._pos         = start
        self._steps       = 0
        self._accum_cost  = 0.0   # full scoring cost accumulated this episode
        self._last_action = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._endpoint_mode == "fixed_set":
            pair = self._endpoint_list[self._endpoint_idx % len(self._endpoint_list)]
            self.start, self.goal = pair
            self._endpoint_idx += 1
        elif self._endpoint_mode == "random":
            # Draw two distinct valid cells each episode
            n = len(self._valid_cells)
            i, j = self.np_random.integers(0, n, size=2)
            while i == j:
                j = self.np_random.integers(0, n)
            self.start     = tuple(self._valid_cells[i])
            self.goal      = tuple(self._valid_cells[j])
            self.max_steps = self._random_max_steps

        self._pos         = self.start
        self._steps       = 0
        self._accum_cost  = 0.0
        self._last_action = 0
        return self._get_obs(), {}

    def step(self, action: int):
        dr, dc, dist = ACTIONS[int(action)]
        r, c = self._pos
        nr, nc = r + int(dr), c + int(dc)

        # ── Out of bounds ──────────────────────────────────────────────────────
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return (self._get_obs(),
                    self.reward_fn.failure_penalty,
                    True, False,
                    {"reason": "oob", "outcome": "oob"})

        # ── No-data cell ──────────────────────────────────────────────────────
        if np.isnan(self.elev[nr, nc]):
            return (self._get_obs(),
                    self.reward_fn.failure_penalty,
                    True, False,
                    {"reason": "nan", "outcome": "nan"})

        new_pos      = (nr, nc)
        current_elev = float(self.elev[nr, nc])
        elev_diff    = current_elev - float(self.elev[r, c])
        patch        = self._get_patch(nr, nc)
        patch_min    = float(patch.min() * self._elev_range + self._elev_min)

        reached_goal = new_pos == self.goal
        reward = self.reward_fn.compute(
            self._pos, new_pos, self.goal,
            dist, elev_diff,
            reached_goal=reached_goal,
            current_elev=current_elev,
            local_patch_min=patch_min,
            elev_min=self._elev_min,
            elev_range=self._elev_range,
        )

        # Track full scoring cost (same formula as A*)
        self._accum_cost  += self.reward_fn.step_cost(dist, elev_diff)
        self._pos          = new_pos
        self._last_action  = int(action)
        self._steps       += 1

        truncated = (not reached_goal) and (self._steps >= self.max_steps)
        info = {}
        if reached_goal:
            info["outcome"] = "success"
        elif truncated:
            info["outcome"] = "timeout"
        return self._get_obs(), reward, reached_goal, truncated, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        r, c   = self._pos
        gr, gc = self.goal

        # Position and goal
        pos_norm  = np.array([r / self.rows,  c / self.cols],  dtype=np.float32)
        goal_norm = np.array([gr / self.rows, gc / self.cols], dtype=np.float32)

        # Distance and direction to goal
        dr_g  = gr - r
        dc_g  = gc - c
        dist_g = np.sqrt(dr_g ** 2 + dc_g ** 2)
        dist_norm = np.array([dist_g / self._max_dist], dtype=np.float32)
        angle     = np.arctan2(dc_g, dr_g)
        direction = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

        # Last action and accumulated cost (both normalised to [0, 1])
        last_act = np.array([self._last_action / 8.0], dtype=np.float32)
        accum    = np.array([min(self._accum_cost / 50_000.0, 1.0)], dtype=np.float32)

        patch = self._get_patch(r, c)
        coarse = self._get_coarse_patch(r, c)
        return np.concatenate(
            [pos_norm, goal_norm, dist_norm, direction, last_act, accum,
             patch.ravel(), coarse.ravel()]
        )

    def _get_patch(self, r: int, c: int) -> np.ndarray:
        """Extract and normalise an 11×11 elevation window centred on (r, c)."""
        h  = self._half
        ps = self.patch_size
        r0, r1 = r - h, r + h + 1
        c0, c1 = c - h, c + h + 1

        # Clamp to grid bounds
        pr0 = max(0, r0);  pr1 = min(self.rows, r1)
        pc0 = max(0, c0);  pc1 = min(self.cols, c1)

        patch = np.zeros((ps, ps), dtype=np.float32)

        # Offset of valid region inside the patch array
        dr0 = pr0 - r0;  dc0 = pc0 - c0
        dr1 = dr0 + (pr1 - pr0)
        dc1 = dc0 + (pc1 - pc0)

        region = self.elev[pr0:pr1, pc0:pc1]
        normed = (region - self._elev_min) / self._elev_range
        normed = np.nan_to_num(normed, nan=0.0).astype(np.float32)
        patch[dr0:dr1, dc0:dc1] = normed
        return patch

    def _get_coarse_patch(self, r: int, c: int) -> np.ndarray:
        """
        Coarse-scale elevation patch sampled at stride `coarse_stride`.
        Same normalisation and out-of-bounds handling as `_get_patch`.
        """
        ps      = self.patch_size_coarse
        stride  = self.coarse_stride
        h       = self._half_coarse
        patch   = np.zeros((ps, ps), dtype=np.float32)

        for i in range(ps):
            for j in range(ps):
                rr = r + (i - h) * stride
                cc = c + (j - h) * stride
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    v = self.elev[rr, cc]
                    if not np.isnan(v):
                        patch[i, j] = (v - self._elev_min) / self._elev_range
        return patch
