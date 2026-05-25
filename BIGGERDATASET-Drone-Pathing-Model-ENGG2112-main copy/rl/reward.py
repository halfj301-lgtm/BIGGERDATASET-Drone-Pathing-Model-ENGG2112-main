import numpy as np
from rl.config import (
    W_DIST, W_ELEV, W_POWER, COST_SCALE, SHAPING_SCALE,
    GOAL_BONUS, FAILURE_PENALTY, TIME_PENALTY,
    W_ALTITUDE, W_VALLEY,
    BASE_POWER_PER_M, CLIMB_POWER_PER_M, DEM_STEP,
)


class RewardFunction:
    """
    Modular reward function for drone pathfinding.

    To add a new cost term in the future:
      1. Add a weight constant to config.py
      2. Add the term inside compute() below
      3. Retrain — nothing else changes.
    """

    def __init__(
        self,
        w_dist=W_DIST,
        w_elev=W_ELEV,
        w_power=W_POWER,
        cost_scale=COST_SCALE,
        shaping_scale=SHAPING_SCALE,
        goal_bonus=GOAL_BONUS,
        failure_penalty=FAILURE_PENALTY,
        base_power=BASE_POWER_PER_M,
        climb_power=CLIMB_POWER_PER_M,
    ):
        self.w_dist          = w_dist
        self.w_elev          = w_elev
        self.w_power         = w_power
        self.cost_scale      = cost_scale
        self.shaping_scale   = shaping_scale
        self.goal_bonus      = goal_bonus
        self.failure_penalty = failure_penalty
        self.time_penalty    = TIME_PENALTY
        self.w_altitude      = W_ALTITUDE
        self.w_valley        = W_VALLEY
        self.base_power      = base_power
        self.climb_power     = climb_power

    # ── Public helpers (also used for evaluation metrics) ─────────────────────

    def step_cost(self, dist: float, elev_diff: float) -> float:
        """
        Full scoring formula including power — matches A* search cost exactly.
        elev_diff : signed elevation change (positive = climbing)
        """
        return (
            self.w_dist  * dist
            + self.w_elev  * abs(elev_diff)
            + self.w_power * self.power_usage(dist, elev_diff)
        )

    def power_usage(self, dist_grid: float, elev_diff: float) -> float:
        """
        Simple physics-inspired power model.
        dist_grid : movement distance in grid units (1.0 or 1.414)
        elev_diff : signed elevation change in metres (positive = climbing)
        Returns power in relative watt-like units.
        """
        dist_m = dist_grid * DEM_STEP
        climb  = max(0.0, elev_diff)          # only climbing costs extra
        return self.base_power * dist_m + self.climb_power * climb

    # ── Training reward ───────────────────────────────────────────────────────

    def compute(
        self,
        pos,
        new_pos,
        goal,
        dist: float,
        elev_diff: float,
        reached_goal: bool = False,
        failed: bool = False,
        current_elev: float = 0.0,
        local_patch_min: float = 0.0,
        elev_min: float = 0.0,
        elev_range: float = 1.0,
    ) -> float:
        """
        Returns the scalar reward for one environment step.

        Design:
          • Dense shaping bonus for moving closer to the goal.
          • Step cost penalty mirroring the A* scoring formula.
          • Altitude penalty — high absolute elevation = thin air = less efficient.
          • Valley bonus — being at local elevation minimum incentivises valley-following.
          • Time penalty — constant per-step cost to prevent hovering.
          • Large one-time bonus on reaching goal / penalty for going out-of-bounds.
        """
        if failed:
            return float(self.failure_penalty)

        # ── 1. Potential-based shaping ─────────────────────────────────────────
        old_d = np.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)
        new_d = np.sqrt((new_pos[0] - goal[0]) ** 2 + (new_pos[1] - goal[1]) ** 2)
        shaping = (old_d - new_d) * self.shaping_scale

        # ── 2. Step cost ───────────────────────────────────────────────────────
        cost = self.step_cost(dist, elev_diff) / self.cost_scale

        # ── 3. Altitude penalty (normalised 0→1 across DEM) ───────────────────
        norm_elev     = (current_elev - elev_min) / elev_range
        altitude_pen  = self.w_altitude * norm_elev

        # ── 4. Valley bonus (how close to local patch minimum) ────────────────
        valley_deficit = (current_elev - local_patch_min) / elev_range
        valley_bon     = self.w_valley * (1.0 - valley_deficit)

        reward = shaping - cost - altitude_pen + valley_bon + self.time_penalty

        # ── 5. Terminal signals ────────────────────────────────────────────────
        if reached_goal:
            reward += self.goal_bonus

        return float(reward)
