import heapq
import time
import numpy as np


# 8-connected neighbourhood: (row_offset, col_offset, move_distance)
_NEIGHBOURS = [
    (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    ( 0, -1, 1.0),                 ( 0, 1, 1.0),
    ( 1, -1, 1.414), ( 1, 0, 1.0), ( 1, 1, 1.414),
]


def _heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(elev, start, goal, slope_weight=5.0,
          power_weight=0.0, base_power=0.0, climb_power=0.0, step_m=4,
          verbose=True):
    """
    A* pathfinder on a 2-D elevation grid.

    Parameters
    ----------
    elev          : 2-D float32/float64 array, NaN = impassable
    start, goal   : (row, col) tuples in elev coordinate space
    slope_weight  : penalty per unit of absolute elevation change (W_ELEV)
    power_weight  : weight on power term (W_POWER); 0 = disabled
    base_power    : baseline power per metre of horizontal travel
    climb_power   : extra power per metre of altitude gained
    step_m        : metres per grid cell (DEM_STEP)
    verbose       : print progress every 50 k nodes

    Returns
    -------
    path      : list of (row, col) from start to goal, or None
    duration  : wall-clock seconds the search took
    n_visited : number of nodes expanded
    """
    rows, cols = elev.shape
    open_heap  = []
    heapq.heappush(open_heap, (0.0, start))

    came_from = {}
    g_score   = {start: 0.0}
    visited   = 0
    t0        = time.perf_counter()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            duration = time.perf_counter() - t0
            if verbose:
                print(f"  A* found path: {len(path)} steps, "
                      f"{visited:,} nodes visited, {duration:.3f}s")
            return path, duration, visited

        visited += 1
        if verbose and visited % 50_000 == 0:
            print(f"  A* searching … {visited:,} nodes visited")

        r, c = current
        for dr, dc, dist in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if np.isnan(elev[nr, nc]):
                continue

            elev_signed = float(elev[nr, nc]) - float(elev[r, c])
            climb       = max(0.0, elev_signed)
            power       = power_weight * (base_power * dist * step_m + climb_power * climb)
            step_cost   = dist + slope_weight * abs(elev_signed) + power
            tentative_g = g_score[current] + step_cost

            neighbour = (nr, nc)
            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g
                f = tentative_g + _heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f, neighbour))

    duration = time.perf_counter() - t0
    if verbose:
        print("  A* found no path.")
    return None, duration, visited


def path_metrics(path, elev, reward_fn, step_m=4):
    """
    Compute evaluation metrics for any path (A* or RL).

    Returns a dict with:
      total_dist_m   : total ground distance in metres
      elev_gain_m    : cumulative positive elevation gain in metres
      total_cost     : sum of A* step costs (dist + slope_weight * |elev_diff|)
      power_usage    : cumulative power in reward_fn units
      n_steps        : number of path steps
    """
    if path is None or len(path) < 2:
        return dict(total_dist_m=0, elev_gain_m=0,
                    total_cost=0, power_usage=0, n_steps=0)

    total_dist  = 0.0
    elev_gain   = 0.0
    total_cost  = 0.0
    total_power = 0.0

    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        dr = r1 - r0
        dc = c1 - c0
        dist      = np.sqrt(dr ** 2 + dc ** 2)
        elev_diff = float(elev[r1, c1]) - float(elev[r0, c0])

        total_dist  += dist * step_m
        if elev_diff > 0:
            elev_gain += elev_diff
        total_cost  += reward_fn.step_cost(dist, elev_diff)  # signed — power uses climb only
        total_power += reward_fn.power_usage(dist, elev_diff)

    return dict(
        total_dist_m = total_dist,
        elev_gain_m  = elev_gain,
        total_cost   = total_cost,
        power_usage  = total_power,
        n_steps      = len(path) - 1,
    )
