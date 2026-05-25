"""
Interactive endpoint picker — A* vs RL comparison.

Usage:
    python rl/compare.py
    python rl/compare.py --model models/checkpoint_2000000_steps.zip

1. A terrain map opens — click once for START (green), once for GOAL (purple).
2. A* and the RL model both run on those endpoints.
3. Both paths are overlaid on the map and metrics are printed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize

from stable_baselines3 import PPO

from rl.config import (
    MODELS_DIR, DEM_STEP, W_ELEV, W_POWER,
    BASE_POWER_PER_M, CLIMB_POWER_PER_M, CLEARANCE_M, SURFACE_STEP,
)
from rl.dem_loader import load_dem, subsample
from rl.astar import astar, path_metrics
from rl.reward import RewardFunction
from rl.environment import DronePathEnv
from rl.evaluate import run_rl

C_ASTAR = "#2196F3"
C_RL    = "#FF5722"
C_START = "#4CAF50"
C_GOAL  = "#9C27B0"


def snap_to_valid(row: float, col: float, elev: np.ndarray):
    """Snap a clicked (row, col) to the nearest non-NaN grid cell."""
    r = int(np.clip(round(row), 0, elev.shape[0] - 1))
    c = int(np.clip(round(col), 0, elev.shape[1] - 1))
    if not np.isnan(elev[r, c]):
        return (r, c)
    valid = np.argwhere(~np.isnan(elev))
    dists = np.abs(valid[:, 0] - r) + np.abs(valid[:, 1] - c)
    nearest = valid[np.argmin(dists)]
    return tuple(nearest)


def pick_points(elev_sub):
    """Show terrain map and collect two clicks — returns (start, goal)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        "Click START point, then GOAL point\n(close window when done)",
        fontsize=12
    )

    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(elev_sub, nan=np.nanmin(elev_sub)), vert_exag=2)
    ax.imshow(hillshade, cmap="gray", alpha=0.4)
    im = ax.imshow(elev_sub, cmap="terrain", alpha=0.7)
    fig.colorbar(im, ax=ax, label="Elevation (m AHD)", shrink=0.7)
    ax.contour(elev_sub, levels=25, colors="black", linewidths=0.2, alpha=0.25)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")

    markers = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        col, row = event.xdata, event.ydata
        pt = snap_to_valid(row, col, elev_sub)

        if len(markers) == 0:
            color, label = C_START, "Start"
        elif len(markers) == 1:
            color, label = C_GOAL, "Goal"
        else:
            return

        markers.append(pt)
        shape = "o" if len(markers) == 1 else "s"
        ax.plot(pt[1], pt[0], shape, color=color, markersize=12,
                label=label, zorder=10)
        ax.legend(loc="upper right", fontsize=10)
        fig.canvas.draw()
        print(f"  {label}: row={pt[0]}, col={pt[1]},  elev={elev_sub[pt]:.1f} m")

        if len(markers) == 2:
            fig.suptitle("Points selected — closing in 5 s …", fontsize=12)
            fig.canvas.draw()
            t = fig.canvas.new_timer(interval=5_000)
            t.add_callback(plt.close, fig)
            t.single_shot = True
            t.start()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(markers) < 2:
        sys.exit("Need two points — start and goal. Run again.")

    return markers[0], markers[1]


def plot_comparison(elev_sub, astar_path, rl_paths, start, goal):
    """Show 2-D and 3-D comparison. rl_paths is a list of paths."""
    multi = len(rl_paths) > 1

    # ── 2-D overlay ───────────────────────────────────────────────────────────
    fig2d, ax = plt.subplots(figsize=(12, 10))
    title = ("Path Variance — RL runs overlaid" if multi
             else "Path Comparison — A* vs RL (PPO)")
    fig2d.suptitle(title, fontsize=13)

    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(elev_sub, nan=np.nanmin(elev_sub)), vert_exag=2)
    ax.imshow(hillshade, cmap="gray", alpha=0.4)
    im = ax.imshow(elev_sub, cmap="terrain", alpha=0.7)
    fig2d.colorbar(im, ax=ax, label="Elevation (m AHD)", shrink=0.7)
    ax.contour(elev_sub, levels=25, colors="black", linewidths=0.2, alpha=0.25)

    if astar_path:
        rs = [p[0] for p in astar_path]
        cs = [p[1] for p in astar_path]
        ax.plot(cs, rs, color=C_ASTAR, linewidth=2.0, label="A* path", zorder=5)

    for i, rl_path in enumerate(rl_paths):
        rs = [p[0] for p in rl_path]
        cs = [p[1] for p in rl_path]
        alpha = 0.35 if multi else 1.0
        label = f"RL run {i+1}" if multi else "RL (PPO) path"
        ax.plot(cs, rs, color=C_RL, linewidth=1.5 if multi else 2.0,
                linestyle="--", alpha=alpha,
                label=label if (not multi or i == 0) else "_nolegend_", zorder=6)

    if multi:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color=C_ASTAR, linewidth=2, label="A* path"),
            Line2D([0], [0], color=C_RL, linewidth=1.5, linestyle="--",
                   alpha=0.6, label=f"RL paths (n={len(rl_paths)})"),
        ], loc="upper right", fontsize=10)
    else:
        ax.legend(loc="upper right", fontsize=10)

    ax.plot(start[1], start[0], "o", color=C_START, markersize=12,
            label="Start", zorder=7)
    ax.plot(goal[1],  goal[0],  "s", color=C_GOAL,  markersize=12,
            label="Goal",  zorder=7)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")

    # ── 3-D overlay ───────────────────────────────────────────────────────────
    surf           = elev_sub[::SURFACE_STEP, ::SURFACE_STEP].copy()
    rows_s, cols_s = surf.shape
    cell_m         = DEM_STEP * SURFACE_STEP
    X, Y           = np.meshgrid(np.arange(cols_s) * cell_m,
                                  np.arange(rows_s) * cell_m)
    norm      = Normalize(vmin=np.nanmin(surf), vmax=np.nanmax(surf))
    surf_rgba = plt.cm.terrain(norm(np.nan_to_num(surf, nan=np.nanmin(surf))))

    fig3d = plt.figure(figsize=(13, 9))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    fig3d.suptitle("3-D View — drag to rotate, scroll to zoom", fontsize=11)

    ax3d.plot_surface(X, Y, surf, facecolors=surf_rgba,
                      linewidth=0, antialiased=False, alpha=0.85, shade=True)

    def _xyz(path):
        return ([p[1]*DEM_STEP for p in path],
                [p[0]*DEM_STEP for p in path],
                [elev_sub[p] + CLEARANCE_M for p in path])

    if astar_path:
        px, py, pz = _xyz(astar_path)
        ax3d.plot(px, py, pz, color=C_ASTAR, linewidth=2,
                  label=f"A* (+{CLEARANCE_M}m)", zorder=10)

    for i, rl_path in enumerate(rl_paths):
        px, py, pz = _xyz(rl_path)
        alpha = 0.35 if multi else 1.0
        label = f"RL paths (n={len(rl_paths)})" if (multi and i == 0) else (
                "RL (PPO)" if not multi else "_nolegend_")
        ax3d.plot(px, py, pz, color=C_RL, linewidth=1.5 if multi else 2,
                  linestyle="--", alpha=alpha, label=label, zorder=11)

    ax3d.scatter([start[1]*DEM_STEP], [start[0]*DEM_STEP],
                 [elev_sub[start]+CLEARANCE_M],
                 color=C_START, s=120, depthshade=False, label="Start", zorder=12)
    ax3d.scatter([goal[1]*DEM_STEP], [goal[0]*DEM_STEP],
                 [elev_sub[goal]+CLEARANCE_M],
                 color=C_GOAL, s=120, marker="s", depthshade=False,
                 label="Goal", zorder=12)

    ax3d.set_xlabel("East (m)",          fontsize=9, labelpad=8)
    ax3d.set_ylabel("North (m)",         fontsize=9, labelpad=8)
    ax3d.set_zlabel("Elevation (m AHD)", fontsize=9, labelpad=8)
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.view_init(elev=45, azim=45)

    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])
    fig3d.colorbar(sm, ax=ax3d, shrink=0.45, aspect=15,
                   label="Elevation (m AHD)")

    plt.tight_layout()
    plt.show(block=False)
    deadline = time.time() + 60
    while plt.get_fignums() and time.time() < deadline:
        plt.pause(0.5)
    plt.close("all")


def main():
    def _latest_model():
        candidates = sorted(MODELS_DIR.glob("run_*/ppo_final.zip"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])
        return str(MODELS_DIR / "ppo_final.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to model .zip. Defaults to most recent run.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Run RL this many times (>1 uses stochastic sampling to show variance)")
    args = parser.parse_args()
    if args.model is None:
        args.model = _latest_model()
        print(f"  Using latest model: {args.model}")

    print("Loading terrain …")
    elevation = load_dem(verbose=False)
    elev_sub  = subsample(elevation)
    reward_fn = RewardFunction()

    print("\nClick START then GOAL on the map.")
    start, goal = pick_points(elev_sub)
    print(f"\nStart: {start}  →  Goal: {goal}")

    print("\n── Running A* ────────────────────────────────────────────────────")
    astar_path, astar_time, visited = astar(
        elev_sub, start, goal,
        slope_weight = W_ELEV,
        power_weight = W_POWER,
        base_power   = BASE_POWER_PER_M,
        climb_power  = CLIMB_POWER_PER_M,
        step_m       = DEM_STEP,
        verbose      = True,
    )
    if astar_path is None:
        print("A* found no path between these points.")

    print(f"\n── Loading RL model: {args.model} ───────────────────────────────")
    try:
        model = PPO.load(args.model)
    except FileNotFoundError:
        sys.exit(f"Model not found: {args.model}\nRun python -m rl.train first.")

    max_steps = int(len(astar_path) * 3) if astar_path else int(
        np.sqrt(elev_sub.shape[0]**2 + elev_sub.shape[1]**2) * 3
    )

    print("\n── Running RL inference ──────────────────────────────────────────")
    deterministic = args.runs == 1
    if not deterministic:
        print(f"  Stochastic mode — running {args.runs} times to show variance")

    rl_paths = []
    for i in range(args.runs):
        path, rl_time, rl_success = run_rl(model, elev_sub, start, goal,
                                            reward_fn, max_steps,
                                            deterministic=deterministic)
        rl_paths.append(path)
        print(f"  Run {i+1}/{args.runs}: {len(path)} steps, "
              f"{rl_time*1000:.1f} ms, goal reached: {rl_success}")

    rl_path = rl_paths[0]  # use first run for metrics

    if astar_path:
        astar_m = path_metrics(astar_path, elev_sub, reward_fn, step_m=DEM_STEP)
        rl_m    = path_metrics(rl_path,    elev_sub, reward_fn, step_m=DEM_STEP)
        print(f"\n{'Metric':<22} {'A*':>12} {'RL':>12}")
        print("-" * 48)
        for key in ["n_steps", "total_dist_m", "elev_gain_m", "total_cost", "power_usage"]:
            print(f"  {key:<20} {astar_m[key]:>12.1f} {rl_m[key]:>12.1f}")
        print(f"\n  Compute time      A*: {astar_time*1000:.0f} ms    "
              f"RL: {rl_time*1000:.1f} ms  "
              f"(RL is {astar_time/rl_time:.1f}× faster)")

    plot_comparison(elev_sub, astar_path, rl_paths, start, goal)


if __name__ == "__main__":
    main()
