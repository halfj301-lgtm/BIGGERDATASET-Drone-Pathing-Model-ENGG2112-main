"""
Compare the trained PPO agent against the A* baseline.

Usage:
    python -m rl.evaluate
    python -m rl.evaluate --model models/checkpoint_2000000_steps.zip

Outputs (written to results/<run_name>/ derived from the model path):
    report_paths_2d.png      — 2-D path overlay on terrain map
    report_elevation.png     — elevation profile comparison
    report_metrics.png       — bar-chart of all key metrics
    report_3d.png            — 3-D terrain view with both paths
    animation_rl.mp4         — step-by-step 3-D animation of RL path
    metrics.txt              — plain-text summary for copy-pasting
    astar_path.npy           — saved path arrays (loaded by view3d.py)
    rl_path.npy
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO

from rl.config import (MODELS_DIR, RESULTS_DIR, W_ELEV, W_POWER,
                       BASE_POWER_PER_M, CLIMB_POWER_PER_M, DEM_STEP, CLEARANCE_M)
from rl.dem_loader import load_all
from rl.astar import astar, path_metrics
from rl.reward import RewardFunction
from rl.environment import DronePathEnv
from rl import visualize


# ── RL inference ──────────────────────────────────────────────────────────────

def run_rl(model, elev_sub, start, goal, reward_fn, max_steps, deterministic=False):
    """
    Run the trained PPO policy from start to goal.
    Returns (path, duration_s, success).
    """
    env = DronePathEnv(elev_sub, start, goal, reward_fn, max_steps)
    obs, _ = env.reset()

    path    = [start]
    done    = False
    t0      = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(int(action))
        path.append(env._pos)
        done = terminated or truncated

    duration = time.perf_counter() - t0
    success  = (env._pos == goal)
    return path, duration, success


# ── Metrics summary text ───────────────────────────────────────────────────────

def _fmt(label, astar_val, rl_val, unit="", fmt=".1f"):
    delta = ((rl_val - astar_val) / astar_val * 100) if astar_val != 0 else 0
    sign  = "+" if delta >= 0 else ""
    return (f"  {label:<22} A*: {astar_val:{fmt}} {unit}   "
            f"RL: {rl_val:{fmt}} {unit}   ({sign}{delta:.1f}%)")


def print_and_save_metrics(astar_m, rl_m, astar_time, rl_time, rl_success, out_path):
    lines = [
        "=" * 62,
        "  A* vs PPO-RL  —  Katoomba Terrain Path Comparison",
        "=" * 62,
        "",
        _fmt("Path steps",      astar_m["n_steps"],      rl_m["n_steps"],      fmt="d"),
        _fmt("Distance (m)",    astar_m["total_dist_m"], rl_m["total_dist_m"], unit="m"),
        _fmt("Elevation gain",  astar_m["elev_gain_m"],  rl_m["elev_gain_m"],  unit="m"),
        _fmt("A* path cost",    astar_m["total_cost"],   rl_m["total_cost"]),
        _fmt("Power usage",     astar_m["power_usage"],  rl_m["power_usage"],  fmt=".3f"),
        "",
        f"  {'Compute time':<22} A*: {astar_time*1000:.1f} ms   "
        f"RL: {rl_time*1000:.1f} ms   "
        f"(RL is {astar_time/rl_time:.1f}× faster)" if rl_time > 0 else "",
        f"  RL goal reached:       {'YES ✓' if rl_success else 'NO ✗ (path truncated)'}",
        "",
        "=" * 62,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def _latest_model():
    """Return path to the most recently saved ppo_final.zip under models/."""
    candidates = sorted(MODELS_DIR.glob("run_*/ppo_final.zip"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])
    legacy = MODELS_DIR / "ppo_final.zip"
    return str(legacy)


def main():
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a saved SB3 model (.zip). Defaults to the most recent run.",
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = _latest_model()
        print(f"  Using latest model: {args.model}")

    # Derive output folder from the model's run directory
    parent_name = Path(args.model).parent.name
    if parent_name.startswith("run_"):
        eval_dir = RESULTS_DIR / parent_name
    else:
        from datetime import datetime
        eval_dir = RESULTS_DIR / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results folder: {eval_dir}/")

    # ── 1. Terrain + start/goal ────────────────────────────────────────────────
    print("\n── Loading terrain data ──────────────────────────────────────────")
    elev_sub, start, goal = load_all(verbose=True)
    reward_fn = RewardFunction()

    # ── 2. A* baseline ────────────────────────────────────────────────────────
    print("\n── Running A* baseline ───────────────────────────────────────────")
    astar_path, astar_time, _ = astar(
        elev_sub, start, goal,
        slope_weight = W_ELEV,
        power_weight = W_POWER,
        base_power   = BASE_POWER_PER_M,
        climb_power  = CLIMB_POWER_PER_M,
        step_m       = DEM_STEP,
        verbose      = True,
    )
    if astar_path is None:
        sys.exit("A* found no path — aborting.")

    astar_m = path_metrics(astar_path, elev_sub, reward_fn, step_m=DEM_STEP)

    # ── 3. Load RL model ──────────────────────────────────────────────────────
    print(f"\n── Loading RL model: {args.model} ────────────────────────────────")
    try:
        model = PPO.load(args.model)
    except FileNotFoundError:
        sys.exit(f"Model not found: {args.model}\nRun  python -m rl.train  first.")

    max_steps = int(len(astar_path) * 3)

    # ── 4. RL inference ───────────────────────────────────────────────────────
    print("\n── Running RL inference ──────────────────────────────────────────")
    rl_path, rl_time, rl_success = run_rl(
        model, elev_sub, start, goal, reward_fn, max_steps
    )
    rl_m = path_metrics(rl_path, elev_sub, reward_fn, step_m=DEM_STEP)
    print(f"  RL path: {len(rl_path)} steps, {rl_time*1000:.1f} ms, "
          f"goal reached: {rl_success}")

    # ── 5. Save paths for interactive viewer ─────────────────────────────────
    np.save(str(eval_dir / "astar_path.npy"), np.array(astar_path))
    np.save(str(eval_dir / "rl_path.npy"),    np.array(rl_path))

    # ── 6. Print + save metrics text ──────────────────────────────────────────
    print_and_save_metrics(
        astar_m, rl_m, astar_time, rl_time, rl_success,
        str(eval_dir / "metrics.txt"),
    )

    # ── 7. Generate report figures ────────────────────────────────────────────
    print("\n── Generating report figures ─────────────────────────────────────")

    visualize.plot_paths_2d(
        elev_sub, astar_path, rl_path, start, goal,
        save_path=str(eval_dir / "report_paths_2d.png"),
    )

    visualize.plot_elevation_profiles(
        elev_sub, astar_path, rl_path,
        step_m=DEM_STEP,
        save_path=str(eval_dir / "report_elevation.png"),
    )

    visualize.plot_metrics_bar(
        astar_m, rl_m, astar_time, rl_time,
        save_path=str(eval_dir / "report_metrics.png"),
    )

    visualize.plot_paths_3d(
        elev_sub, astar_path, rl_path, start, goal,
        clearance_m=CLEARANCE_M,
        save_path=str(eval_dir / "report_3d.png"),
    )

    print("\n── Generating 3-D animation (this may take a minute) ─────────────")
    visualize.animate_rl_path_3d(
        elev_sub, rl_path, astar_path, start, goal,
        clearance_m=CLEARANCE_M,
        save_path=str(eval_dir / "animation_rl.mp4"),
    )

    print(f"\n── All outputs saved to  {eval_dir}/")
    print("   report_paths_2d.png  |  report_elevation.png  |  report_metrics.png")
    print("   report_3d.png        |  animation_rl.mp4       |  metrics.txt")


if __name__ == "__main__":
    main()
