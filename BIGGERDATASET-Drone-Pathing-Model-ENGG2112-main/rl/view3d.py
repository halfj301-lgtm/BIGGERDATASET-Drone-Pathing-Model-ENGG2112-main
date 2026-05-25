"""
Interactive 3-D path viewer.

Usage:
    python view3d.py

Loads the saved paths from results/ and opens an interactive matplotlib
window — drag to rotate, scroll to zoom.

Run  python -m rl.evaluate  first to generate the path files.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rl.config import DEM_STEP, SURFACE_STEP, CLEARANCE_M, RESULTS_DIR
from rl.dem_loader import load_all

C_ASTAR = "#2196F3"
C_RL    = "#FF5722"
C_START = "#4CAF50"
C_GOAL  = "#9C27B0"


def _latest_eval_dir():
    """Return the most recently modified results/run_*/ folder, or RESULTS_DIR as fallback."""
    candidates = sorted(RESULTS_DIR.glob("run_*/astar_path.npy"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].parent
    return RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=None,
        help="Results folder to load paths from (e.g. results/run_20241215_123456). "
             "Defaults to the most recently evaluated run.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results) if args.results else _latest_eval_dir()

    # ── Load paths ────────────────────────────────────────────────────────────
    astar_file = results_dir / "astar_path.npy"
    rl_file    = results_dir / "rl_path.npy"

    if not astar_file.exists() or not rl_file.exists():
        sys.exit(
            f"Path files not found in {results_dir}/\n"
            "Run  python -m rl.evaluate  first."
        )

    astar_path = [tuple(p) for p in np.load(str(astar_file))]
    rl_path    = [tuple(p) for p in np.load(str(rl_file))]

    # ── Load terrain ──────────────────────────────────────────────────────────
    print("Loading terrain …")
    elev_sub, start, goal = load_all(verbose=False)

    # ── Build surface grid ────────────────────────────────────────────────────
    surf           = elev_sub[::SURFACE_STEP, ::SURFACE_STEP].copy()
    rows_s, cols_s = surf.shape
    cell_m         = DEM_STEP * SURFACE_STEP
    X              = np.arange(cols_s) * cell_m
    Y              = np.arange(rows_s) * cell_m
    X, Y           = np.meshgrid(X, Y)

    norm      = Normalize(vmin=np.nanmin(surf), vmax=np.nanmax(surf))
    surf_rgba = plt.cm.terrain(norm(np.nan_to_num(surf, nan=np.nanmin(surf))))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig  = plt.figure(figsize=(14, 9))
    ax   = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, surf, facecolors=surf_rgba,
                    linewidth=0, antialiased=False, alpha=0.85, shade=True)

    def _xyz(path):
        px = [p[1] * DEM_STEP for p in path]
        py = [p[0] * DEM_STEP for p in path]
        pz = [elev_sub[p] + CLEARANCE_M for p in path]
        return px, py, pz

    if astar_path:
        px, py, pz = _xyz(astar_path)
        ax.plot(px, py, pz, color=C_ASTAR, linewidth=2.0,
                label=f"A* path (+{CLEARANCE_M}m clearance)", zorder=10)

    if rl_path:
        px, py, pz = _xyz(rl_path)
        ax.plot(px, py, pz, color=C_RL, linewidth=2.0,
                label="RL (PPO) path", linestyle="--", zorder=11)

    ax.scatter([start[1]*DEM_STEP], [start[0]*DEM_STEP],
               [elev_sub[start] + CLEARANCE_M],
               color=C_START, s=120, zorder=12, label="Start", depthshade=False)
    ax.scatter([goal[1]*DEM_STEP], [goal[0]*DEM_STEP],
               [elev_sub[goal] + CLEARANCE_M],
               color=C_GOAL, s=120, marker="s", zorder=12, label="Goal", depthshade=False)

    ax.set_xlabel("East (m)",          fontsize=9, labelpad=8)
    ax.set_ylabel("North (m)",         fontsize=9, labelpad=8)
    ax.set_zlabel("Elevation (m AHD)", fontsize=9, labelpad=8)
    ax.set_title("3-D Terrain View — A* vs RL Path  (drag to rotate, scroll to zoom)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=45, azim=45)

    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, aspect=15, label="Elevation (m AHD)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
