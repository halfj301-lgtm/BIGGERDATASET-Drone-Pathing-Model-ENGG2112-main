"""
All visualisation helpers used by evaluate.py.

Each function saves one figure to disk and closes cleanly.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.animation import FuncAnimation

from rl.config import DEM_STEP, CLEARANCE_M, SURFACE_STEP, ANIM_TARGET_FRAMES, ANIM_FPS


# ── Colour palette ────────────────────────────────────────────────────────────
C_ASTAR = "#2196F3"    # blue
C_RL    = "#FF5722"    # deep orange
C_START = "#4CAF50"    # green
C_GOAL  = "#9C27B0"    # purple


# ── 1. 2-D path overlay ───────────────────────────────────────────────────────

def plot_paths_2d(elev_sub, astar_path, rl_path, start, goal, save_path):
    """Hillshade terrain map with A* and RL paths overlaid."""
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_axes([0.05, 0.05, 0.80, 0.90])
    cb_ax = fig.add_axes([0.87, 0.05, 0.02, 0.90])

    ls        = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(elev_sub, nan=np.nanmin(elev_sub)),
                             vert_exag=2)
    ax.imshow(hillshade, cmap="gray", alpha=0.4)
    im = ax.imshow(elev_sub, cmap="terrain", alpha=0.7)
    fig.colorbar(im, cax=cb_ax, label="Elevation (m AHD)")
    ax.contour(elev_sub, levels=25, colors="black", linewidths=0.2, alpha=0.25)

    # Paths
    if astar_path:
        rs = [p[0] for p in astar_path]
        cs = [p[1] for p in astar_path]
        ax.plot(cs, rs, color=C_ASTAR, linewidth=2.0, label="A* path", zorder=5)

    if rl_path:
        rs = [p[0] for p in rl_path]
        cs = [p[1] for p in rl_path]
        ax.plot(cs, rs, color=C_RL, linewidth=2.0, label="RL (PPO) path",
                zorder=6, linestyle="--")

    # Markers
    ax.plot(start[1], start[0], "o", color=C_START, markersize=12,
            label="Start", zorder=7)
    ax.plot(goal[1],  goal[0],  "s", color=C_GOAL,  markersize=12,
            label="Goal",  zorder=7)

    ax.set_title("Path Comparison — A* vs PPO Reinforcement Learning", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Grid column"); ax.set_ylabel("Grid row")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── 2. Elevation profiles ─────────────────────────────────────────────────────

def _profile(path, elev_sub, step_m):
    """Return (distances_m, elevations_m) arrays for a path."""
    if not path or len(path) < 2:
        return np.array([0.0]), np.array([float(elev_sub[path[0]])] if path else [0.0])
    elevs = [float(elev_sub[p]) for p in path]
    dists = np.cumsum(
        [0.0] + [
            np.sqrt((path[i+1][0] - path[i][0])**2 +
                    (path[i+1][1] - path[i][1])**2) * step_m
            for i in range(len(path) - 1)
        ]
    )
    return dists, np.array(elevs)


def plot_elevation_profiles(elev_sub, astar_path, rl_path, step_m=DEM_STEP,
                             save_path=None):
    fig, ax = plt.subplots(figsize=(13, 5))

    if astar_path:
        d, e = _profile(astar_path, elev_sub, step_m)
        ax.plot(d, e, color=C_ASTAR, linewidth=2, label="A* path")
        ax.fill_between(d, e, alpha=0.15, color=C_ASTAR)

    if rl_path:
        d, e = _profile(rl_path, elev_sub, step_m)
        ax.plot(d, e, color=C_RL, linewidth=2, label="RL (PPO) path",
                linestyle="--")
        ax.fill_between(d, e, alpha=0.15, color=C_RL)

    ax.set_xlabel("Distance along path (m)", fontsize=11)
    ax.set_ylabel("Elevation (m AHD)", fontsize=11)
    ax.set_title("Elevation Profile Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ── 3. Metrics bar chart ──────────────────────────────────────────────────────

def plot_metrics_bar(astar_m, rl_m, astar_time, rl_time, save_path=None):
    labels = [
        "Path steps", "Distance (m)",
        "Elev gain (m)", "A* path cost",
        "Power usage", "Compute time (ms)",
    ]
    astar_vals = [
        astar_m["n_steps"],
        astar_m["total_dist_m"],
        astar_m["elev_gain_m"],
        astar_m["total_cost"],
        astar_m["power_usage"],
        astar_time * 1000,
    ]
    rl_vals = [
        rl_m["n_steps"],
        rl_m["total_dist_m"],
        rl_m["elev_gain_m"],
        rl_m["total_cost"],
        rl_m["power_usage"],
        rl_time * 1000,
    ]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_a = ax.bar(x - width/2, astar_vals, width, label="A*",       color=C_ASTAR, alpha=0.85)
    bars_r = ax.bar(x + width/2, rl_vals,    width, label="RL (PPO)", color=C_RL,    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("A* vs PPO-RL — Key Metrics Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bar in [*bars_a, *bars_r]:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.01,
            f"{h:.1f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ── 4. 3-D static view ────────────────────────────────────────────────────────

def plot_paths_3d(elev_sub, astar_path, rl_path, start, goal,
                  clearance_m=CLEARANCE_M, save_path=None):
    surf       = elev_sub[::SURFACE_STEP, ::SURFACE_STEP].copy()
    rows_s, cols_s = surf.shape
    cell_m     = DEM_STEP * SURFACE_STEP
    X          = np.arange(cols_s) * cell_m
    Y          = np.arange(rows_s) * cell_m
    X, Y       = np.meshgrid(X, Y)

    norm       = Normalize(vmin=np.nanmin(surf), vmax=np.nanmax(surf))
    surf_rgba  = plt.cm.terrain(norm(np.nan_to_num(surf, nan=np.nanmin(surf))))

    fig  = plt.figure(figsize=(14, 9))
    ax3d = fig.add_subplot(111, projection="3d")

    ax3d.plot_surface(X, Y, surf, facecolors=surf_rgba,
                      linewidth=0, antialiased=False, alpha=0.85, shade=True)

    def _path_xyz(path, offset=clearance_m):
        px = [p[1] * DEM_STEP for p in path]
        py = [p[0] * DEM_STEP for p in path]
        pz = [elev_sub[p] + offset for p in path]
        return px, py, pz

    if astar_path:
        px, py, pz = _path_xyz(astar_path)
        ax3d.plot(px, py, pz, color=C_ASTAR, linewidth=2.0, zorder=10,
                  label=f"A* path (+{clearance_m}m)")

    if rl_path:
        px, py, pz = _path_xyz(rl_path)
        ax3d.plot(px, py, pz, color=C_RL, linewidth=2.0, zorder=11,
                  label="RL (PPO) path", linestyle="--")

    # Start / goal markers
    ax3d.scatter([start[1]*DEM_STEP], [start[0]*DEM_STEP],
                 [elev_sub[start] + clearance_m],
                 color=C_START, s=120, zorder=12, label="Start", depthshade=False)
    ax3d.scatter([goal[1]*DEM_STEP], [goal[0]*DEM_STEP],
                 [elev_sub[goal] + clearance_m],
                 color=C_GOAL, s=120, marker="s", zorder=12, label="Goal", depthshade=False)

    ax3d.set_xlabel("East (m)", fontsize=9, labelpad=8)
    ax3d.set_ylabel("North (m)", fontsize=9, labelpad=8)
    ax3d.set_zlabel("Elevation (m AHD)", fontsize=9, labelpad=8)
    ax3d.set_title("3-D Terrain View — A* vs RL Path", fontsize=12)
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.view_init(elev=45, azim=45)

    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.45, aspect=15, label="Elevation (m AHD)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ── 5. 3-D step-by-step animation ────────────────────────────────────────────

def animate_rl_path_3d(elev_sub, rl_path, astar_path, start, goal,
                        clearance_m=CLEARANCE_M, save_path=None):
    """
    Renders an MP4 animation of the RL drone flying its path over the 3-D
    terrain, with the A* path shown as a static blue reference.

    Requires ffmpeg. Falls back to GIF (via Pillow) if ffmpeg is unavailable.
    """
    if not rl_path or len(rl_path) < 2:
        print("  [animation] RL path is empty — skipping.")
        return

    # ── Build surface ─────────────────────────────────────────────────────────
    surf       = elev_sub[::SURFACE_STEP, ::SURFACE_STEP].copy()
    rows_s, cols_s = surf.shape
    cell_m     = DEM_STEP * SURFACE_STEP
    X          = np.arange(cols_s) * cell_m
    Y          = np.arange(rows_s) * cell_m
    X, Y       = np.meshgrid(X, Y)
    norm       = Normalize(vmin=np.nanmin(surf), vmax=np.nanmax(surf))
    surf_rgba  = plt.cm.terrain(norm(np.nan_to_num(surf, nan=np.nanmin(surf))))

    # ── Pre-compute RL path in world coords ───────────────────────────────────
    rl_x = np.array([p[1] * DEM_STEP for p in rl_path], dtype=float)
    rl_y = np.array([p[0] * DEM_STEP for p in rl_path], dtype=float)
    rl_z = np.array([elev_sub[p] + clearance_m for p in rl_path], dtype=float)

    # ── Subsample to target frame count ───────────────────────────────────────
    n     = len(rl_path)
    step  = max(1, n // ANIM_TARGET_FRAMES)
    idxs  = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    # ── Set up figure ─────────────────────────────────────────────────────────
    fig  = plt.figure(figsize=(13, 8))
    ax3d = fig.add_subplot(111, projection="3d")

    ax3d.plot_surface(X, Y, surf, facecolors=surf_rgba,
                      linewidth=0, antialiased=False, alpha=0.80, shade=True)

    # Static A* reference
    if astar_path:
        ax = [p[1] * DEM_STEP for p in astar_path]
        ay = [p[0] * DEM_STEP for p in astar_path]
        az = [elev_sub[p] + clearance_m for p in astar_path]
        ax3d.plot(ax, ay, az, color=C_ASTAR, linewidth=1.5,
                  alpha=0.6, label="A* reference", zorder=5)

    # Start / goal markers (static)
    ax3d.scatter([start[1]*DEM_STEP], [start[0]*DEM_STEP],
                 [elev_sub[start] + clearance_m],
                 color=C_START, s=100, zorder=20, depthshade=False)
    ax3d.scatter([goal[1]*DEM_STEP], [goal[0]*DEM_STEP],
                 [elev_sub[goal] + clearance_m],
                 color=C_GOAL, s=100, marker="s", zorder=20, depthshade=False)

    # Dynamic RL trail and drone marker
    rl_trail,  = ax3d.plot([], [], [], color=C_RL, linewidth=2.0,
                           zorder=10, label="RL path")
    rl_dot,    = ax3d.plot([], [], [], "o", color=C_RL, markersize=8,
                           zorder=11)

    ax3d.set_xlabel("East (m)",  fontsize=9, labelpad=6)
    ax3d.set_ylabel("North (m)", fontsize=9, labelpad=6)
    ax3d.set_zlabel("Elevation (m AHD)", fontsize=9, labelpad=6)
    ax3d.legend(loc="upper left", fontsize=9)
    ax3d.view_init(elev=45, azim=45)

    title_obj = ax3d.set_title("", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.45, aspect=15, label="Elevation (m AHD)")

    # ── Animation update function ─────────────────────────────────────────────
    def update(frame_idx):
        idx = idxs[frame_idx]
        rl_trail.set_data(rl_x[:idx+1], rl_y[:idx+1])
        rl_trail.set_3d_properties(rl_z[:idx+1])
        rl_dot.set_data([rl_x[idx]], [rl_y[idx]])
        rl_dot.set_3d_properties([rl_z[idx]])
        pct = idx / max(n - 1, 1) * 100
        title_obj.set_text(
            f"RL Drone Path  —  step {idx}/{n-1}  ({pct:.0f}%)"
        )
        return rl_trail, rl_dot, title_obj

    anim = FuncAnimation(
        fig, update,
        frames=len(idxs),
        interval=1000 / ANIM_FPS,
        blit=False,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    saved = False
    if save_path.endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=ANIM_FPS, bitrate=1800)
            anim.save(save_path, writer=writer, dpi=100)
            saved = True
        except Exception as e:
            warnings.warn(f"ffmpeg unavailable ({e}); falling back to GIF.")

    if not saved:
        gif_path = save_path.replace(".mp4", ".gif")
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=ANIM_FPS)
            anim.save(gif_path, writer=writer, dpi=80)
            save_path = gif_path
        except Exception as e:
            print(f"  [animation] Could not save animation: {e}")
            plt.close()
            return

    plt.close()
    print(f"  Saved → {save_path}")
