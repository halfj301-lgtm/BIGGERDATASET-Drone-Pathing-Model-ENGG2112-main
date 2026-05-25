"""
Shared DEM loader.

Reproduces the exact same start/goal as pathfinder-3d.py (seed=42).
Both train.py and evaluate.py import from here so the DEM is only
loaded once per script and the start/goal are always consistent.
"""

import glob
import zipfile
import numpy as np
import rasterio
from rasterio.merge import merge

from rl.config import ZIP_PATH, EXTRACT_DIR, DEM_STEP, SEED


def load_dem(verbose=True) -> np.ndarray:
    """
    Extract zip (if needed), mosaic all .tif tiles, and return a
    float32 elevation grid with no-data cells replaced by NaN.
    """
    # Extract zip only if the output directory doesn't already exist
    if not EXTRACT_DIR.exists():
        if verbose:
            print(f"Extracting {ZIP_PATH.name} …")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

    if verbose:
        print("Loading DEM tiles …")

    tif_files = glob.glob(str(EXTRACT_DIR / "**" / "*.tif"), recursive=True)
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files found under {EXTRACT_DIR}. "
            "Check that DATA_1349498.zip extracts into koziousco_dem/."
        )

    datasets       = [rasterio.open(f) for f in tif_files]
    mosaic, _      = merge(datasets)
    elevation      = mosaic[0].astype(np.float32)
    elevation[elevation < -9000] = np.nan

    for ds in datasets:
        ds.close()

    if verbose:
        print(f"  DEM shape  : {elevation.shape[0]:,} × {elevation.shape[1]:,} px")
        print(f"  Elev range : {np.nanmin(elevation):.0f} m – {np.nanmax(elevation):.0f} m")

    return elevation


def subsample(elevation: np.ndarray, step: int = DEM_STEP) -> np.ndarray:
    """Return a subsampled copy of the elevation grid (1 cell = step metres)."""
    return elevation[::step, ::step].copy()


def get_start_goal(elev_sub: np.ndarray, seed: int = SEED):
    """
    Reproduce the same random start/goal as pathfinder-3d.py.
    Uses numpy default_rng(seed=42) and draws from valid (non-NaN) cells.
    """
    rng       = np.random.default_rng(seed=seed)
    valid     = np.argwhere(~np.isnan(elev_sub))
    idx       = rng.choice(len(valid), size=2, replace=False)
    start     = tuple(valid[idx[0]])
    goal      = tuple(valid[idx[1]])
    return start, goal


def get_endpoint_set(elev_sub: np.ndarray, seeds):
    """Return a list of (start, goal) pairs, one per seed."""
    return [get_start_goal(elev_sub, seed=s) for s in seeds]


def load_all(verbose=True):
    """Convenience wrapper: load DEM, subsample, return (elev_sub, start, goal)."""
    elevation = load_dem(verbose=verbose)
    elev_sub  = subsample(elevation)
    start, goal = get_start_goal(elev_sub)

    if verbose:
        print(f"  Subsampled : {elev_sub.shape[0]:,} × {elev_sub.shape[1]:,} "
              f"cells  ({DEM_STEP}m/cell)")
        print(f"  Start      : row={start[0]}, col={start[1]}  "
              f"elev={elev_sub[start]:.1f} m")
        print(f"  Goal       : row={goal[0]},  col={goal[1]}   "
              f"elev={elev_sub[goal]:.1f} m")

    return elev_sub, start, goal
