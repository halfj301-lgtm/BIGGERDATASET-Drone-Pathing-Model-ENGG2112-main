from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
ZIP_PATH    = BASE_DIR / "data" / "10X10DATA_1838653.zip"
EXTRACT_DIR = BASE_DIR / "data" / "koziousco_dem"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ── DEM settings (must match pathfinder-3d.py) ────────────────────────────────
DEM_STEP     = 4    # subsampling: 1 grid cell = 4 metres on the ground
CLEARANCE_M  = 50   # drone flies this many metres above terrain surface
SURFACE_STEP = 3    # extra downsampling for 3D surface render (keeps it fast)
SEED         = 42   # reproduces the same start/goal as pathfinder-3d.py

# ── Environment ───────────────────────────────────────────────────────────────
PATCH_SIZE           = 21    # local elevation context window (21×21 cells)
PATCH_SIZE_COARSE    = 21    # second elevation context window, sampled at coarse stride
COARSE_STRIDE        = 8     # cells between samples in the coarse patch (84m/cell at DEM_STEP=4 → ~672m radius)
MAX_STEPS_MULTIPLIER = 3     # episode time limit = this × A* path length (fixed mode)
N_ENVS               = 12    # parallel training environments (matches i7-12700K P-core count)
ENDPOINT_MODE        = "fixed_single"  # "fixed_single" | "fixed_set" | "random"
ENDPOINT_SEEDS       = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
N_ENDPOINT_PAIRS     = 10              # how many seeds from ENDPOINT_SEEDS to use

# ── Reward weights (all in one place — change here, nowhere else) ─────────────
W_DIST          = 1.0    # movement distance cost (mirrors A* dist)
W_ELEV          = 5.0    # elevation-change cost (mirrors A* slope_weight=5)
W_POWER         = 0.3    # power-usage penalty weight
COST_SCALE      = 100.0  # divides raw step cost to keep rewards in [-1, +1] range
SHAPING_SCALE   = 0.2    # scales the distance-to-goal shaping bonus
                          # Raised from 0.05 in Step 6 — see docs/rationale.md.
GOAL_BONUS      = 50.0   # one-time reward for reaching the goal cell
FAILURE_PENALTY = -50.0  # reward for going out-of-bounds or into a no-data cell
TIME_PENALTY    = -0.01  # constant per-step penalty — keeps agent searching, breaks local optima
                          # Reduced from -0.05 in Step 6 so it doesn't drown shaping.
W_ALTITUDE      = 0.0    # penalty for high absolute elevation (thin air = less efficient)
                          # Disabled in Step 6 — see docs/rationale.md. Was poisoning the reward
                          # landscape (paid every step regardless of direction).
W_VALLEY        = 0.0    # bonus for being at local elevation minimum (valley-following)
                          # Disabled — see docs/rationale.md (Step 2). The bonus dominated
                          # shaping and step-cost signals, producing hovering exploits.
                          # Set back to 0.3 to reproduce the pre-rework reward.

# ── Power model ───────────────────────────────────────────────────────────────
BASE_POWER_PER_M  = 0.1   # baseline power per metre of horizontal travel (W/m)
CLIMB_POWER_PER_M = 0.5   # extra power per metre of altitude gained (W/m)
# Descent saves no power for now; add a DESCENT_SAVING constant here later.

# ── PPO hyperparameters ───────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000  # Phase 1 fixed endpoints
PPO_N_STEPS     = 4096        # steps collected per env before each update
PPO_BATCH_SIZE  = 16384       # larger minibatch = GPU works harder per update
PPO_N_EPOCHS    = 10
PPO_LR          = 1e-4
PPO_GAMMA       = 0.99        # discount factor
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP_RANGE  = 0.1
PPO_HIDDEN      = [256, 256]  # actor + critic hidden layer sizes

# ── Callbacks ─────────────────────────────────────────────────────────────────
CHECKPOINT_FREQ = 200_000   # save model checkpoint every N timesteps
PLOT_FREQ       = 50_000    # redraw and save training-curve PNG every N timesteps

# ── Early stopping ─────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 15         # consecutive checks with no improvement → stop
EARLY_STOP_MIN_DELTA = 5.0        # minimum reward improvement to reset patience counter
EARLY_STOP_MIN_STEPS = 1_000_000  # never stop before this many timesteps

# ── Animation ─────────────────────────────────────────────────────────────────
ANIM_TARGET_FRAMES = 300    # subsample path to ≈ this many animation frames
ANIM_FPS           = 30
