"""
Train a PPO agent to fly from start to goal over the Katoomba DEM.

Usage:
    python -m rl.train

Outputs (all written to results/ and models/):
    models/checkpoint_<N>_steps.zip   — periodic checkpoints
    models/ppo_final.zip              — final trained model
    results/training_curve.png        — live-updating reward/length plot
    results/tensorboard/              — TensorBoard logs (run: tensorboard --logdir results/tensorboard)
"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive: safe for background training
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn

from rl.config import (
    MODELS_DIR, RESULTS_DIR,
    N_ENVS, TOTAL_TIMESTEPS,
    PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS, PPO_LR,
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_RANGE, PPO_HIDDEN,
    CHECKPOINT_FREQ, PLOT_FREQ,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, EARLY_STOP_MIN_STEPS,
    W_ELEV, W_POWER, BASE_POWER_PER_M, CLIMB_POWER_PER_M, DEM_STEP,
    ENDPOINT_MODE, ENDPOINT_SEEDS, N_ENDPOINT_PAIRS,
)
from rl.dem_loader import load_all, get_endpoint_set
from rl.astar import astar
from rl.reward import RewardFunction
from rl.environment import DronePathEnv
from rl.policy import DronePatchCNN


# ── Resume training learning rate (catastrophic-forgetting safeguard) ────────
RESUME_LR = 5e-5


# ── Callback: live training-curve PNG ─────────────────────────────────────────

class TrainingPlotCallback(BaseCallback):
    """Saves a training-curve PNG every PLOT_FREQ timesteps."""

    def __init__(self, results_dir, plot_freq=PLOT_FREQ, verbose=0):
        super().__init__(verbose)
        self.results_dir   = results_dir
        self.plot_freq     = plot_freq
        self._ep_rewards   = []
        self._ep_lengths   = []
        self._ep_outcomes  = []
        self._ep_successes = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(float(ep["r"]))
                self._ep_lengths.append(int(ep["l"]))
                outcome = info.get("outcome", "unknown")
                self._ep_outcomes.append(outcome)

        if self.n_calls % self.plot_freq == 0 and len(self._ep_rewards) > 10:
            self._save_plot()
        return True

    def _save_plot(self):
        rewards = np.array(self._ep_rewards)
        lengths = np.array(self._ep_lengths)
        outcomes = np.array(self._ep_outcomes)
        window  = min(100, len(rewards))

        # Compute success rate (binary: 1 if "success", else 0)
        successes = (outcomes == "success").astype(float)
        roll_s = np.convolve(successes, np.ones(window) / window, mode="valid")
        latest_rate = roll_s[-1] if len(roll_s) > 0 else 0.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(
            f"PPO Training  —  {self.num_timesteps:,} / {TOTAL_TIMESTEPS:,} timesteps  —  success {latest_rate:.1%}",
            fontsize=13,
        )

        # Reward curve
        roll_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(rewards, alpha=0.25, color="steelblue", linewidth=0.6)
        ax1.plot(range(window - 1, len(rewards)), roll_r,
                 color="steelblue", linewidth=2,
                 label=f"{window}-ep rolling mean")
        ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax1.set_ylabel("Episode Reward", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Episode-length curve (shorter = agent reaches goal faster)
        roll_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax2.plot(lengths, alpha=0.25, color="darkorange", linewidth=0.6)
        ax2.plot(range(window - 1, len(lengths)), roll_l,
                 color="darkorange", linewidth=2)
        ax2.set_ylabel("Episode Length (steps)", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Success rate curve
        ax3.plot(range(window - 1, len(successes)), roll_s,
                 color="darkgreen", linewidth=2,
                 label=f"{window}-ep rolling mean")
        ax3.axhline(1.0, color="lightgray", linewidth=0.8, linestyle=":")
        ax3.set_ylabel("Success rate", fontsize=11)
        ax3.set_xlabel("Episode", fontsize=11)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)

        plt.tight_layout()
        out = os.path.join(self.results_dir, "training_curve.png")
        plt.savefig(out, dpi=100)
        plt.close()
        if self.verbose > 0:
            print(f"  [plot] saved → {out}")


# ── Callback: early stopping on reward plateau ────────────────────────────────

class EarlyStoppingCallback(BaseCallback):
    """Stops training when rolling mean reward stops improving."""

    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA,
                 min_timesteps=EARLY_STOP_MIN_STEPS, check_freq=PLOT_FREQ, verbose=1):
        super().__init__(verbose)
        self.patience      = patience
        self.min_delta     = min_delta
        self.min_timesteps = min_timesteps
        self.check_freq    = check_freq
        self._best_mean    = -np.inf
        self._no_improve   = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if self.num_timesteps < self.min_timesteps:
            return True
        if not self.model.ep_info_buffer:
            return True

        mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

        if mean_reward > self._best_mean + self.min_delta:
            self._best_mean  = mean_reward
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self.verbose:
                print(f"  [early stop] No improvement for {self._no_improve}/{self.patience} "
                      f"checks  (best mean: {self._best_mean:.1f})")

        if self._no_improve >= self.patience:
            print(f"\n── Early stopping triggered at {self.num_timesteps:,} timesteps "
                  f"(best mean reward: {self._best_mean:.1f})")
            return False

        return True


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(elev_sub, start, goal, reward_fn, max_steps, rank,
             endpoint_mode="fixed_single", endpoint_list=None):
    """Returns a callable that creates one DronePathEnv (required by SubprocVecEnv)."""
    def _init():
        set_random_seed(rank)
        return DronePathEnv(elev_sub, start, goal, reward_fn, max_steps,
                            endpoint_mode=endpoint_mode, endpoint_list=endpoint_list)
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Parse command-line arguments ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train PPO drone-path agent.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing ppo_final.zip to resume from. "
             "When set, learning rate is lowered to 5e-5.",
    )
    args = parser.parse_args()

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = MODELS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n── Run folder: models/{run_name}/")

    # ── 1. Load DEM and reproduce start/goal ──────────────────────────────────
    print("\n── Loading terrain data ──────────────────────────────────────────")
    elev_sub, start, goal = load_all(verbose=True)

    # ── 2. A* baseline (gives us max_steps and a reference cost) ─────────────
    reward_fn = RewardFunction()

    if ENDPOINT_MODE == "fixed_set":
        endpoint_pairs = get_endpoint_set(elev_sub, ENDPOINT_SEEDS[:N_ENDPOINT_PAIRS])
        print(f"\n── Running A* on {len(endpoint_pairs)} endpoint pairs ─────────────────")
        astar_lengths = []
        for i, (s, g) in enumerate(endpoint_pairs):
            path, _, _ = astar(
                elev_sub, s, g,
                slope_weight = W_ELEV,
                power_weight = W_POWER,
                base_power   = BASE_POWER_PER_M,
                climb_power  = CLIMB_POWER_PER_M,
                step_m       = DEM_STEP,
                verbose      = False,
            )
            if path:
                astar_lengths.append(len(path))
                print(f"  Pair {i+1:2d}: start={s}, goal={g}, A* len={len(path)}")
            else:
                print(f"  Pair {i+1:2d}: start={s}, goal={g}, A* FAILED — skipping")
        if not astar_lengths:
            sys.exit("A* found no valid paths for any endpoint pair.")
        max_steps = int(max(astar_lengths) * 3)
        print(f"  max_steps = {max_steps}  (3 × max A* len {max(astar_lengths)})")

    elif ENDPOINT_MODE == "random":
        endpoint_pairs = []
        rows, cols = elev_sub.shape
        max_steps = int(np.sqrt(rows ** 2 + cols ** 2) * 3)
        print(f"\n── Random endpoints mode — max_steps = {max_steps} (3 × grid diagonal)")

    else:  # fixed_single
        endpoint_pairs = []
        print("\n── Running A* baseline ───────────────────────────────────────────")
        astar_path, astar_time, astar_visited = astar(
            elev_sub, start, goal,
            slope_weight = W_ELEV,
            power_weight = W_POWER,
            base_power   = BASE_POWER_PER_M,
            climb_power  = CLIMB_POWER_PER_M,
            step_m       = DEM_STEP,
            verbose      = True,
        )
        if astar_path is None:
            sys.exit("A* found no path — check start/goal validity.")
        max_steps = int(len(astar_path) * 3)
        print(f"  max_steps = {max_steps}  (3 × A* length of {len(astar_path)})")

    # ── 3. Device setup ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32: ~2x faster matmul on Ampere/Ada, negligible precision loss
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True  # auto-tune kernels for fixed input shapes
        torch.set_num_threads(4)                       # leave CPU cores free for SubprocVecEnv workers
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n── Device: CUDA ({gpu_name}) — TF32 + cuDNN benchmark enabled")
    else:
        device = "cpu"
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        print(f"\n── Device: CPU ({torch.get_num_threads()} PyTorch threads)")

    # ── 3b. Warn if random endpoints without resume ───────────────────────────
    if ENDPOINT_MODE == "random" and not args.resume:
        print("\n  WARNING: ENDPOINT_MODE=random without --resume. Phase 3 should normally "
              "warm-start from a Phase 2 model. Continue only if intentional.\n")

    # ── 4. Create vectorised environments ─────────────────────────────────────
    print(f"\n── Creating {N_ENVS} parallel environments …")
    print(f"  Mode: {ENDPOINT_MODE}")
    env_fns = [make_env(elev_sub, start, goal, reward_fn, max_steps, i,
                        endpoint_mode=ENDPOINT_MODE,
                        endpoint_list=endpoint_pairs if ENDPOINT_MODE == "fixed_set" else None)
               for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)   # records episode stats for callbacks

    # ── 5. Build or load PPO model ────────────────────────────────────────────
    print("\n── Building PPO model ────────────────────────────────────────────")

    if args.resume:
        print(f"\n── Resuming from: {args.resume} ──────────────────────────────────────")
        print(f"  Lowered LR to {RESUME_LR} (catastrophic-forgetting safeguard)")
        model = PPO.load(args.resume, env=vec_env, device=device)
        model.learning_rate = RESUME_LR
        # SB3 caches a learning-rate schedule callable; refresh it:
        model.lr_schedule = get_schedule_fn(RESUME_LR)
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Policy parameters: {total_params:,}")
    else:
        # Step 7 (see docs/rationale.md): CNN feature extractor temporarily disabled.
        # Random-init CNN was drowning the 9 scalar goal-direction features in noise,
        # producing a near-random initial policy. Reverting to MLP over the flat
        # 891-dim obs (still includes the multi-scale patch). Re-enable later with
        # a smaller CNN once the reward + obs are proven.
        policy_kwargs = dict(
            net_arch                  = PPO_HIDDEN,
            activation_fn             = torch.nn.ReLU,
            # features_extractor_class  = DronePatchCNN,
            # features_extractor_kwargs = dict(features_dim=128),
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps         = PPO_N_STEPS,
            batch_size      = PPO_BATCH_SIZE,
            n_epochs        = PPO_N_EPOCHS,
            learning_rate   = PPO_LR,
            gamma           = PPO_GAMMA,
            gae_lambda      = PPO_GAE_LAMBDA,
            clip_range      = PPO_CLIP_RANGE,
            ent_coef        = 0.01,   # Step 7: encourage exploration so policy doesn't collapse
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            device          = device,
            tensorboard_log = str(RESULTS_DIR / "tensorboard"),
        )
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Policy parameters: {total_params:,}")

    # ── 6. Callbacks ──────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = str(run_dir),
        name_prefix = "checkpoint",
        verbose     = 1,
    )
    plot_cb = TrainingPlotCallback(
        results_dir = str(RESULTS_DIR),
        plot_freq   = max(PLOT_FREQ // N_ENVS, 1),
        verbose     = 1,
    )
    early_stop_cb = EarlyStoppingCallback(verbose=1)

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n── Training for {TOTAL_TIMESTEPS:,} timesteps ────────────────────")
    print("   TensorBoard: tensorboard --logdir results/tensorboard")
    print("   Live curve : results/training_curve.png  (refreshes every "
          f"{PLOT_FREQ:,} steps)\n")

    t_start = time.perf_counter()
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, plot_cb, early_stop_cb],
        progress_bar    = True,
    )
    t_train = time.perf_counter() - t_start

    # ── 8. Save final model ───────────────────────────────────────────────────
    final_path = str(run_dir / "ppo_final")
    model.save(final_path)
    print(f"\n── Training complete in {t_train/60:.1f} min")
    print(f"   Final model saved → {final_path}.zip")
    print("   Run  python -m rl.evaluate  to compare against A*.")

    vec_env.close()


if __name__ == "__main__":
    main()
