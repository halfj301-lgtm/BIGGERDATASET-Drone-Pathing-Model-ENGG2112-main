"""
Custom feature extractor for DronePathEnv observations.

The env returns a flat float32 vector laid out as:
    [0:N_SCALARS]                                — scalar state features
    [N_SCALARS:N_SCALARS + PATCH²]              — flattened fine elevation patch (PATCH × PATCH)
    [N_SCALARS + PATCH²:N_SCALARS + 2×PATCH²]   — flattened coarse elevation patch (PATCH × PATCH, stride COARSE_STRIDE)

This extractor splits that vector, reshapes both patches back to 2D form,
stacks them on the channel dimension to create a 2-channel image,
runs it through a small CNN, then concatenates the CNN output with the
scalar features before projecting to `features_dim`. The resulting
features are fed into PPO's standard MLP policy/value heads.

Why this exists:
    The flat-MLP baseline cannot exploit 2D adjacency in the elevation
    patches. Adding a CNN encoder gives the policy access to local terrain
    structure (ridges, valleys, slopes) at multiple scales — so the agent
    can navigate both locally and strategically.

See docs/rationale.md (Step 4) for the design discussion.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.config import PATCH_SIZE, PATCH_SIZE_COARSE


# Number of scalar features at the start of the obs vector.
# Must match `DronePathEnv._get_obs()` in rl/environment.py.
N_SCALARS = 9


class DronePatchCNN(BaseFeaturesExtractor):
    """
    CNN-based feature extractor for DronePathEnv.

    Input:  flat obs vector of shape (batch, N_SCALARS + 2×PATCH_SIZE²)
            (9 scalars + fine patch + coarse patch, both PATCH_SIZE × PATCH_SIZE)
    Output: feature vector of shape (batch, features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self._patch_size = PATCH_SIZE
        self._n_scalars  = N_SCALARS

        if PATCH_SIZE != PATCH_SIZE_COARSE:
            raise ValueError(
                f"DronePatchCNN currently requires PATCH_SIZE == PATCH_SIZE_COARSE; "
                f"got {PATCH_SIZE} vs {PATCH_SIZE_COARSE}."
            )

        expected_dim = self._n_scalars + 2 * self._patch_size * self._patch_size
        actual_dim   = int(observation_space.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                f"DronePatchCNN: expected obs dim {expected_dim} "
                f"(N_SCALARS={self._n_scalars} + 2×PATCH_SIZE²={2*self._patch_size**2} for fine+coarse patches), "
                f"got {actual_dim}. Check rl/environment.py and rl/config.py."
            )

        # Small CNN over the elevation patches (fine + coarse stacked on channel dim).
        # Channels: 2 → 16 → 32 → 32 (stride-2 downsample at the last conv).
        # For PATCH_SIZE=21 the output spatial dim is ceil(21/2) = 11, so
        # cnn output features = 32 * 11 * 11 = 3872 (computed dynamically below).
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim by running a dummy tensor through it.
        with th.no_grad():
            dummy = th.zeros(1, 2, self._patch_size, self._patch_size)
            cnn_out_dim = int(self.cnn(dummy).shape[1])

        # Project (CNN features + scalar features) → features_dim.
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + self._n_scalars, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs shape: (batch, N_SCALARS + 2×PATCH_SIZE²)
        p2 = self._patch_size * self._patch_size
        scalars = obs[:, : self._n_scalars]
        fine    = obs[:, self._n_scalars : self._n_scalars + p2].reshape(
            -1, 1, self._patch_size, self._patch_size
        )
        coarse  = obs[:, self._n_scalars + p2 :].reshape(
            -1, 1, self._patch_size, self._patch_size
        )
        patches  = th.cat([fine, coarse], dim=1)   # (batch, 2, P, P)
        cnn_feat = self.cnn(patches)
        combined = th.cat([scalars, cnn_feat], dim=1)
        return self.head(combined)
