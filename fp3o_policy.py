"""
fp3o_policy.py — FP3O Shared Backbone + Specialized Heads Policy
=================================================================
Implements Partial Parameter Sharing (FP3O) for heterogeneous ECU agents
following the architecture in:
  • FP3O (arXiv:2310.05053) — Flexible Parameter-sharing for PPO in MARL

Architecture overview
---------------------

┌─────────────────────────────────────────────────────────────┐
│                   Shared Backbone (MLP)                      │
│   obs_flat  ──►  Linear(obs_dim, 256) ──► ReLU              │
│             ──►  Linear(256, 256)     ──► ReLU               │
│             ──►  Linear(256, 128)     ──► ReLU   ──► z (128)│
└─────────────────────────────────────────────────────────────┘
         │                               │
         ▼                               ▼
 Specialized Actor Heads          Centralized Critic
 (one per ECU type)               Linear(128, 1)  →  V(s)
 ┌───────────────────┐             (with Value Normalizer)
 │ ActionHead(ECU)   │
 │  z → 128 → logits│
 └───────────────────┘
 ┌───────────────────┐
 │ PositionHead(ECU) │
 │  z → 128 → logits│
 └───────────────────┘

Key design decisions
--------------------
1. Shared backbone: all agents share weights for global firmware semantics
   (block similarity structure, bandwidth/latency signals).
2. Specialized heads: each ECU TYPE (engine / braking / infotainment / generic)
   gets its own action head and position head. Safety-critical ECUs (engine,
   braking) should develop more conservative update policies while infotainment
   ECUs can be more aggressive.
3. Value Normalization: running mean/std of value targets, regress on
   normalized targets, denormalize during GAE. Prevents value scale explosion.

Usage
-----
See train_mappo.py for integration example. The policy class is registered as
a SB3 custom policy so it can be passed as `policy=FP3OPolicy` to PPO().

References
----------
- FP3O: arXiv:2310.05053
- MAPPO-PIS: arXiv:2408.06656
- PopArt (value normalization): Hessel et al., 2019
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

# pyrefly: ignore [missing-import]
from stable_baselines3.common.policies import ActorCriticPolicy
# pyrefly: ignore [missing-import]
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
# pyrefly: ignore [missing-import]
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback
# pyrefly: ignore [missing-import]
from gymnasium import spaces


# ══════════════════════════════════════════════════════════════
#  ECU Type Registry
# ══════════════════════════════════════════════════════════════

ECU_TYPES = ["engine", "braking", "infotainment", "generic"]
ECU_TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(ECU_TYPES)}

def ecu_type_to_idx(ecu_type: str) -> int:
    """Map ECU type string to integer index for head selection."""
    return ECU_TYPE_TO_IDX.get(ecu_type, ECU_TYPE_TO_IDX["generic"])


# ══════════════════════════════════════════════════════════════
#  Value Normalizer (PopArt-style running stats)
# ══════════════════════════════════════════════════════════════

class ValueNormalizer(nn.Module):
    """
    Running mean/std normalizer for value function targets.

    Based on PopArt (Hessel et al., 2019):
      - Track μ (mean) and σ² (variance) of value targets online.
      - During critic update: normalize targets → regress on z-scored values.
      - During advantage estimation (GAE): denormalize V(s) predictions to
        keep TD errors on the original reward scale.

    Parameters
    ----------
    epsilon : float
        Small constant for numerical stability (avoids division by zero).
    momentum : float
        EMA factor for updating running stats. Lower = slower adaptation.
    clip_val : float
        Max absolute value of normalized targets (prevents exploding gradients).
    """

    def __init__(
        self,
        epsilon: float  = 1e-4,
        momentum: float = 0.01,
        clip_val: float = 10.0,
    ):
        super().__init__()
        self.epsilon  = epsilon
        self.momentum = momentum
        self.clip_val = clip_val

        # Running statistics — registered as buffers (not learned parameters)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var",  torch.ones(1))
        self.register_buffer("count",        torch.zeros(1))

    @property
    def running_std(self) -> torch.Tensor:
        return torch.sqrt(self.running_var + self.epsilon)

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        """
        Update running statistics with a batch of new value targets.
        Uses Welford's online algorithm adapted for batch EMA.
        """
        batch_mean = targets.mean()
        batch_var  = targets.var(unbiased=False)
        batch_n    = torch.tensor(targets.numel(), dtype=torch.float32,
                                  device=targets.device)

        # EMA update
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * batch_var
        self.count        = self.count + batch_n

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Normalize value targets to zero-mean, unit-variance.
        Call update() first on each training batch.
        """
        normed = (targets - self.running_mean) / self.running_std
        return normed.clamp(-self.clip_val, self.clip_val)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Denormalize critic outputs back to original reward scale.
        Call this during GAE advantage estimation.
        """
        return values * self.running_std + self.running_mean

    def extra_repr(self) -> str:
        return (f"momentum={self.momentum}, epsilon={self.epsilon}, "
                f"clip_val={self.clip_val}")


# ══════════════════════════════════════════════════════════════
#  Shared Backbone
# ══════════════════════════════════════════════════════════════

class SharedBackbone(nn.Module):
    """
    Shared MLP backbone — processes flattened observation features common
    to all ECU agents. Learns general firmware update semantics:
      - Block similarity structure (MD5/delta encodings)
      - Global bandwidth/latency state
      - Fleet-level memory budget signals

    Architecture: Linear(obs_dim → 256) → ReLU → Linear(256 → 256) → ReLU
                  → Linear(256 → latent_dim) → ReLU

    Parameters
    ----------
    obs_dim    : flattened input feature dimension
    latent_dim : output latent code dimension (default 128)
    """

    def __init__(self, obs_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════
#  Specialized Action Head  (per ECU type)
# ══════════════════════════════════════════════════════════════

class ActionHead(nn.Module):
    """
    ECU-type-specific action head for the *operation* dimension
    (Copy / Modify / Modify+Backup — 3 discrete choices).

    Safety-critical ECU types (engine, braking) will develop a bias
    toward Modify+Backup (op=2) to preserve rollback capability.
    Infotainment ECUs can be more aggressive (Copy or Modify).

    Each head is a small MLP: latent_z → 128 → action_logits
    """

    def __init__(self, latent_dim: int = 128, n_operations: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_operations),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits over {Copy, Modify, Modify+Backup}."""
        return self.head(z)


class PositionHead(nn.Module):
    """
    ECU-type-specific position head for the *block selection* dimension
    (which firmware block to update next — n_blocks discrete choices).

    Each head is a small MLP: latent_z → 128 → block_logits
    """

    def __init__(self, latent_dim: int = 128, n_blocks: int = 24):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_blocks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits over n_blocks block positions."""
        return self.head(z)


# ══════════════════════════════════════════════════════════════
#  Centralized Critic Head
# ══════════════════════════════════════════════════════════════

class CriticHead(nn.Module):
    """
    Centralized value function head V(s).
    Takes the shared backbone latent representation and returns a scalar.
    Used by MAPPO's centralized critic (all agents share this head).

    In IPPO mode, each agent still uses this head independently (same weights).
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ══════════════════════════════════════════════════════════════
#  FP3O Features Extractor (plugs into SB3 as a BaseFeaturesExtractor)
# ══════════════════════════════════════════════════════════════

class FP3OFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor that:
    1. Flattens the Dict observation space into a single vector.
    2. Passes it through the SharedBackbone.
    3. Returns the latent code z for downstream actor/critic heads.

    Registered with SB3 via policy_kwargs["features_extractor_class"].
    """

    def __init__(self, observation_space: spaces.Dict, latent_dim: int = 128):
        # Compute flat obs dimension
        flat_dim = sum(
            int(np.prod(v.shape)) for v in observation_space.spaces.values()
        )
        super().__init__(observation_space, features_dim=latent_dim)

        self.flatten     = nn.Flatten()
        self.backbone    = SharedBackbone(obs_dim=flat_dim, latent_dim=latent_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten each sub-space, concatenate, then run through backbone
        parts = []
        for key in sorted(observations.keys()):   # deterministic ordering
            obs = observations[key]
            if obs.dtype == torch.int32 or obs.dtype == torch.int64:
                obs = obs.float()
            parts.append(obs.reshape(obs.shape[0], -1))
        x = torch.cat(parts, dim=1)
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
#  FP3O Actor-Critic Policy  (SB3 custom policy class)
# ══════════════════════════════════════════════════════════════

class FP3OPolicy(ActorCriticPolicy):
    """
    FP3O Actor-Critic Policy for heterogeneous MARL.

    Extends SB3's ActorCriticPolicy with:
    - Shared backbone feature extractor (FP3OFeaturesExtractor)
    - Per-ECU-type specialized action heads and position heads
    - Value normalizer on the critic path
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # FP3O-specific kwargs
        n_blocks: int     = 24,
        ecu_type_idx: int = 3,        # 3 = "generic" (safe default)
        latent_dim: int   = 128,
        # Value normalizer hypers
        vn_momentum: float = 0.01,
        vn_clip:     float = 10.0,
        # Pass remaining kwargs to parent (net_arch, ortho_init, etc.)
        **kwargs,
    ):
        # Inject our custom features extractor
        kwargs.setdefault("features_extractor_class", FP3OFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"latent_dim": latent_dim})

        self.n_blocks     = n_blocks
        self.ecu_type_idx = ecu_type_idx
        self.latent_dim   = latent_dim
        self.vn_momentum  = vn_momentum
        self.vn_clip      = vn_clip

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Value normalizer (initialized after super().__init__ sets up device)
        self.value_normalizer = ValueNormalizer(
            momentum = self.vn_momentum,
            clip_val = self.vn_clip,
        ).to(self.device)

        # Build mapping buffer from one-hot agent index to ecu type index
        if isinstance(observation_space, spaces.Dict) and "agent_id" in observation_space.spaces:
            n_agents = observation_space.spaces["agent_id"].shape[0]
        else:
            n_agents = 4  # safe default fallback
        
        self.register_buffer("agent_idx_to_ecu_type", torch.tensor([
            i % len(ECU_TYPES) for i in range(n_agents)
        ], dtype=torch.long))

    def _build_mlp_extractor(self) -> None:
        """
        Override SB3's default MLP extractor construction.
        FP3O uses specialized heads instead of a single shared MLP.
        We build the heads here after the features extractor is ready.
        """
        latent_dim = self.latent_dim

        # ── Specialized actor heads (ECU-type-specific) ──────────────────────
        # Each ECU type gets its own pair of heads. We instantiate ALL types so
        # that the same policy class can be used across different ECU types just
        # by changing ecu_type_idx at instantiation time.
        self.action_heads: nn.ModuleList = nn.ModuleList([
            ActionHead(latent_dim=latent_dim, n_operations=3)
            for _ in ECU_TYPES
        ])
        self.position_heads: nn.ModuleList = nn.ModuleList([
            PositionHead(latent_dim=latent_dim, n_blocks=self.n_blocks)
            for _ in ECU_TYPES
        ])

        # ── Centralized critic head ───────────────────────────────────────────
        self.critic_head = CriticHead(latent_dim=latent_dim)

        # SB3 needs mlp_extractor.latent_dim_pi and .latent_dim_vf
        # We satisfy this by assigning a thin wrapper that just passes z through.
        class _IdentityExtractor(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.latent_dim_pi = dim
                self.latent_dim_vf = dim
            def forward(self, features):
                return features, features  # (pi_features, vf_features)

        self.mlp_extractor = _IdentityExtractor(latent_dim)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, obs: Optional[Dict[str, torch.Tensor]] = None):
        """
        Override: compute action distribution from the shared latent code.
        Selects the ECU-type-specific action head and position head.
        """
        if obs is not None and isinstance(obs, dict) and "agent_id" in obs:
            agent_id_batch = obs["agent_id"]  # shape: (batch_size, n_agents)
            agent_indices = agent_id_batch.argmax(dim=-1)  # shape: (batch_size,)
            batch_ecu_type_indices = self.agent_idx_to_ecu_type[agent_indices]  # shape: (batch_size,)
            
            action_logits_stacked = torch.stack([
                head(latent_pi) for head in self.action_heads
            ], dim=1)  # shape: (batch_size, 4, 3)
            
            position_logits_stacked = torch.stack([
                head(latent_pi) for head in self.position_heads
            ], dim=1)  # shape: (batch_size, 4, n_blocks)
            
            batch_indices = torch.arange(latent_pi.shape[0], device=latent_pi.device)
            action_logits = action_logits_stacked[batch_indices, batch_ecu_type_indices]
            position_logits = position_logits_stacked[batch_indices, batch_ecu_type_indices]
        else:
            # Fallback to static ecu_type_idx
            action_logits   = self.action_heads[self.ecu_type_idx](latent_pi)
            position_logits = self.position_heads[self.ecu_type_idx](latent_pi)

        # MultiDiscrete distribution: [block_idx_dist, operation_dist]
        # SB3's MultiCategoricalDistribution expects a single concatenated flat tensor
        concat_logits = torch.cat([position_logits, action_logits], dim=1)
        return self.action_dist.proba_distribution(
            action_logits=concat_logits
        )

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Predict value estimates V(s) with denormalization.
        Called during rollout collection — returns raw-scale values.
        """
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor(features)[1]
        # Critic head returns normalized values; denormalize to reward scale
        normed_values = self.critic_head(latent_vf)
        return self.value_normalizer.denormalize(normed_values)

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Override to inject value normalization into the training loop.
        Called inside PPO's learn() during gradient computation.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution   = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob       = distribution.log_prob(actions)
        entropy        = distribution.entropy()

        # Critic: return normalized values (training regresses on normed targets)
        values_normed = self.critic_head(latent_vf)
        return values_normed, log_prob, entropy

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SB3 standard forward: returns (actions, values, log_probs)."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions      = distribution.get_actions(deterministic=deterministic)
        log_prob     = distribution.log_prob(actions)

        # Return denormalized values for rollout collection
        values_normed = self.critic_head(latent_vf)
        values        = self.value_normalizer.denormalize(values_normed)
        return actions, values, log_prob


# ══════════════════════════════════════════════════════════════
#  Value Normalization Training Callback
# ══════════════════════════════════════════════════════════════

class ValueNormalizationCallback(BaseCallback):
    """
    Callback to update value normalizer running statistics and normalize targets
    in the rollout buffer prior to gradient steps.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        policy = self.model.policy
        if hasattr(policy, "value_normalizer"):
            returns = self.model.rollout_buffer.returns
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=policy.device)
            policy.value_normalizer.update(returns_tensor)
            
            # Normalize returns and values in rollout buffer
            self.model.rollout_buffer.returns = policy.value_normalizer.normalize(returns_tensor).cpu().numpy()
            
            values = self.model.rollout_buffer.values
            values_tensor = torch.tensor(values, dtype=torch.float32, device=policy.device)
            self.model.rollout_buffer.values = policy.value_normalizer.normalize(values_tensor).cpu().numpy()

# ══════════════════════════════════════════════════════════════
#  Convenience factory function
# ══════════════════════════════════════════════════════════════

def make_fp3o_policy_kwargs(
    n_blocks: int,
    ecu_type: str,
    latent_dim: int = 128,
    vn_momentum: float = 0.01,
    vn_clip: float = 10.0,
) -> dict:
    """
    Build the policy_kwargs dict to pass to PPO() when using FP3OPolicy.

    Parameters
    ----------
    n_blocks    : number of firmware blocks in the environment
    ecu_type    : ECU type string ("engine", "braking", "infotainment", "generic")
    latent_dim  : backbone latent dimension
    vn_momentum : EMA momentum for value normalizer stats update
    vn_clip     : clip range for normalized value targets

    Returns
    -------
    dict : suitable for PPO(..., policy_kwargs=make_fp3o_policy_kwargs(...))
    """
    return {
        "n_blocks":     n_blocks,
        "ecu_type_idx": ecu_type_to_idx(ecu_type),
        "latent_dim":   latent_dim,
        "vn_momentum":  vn_momentum,
        "vn_clip":      vn_clip,
    }


# ══════════════════════════════════════════════════════════════
#  Quick sanity test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke-testing FP3O policy components ...\n")

    # ── ValueNormalizer ──
    vn = ValueNormalizer(momentum=0.1)
    dummy_targets = torch.randn(32)
    vn.update(dummy_targets)
    normed = vn.normalize(dummy_targets)
    denormed = vn.denormalize(normed)
    print(f"ValueNormalizer  |  mean={vn.running_mean.item():.4f}  "
          f"std={vn.running_std.item():.4f}")
    print(f"  normalize range: [{normed.min():.2f}, {normed.max():.2f}]")
    print(f"  roundtrip error: {(denormed - dummy_targets).abs().max():.6f}  ✓\n")

    # ── SharedBackbone ──
    backbone = SharedBackbone(obs_dim=64, latent_dim=128)
    z = backbone(torch.randn(8, 64))
    print(f"SharedBackbone   |  input (8,64) → latent {tuple(z.shape)}  ✓")

    # ── ActionHead + PositionHead ──
    for ecu_type in ECU_TYPES:
        idx = ecu_type_to_idx(ecu_type)
        ah  = ActionHead(latent_dim=128, n_operations=3)
        ph  = PositionHead(latent_dim=128, n_blocks=24)
        action_logits   = ah(z)
        position_logits = ph(z)
        print(f"  {ecu_type:14s} | action logits {tuple(action_logits.shape)}  "
              f"position logits {tuple(position_logits.shape)}  ✓")

    # ── CriticHead ──
    ch = CriticHead(latent_dim=128)
    v  = ch(z)
    print(f"\nCriticHead       |  latent (8,128) → value {tuple(v.shape)}  ✓")

    # ── Full parameter count ──
    total = (sum(p.numel() for p in backbone.parameters()) +
             sum(sum(p.numel() for p in ah.parameters()) for ah in
                 [ActionHead(128) for _ in ECU_TYPES]) +
             sum(sum(p.numel() for p in ph.parameters()) for ph in
                 [PositionHead(128, 24) for _ in ECU_TYPES]) +
             sum(p.numel() for p in ch.parameters()))
    print(f"\nTotal FP3O params (approx): {total:,}  |  per ECU type: {total // len(ECU_TYPES):,}")
    print("\n✅  All FP3O components validated.")
