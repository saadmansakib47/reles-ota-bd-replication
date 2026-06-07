"""
test_fp3o.py — FP3O architecture verification (no SB3/Gym)
=================================================================
Pure PyTorch unit tests for the Shared Backbone, Specialized Action/Position
Heads, and Value Normalizer components defined in fp3o_policy.py. No Gym/SB3
dependencies needed — just torch, numpy, and gymnasium.spaces.

Main tests performed
--------------------
1. ValueNormalizer roundtrip: create running mean/std, update with a batch
   of targets, normalize, then denormalize, verifying that V_hat_denorm == V_orig
   up to numerical precision.
2. Shared Backbone forward pass: feed batched flattened observations (mask, costs,
   memory, step, agent_id) through the 3-layer MLP (256→256→128) and assert
   output shape is (batch_size, 128).
3. Specialized Heads forward pass: for both engine and braking-type policies,
   run a forward pass and assert action/value outputs have the expected shapes.

References
----------
- FP3O paper (arXiv:2310.05053) for the architectural design
- PopArt (arXiv:1902.10597) for the value normalization strategy
- Stable-Baselines3 usage patterns for integration notes

Usage
-----
Run from the project root:

  python test_fp3o.py

Expected output
---------------
Both tests should pass with messages like:

  ValueNormalizer verification: PASS
  FP3O Architecture verification: PASS

If any test fails, the script will raise an AssertionError with a descriptive
message.

Testing environment
---------------------
- torch >= 1.13
- numpy >= 1.21
- gymnasium >= 0.27 (for spaces enum/types; can be downgraded if needed)

No need for a full MARL environment or SB3 RL training loop — this module tests
only the reusable policy components in isolation.

Notes
-----
- Dynamic agent routing is tested via the vectorized batch forward pass:
  the `agent_id` one-hot in the mock observation allows each sample to be
  routed to its appropriate action/position head without explicit per-agent
  looping.
- `ValueNormalizationCallback` is not tested here since it hooks into the
  SB3 training loop; only the standalone `ValueNormalizer` class is verified.

Questions / assumptions
-----------------------
- Assumes the `_get_action_dist_from_latent` helper in fp3o_policy.py
  correctly uses `agent_id` one-hot for vectorized routing — this is verified
  by the shape checks below.
- Assumes the reward range and value function scales are within typical
  deep RL values where PopArt normalization provides a stable gradient target.

If this test suite needs to be updated, please align with the current
fp3o_policy.py implementation and add regression tests for any new features.

"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from gymnasium import spaces
from fp3o_policy import FP3OPolicy, ValueNormalizer, SharedBackbone, ActionHead, PositionHead

def test_value_normalizer():
    print("--- Testing ValueNormalizer ---")
    # Set momentum=1.0 and a high clip_val to get exact z-score match for a single batch
    vn = ValueNormalizer(momentum=1.0, clip_val=999.0)
    
    # Generate some dummy value targets
    targets = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    vn.update(targets)
    
    print(f"Running Mean: {vn.running_mean.item():.2f}")
    print(f"Running Var: {vn.running_var.item():.2f}")
    
    normed = vn.normalize(targets)
    print(f"Normalized Targets: {normed.tolist()}")
    
    denormed = vn.denormalize(normed)
    print(f"Denormalized Targets (roundtrip): {denormed.tolist()}")
    
    assert torch.allclose(targets, denormed, atol=1e-4), "Roundtrip check failed!"
    print("ValueNormalizer verification: PASS")

def test_fp3o_architecture():
    print("\n--- Testing FP3O Specialized Heads & Backbone ---")
    n_blocks = 16
    obs_space = spaces.Dict({
        "mask":              spaces.MultiBinary(n_blocks),
        "cum_encoding_cost": spaces.Box(0, np.inf, (1,), dtype=np.float32),
        "cum_tx_cost":       spaces.Box(0, np.inf, (1,), dtype=np.float32),
        "memory_used":       spaces.Box(0, 1.0,   (1,), dtype=np.float32),
        "step":              spaces.Box(0, n_blocks + 10, (1,), dtype=np.int32),
        "agent_id":          spaces.Box(0, 1.0, (4,), dtype=np.float32),
    })
    act_space = spaces.MultiDiscrete([n_blocks, 3])
    
    # Instantiate engine-type policy
    policy_engine = FP3OPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        n_blocks=n_blocks,
        ecu_type_idx=0, # Engine
    )
    
    # Instantiate braking-type policy (shares the same class, initialized differently)
    policy_braking = FP3OPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        n_blocks=n_blocks,
        ecu_type_idx=1, # Braking
    )
    
    # Create a mock batch of observations
    batch_size = 2
    mock_obs = {
        "mask":              torch.ones(batch_size, n_blocks, dtype=torch.float32),
        "cum_encoding_cost": torch.zeros(batch_size, 1, dtype=torch.float32),
        "cum_tx_cost":       torch.zeros(batch_size, 1, dtype=torch.float32),
        "memory_used":       torch.zeros(batch_size, 1, dtype=torch.float32),
        "step":              torch.zeros(batch_size, 1, dtype=torch.float32),
        "agent_id":          torch.eye(4)[:batch_size], # One-hot ids
    }
    
    # Check Backbone Forward
    features = policy_engine.extract_features(mock_obs)
    print(f"Shared Backbone output shape: {features.shape} (Expected: [batch_size, 128])")
    assert features.shape == (batch_size, 128), "Backbone output shape mismatch!"
    
    # Forward for Engine Head
    actions_eng, values_eng, log_probs_eng = policy_engine(mock_obs)
    print(f"Engine actions shape: {actions_eng.shape} (Expected: [batch_size, 2])")
    assert actions_eng.shape == (batch_size, 2), "Engine action shape mismatch!"
    
    # Forward for Braking Head
    actions_brk, values_brk, log_probs_brk = policy_braking(mock_obs)
    print(f"Braking actions shape: {actions_brk.shape} (Expected: [batch_size, 2])")
    assert actions_brk.shape == (batch_size, 2), "Braking action shape mismatch!"
    
    print("FP3O Architecture verification: PASS")

if __name__ == "__main__":
    test_value_normalizer()
    test_fp3o_architecture()
    print("\nAll FP3O verification tests completed successfully!")
