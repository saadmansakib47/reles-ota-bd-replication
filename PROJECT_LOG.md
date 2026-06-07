# Project Log

This file tracks the major implementation updates for the ReLES-OTA replication project. Add a new dated entry for each milestone so the team can keep a clean history of what changed, why it changed, and how it was verified.

## 2026-06-07 — Phase 2, Step 2: FP3O Architecture & Environment Migration

### Completed Work
- **Conda Environment Migration**: Created dedicated `reles-ota` Conda environment (Python 3.12) to bypass Visual C++ build tool requirements on Windows; all dependencies install from precompiled binary wheels.
- **Shared Backbone**: Verified `SharedBackbone` (3-layer MLP: 256→256→128 with ReLU) processes flattened observations shared across all agents — firmware block masks, cumulative encoding/TX costs, memory budget, step counter, and agent one-hot ID.
- **Specialized Heads (FP3O)**: Completed Partial Parameter Sharing architecture in [fp3o_policy.py](file:///d:/Thesis/reles-ota-bd-replication/fp3o_policy.py):
  - `ActionHead` per ECU type (engine / braking / infotainment / generic) → logits over {Copy, Modify, Modify+Backup}
  - `PositionHead` per ECU type → logits over N firmware block positions
  - Dynamic routing in `_get_action_dist_from_latent`: extracts agent indices from `agent_id` one-hot in observation batch, selects the correct head per sample in a vectorized batch.
- **Value Normalization**: Implemented `ValueNormalizer` (PopArt-style EMA running mean/std) and `ValueNormalizationCallback` (SB3 `BaseCallback`):
  - `update()` called per rollout; normalizes rollout buffer returns and values before critic gradient steps.
  - `predict_values()` and `forward()` denormalize critic output to reward scale for GAE advantage estimation.
- **Training Integration**: [train_mappo.py](file:///d:/Thesis/reles-ota-bd-replication/train_mappo.py) IPPO path now uses `FP3OPolicy` and `ValueNormalizationCallback` instead of `MultiInputPolicy`.
- **Console Encoding Fix**: Added `sys.stdout.reconfigure(encoding='utf-8')` to `test_marl_env.py` and `train_mappo.py` to prevent `UnicodeEncodeError` on Windows cp1252 consoles.
- **Codebase Cleanup**: Removed duplicate `FP3OPolicy` stub class, unused imports (`F`, `FlattenExtractor`, `Type`, `List`), and moved all imports to module level.

### Verification
- `python test_fp3o.py` — ValueNormalizer roundtrip PASS; FP3O backbone/heads shape verification PASS.
- `python test_marl_env.py` — All 5 tests PASS (API compliance, random episodes, death masking, stochastic latency, scalability).
- `python train_mappo.py --mode ippo --n_agents 4 --n_blocks 6 --timesteps 5000` — IPPO training with FP3OPolicy completed successfully; model saved.

### Notes
- Dynamic head routing uses `agent_id` one-hot in the observation to route each sample in a vectorized batch to its correct ECU-type specialized head, enabling truly heterogeneous multi-agent training under a single policy instance.
- `ValueNormalizationCallback` operates on the rollout buffer's `returns` and `values` tensors after GAE computation, ensuring the critic regresses on z-scored targets while `predict_values()` always denormalizes back to raw reward scale.

## 2026-06-06 — Phase 2, Step 1 (Continuation): Dependency Setup & Wrapper Integration Fixes

### Completed Work
- Resolved dependencies on Windows Python 3.12, including precompiled `tinyscaler` and `supersuit` setup.
- Added `render_mode` parameter and attribute to [MultiAgentOTAEnv](file:///d:/Thesis/reles-ota-bd-replication/marl_ota_env.py#L44) for compatibility with SuperSuit's wrappers.
- Configured `MarkovVectorEnv` wrapper with `black_death=True` in [train_mappo.py](file:///d:/Thesis/reles-ota-bd-replication/train_mappo.py) to support variable agent lifetimes and align with our death masking strategy.
- Patched `ConcatVecEnv`'s missing `seed` method to resolve the AttributeError in Stable Baselines 3 (`set_random_seed`).

### Verification
- Ran `python test_marl_env.py` under Python 3.12 successfully (all Parallel API compliance and death masking checks passed).
- Successfully ran IPPO training via `python train_mappo.py --mode ippo --n_agents 2 --n_blocks 6 --timesteps 5000` (completed log stats and model saving without errors).

### Notes
- Seeding for Gym 0.26+ vectorized environments is now safely bypassed with a dummy `.seed()` method on the vectorized wrapper.
- `black_death=True` is required when wrapping multi-agent environments with varying agent counts under SuperSuit's Markov vector wrappers.

## 2026-06-06 — Phase 2, Step 1: Multi-Agent Environment Scaling

### Completed Work
- Added a new PettingZoo Parallel API multi-agent environment in `marl_ota_env.py`.
- Supported N concurrent ECU agents, each with its own firmware block list, mask, and episode state.
- Implemented death masking so finished agents keep returning a fixed zero-vector observation with only `agent_id` preserved.
- Added stochastic network latency support through a shared physics module in `ota_core.py`.
- Kept the original single-agent environment in `ota_env.py` intact for Phase 1 comparison.
- Added a multi-agent validation suite in `test_marl_env.py`.
- Added a training scaffold / random baseline runner in `train_mappo.py`.
- Updated `requirements.txt` with PettingZoo, SuperSuit, and SciPy dependencies.

### Verification
- Ran `python test_marl_env.py` successfully.
- Confirmed PettingZoo parallel API compliance.
- Confirmed death masking works for terminated agents.
- Confirmed stochastic latency differs from fixed-latency behavior under identical actions.
- Confirmed scalability checks pass for multiple agent counts.
- Ran `python train_mappo.py --mode random --n_agents 4 --n_blocks 6 --episodes 2` successfully.

### Notes
- The multi-agent environment is implemented as a new file rather than refactoring the single-agent env in place.
- The current training entrypoint supports random rollout verification and an IPPO scaffold, while a full MAPPO learner remains a later step.

## Suggested Update Format

Use this template for future entries:

### YYYY-MM-DD — Short Milestone Title

#### Completed Work
- Item 1
- Item 2
- Item 3

#### Verification
- Test or command run
- Result

#### Notes
- Any design decisions, caveats, or follow-up work
