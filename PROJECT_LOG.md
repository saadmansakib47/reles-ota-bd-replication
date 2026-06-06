# Project Log

This file tracks the major implementation updates for the ReLES-OTA replication project. Add a new dated entry for each milestone so the team can keep a clean history of what changed, why it changed, and how it was verified.

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
