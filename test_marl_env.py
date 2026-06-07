"""
test_marl_env.py — MARL Environment Validation Suite
=====================================================
Sanity tests for MultiAgentOTAEnv (Phase 2, Step 1).

Tests
-----
1. PettingZoo Parallel API compliance check
2. Full episode with random actions — 3 independent runs
3. Death masking validation — terminated agents return zero-vector obs
4. Stochastic latency variance check — costs should vary across episodes
5. Per-agent stats table at end of each episode

Run with:
    python test_marl_env.py
"""

import numpy as np
import time
import sys
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from marl_ota_env import MultiAgentOTAEnv


# ══════════════════════════════════════════════════════════════
#  Helper: run one full episode, return stats
# ══════════════════════════════════════════════════════════════

def run_episode(env: MultiAgentOTAEnv, seed: int = None, verbose: bool = False) -> dict:
    """Run one episode with random actions. Returns per-agent result dict."""
    obs, _ = env.reset(seed=seed)
    step_count = 0
    agent_stats = {a: {"payload": 0.0, "memory": 0.0, "steps": 0} for a in env.possible_agents}

    while env.agents:
        # Only pass actions for ALIVE agents
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        step_count += 1

        if verbose and step_count % 5 == 0:
            env.render()

        # Collect final info when agents terminate
        for agent in env.possible_agents:
            info = infos.get(agent, {})
            if info.get("done") or info.get("truncated"):
                agent_stats[agent]["payload"] = info.get("payload_bytes", 0.0)
                agent_stats[agent]["memory"]  = info.get("memory_used", 0.0)
                agent_stats[agent]["steps"]   = env.current_step[agent]

    # Catch any agents that never fired a "done" info
    for agent in env.possible_agents:
        if agent_stats[agent]["payload"] == 0.0:
            agent_stats[agent]["payload"] = env.cum_enc_cost[agent] + env.cum_tx_cost[agent]
            agent_stats[agent]["memory"]  = env.cum_memory[agent]
            agent_stats[agent]["steps"]   = env.current_step[agent]

    return {"global_steps": step_count, "agents": agent_stats}


# ══════════════════════════════════════════════════════════════
#  TEST 1 — PettingZoo API Compliance
# ══════════════════════════════════════════════════════════════

def test_api_compliance():
    print("\n" + "═" * 60)
    print("  TEST 1: PettingZoo Parallel API Compliance")
    print("═" * 60)

    try:
        from pettingzoo.test import parallel_api_test
        env = MultiAgentOTAEnv(n_agents=3, n_blocks=6, bd_mode=False, stochastic_latency=False)
        parallel_api_test(env, num_cycles=10)
        print("  ✅  parallel_api_test PASSED")
        return True
    except ImportError:
        print("  ⚠️  pettingzoo.test not available — skipping API test")
        return True
    except Exception as e:
        print(f"  ❌  parallel_api_test FAILED: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  TEST 2 — Full Random Episodes (3 runs)
# ══════════════════════════════════════════════════════════════

def test_random_episodes():
    print("\n" + "═" * 60)
    print("  TEST 2: Full Random Episodes (3 runs, 4 agents, 12 blocks)")
    print("═" * 60)

    env = MultiAgentOTAEnv(
        n_agents=4, n_blocks=12, bd_mode=True,
        stochastic_latency=True
    )

    all_passed = True
    for run_id in range(3):
        t0     = time.time()
        result = run_episode(env, seed=run_id * 10)
        elapsed = time.time() - t0

        print(f"\n  Run {run_id + 1}  |  global_steps={result['global_steps']}  "
              f"|  time={elapsed:.2f}s")
        print(f"  {'Agent':8s}  {'Payload':>10s}  {'Memory':>10s}  {'Steps':>6s}")
        print(f"  {'-'*40}")

        for agent, stats in result["agents"].items():
            status = "✓" if stats["steps"] > 0 else "?"
            print(f"  {agent:8s}  {stats['payload']:10.1f}  {stats['memory']:10.1f}  "
                  f"{stats['steps']:6d}  {status}")

        # All agents should have been terminated
        any_unfinished = any(s["steps"] == 0 for s in result["agents"].values())
        if any_unfinished:
            print("  ⚠️  Some agents show 0 steps — may have been instantly truncated")
        else:
            print(f"\n  ✅  Run {run_id + 1} completed cleanly")

    return all_passed


# ══════════════════════════════════════════════════════════════
#  TEST 3 — Death Masking Validation
# ══════════════════════════════════════════════════════════════

def test_death_masking():
    print("\n" + "═" * 60)
    print("  TEST 3: Death Masking — Terminated Agents Return Zero-Vector Obs")
    print("═" * 60)

    # Use very few blocks so agents finish quickly and we can observe masking
    env = MultiAgentOTAEnv(n_agents=4, n_blocks=4, bd_mode=False, stochastic_latency=False)
    obs, _ = env.reset(seed=99)

    death_mask_verified = defaultdict(bool)
    step = 0

    while env.agents or step < 30:
        if not env.agents:
            break

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        step += 1

        # Check: terminated agents must have zero-vector obs (except agent_id)
        for agent in env.possible_agents:
            if terms[agent] or truncs[agent]:
                a_obs = obs[agent]
                mask_sum = float(np.sum(a_obs["mask"]))
                enc_sum  = float(a_obs["cum_encoding_cost"][0])
                tx_sum   = float(a_obs["cum_tx_cost"][0])
                mem_sum  = float(a_obs["memory_used"][0])
                id_sum   = float(np.sum(a_obs["agent_id"]))   # should be 1.0 (one-hot)

                zeros_ok  = (mask_sum == 0) and (enc_sum == 0) and (tx_sum == 0) and (mem_sum == 0)
                id_ok     = abs(id_sum - 1.0) < 1e-5

                if not death_mask_verified[agent]:
                    if zeros_ok and id_ok:
                        print(f"  ✅  {agent} death mask CORRECT at step {step}  "
                              f"(all zeros, agent_id={a_obs['agent_id'].tolist()})")
                        death_mask_verified[agent] = True
                    else:
                        print(f"  ❌  {agent} death mask WRONG at step {step}  "
                              f"zeros_ok={zeros_ok}, id_ok={id_ok} id_sum={id_sum:.3f}")

    unverified = [a for a in env.possible_agents if not death_mask_verified[a]]
    if unverified:
        print(f"  ⚠️  Could not verify death mask for: {unverified} (may not have died in test)")
    else:
        print("\n  ✅  All agents' death masks verified correctly!")

    return len(unverified) == 0


# ══════════════════════════════════════════════════════════════
#  TEST 4 — Stochastic Latency Variance
# ══════════════════════════════════════════════════════════════

def test_stochastic_latency():
    print("\n" + "═" * 60)
    print("  TEST 4: Stochastic Latency — TX costs should differ from fixed latency")
    print("═" * 60)

    env_stoch = MultiAgentOTAEnv(n_agents=2, n_blocks=8, bd_mode=False, stochastic_latency=True)
    env_fixed = MultiAgentOTAEnv(n_agents=2, n_blocks=8, bd_mode=False, stochastic_latency=False)

    # Build one deterministic action trace, then replay it in both envs.
    # This isolates latency sampling from random-policy noise.
    trace_env = MultiAgentOTAEnv(n_agents=2, n_blocks=8, bd_mode=False, stochastic_latency=False)
    trace_env.reset(seed=123)
    action_trace = []
    while trace_env.agents:
        actions = {agent: trace_env.action_space(agent).sample() for agent in trace_env.agents}
        action_trace.append(actions)
        trace_env.step(actions)

    env_stoch.reset(seed=123)
    env_fixed.reset(seed=123)

    stoch_payloads = []
    fixed_payloads = []

    for actions in action_trace:
        _, _, _, _, infos_stoch = env_stoch.step(actions)
        _, _, _, _, infos_fixed = env_fixed.step(actions)

        stoch_payloads.append(np.mean([
            infos_stoch.get(agent, {}).get("payload_bytes", 0.0)
            for agent in env_stoch.possible_agents
        ]))
        fixed_payloads.append(np.mean([
            infos_fixed.get(agent, {}).get("payload_bytes", 0.0)
            for agent in env_fixed.possible_agents
        ]))

    stoch_payloads = np.array(stoch_payloads, dtype=np.float64)
    fixed_payloads = np.array(fixed_payloads, dtype=np.float64)
    diff = np.abs(stoch_payloads - fixed_payloads)

    print(f"\n  Mean absolute payload difference: {diff.mean():.2f}")
    print(f"  Max  absolute payload difference: {diff.max():.2f}")

    if np.any(diff > 1e-6):
        print("  ✅  Stochastic env differs from fixed-latency env as expected")
        return True
    else:
        print("  ⚠️  Stochastic env matches fixed-latency env — check latency sampling")
        return False


# ══════════════════════════════════════════════════════════════
#  TEST 5 — N-Agent Scalability
# ══════════════════════════════════════════════════════════════

def test_scalability():
    print("\n" + "═" * 60)
    print("  TEST 5: N-Agent Scalability (2 → 4 → 8 → 16 agents)")
    print("═" * 60)

    print(f"\n  {'N Agents':>10s}  {'Time (s)':>10s}  {'Global Steps':>14s}  {'Status':>8s}")
    print(f"  {'-'*50}")

    all_ok = True
    for n in [2, 4, 8, 16]:
        env = MultiAgentOTAEnv(n_agents=n, n_blocks=12, bd_mode=True, stochastic_latency=True)
        t0 = time.time()
        try:
            result = run_episode(env, seed=7)
            elapsed = time.time() - t0
            print(f"  {n:>10d}  {elapsed:>10.3f}  {result['global_steps']:>14d}  {'✅ OK':>8s}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {n:>10d}  {elapsed:>10.3f}  {'—':>14s}  {'❌ ERR':>8s}  {e}")
            all_ok = False

    return all_ok


# ══════════════════════════════════════════════════════════════
#  Main runner
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   MultiAgentOTAEnv — Phase 2 Step 1 Validation Suite      ║")
    print("╚" + "═" * 58 + "╝")

    results = {}

    results["api_compliance"]       = test_api_compliance()
    results["random_episodes"]      = test_random_episodes()
    results["death_masking"]        = test_death_masking()
    results["stochastic_latency"]   = test_stochastic_latency()
    results["scalability"]          = test_scalability()

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    print("═" * 60)
    if all_passed:
        print("  🎉  ALL TESTS PASSED — MultiAgentOTAEnv is ready!")
    else:
        print("  ⚠️  Some tests failed — review output above.")
    print("═" * 60 + "\n")
