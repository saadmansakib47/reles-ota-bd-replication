"""
train_mappo.py — MAPPO/IPPO Training Scaffold for Phase 2
==========================================================
Training script for multi-agent OTA update learning.

Phase 2 Step 1 (current):
  - IPPO (Independent PPO): each agent has its own PPO policy,
    no parameter sharing, simplest MARL baseline.
  - Uses SuperSuit to wrap PettingZoo env into SB3-compatible VecEnv.

Phase 2 Steps 2+:
  - MAPPO: centralized critic, decentralized actors.
  - Parameter sharing across agents.

Usage
-----
    # IPPO training (default, recommended for Step 1)
    python train_mappo.py --mode ippo --n_agents 4 --timesteps 100000

    # Quick sanity run
    python train_mappo.py --mode random --n_agents 4

References
----------
- MAPPO-PIS  (arXiv:2408.06656)
- FP3O       (arXiv:2310.05053)
- MARL-CC    (arXiv:2511.17653)
"""

import argparse
import json
# pyrefly: ignore [missing-import]
import numpy as np
import time
import sys
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── Env imports ──
from marl_ota_env import MultiAgentOTAEnv


# ══════════════════════════════════════════════════════════════
#  Random-agent baseline (no learning — sanity/benchmark only)
# ══════════════════════════════════════════════════════════════

def run_random_marl(n_agents: int = 4, n_blocks: int = 16, n_episodes: int = 20,
                    bd_mode: bool = False) -> dict:
    """
    Run N episodes with purely random agents.
    Used as the MARL baseline (equivalent to Phase 1's random baseline).
    """
    print(f"\n Running Random Multi-Agent Baseline")
    print(f"   n_agents={n_agents}, n_blocks={n_blocks}, n_episodes={n_episodes}, bd_mode={bd_mode}")

    env = MultiAgentOTAEnv(
        n_agents=n_agents, n_blocks=n_blocks,
        bd_mode=bd_mode, stochastic_latency=bd_mode
    )

    episode_payloads     = []
    episode_memories     = []
    episode_steps        = []
    per_agent_payloads   = defaultdict(list)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        step   = 0
        ep_payload = 0.0

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, _, terms, truncs, infos = env.step(actions)
            step += 1

        # Aggregate after episode
        total_payload = sum(env.cum_enc_cost[a] + env.cum_tx_cost[a] for a in env.possible_agents)
        total_memory  = sum(env.cum_memory[a] for a in env.possible_agents)
        episode_payloads.append(total_payload)
        episode_memories.append(total_memory)
        episode_steps.append(step)

        for agent in env.possible_agents:
            per_agent_payloads[agent].append(env.cum_enc_cost[agent] + env.cum_tx_cost[agent])

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"fleet_payload={total_payload:.1f}  "
                  f"steps={step}")

    results = {
        "method":          "random_marl",
        "n_agents":        n_agents,
        "n_episodes":      n_episodes,
        "bd_mode":         bd_mode,
        "mean_fleet_payload": float(np.mean(episode_payloads)),
        "std_fleet_payload":  float(np.std(episode_payloads)),
        "mean_fleet_memory":  float(np.mean(episode_memories)),
        "mean_episode_steps": float(np.mean(episode_steps)),
        "per_agent_mean_payload": {
            a: float(np.mean(per_agent_payloads[a])) for a in env.possible_agents
        },
    }

    print(f"\n  Mean fleet payload : {results['mean_fleet_payload']:.2f} ± {results['std_fleet_payload']:.2f}")
    print(f"  Mean fleet memory  : {results['mean_fleet_memory']:.2f}")
    print(f"  Mean episode steps : {results['mean_episode_steps']:.1f}")
    return results


# ══════════════════════════════════════════════════════════════
#  IPPO training (Independent PPO via SuperSuit + SB3)
# ══════════════════════════════════════════════════════════════

def train_ippo(
    n_agents: int       = 4,
    n_blocks: int       = 16,
    bd_mode: bool       = False,
    total_timesteps: int = 200_000,
    save_dir: str       = "results/marl_models",
) -> None:
    """
    Train Independent PPO (IPPO) on the multi-agent OTA env.

    IPPO treats each agent as an independent learner with its own
    PPO policy. No parameter sharing, no centralized critic.
    Simple but effective baseline for MARL — often competitive with MAPPO.

    SuperSuit is used to:
      1. Convert PettingZoo Parallel → AEC (parallel_to_aec)
      2. Wrap into a Gymnasium-compatible VecEnv for SB3
    """
    try:
        # pyrefly: ignore [import, missing-import]
        import supersuit as ss
        # pyrefly: ignore [import, missing-import]
        from stable_baselines3 import PPO
        # pyrefly: ignore [import, missing-import]
        from stable_baselines3.common.vec_env import VecMonitor
        # pyrefly: ignore [import, missing-import]
        from fp3o_policy import FP3OPolicy, ValueNormalizationCallback, make_fp3o_policy_kwargs
    except ImportError as e:
        print(f"  ❌  Missing dependency: {e}")
        print("  Install with:  pip install supersuit stable-baselines3")
        return

    print(f"\n Training IPPO on MultiAgentOTAEnv")
    print(f"   n_agents={n_agents}, n_blocks={n_blocks}, bd_mode={bd_mode}")
    print(f"   total_timesteps={total_timesteps:,}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Build env ──
    def make_env():
        return MultiAgentOTAEnv(
            n_agents=n_agents, n_blocks=n_blocks,
            bd_mode=bd_mode, stochastic_latency=bd_mode
        )

    raw_env = make_env()

    # SuperSuit: PettingZoo Parallel → SB3-compatible VecEnv
    # MarkovVectorEnv handles variable-length episodes via black_death masking.
    
    # pyrefly: ignore [missing-import]
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv
    env = MarkovVectorEnv(raw_env, black_death=True)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # ── Patch missing VecEnv interface methods onto ConcatVecEnv ──────────────
    # SB3's VecEnvWrapper.__init__ calls get_attr("render_mode") on the inner
    # env; ConcatVecEnv does not implement get_attr / set_attr / env_method,
    # which causes AttributeError inside VecMonitor.
    # Setting render_mode explicitly silences the SB3 UserWarning.
    env.render_mode = None
    if not hasattr(env, "get_attr"):
        def _get_attr(attr_name, indices=None):
            val  = getattr(env, attr_name, None)
            idxs = list(range(env.num_envs)) if indices is None else (
                [indices] if isinstance(indices, int) else list(indices)
            )
            return [val for _ in idxs]
        def _set_attr(attr_name, value, indices=None): pass
        def _env_method(method_name, *method_args, indices=None, **method_kwargs): return []
        env.get_attr   = _get_attr
        env.set_attr   = _set_attr
        env.env_method = _env_method

    # Dummy seed stub required by older SB3 internals.
    env.seed = lambda seed=None: None
    env = VecMonitor(env)

    # ── PPO model ──
    policy_kwargs = make_fp3o_policy_kwargs(
        n_blocks=n_blocks,
        ecu_type="generic"
    )

    model = PPO(
        policy          = FP3OPolicy,
        env             = env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = f"{save_dir}/logs/ippo_{'bd' if bd_mode else 'generic'}",
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        gae_lambda      = 0.95,
        gamma           = 0.99,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
    )
    # Seed RNGs without triggering env.seed() propagation (SB3 >=2.x)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\n  Starting IPPO training...")
    t0 = time.time()
    callback = ValueNormalizationCallback()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    elapsed = time.time() - t0

    tag = "bd" if bd_mode else "generic"
    model.save(f"{save_dir}/ippo_{tag}_final")
    print(f"\n  Training done in {elapsed:.1f}s  →  saved to {save_dir}/ippo_{tag}_final")
    env.close()


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 MARL Training — OTA Update Environment")
    parser.add_argument("--mode",       choices=["random", "ippo"], default="random",
                        help="Training mode: 'random' for baseline, 'ippo' for IPPO training")
    parser.add_argument("--n_agents",   type=int, default=4,       help="Number of ECU agents")
    parser.add_argument("--n_blocks",   type=int, default=16,      help="Firmware blocks per agent")
    parser.add_argument("--bd_mode",    action="store_true",        help="Enable BD network parameters")
    parser.add_argument("--timesteps",  type=int, default=100_000, help="IPPO training timesteps")
    parser.add_argument("--episodes",   type=int, default=20,      help="Episodes for random baseline")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 55 + "╗")
    print("║   Phase 2 — Multi-Agent OTA Training                  ║")
    print("╚" + "═" * 55 + "╝")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Agents: {args.n_agents} | Blocks/agent: {args.n_blocks} | BD mode: {args.bd_mode}")

    if args.mode == "random":
        results = run_random_marl(
            n_agents   = args.n_agents,
            n_blocks   = args.n_blocks,
            n_episodes = args.episodes,
            bd_mode    = args.bd_mode,
        )
        out_path = Path("results/marl_random_baseline.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    elif args.mode == "ippo":
        train_ippo(
            n_agents        = args.n_agents,
            n_blocks        = args.n_blocks,
            bd_mode         = args.bd_mode,
            total_timesteps = args.timesteps,
        )


if __name__ == "__main__":
    main()
