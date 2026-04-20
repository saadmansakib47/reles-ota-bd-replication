import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hashlib
import json
import random
import time
from pathlib import Path

class OTAEnv(gym.Env):
    """
    ReLES-OTA Custom Gymnasium Environment (faithful replication + BD adaptation)
    """

    def __init__(self, n_blocks: int = 24, block_size: int = 4096, bd_mode: bool = False, bd_params_path: str = "bd_params.json"):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.bd_mode = bd_mode

        self.bd_params = self._load_bd_params(bd_params_path)

        # Dummy firmware blocks
        self.old_blocks = [np.random.bytes(block_size) for _ in range(n_blocks)]
        self.new_blocks = [np.random.bytes(block_size) for _ in range(n_blocks)]
        self.old_hashes = [hashlib.md5(b).digest() for b in self.old_blocks]
        self.new_hashes = [hashlib.md5(b).digest() for b in self.new_blocks]

        # Pre-compute similarity bias for speed
        self.similarity_bias = np.random.uniform(0.4, 0.9, n_blocks)

        # Observation space
        self.observation_space = spaces.Dict({
            "mask": spaces.MultiBinary(n_blocks),
            "cum_encoding_cost": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "cum_tx_cost": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "memory_used": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "step": spaces.Box(low=0, high=n_blocks, shape=(1,), dtype=np.int32),
        })

        self.action_space = spaces.MultiDiscrete([n_blocks, 3])

        self.episode_start_time = None
        self.reset()

    def _load_bd_params(self, path: str):
        default = {
            "latency_base_ms": 60.0,
            "packet_loss_rate": 0.01,
            "memory_budget_fraction": 1.0,
            "bandwidth_mbps": 50.0
        }
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                    default.update(loaded)
            except:
                pass
        return default

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        super().reset(seed=seed)
        self.current_step = 0
        self.mask = np.ones(self.n_blocks, dtype=np.int32)
        self.cum_encoding_cost = 0.0
        self.cum_tx_cost = 0.0
        self.cum_memory = 0.0
        self.processed_blocks = []
        self.episode_start_time = time.time()   # Start timer
        
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "mask": self.mask.copy(),
            "cum_encoding_cost": np.array([self.cum_encoding_cost], dtype=np.float32),
            "cum_tx_cost": np.array([self.cum_tx_cost], dtype=np.float32),
            "memory_used": np.array([min(self.cum_memory / max(self.bd_params["memory_budget_fraction"], 0.01), 1.0)], dtype=np.float32),
            "step": np.array([self.current_step], dtype=np.int32),
        }

    def _estimate_delta_size(self, block_idx: int, operation: int) -> float:
        similarity = self.similarity_bias[block_idx]
        if operation == 0:      # Copy
            similarity = min(0.95, similarity + 0.25)
        elif operation == 2:    # MB
            similarity -= 0.18
        
        base_delta = self.block_size * (1.0 - similarity)
        if operation == 2:
            base_delta *= 1.45
        return max(64.0, base_delta)

    def _calculate_tx_cost(self, payload_bytes: float) -> float:
        latency_factor = 1.0 + (self.bd_params["latency_base_ms"] / 800.0)
        loss_factor = 1.0 + (self.bd_params["packet_loss_rate"] * 6.0)
        bandwidth_factor = 800.0 / max(self.bd_params["bandwidth_mbps"], 5.0)
        
        tx_cost = payload_bytes * latency_factor * loss_factor * bandwidth_factor * 0.0008
        return tx_cost

    def step(self, action):
        # === Time Safety Timeout (30 seconds max per episode) ===
        if self.episode_start_time is not None:
            elapsed = time.time() - self.episode_start_time
            if elapsed > 30.0:
                print(f"⚠️ Episode timeout after {elapsed:.1f}s")
                return self._get_obs(), -200.0, True, False, {"early_termination": "timeout"}

        # === Step limit safety ===
        if self.current_step > self.n_blocks + 10:
            return self._get_obs(), -100.0, True, False, {"early_termination": "step_limit"}

        block_idx, operation = action

        # Invalid action
        if self.mask[block_idx] == 0 or block_idx >= self.n_blocks:
            return self._get_obs(), -50.0, False, False, {"invalid_action": True}

        # Calculate costs
        delta_size = self._estimate_delta_size(block_idx, operation)
        encoding_cost = delta_size * (1.25 if operation == 2 else 1.0)
        tx_cost = self._calculate_tx_cost(delta_size)
        memory_overhead = delta_size * (2.6 if operation == 2 else 1.9)

        # Update
        self.cum_encoding_cost += encoding_cost
        self.cum_tx_cost += tx_cost
        self.cum_memory += memory_overhead

        self.mask[block_idx] = 0
        self.processed_blocks.append(block_idx)
        self.current_step += 1

        # Reward
        total_cost = encoding_cost + tx_cost + memory_overhead * 0.3
        reward = -total_cost

        done = np.all(self.mask == 0)
        if done:
            success_bonus = 250.0 * (1.0 - (self.cum_encoding_cost + self.cum_tx_cost) / (self.n_blocks * self.block_size * 0.7))
            reward += success_bonus

        info = {
            "payload_bytes": self.cum_encoding_cost + self.cum_tx_cost,
            "memory_used": self.cum_memory,
            "blocks_processed": len(self.processed_blocks)
        }

        return self._get_obs(), reward, done, False, info

    def render(self):
        total = self.cum_encoding_cost + self.cum_tx_cost
        print(f"Step {self.current_step:2d}/{self.n_blocks} | Payload: {total:7.1f} | Memory: {self.cum_memory:6.1f}")