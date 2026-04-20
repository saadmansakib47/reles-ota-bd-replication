import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hashlib
import json
from pathlib import Path

class OTAEnv(gym.Env):
    """
    ReLES-OTA Custom Gymnasium Environment (faithful replication + BD adaptation)
    Paper: ReLES-OTA (Bhattacharjee et al., 2025)
    Models full-system OTA as block-wise sequential construction.
    """

    def __init__(self, n_blocks: int = 32, block_size: int = 4096, bd_mode: bool = False, bd_params_path: str = "bd_params.json"):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.bd_mode = bd_mode

        # Load BD parameters if available
        self.bd_params = self._load_bd_params(bd_params_path)

        # Dummy old/new images (in real runs: load actual firmware binaries)
        self.old_blocks = [np.random.bytes(block_size) for _ in range(n_blocks)]
        self.new_blocks = [np.random.bytes(block_size) for _ in range(n_blocks)]  # simulate differences

        self.old_hashes = [hashlib.md5(b).digest() for b in self.old_blocks]
        self.new_hashes = [hashlib.md5(b).digest() for b in self.new_blocks]

        # Observation space (matches paper: mask + costs + memory)
        self.observation_space = spaces.Dict({
            "mask": spaces.MultiBinary(n_blocks),           # 1 = unprocessed
            "cum_encoding_cost": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "cum_tx_cost": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "memory_used": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "step": spaces.Box(low=0, high=n_blocks, shape=(1,), dtype=np.int32),
        })

        # Action space: (block_index to process, operation_type)
        # operation: 0 = Copy (identical), 1 = Modify (M), 2 = Modify+Backup (MB)
        self.action_space = spaces.MultiDiscrete([n_blocks, 3])

        self.reset()

    def _load_bd_params(self, path: str):
        default = {
            "latency_base_ms": 60.0,
            "packet_loss_rate": 0.01,
            "memory_budget_fraction": 1.0,
            "bandwidth_mbps": 50.0
        }
        if Path(path).exists():
            with open(path, 'r') as f:
                loaded = json.load(f)
                default.update(loaded)
        return default

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.mask = np.ones(self.n_blocks, dtype=np.int32)  # 1 = unprocessed
        self.cum_encoding_cost = 0.0
        self.cum_tx_cost = 0.0
        self.cum_memory = 0.0
        self.processed_blocks = []

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "mask": self.mask.copy(),
            "cum_encoding_cost": np.array([self.cum_encoding_cost], dtype=np.float32),
            "cum_tx_cost": np.array([self.cum_tx_cost], dtype=np.float32),
            "memory_used": np.array([self.cum_memory / max(self.bd_params["memory_budget_fraction"], 0.01)], dtype=np.float32),
            "step": np.array([self.current_step], dtype=np.int32),
        }

    def render(self):
        print(f"Step: {self.current_step}/{self.n_blocks} | Processed: {len(self.processed_blocks)} | "
              f"Total Cost: {self.cum_encoding_cost + self.cum_tx_cost:.2f}")

    # WE WILL IMPLEMENT STEP() IN CHECKPOINT 2
    def step(self, action):
        raise NotImplementedError("Implement in next checkpoint")