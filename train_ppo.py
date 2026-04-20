import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import os
from pathlib import Path
from ota_env import OTAEnv

# ====================== CONFIG ======================
N_BLOCKS = 16 
TOTAL_TIMESTEPS = 50_000          # Start small, increase later
N_ENVS = 2                         # Vectorized environments for speed
EVAL_FREQ = 10_000
SAVE_FREQ = 25_000

Path("results/models").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)

# ====================== TRAINING FUNCTION ======================
def train_ppo(version: str = "generic", bd_mode: bool = False):
    print(f"\n Starting PPO Training → {version.upper()} version (BD={bd_mode})")
    
    # Create vectorized environment
    def make_env():
        return OTAEnv(n_blocks=N_BLOCKS, bd_mode=bd_mode)
    
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    
    # PPO Model
    model = PPO(
        policy="MultiInputPolicy",      # Important: handles Dict observation space
        env=vec_env,
        verbose=1,
        tensorboard_log=f"results/logs/ppo_{version}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f"results/models/{version}_best",
        log_path=f"results/logs/{version}_eval",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"results/models/{version}_checkpoints/",
        name_prefix=f"ppo_{version}"
    )
    
    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"results/models/ppo_{version}_final")
    print(f" Training completed for {version} version.")
    return model


# ====================== QUICK TEST RUN ======================
if __name__ == "__main__":
    # Generic already trained → now train BD with real parameters
    print(" Training BD version with real Bangladesh parameters from PDF...")
    model_bd = train_ppo(version="bd", bd_mode=True)
    
    print("\n BD training completed!")
    print("Models saved in results/models/")
    print("Next step: run experiment_runner.py")