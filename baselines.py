import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ota_env import OTAEnv
import json
from pathlib import Path

class BaselineRunner:
    def __init__(self, n_blocks=32, bd_mode=False):
        self.n_blocks = n_blocks
        self.bd_mode = bd_mode
        self.env = None
        self.reset_env()

    def reset_env(self):
        self.env = OTAEnv(n_blocks=self.n_blocks, bd_mode=self.bd_mode)

    def run_random_baseline(self, num_episodes=50):
        """Random action baseline"""
        payloads = []
        memories = []
        
        for ep in range(num_episodes):
            self.reset_env()
            total_reward = 0
            done = False
            
            while not done:
                action = self.env.action_space.sample()
                _, reward, done, _, info = self.env.step(action)
                total_reward += reward
            
            payloads.append(info.get('payload_bytes', 0))
            memories.append(info.get('memory_used', 0))
        
        return {
            'mean_payload': np.mean(payloads),
            'std_payload': np.std(payloads),
            'mean_memory': np.mean(memories),
            'name': 'Random'
        }

    def run_sequential_baseline(self, num_episodes=20):
        """Always process blocks in order 0→1→2... with random operation"""
        payloads = []
        memories = []
        
        for ep in range(num_episodes):
            self.reset_env()
            done = False
            
            while not done:
                # Choose next unprocessed block in order
                unprocessed = np.where(self.env.mask == 1)[0]
                if len(unprocessed) == 0:
                    break
                block_idx = unprocessed[0]  # sequential
                operation = np.random.randint(0, 3)  # random op
                action = [block_idx, operation]
                
                _, _, done, _, info = self.env.step(action)
            
            payloads.append(info.get('payload_bytes', 0))
            memories.append(info.get('memory_used', 0))
        
        return {
            'mean_payload': np.mean(payloads),
            'std_payload': np.std(payloads),
            'mean_memory': np.mean(memories),
            'name': 'Sequential'
        }

    def run_multiple_episodes(self, num_episodes=30):
        """Run random policy for statistics"""
        results = self.run_random_baseline(num_episodes)
        return results


# ==================== Plotting & Comparison ====================
def plot_baseline_comparison(generic_results, bd_results=None):
    plt.figure(figsize=(12, 5))

    # Payload comparison
    plt.subplot(1, 2, 1)
    names = [generic_results['name']]
    payloads = [generic_results['mean_payload']]
    if bd_results:
        names.append(bd_results['name'] + " (BD)")
        payloads.append(bd_results['mean_payload'])
    
    plt.bar(names, payloads, color=['skyblue', 'orange'])
    plt.ylabel('Mean Payload Cost')
    plt.title('Payload Size Comparison')
    plt.grid(axis='y', alpha=0.3)

    # Memory comparison
    plt.subplot(1, 2, 2)
    memories = [generic_results['mean_memory']]
    if bd_results:
        memories.append(bd_results['mean_memory'])
    
    plt.bar(names, memories, color=['skyblue', 'orange'])
    plt.ylabel('Mean Memory Overhead')
    plt.title('Memory Usage Comparison')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved to results/baseline_comparison.png")


# ==================== Quick Run ====================
if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    
    print("Running Generic baselines...")
    runner = BaselineRunner(n_blocks=24, bd_mode=False)
    generic_stats = runner.run_random_baseline(num_episodes=30)
    seq_stats = runner.run_sequential_baseline(num_episodes=15)
    
    print(f"\nGeneric Random Baseline:")
    print(f"   Mean Payload : {generic_stats['mean_payload']:.1f}")
    print(f"   Mean Memory  : {generic_stats['mean_memory']:.1f}")
    
    print(f"\nSequential Baseline:")
    print(f"   Mean Payload : {seq_stats['mean_payload']:.1f}")
    
    # Optional BD run (if bd_params.json is good)
    if Path("bd_params.json").exists():
        print("\nRunning BD version...")
        bd_runner = BaselineRunner(n_blocks=24, bd_mode=True)
        bd_stats = bd_runner.run_random_baseline(20)
        plot_baseline_comparison(generic_stats, bd_stats)
    else:
        plot_baseline_comparison(generic_stats)