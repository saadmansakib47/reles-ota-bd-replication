from ota_env import OTAEnv

env = OTAEnv(n_blocks=16, bd_mode=False)
obs, _ = env.reset()
print("Environment created successfully!")

total_reward = 0.0
done = False

for step in range(30):          # max 30 steps
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    
    if done:
        print("\n Episode finished successfully!")
        break

print(f"\nTotal reward: {total_reward:.2f}")
print(f"Final payload cost: {info.get('payload_bytes', 0):.1f}")
print(" Checkpoint 2 test completed!")