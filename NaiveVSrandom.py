import time
import gym
import numpy as np

def naive_policy(obs):
    angle = obs[0][2] if isinstance(obs[0], np.ndarray) else 0  
    return 0 if angle < 0 else 1

def random_policy(obs):
    return 0 if np.random.uniform() < 0.5 else 1

def naive_main(policy):
    env = gym.make("CartPole-v0")
    obs = env.reset()
    env.render()

    totals = []
    for episode in range(100):
        episode_rewards = 0
        obs = env.reset()
        for step in range(10000):
            action = policy(obs)
            step_result = env.step(action)
            next_obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            env.render()
            time.sleep(0.1)
            episode_rewards += reward
            if done:
                print("Game over. Number of steps =", step)
                env.render()
                time.sleep(3.14)
                break
            obs = next_obs  
        totals.append(episode_rewards)
    print(f"Mean: {np.mean(totals)}, Std: {np.std(totals)}, Min: {np.min(totals)}, Max: {np.max(totals)}")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs = env.reset()
    obs_dimension = len(obs[0]) if isinstance(obs[0], np.ndarray) else 0
    print("Observation:", obs)
    print("Observation dimension:", len(obs) if isinstance(obs, tuple) else 0)

    if obs_dimension == 4:
        naive_main(naive_policy)
        # Uncomment the line below to test the random policy
        # naive_main(random_policy)
    else:
        print("Observation space dimension mismatch. Expected 4, got:", obs_dimension)
