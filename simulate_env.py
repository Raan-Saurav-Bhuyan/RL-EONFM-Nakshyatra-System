import gymnasium as gym
import eon_env # Important: This registers the environment

if __name__ == '__main__':
    # Create the environment: --->
    env = gym.make('EON-v0')

    # Reset the environment to get the initial state: --->
    observation, info = env.reset()

    print("--- Initial State ---")
    print("Observation shape:", observation.shape)
    print("Initial GSNR values (first 5 lightpaths):")
    print(observation[:5, 0])

    terminated = False
    truncated = False
    total_reward = 0

    # Run the simulation for a few steps (days): --->
    for day in range(1, 50):
        if terminated or truncated:
            break

        # Action is 0 (continue simulation): --->
        action = env.action_space.sample()

        # Take a step: --->
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Render the state to the console: --->
        env.render()

    print("\n--- Simulation Finished ---")
    print(f"Total steps: {info['step']}")
    print(f"Final total reward: {total_reward:.2f}")

    # The final 'observation' is the matrix of OPM metrics: --->
    print("\nFinal observation matrix (state) for clustering:")
    print(observation)

    env.close()
