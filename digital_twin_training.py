# Import libraries: --->
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError("TensorBoard is required for logging but is not installed. Please run 'pip install tensorboard'.")

# Import custom modules: --->
import gymnasium as gym
import eon_env
from eon_env.v1.rl_wrapper import Surrogate_Reward_Wrapper
from PPO import PPOAgent

if __name__ == '__main__':
    # 1. Initialize Base Environment and apply RL Wrapper: --->
    base_env = gym.make('EON-v0')
    env = Surrogate_Reward_Wrapper(base_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Initialized PPO Agent | State Dim: {state_dim} | Action Dim: {action_dim}")
    agent = PPOAgent(state_dim, action_dim)

    # Initialize TensorBoard Writer: --->
    writer = SummaryWriter(log_dir="runs/ppo_eon_training")

    max_episodes = 100
    max_steps = 400             # <--- Each episodes has a max. of 365 days of run (determined by max. simulation run)
    update_timestep = 400

    time_step = 0

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            print(f"\nStep = {step}")
            time_step += 1
            action = agent.select_action(state)

            state, reward, terminated, truncated, _ = env.step(action)

            agent.buffer['rewards'].append(reward)
            agent.buffer['is_terminals'].append(terminated)
            ep_reward += reward

            if time_step % update_timestep == 0:
                a_loss, c_loss, t_loss = agent.update()
                print(f"\nUpdated the agent: Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f} | Total loss: {t_loss:.4f}\n")

                # Log losses to TensorBoard: --->
                writer.add_scalar('Loss/Actor', a_loss, time_step)
                writer.add_scalar('Loss/Critic', c_loss, time_step)
                writer.add_scalar('Loss/Total', t_loss, time_step)

            # Ensure logging happens if terminated, truncated, or we run out of steps
            if terminated or truncated or step == max_steps - 1:
                print(f"\nEpisode: {ep:3d} | Reward: {ep_reward:7.2f} | Steps: {step+1:3d} | Action Taken: {action}\n")
                writer.add_scalar('Reward/Episode', ep_reward, ep)

                break

    writer.close()
