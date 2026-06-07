import os
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError("TensorBoard is required for logging but is not installed. Please run 'pip install tensorboard'.")

# Import custom modules: --->
from eon_env.v2.temporal_mdp_wrapper import TemporalEONEnvV2
from PPO.CNN_PPO import PPOAgentCNN

if __name__ == '__main__':
    # Initialize Temporal MDP Environment for Predictive Maintenance: --->
    env = TemporalEONEnvV2()

    state_dim_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Initialized CNN-PPO Agent | State Shape: {state_dim_shape} | Action Dim: {action_dim}")
    agent = PPOAgentCNN(action_dim = action_dim)

    # Initialize TensorBoard Writer with timestamp: --->
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/ppo_cnn_eon_v2_{current_time}")

    # Training Hyperparameters: --->
    # (Note: In the V2 Temporal MDP, 1 'step' encapsulates 10 years of simulated operation.
    # The environment immediately terminates after evaluating the predictive maintenance action.)
    max_episodes = 100
    update_timestep = 3  # Update policy on every episodes

    time_step = 0

    print("\n--- Starting Pre-Training of CNN PPO Agent ---")
    for ep in range(1, max_episodes + 1):
        print(f"\n--- Episode {ep}/{max_episodes} (Simulating 10 Years...) ---")

        # Resetting simulates 10 full years and returns the (10, 5, 18) temporal stack: --->
        state, _ = env.reset()

        time_step += 1
        action = agent.select_action(state)

        # Step executes the predictive maintenance decision and returns immediate reward: --->
        next_state, reward, terminated, truncated, info = env.step(action)

        # Record transitions for PPO buffer: --->
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(terminated)

        # Retrieve diagnostic metrics: --->
        degradation = info.get('degradation_db', 0.0)
        action_name = "Isolate/Maintain" if action == 1 else "Monitor"
        print(f"Action Taken: {action_name} | Degradation: {degradation:.2f} dB | Reward: {reward:.2f}")

        # Log metrics to TensorBoard: --->
        writer.add_scalar('Reward/Episode_Reward', reward, ep)
        writer.add_scalar('Environment/Degradation_dB', degradation, ep)
        writer.add_scalar('Action/Action_Chosen', action, ep)

        # Update the PPO Agent: --->
        if time_step % update_timestep == 0 and time_step != 1:
            print("\n[Updating CNN Actor-Critic Networks...]")
            a_loss, c_loss, t_loss = agent.update()
            print(f"Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f} | Total loss: {t_loss:.4f}")

            # Log losses: --->
            writer.add_scalar('Loss/Actor', a_loss, time_step)
            writer.add_scalar('Loss/Critic', c_loss, time_step)
            writer.add_scalar('Loss/Total', t_loss, time_step)

    writer.close()
    print("\n--- Pre-Training Complete ---")
