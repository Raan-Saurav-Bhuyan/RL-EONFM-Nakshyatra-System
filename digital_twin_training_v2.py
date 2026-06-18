import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.tensorboard import SummaryWriter

# Import custom modules: --->
from eon_env.v2.temporal_mdp_wrapper import TemporalEONEnvV2
from eon_env.v2.localization_mdp_wrapper import ComponentLocalizationEnv
from PPO.CNN_PPO import PPOAgentCNN
from PPO.GNN_PPO.ppo_agent_gnn import PPOAgentGNN

if __name__ == '__main__':
    # Create directory for classification visualizations --->
    os.makedirs("visualizations/classification_metrics", exist_ok=True)

    # Initialize Temporal MDP Environment for Predictive Maintenance: --->
    env = TemporalEONEnvV2()

    state_dim_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Initialized CNN-PPO Agent | State Shape: {state_dim_shape} | Action Dim: {action_dim}")
    agent = PPOAgentCNN(action_dim = action_dim)
    gnn_agent = PPOAgentGNN()

    # Initialize TensorBoard Writer with timestamp: --->
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/ppo_cnn_eon_v2_{current_time}")

     # Training Hyperparameters: --->
    # (Note: In the V2 Temporal MDP, 1 'step' encapsulates 10-20 years of simulated operation.
    # The environment immediately terminates after evaluating the predictive maintenance action.)
    time_step = 0
    max_episodes = 100

    # Accumulate a small batch of episodes before updating the detection agent CNN=based PPO Actor-Critic RL agent
    # to stabilize gradients: --->
    update_timestep = 3

    # Arrays to store tracking history for classification metrics: --->
    y_true_binary = []
    y_pred_binary = []

    print("\n--- Starting Pre-Training of CNN PPO Agent ---")

    for ep in range(1, max_episodes + 1):
        print(f"\n--- Episode {ep}/{max_episodes} ---")

        # Resetting simulates 10 full years and returns the (10, 5, 18) temporal stack: --->
        state, _ = env.reset()
        print(f"Simulated {env.simulated_years} Years randomly.")

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

        # Initialize and run the GNN Agent for Localization: --->
        if action == 1:
            print("\n[Maintenance Triggered] Passing control to Low-Level GNN Agent...")
            loc_env = ComponentLocalizationEnv(env.simulator)
            loc_state, loc_info = loc_env.reset()
            adj = loc_info['adjacency']

            gnn_reward_sum = 0
            loc_done = False

            while not loc_done:
                loc_action = gnn_agent.select_action(loc_state, adj)
                next_loc_state, loc_reward, loc_term, loc_trunc, l_info = loc_env.step(loc_action)

                gnn_agent.buffer.rewards.append(loc_reward)
                gnn_agent.buffer.is_terminals.append(loc_term or loc_trunc)

                # Track performance strictly as binary task: hit (1) or miss (0) the fault: --->
                is_faulty = l_info['is_faulty']
                y_true_binary.append(1 if is_faulty else 0)

                # Agent predicted this node had a fault by selecting it: --->
                y_pred_binary.append(1)

                loc_state = next_loc_state
                gnn_reward_sum += loc_reward
                loc_done = loc_term or loc_trunc

            writer.add_scalar('GNN_Agent/Reward', gnn_reward_sum, ep)
            print(f"GNN Agent finished with total reward: {gnn_reward_sum}")

        # Update the PPO Agent: --->
        if time_step % update_timestep == 0:
            print("\n[Updating CNN Actor-Critic Networks...]")
            a_loss, c_loss, t_loss = agent.update()
            print(f"CNN Agent updated: Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f}")

            # Log losses: --->
            writer.add_scalar('CNN_Loss/Actor', a_loss, time_step)
            writer.add_scalar('CNN_Loss/Critic', c_loss, time_step)
            writer.add_scalar('CNN_Loss/Total', t_loss, time_step)

            # Perform update for GNN agent only if it was triggered: --->
            if len(gnn_agent.buffer.states) > 0:
                print("[Updating GNN Actor-Critic Networks...]")
                g_a_loss, g_c_loss, g_t_loss = gnn_agent.update()
                writer.add_scalar('GNN_Loss/Total', g_t_loss, time_step)
            else:
                print("[Skipping GNN Update: No localization actions taken in this cycle]")

    # Post-training Classification Analytics for GNN Agent --->
    if y_true_binary:
        print("\n--- Generating Classification Metrics for GNN Localization Agent ---")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)

        metrics_dict = {'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Accuracy': acc}

        # Matplotlib visualization: --->
        fig, ax = plt.subplots(figsize = (8, 5))
        bars = ax.bar(metrics_dict.keys(), metrics_dict.values(), color = ['blue', 'orange', 'green', 'red'])

        ax.set_ylim(0, 1.1)
        ax.set_title('GNN Agent: Spatial Fault Localization Performance')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha = 'center', va = 'bottom')

        # Save Matplotlib visualization as PNG images inside the specified directory: --->
        plot_path = f"visualizations/classification_metrics/gnn_performance_{current_time}.png"
        plt.savefig(plot_path)
        print(f"Classification metrics visualization saved to {plot_path}")

    writer.close()
    print("\n--- Pre-Training Complete ---")
