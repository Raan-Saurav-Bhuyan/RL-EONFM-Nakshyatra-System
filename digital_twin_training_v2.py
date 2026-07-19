import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import custom modules: --->
from eon_env.v2.temporal_mdp_wrapper import TemporalEONEnvV2
from eon_env.v2.localization_mdp_wrapper import ComponentLocalizationEnv
from PPO.CNN_PPO import PPOAgentCNN, DetectionEvalTracker
from PPO.GNN_PPO import PPOAgentGNN, LocalizationEvalTracker

# ═══════════════════════════════════════════════════════════════════
# Kill-switch global variables for agent performance evaluation.
# Set to False to disable the respective feature.
# ═══════════════════════════════════════════════════════════════════
DET_EVAL_ENABLED = True       # Enable/disable detection performance metrics collection + plotting
DET_EVAL_TENSORBOARD = True   # Enable/disable TensorBoard logging of detection eval metrics
LOC_EVAL_ENABLED = True       # Enable/disable localization performance metrics collection + plotting
LOC_EVAL_TENSORBOARD = True   # Enable/disable TensorBoard logging of localization eval metrics

if __name__ == '__main__':
    # Create directory for classification visualizations --->
    os.makedirs("visualizations/classification_metrics", exist_ok = True)

    # Create directories for best-model checkpoints: --->
    cnn_save_dir = "models/CNN_PPO"
    gnn_save_dir = "models/GNN_PPO"
    os.makedirs(cnn_save_dir, exist_ok = True)
    os.makedirs(gnn_save_dir, exist_ok = True)

    # Initialize Temporal MDP Environment for Predictive Maintenance: --->
    env = TemporalEONEnvV2()

    state_dim_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Initialized CNN-PPO Agent | State Shape: {state_dim_shape} | Action Dim: {action_dim}")
    agent = PPOAgentCNN(action_dim = action_dim, save_dir = cnn_save_dir)
    gnn_agent = PPOAgentGNN(save_dir = gnn_save_dir)

    # Detection agent performance evaluation tracker: --->
    det_tracker = DetectionEvalTracker() if DET_EVAL_ENABLED else None

    # Localization agent performance evaluation tracker: --->
    loc_tracker = LocalizationEvalTracker() if LOC_EVAL_ENABLED else None

    # Initialize TensorBoard Writer with timestamp: --->
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/ppo_eon_v2_{current_time}")

     # Training Hyperparameters: --->
    # (Note: In the V2 Temporal MDP, 1 'step' encapsulates 10-20 years of simulated operation.
    # The environment immediately terminates after evaluating the predictive maintenance action.)
    time_step = 0
    max_episodes = 500

    # Accumulate a small batch of episodes before updating the detection agent CNN=based PPO Actor-Critic RL agent
    # to stabilize gradients: --->
    update_timestep = 3

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
        n_failed = info.get('n_failed_lightpaths', 0)
        action_name = "Isolate/Maintain" if action == 1 else "Monitor"
        print(f"Action Taken: {action_name} | Degradation: {degradation:.2f} dB | Reward: {reward:.2f}")

        # Record episode for detection eval tracker: --->
        if det_tracker is not None:
            det_tracker.record_episode(ep, action, reward, degradation, n_failed)

        # Log metrics to TensorBoard: --->
        writer.add_scalar('Reward/Episode_Reward', reward, ep)
        writer.add_scalar('Environment/Degradation_dB', degradation, ep)
        writer.add_scalar('Action/Action_Chosen', action, ep)

        # Log detection eval metrics to TensorBoard: --->
        if det_tracker is not None and DET_EVAL_TENSORBOARD:
            det_tracker.log_to_tensorboard(writer, ep)

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

                # Record inspection step for localization eval tracker: --->
                if loc_tracker is not None:
                    loc_tracker.record_step(l_info['is_faulty'])

                loc_state = next_loc_state
                gnn_reward_sum += loc_reward
                loc_done = loc_term or loc_trunc

            # Finalize localization episode metrics: --->
            if loc_tracker is not None:
                loc_tracker.finalize_episode(gnn_reward_sum, loc_env.num_components, l_info['true_faults_count'])

            # Log localization eval metrics to TensorBoard: --->
            if loc_tracker is not None and LOC_EVAL_TENSORBOARD:
                loc_tracker.log_to_tensorboard(writer, ep)

            writer.add_scalar('GNN_Agent/Reward', gnn_reward_sum, ep)
            print(f"GNN Agent finished with total reward: {gnn_reward_sum}")

        # Update the PPO Agent: --->
        if time_step % update_timestep == 0:
            print("\n[Updating CNN Actor-Critic Networks...]")
            a_loss, c_loss, t_loss = agent.update()
            print(f"CNN Agent updated: Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f} | Total loss: {t_loss:.4f}")

            # Record PPO losses for detection eval tracker: --->
            if det_tracker is not None:
                det_tracker.record_losses(time_step, a_loss, c_loss, t_loss)

            # Log CNN-PPO losses: --->
            writer.add_scalar('CNN_Loss/Actor', a_loss, time_step)
            writer.add_scalar('CNN_Loss/Critic', c_loss, time_step)
            writer.add_scalar('CNN_Loss/Total', t_loss, time_step)

            # Perform update for GNN agent only if it was triggered: --->
            if len(gnn_agent.buffer.states) > 0:
                print("[Updating GNN Actor-Critic Networks...]")
                g_a_loss, g_c_loss, g_t_loss = gnn_agent.update()
                print(f"GNN Agent updated: Actor loss: {g_a_loss:.4f} | Critic loss: {g_c_loss:.4f} | Total loss: {g_t_loss:.4f}")

                # Record GNN PPO losses for localization eval tracker: --->
                if loc_tracker is not None:
                    loc_tracker.record_losses(time_step, g_a_loss, g_c_loss, g_t_loss)

                # Log GNN-PPO losses: --->
                writer.add_scalar('GNN_Loss/Actor', g_a_loss, time_step)
                writer.add_scalar('GNN_Loss/Critic', g_c_loss, time_step)
                writer.add_scalar('GNN_Loss/Total', g_t_loss, time_step)
            else:
                print("[Skipping GNN Update: No localization actions taken in this cycle]")

    # Post-training: Generate detection agent performance evaluation plots: --->
    if det_tracker is not None:
        det_tracker.generate_plots("visualizations/detection_plots")

    # Post-training: Generate localization agent performance evaluation plots: --->
    if loc_tracker is not None:
        loc_tracker.generate_plots("visualizations/classification_plots")

    writer.close()

    # Final summary of best model checkpoints: --->
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE — MODEL SUMMARY")
    print("=" * 70)

    cnn_ckpt = os.path.join(cnn_save_dir, "best_cnn_ppo.pt")
    if os.path.isfile(cnn_ckpt):
        print(f"  [CNN-PPO] Best checkpoint : {cnn_ckpt}")
        print(f"             Best total loss: {agent.best_total_loss:.6f}")
    else:
        print(f"  [CNN-PPO] No checkpoint saved (best_total_loss never improved).")

    gnn_ckpt = os.path.join(gnn_save_dir, "best_gnn_ppo.pt")
    if os.path.isfile(gnn_ckpt):
        print(f"  [GNN-PPO] Best checkpoint : {gnn_ckpt}")
        print(f"             Best total loss: {gnn_agent.best_total_loss:.6f}")
    else:
        print(f"  [GNN-PPO] No checkpoint saved (best_total_loss never improved).")

    print("=" * 70)
    print("--- Pre-Training Complete ---\n")
