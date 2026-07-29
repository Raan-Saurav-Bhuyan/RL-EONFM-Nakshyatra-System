import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import custom modules: --->
from eon_env.v2.temporal_mdp_wrapper import TemporalEONEnvV2
from eon_env.v2.localization_mdp_wrapper import ComponentLocalizationEnv
from PPO.CNN_PPO import PPOAgentCNN, DetectionEvalTracker, pretrain_detection_agent, pretrain_detection_step
from PPO.GNN_PPO import PPOAgentGNN, LocalizationEvalTracker, pretrain_localization_agent, pretrain_localization_step

# ═══════════════════════════════════════════════════════════════════
# Kill-switch global variables for agent performance evaluation.
# Set to False to disable the respective feature.
# ═══════════════════════════════════════════════════════════════════
DET_EVAL_ENABLED = True       # Enable/disable detection performance metrics collection + plotting
DET_EVAL_TENSORBOARD = True   # Enable/disable TensorBoard logging of detection eval metrics
LOC_EVAL_ENABLED = True       # Enable/disable localization performance metrics collection + plotting
LOC_EVAL_TENSORBOARD = True   # Enable/disable TensorBoard logging of localization eval metrics

# =========================================================================
# Pre-training global variables for independent Pre-Training
# for agent model convergence.
# =========================================================================
PRETRAIN_DETECTION = True     # Enable/disable independent pre-training of CNN agent
PRETRAIN_LOCALIZATION = True  # Enable/disable independent pre-training of GNN agent
PRETRAIN_EPISODES = 10      # Number of episodes for independent pre-training


def pretrain_agents_unified(
    env,
    max_episodes = 500,
    pretrain_detection = True,
    pretrain_localization = True,
    det_update_timestep = 3,
    enable_det_eval = True,
    enable_loc_eval = True,
    enable_tb_det = True,
    enable_tb_loc = True):
    """
    Unified pre-training loop that runs the digital twin simulation ONCE
    per episode and trains both the CNN detection agent and GNN localization
    agent sequentially within each episode.

    This halves the simulation cost compared to running two separate
    pre-training passes.
    """
    action_dim = env.action_space.n

    # CNN Detection Agent Setup: --->
    det_agent = None
    det_tracker = None
    det_writer = None
    det_time_step = 0

    if pretrain_detection:
        cnn_save_dir = "models/CNN_PPO"
        os.makedirs(cnn_save_dir, exist_ok = True)

        det_agent = PPOAgentCNN(
            action_dim = action_dim,
            in_channels = env.observation_space.shape[0],
            save_dir = cnn_save_dir, checkpoint_name = "pre_best_cnn_ppo.pt")
        det_tracker = DetectionEvalTracker() if enable_det_eval else None

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        det_writer = SummaryWriter(
            log_dir = f"runs/pre_ppo_detection_eon_v2_{current_time}") if enable_tb_det else None

    # GNN Localization Agent Setup: --->
    loc_agent = None
    loc_tracker = None
    loc_writer = None

    if pretrain_localization:
        gnn_save_dir = "models/GNN_PPO"
        os.makedirs(gnn_save_dir, exist_ok = True)

        loc_agent = PPOAgentGNN(
            save_dir = gnn_save_dir, checkpoint_name = "pre_best_gnn_ppo.pt")
        loc_tracker = LocalizationEvalTracker() if enable_loc_eval else None

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        loc_writer = SummaryWriter(
            log_dir = f"runs/pre_ppo_localization_eon_v2_{current_time}") if enable_tb_loc else None

    # Unified Pre-Training Loop: --->
    print("\n" + "=" * 70)
    print("      STARTING UNIFIED PRE-TRAINING (SINGLE SIMULATION PER EPISODE)")
    agents_str = []
    if pretrain_detection:
        agents_str.append("CNN Detection")
    if pretrain_localization:
        agents_str.append("GNN Localization")
    print(f"      Agents: {' + '.join(agents_str)}")
    print("=" * 70)

    for ep in range(1, max_episodes + 1):
        print(f"\n--- Unified Pre-Training Episode {ep}/{max_episodes} ---")

        # One simulation per episode: --->
        state, _ = env.reset()
        print(f"Simulated {env.simulated_years} Years randomly.")

        # CNN Detection Step: --->
        if det_agent is not None:
            det_time_step += 1
            action, reward, info = pretrain_detection_step(
                det_agent, env, state, det_tracker, ep)

            degradation = info.get('degradation_db', 0.0)

            if det_writer is not None:
                det_writer.add_scalar('PreTrain_Reward/Episode_Reward', reward, ep)
                det_writer.add_scalar('PreTrain_Environment/Degradation_dB', degradation, ep)
                det_writer.add_scalar('PreTrain_Action/Action_Chosen', action, ep)

                if det_tracker is not None:
                    det_tracker.log_to_tensorboard(det_writer, ep)

            # Update CNN-PPO agent: --->
            if det_time_step % det_update_timestep == 0:
                print("\n[Updating CNN Actor-Critic Networks...]")
                a_loss, c_loss, t_loss = det_agent.update()
                print(f"CNN Agent updated: Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f} | Total loss: {t_loss:.4f}")

                if det_tracker is not None:
                    det_tracker.record_losses(det_time_step, a_loss, c_loss, t_loss)

                if det_writer is not None:
                    det_writer.add_scalar('PreTrain_CNN_Loss/Actor', a_loss, det_time_step)
                    det_writer.add_scalar('PreTrain_CNN_Loss/Critic', c_loss, det_time_step)
                    det_writer.add_scalar('PreTrain_CNN_Loss/Total', t_loss, det_time_step)

        # GNN Localization Step (single-step full-graph classification): --->
        if loc_agent is not None:
            loc_reward = pretrain_localization_step(
                loc_agent, env, loc_tracker, ep)

            if loc_tracker is not None and loc_writer is not None:
                loc_tracker.log_to_tensorboard(loc_writer, ep)

            if loc_writer is not None:
                loc_writer.add_scalar('PreTrain_GNN_Agent/Reward', loc_reward, ep)

            # Update GNN Actor-Critic Networks: --->
            if len(loc_agent.buffer.states) > 0:
                g_a_loss, g_c_loss, g_t_loss = loc_agent.update()

                if loc_tracker is not None:
                    loc_tracker.record_losses(ep, g_a_loss, g_c_loss, g_t_loss)

                if loc_writer is not None:
                    loc_writer.add_scalar('PreTrain_GNN_Loss/Actor', g_a_loss, ep)
                    loc_writer.add_scalar('PreTrain_GNN_Loss/Critic', g_c_loss, ep)
                    loc_writer.add_scalar('PreTrain_GNN_Loss/Total', g_t_loss, ep)

            # if ep % 50 == 0 or ep == max_episodes:
            print(f"--- GNN Pre-Training Episode {ep}/{max_episodes} | Reward: {loc_reward:.2f} ---")

    # Post Pre-Training Cleanup: --->
    print("\n" + "=" * 70)
    print("      UNIFIED PRE-TRAINING COMPLETE")
    print("=" * 70)

    if det_tracker is not None:
        det_tracker.generate_plots("visualizations/detection_plots/pre_detection_plots")
    if det_writer is not None:
        det_writer.close()

    if loc_tracker is not None:
        loc_tracker.generate_plots("visualizations/classification_plots/pre_localization_plots")
    if loc_writer is not None:
        loc_writer.close()

    return det_agent, loc_agent


if __name__ == '__main__':
    # Create directory for classification visualizations --->
    os.makedirs("visualizations/classification_metrics", exist_ok = True)

    # Create directories for best-model checkpoints: --->
    cnn_save_dir = "models/CNN_PPO"
    gnn_save_dir = "models/GNN_PPO"
    os.makedirs(cnn_save_dir, exist_ok = True)
    os.makedirs(gnn_save_dir, exist_ok = True)

    # Initialize Temporal MDP Environment for predictive Maintenance: --->
    env = TemporalEONEnvV2()

    state_dim_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # =========================================================================
    # Phase 1 & 2: Unified Pre-Training (single simulation per episode)
    # =========================================================================
    if PRETRAIN_DETECTION or PRETRAIN_LOCALIZATION:
        pretrain_agents_unified(
            env,
            max_episodes = PRETRAIN_EPISODES,
            pretrain_detection = PRETRAIN_DETECTION,
            pretrain_localization = PRETRAIN_LOCALIZATION,
            det_update_timestep = 3,
            enable_det_eval = DET_EVAL_ENABLED,
            enable_loc_eval = LOC_EVAL_ENABLED,
            enable_tb_det = DET_EVAL_TENSORBOARD,
            enable_tb_loc = LOC_EVAL_TENSORBOARD,
        )

    # =========================================================================
    # Phase 3: Hierarchical Integrated Training (HRL-SFDL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("      STARTING HIERARCHICAL INTEGRATED TRAINING (HRL-SFDL)")
    print("=" * 70)

    print(f"Initialized CNN-PPO Agent | State Shape: {state_dim_shape} | Action Dim: {action_dim}")
    
    # Initialize integrated agents (saving to default checkpoint names): --->
    agent = PPOAgentCNN(
        action_dim = action_dim,
        in_channels = env.observation_space.shape[0],
        save_dir = cnn_save_dir, checkpoint_name = "best_cnn_ppo.pt")
    gnn_agent = PPOAgentGNN(save_dir = gnn_save_dir, checkpoint_name = "best_gnn_ppo.pt")

    # Load pre-trained weights if available, then reset best_total_loss so they
    # don't get stuck on an artificially low pre-training loss during integrated training: --->
    pre_cnn_path = os.path.join(cnn_save_dir, "pre_best_cnn_ppo.pt")
    if os.path.isfile(pre_cnn_path):
        agent.load_best_model(path=pre_cnn_path)
        agent.best_total_loss = float('inf')
        print("  -> Reset CNN best_total_loss to infinity for integrated training.")

    pre_gnn_path = os.path.join(gnn_save_dir, "pre_best_gnn_ppo.pt")
    if os.path.isfile(pre_gnn_path):
        gnn_agent.load_best_model(path=pre_gnn_path)
        gnn_agent.best_total_loss = float('inf')
        print("  -> Reset GNN best_total_loss to infinity for integrated training.")

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

    for ep in range(1, max_episodes + 1):
        print(f"\n--- Integrated Training Episode {ep}/{max_episodes} ---")
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

            # Single-step full-graph classification: --->
            loc_action = gnn_agent.select_action(loc_state, adj)
            next_loc_state, loc_reward, loc_term, loc_trunc, l_info = loc_env.step(loc_action)

            gnn_agent.buffer.rewards.append(loc_reward)
            gnn_agent.buffer.is_terminals.append(True)

            # Store ground truth for auxiliary BCE loss: --->
            gnn_agent.store_ground_truth(l_info['ground_truth'])

            # Record classification metrics for localization eval tracker: --->
            if loc_tracker is not None:
                loc_tracker.record_episode_classifications(
                    l_info['tp'], l_info['fp'], l_info['tn'], l_info['fn'])
                loc_tracker.finalize_episode(loc_reward, l_info['f1'])

            # Log localization eval metrics to TensorBoard: --->
            if loc_tracker is not None and LOC_EVAL_TENSORBOARD:
                loc_tracker.log_to_tensorboard(writer, ep)

            writer.add_scalar('GNN_Agent/Reward', loc_reward, ep)
            print(f"GNN Agent classified {loc_env.num_components} components | Reward: {loc_reward:.2f} | F1: {l_info['f1']:.3f} | Faults: {l_info['true_faults_count']}")

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
