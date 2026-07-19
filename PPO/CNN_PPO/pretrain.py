import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .ppo_agent import PPOAgentCNN
from .det_eval_tracker import DetectionEvalTracker

def pretrain_detection_agent(
    env,
    max_episodes = 1500,
    update_timestep = 3,
    enable_eval = True,
    enable_tb = True):
    """
    Independently pre-trains the CNN detection agent on the temporal MDP,
    bypassing the localization (GNN) agent entirely.
    """
    action_dim = env.action_space.n
    cnn_save_dir = "models/CNN_PPO"
    os.makedirs(cnn_save_dir, exist_ok = True)
    
    agent = PPOAgentCNN(
        action_dim = action_dim,
        save_dir = cnn_save_dir, checkpoint_name = "pre_best_cnn_ppo.pt")
    det_tracker = DetectionEvalTracker() if enable_eval else None
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir = f"runs/pre_ppo_detection_eon_v2_{current_time}") if enable_tb else None

    time_step = 0

    print("\n" + "=" * 70)
    print("      STARTING INDEPENDENT PRE-TRAINING: CNN DETECTION AGENT")
    print("=" * 70)

    for ep in range(1, max_episodes + 1):
        print(f"\n--- CNN Pre-Training Episode {ep}/{max_episodes} ---")
        
        state, _ = env.reset()
        print(f"Simulated {env.simulated_years} Years randomly.")

        time_step += 1
        action = agent.select_action(state)

        # Execute predictive maintenance decision (terminates immediately)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(terminated)

        degradation = info.get('degradation_db', 0.0)
        n_failed = info.get('n_failed_lightpaths', 0)
        action_name = "Isolate/Maintain" if action == 1 else "Monitor"
        print(f"Action Taken: {action_name} | Degradation: {degradation:.2f} dB | Reward: {reward:.2f}")

        if det_tracker is not None:
            det_tracker.record_episode(ep, action, reward, degradation, n_failed)

        if writer is not None:
            writer.add_scalar('PreTrain_Reward/Episode_Reward', reward, ep)
            writer.add_scalar('PreTrain_Environment/Degradation_dB', degradation, ep)
            writer.add_scalar('PreTrain_Action/Action_Chosen', action, ep)
            
            if det_tracker is not None:
                det_tracker.log_to_tensorboard(writer, ep)

        # Update CNN-PPO agent
        if time_step % update_timestep == 0:
            print("[Updating CNN Actor-Critic Networks...]")
            a_loss, c_loss, t_loss = agent.update()
            print(f"CNN Agent updated: Actor loss: {a_loss:.4f} | Critic loss: {c_loss:.4f} | Total loss: {t_loss:.4f}")

            if det_tracker is not None:
                det_tracker.record_losses(time_step, a_loss, c_loss, t_loss)

            if writer is not None:
                writer.add_scalar('PreTrain_CNN_Loss/Actor', a_loss, time_step)
                writer.add_scalar('PreTrain_CNN_Loss/Critic', c_loss, time_step)
                writer.add_scalar('PreTrain_CNN_Loss/Total', t_loss, time_step)

    print("\n" + "=" * 70)
    print("      CNN DETECTION AGENT PRE-TRAINING COMPLETE")
    print("=" * 70)

    if det_tracker is not None:
        det_tracker.generate_plots("visualizations/detection_plots/pre_detection_plots")

    if writer is not None:
        writer.close()

    return agent
