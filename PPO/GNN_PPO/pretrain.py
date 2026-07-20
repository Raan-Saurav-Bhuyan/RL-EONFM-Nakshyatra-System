import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from eon_env.v2.localization_mdp_wrapper import ComponentLocalizationEnv
from .ppo_agent_gnn import PPOAgentGNN
from .loc_eval_tracker import LocalizationEvalTracker

def pretrain_localization_agent(env, max_episodes = 1500, enable_eval = True, enable_tb = True):
    """
    Independently pre-trains the GNN localization agent. 
    Forces the localization environment to run on randomly generated 
    states from the temporal MDP, regardless of any detection policy.
    """
    gnn_save_dir = "models/GNN_PPO"
    os.makedirs(gnn_save_dir, exist_ok = True)
    
    gnn_agent = PPOAgentGNN(save_dir = gnn_save_dir, checkpoint_name = "pre_best_gnn_ppo.pt")
    loc_tracker = LocalizationEvalTracker() if enable_eval else None
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir = f"runs/pre_ppo_localization_eon_v2_{current_time}") if enable_tb else None

    print("\n" + "=" * 70)
    print("      STARTING INDEPENDENT PRE-TRAINING: GNN LOCALIZATION AGENT")
    print("=" * 70)

    for ep in range(1, max_episodes + 1):
        # Generate a random state using the Temporal MDP (10-20 years of simulation): --->
        env.reset()
        
        # Initialize localization environment based on the simulated state: --->
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

            if loc_tracker is not None:
                loc_tracker.record_step(l_info['is_faulty'])

            loc_state = next_loc_state
            gnn_reward_sum += loc_reward
            loc_done = loc_term or loc_trunc

        if loc_tracker is not None:
            loc_tracker.finalize_episode(gnn_reward_sum, loc_env.num_components, l_info['true_faults_count'])
            if writer is not None:
                loc_tracker.log_to_tensorboard(writer, ep)
                
        if writer is not None:
            writer.add_scalar('PreTrain_GNN_Agent/Reward', gnn_reward_sum, ep)

        # Update GNN Actor-Critic Networks: --->
        if len(gnn_agent.buffer.states) > 0:
            g_a_loss, g_c_loss, g_t_loss = gnn_agent.update()
            
            if loc_tracker is not None:
                loc_tracker.record_losses(ep, g_a_loss, g_c_loss, g_t_loss)

            if writer is not None:
                writer.add_scalar('PreTrain_GNN_Loss/Actor', g_a_loss, ep)
                writer.add_scalar('PreTrain_GNN_Loss/Critic', g_c_loss, ep)
                writer.add_scalar('PreTrain_GNN_Loss/Total', g_t_loss, ep)

        # if ep % 50 == 0 or ep == max_episodes:
        print(f"--- GNN Pre-Training Episode {ep}/{max_episodes} | Total Reward: {gnn_reward_sum:.2f} ---")

    print("\n" + "=" * 70)
    print("      GNN LOCALIZATION AGENT PRE-TRAINING COMPLETE")
    print("=" * 70)

    if loc_tracker is not None:
        loc_tracker.generate_plots("visualizations/classification_plots/pre_localization_plots")

    if writer is not None:
        writer.close()

    return gnn_agent

def pretrain_localization_step(gnn_agent, env, loc_tracker, ep):
    """
    Executes one GNN localization pre-training episode using
    the simulator state already populated in env (from env.reset()).

    Parameters:
        gnn_agent   : PPOAgentGNN instance.
        env         : TemporalEONEnvV2 instance (already reset, simulator populated).
        loc_tracker : LocalizationEvalTracker instance or None.
        ep          : Current episode number (for logging).

    Returns:
        gnn_reward_sum (float): Total reward accumulated during localization.
    """
    # Initialize localization environment based on the simulated state: --->
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

        if loc_tracker is not None:
            loc_tracker.record_step(l_info['is_faulty'])

        loc_state = next_loc_state
        gnn_reward_sum += loc_reward
        loc_done = loc_term or loc_trunc

    if loc_tracker is not None:
        loc_tracker.finalize_episode(gnn_reward_sum, loc_env.num_components, l_info['true_faults_count'])

    return gnn_reward_sum
