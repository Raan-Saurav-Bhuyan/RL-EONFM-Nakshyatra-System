"""
Background Simulation and RL Training Runner for SDM-EON Digital Twin.
Executes non-blocking background threads for Pre-Training, Integrated Training, and Inference/Evaluation modes.
Emits real-time telemetry events via a thread-safe callback queue.
"""

import os
import time
import threading
import queue
from datetime import datetime
import torch
import numpy as np

from eon_env.v2.temporal_mdp_wrapper import TemporalEONEnvV2
from eon_env.v2.localization_mdp_wrapper import ComponentLocalizationEnv
from PPO.CNN_PPO import PPOAgentCNN, DetectionEvalTracker, pretrain_detection_step
from PPO.GNN_PPO import PPOAgentGNN, LocalizationEvalTracker, pretrain_localization_step
from web_app.backend.topology_parser import validate_and_parse_topology


class TrainingRunner:
    def __init__(self, broadcast_callback=None):
        self.broadcast_callback = broadcast_callback
        self.thread = None
        self.running = False
        self.paused = False
        self.step_once = False
        self.lock = threading.Lock()
        self.step_delay = 0.1  # seconds between steps

        # Session configuration: --->
        self.mode = "integrated"  # "pretrain", "integrated", "inference"
        self.engine_name = "lsh"  # "lsh", "similarity", "contrastive"
        self.topology_path = "nsfnet.json"
        self.max_episodes = 200
        self.simulated_years = 10
        self.auto_step = False
        self.cnn_checkpoint = None
        self.gnn_checkpoint = None

        # Live stats summary: --->
        self.current_episode = 0
        self.total_episodes = 0
        self.status = "idle"  # "idle", "running", "paused", "completed", "error"
        self.error_message = ""

    def start_session(self, config: dict):
        """Starts a new training or inference session in a background thread."""
        with self.lock:
            if self.running:
                self.running = False
                if self.thread:
                    self.thread.join(timeout=1.0)
            
            self.mode = config.get("mode", "integrated")
            self.engine_name = config.get("engine_name", "lsh")
            self.topology_path = config.get("topology_path", "nsfnet.json")
            self.max_episodes = int(config.get("max_episodes", 200))
            self.simulated_years = int(config.get("simulated_years", 10))
            self.auto_step = bool(config.get("auto_step", False))
            self.step_delay = float(config.get("step_delay_ms", 100)) / 1000.0
            self.cnn_checkpoint = config.get("cnn_checkpoint", None)
            self.gnn_checkpoint = config.get("gnn_checkpoint", None)

            self.running = True
            self.paused = False
            self.step_once = False
            self.status = "running"
            self.error_message = ""
            self._emit_event("session_status", self.get_state())

            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def pause_session(self):
        with self.lock:
            if self.running:
                self.paused = True
                self.status = "paused"
                self._emit_event("session_status", self.get_state())

    def resume_session(self):
        with self.lock:
            if self.running:
                self.paused = False
                self.status = "running"
                self._emit_event("session_status", self.get_state())

    def stop_session(self):
        with self.lock:
            self.running = False
            self.paused = False
            self.status = "idle"
            self._emit_event("session_status", self.get_state())

    def step_forward(self):
        with self.lock:
            if self.running and self.paused:
                self.step_once = True
                self.paused = False
                self.status = "running"
                self._emit_event("session_status", self.get_state())

    def set_speed(self, step_delay_ms: float):
        self.step_delay = max(0.0, float(step_delay_ms) / 1000.0)

    def get_state(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "engine_name": self.engine_name,
            "topology_path": self.topology_path,
            "current_episode": self.current_episode,
            "total_episodes": self.max_episodes,
            "simulated_years": self.simulated_years,
            "auto_step": self.auto_step,
            "step_delay_ms": int(self.step_delay * 1000),
            "error_message": self.error_message,
        }

    def _emit_event(self, event_type: str, data: dict):
        if self.broadcast_callback:
            payload = {
                "event": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
            try:
                self.broadcast_callback(payload)
            except Exception as e:
                print(f"[TrainerRunner] Error broadcasting telemetry: {e}")

    def _run_loop(self):
        try:
            # 1. Parse and validate topology JSON: --->
            is_valid, msg, graph_info = validate_and_parse_topology(self.topology_path)
            if not is_valid:
                raise ValueError(f"Invalid topology file '{self.topology_path}': {msg}")

            self._emit_event("topology_loaded", {
                "topology_path": self.topology_path,
                "graph": graph_info
            })

            # 2. Initialize Gym Environment: --->
            env = TemporalEONEnvV2(
                network_json_path=self.topology_path,
                engine_name=self.engine_name,
                simulated_years=self.simulated_years
            )

            state_dim_shape = env.observation_space.shape
            action_dim = env.action_space.n

            # 3. Setup Agent Checkpoint Directories: --->
            cnn_dir = "models/CNN_PPO"
            gnn_dir = "models/GNN_PPO"
            os.makedirs(cnn_dir, exist_ok=True)
            os.makedirs(gnn_dir, exist_ok=True)

            det_agent = PPOAgentCNN(
                action_dim=action_dim,
                in_channels=state_dim_shape[0],
                save_dir=cnn_dir,
                checkpoint_name="best_cnn_ppo.pt"
            )
            loc_agent = PPOAgentGNN(
                save_dir=gnn_dir,
                checkpoint_name="best_gnn_ppo.pt"
            )

            # Load custom checkpoints if provided: --->
            if self.cnn_checkpoint and os.path.isfile(self.cnn_checkpoint):
                det_agent.load_best_model(path=self.cnn_checkpoint)
                print(f"[TrainerRunner] Loaded CNN checkpoint: {self.cnn_checkpoint}")

            if self.gnn_checkpoint and os.path.isfile(self.gnn_checkpoint):
                loc_agent.load_best_model(path=self.gnn_checkpoint)
                print(f"[TrainerRunner] Loaded GNN checkpoint: {self.gnn_checkpoint}")

            # Performance Trackers: --->
            det_tracker = DetectionEvalTracker()
            loc_tracker = LocalizationEvalTracker()

            self._emit_event("session_status", {
                "status": "running",
                "mode": self.mode,
                "engine": self.engine_name,
                "max_episodes": self.max_episodes
            })

            # 4. Mode Branching Execution: --->
            if self.mode == "pretrain":
                self._run_pretrain_loop(env, det_agent, loc_agent, det_tracker, loc_tracker)
            elif self.mode == "integrated":
                self._run_integrated_loop(env, det_agent, loc_agent, det_tracker, loc_tracker)
            elif self.mode == "inference":
                self._run_inference_loop(env, det_agent, loc_agent, det_tracker, loc_tracker)

            self.status = "completed"
            self._emit_event("session_status", {"status": "completed"})

        except Exception as e:
            self.running = False
            self.status = "error"
            self.error_message = str(e)
            print(f"[TrainerRunner Error] {e}")
            self._emit_event("session_status", {"status": "error", "error": str(e)})

    def _run_pretrain_loop(self, env, det_agent, loc_agent, det_tracker, loc_tracker):
        det_time_step = 0
        det_update_timestep = 3

        for ep in range(1, self.max_episodes + 1):
            if not self.running:
                break
            while self.paused and self.running:
                time.sleep(0.1)

            if not self.running:
                break

            if self.step_once:
                self.paused = True
                self.step_once = False
                self.status = "paused"
                self._emit_event("session_status", self.get_state())

            self.current_episode = ep
            state, _ = env.reset()

            # CNN Detection Step: --->
            det_time_step += 1
            action, reward, info = pretrain_detection_step(det_agent, env, state, det_tracker, ep)
            degradation = info.get('degradation_db', 0.0)
            n_failed = info.get('n_failed_lightpaths', 0)

            det_a_loss, det_c_loss, det_t_loss = 0.0, 0.0, 0.0
            if det_time_step % det_update_timestep == 0:
                det_a_loss, det_c_loss, det_t_loss = det_agent.update()
                det_tracker.record_losses(det_time_step, det_a_loss, det_c_loss, det_t_loss)

            # GNN Localization Step: --->
            loc_reward = pretrain_localization_step(loc_agent, env, loc_tracker, ep)
            loc_a_loss, loc_c_loss, loc_t_loss = 0.0, 0.0, 0.0
            if len(loc_agent.buffer.states) > 0:
                loc_a_loss, loc_c_loss, loc_t_loss = loc_agent.update()
                loc_tracker.record_losses(ep, loc_a_loss, loc_c_loss, loc_t_loss)

            # Extract evaluation metrics: --->
            far = self._calculate_far(det_tracker)
            mdr = self._calculate_mdr(det_tracker)
            precision, recall, f1, acc = self._calculate_loc_metrics(loc_tracker)

            # Build telemetry payload: --->
            telemetry = {
                "episode": ep,
                "total_episodes": self.max_episodes,
                "simulated_years": getattr(env, 'simulated_years', 10),
                "mode": "pretrain",
                "detection": {
                    "action": action,
                    "action_name": "Localize/Maintain" if action == 1 else "Monitor",
                    "reward": float(reward),
                    "degradation_db": float(degradation),
                    "n_failed_lightpaths": int(n_failed),
                    "actor_loss": float(det_a_loss),
                    "critic_loss": float(det_c_loss),
                    "total_loss": float(det_t_loss),
                    "far": float(far),
                    "mdr": float(mdr)
                },
                "localization": {
                    "reward": float(loc_reward),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "accuracy": float(acc),
                    "actor_loss": float(loc_a_loss),
                    "critic_loss": float(loc_c_loss),
                    "total_loss": float(loc_t_loss)
                }
            }

            self._emit_event("telemetry_frame", telemetry)
            if self.step_delay > 0:
                time.sleep(self.step_delay)
                
            if not self.auto_step and self.running:
                with self.lock:
                    self.paused = True
                    self.status = "paused"
                    self._emit_event("session_status", self.get_state())

    def _run_integrated_loop(self, env, det_agent, loc_agent, det_tracker, loc_tracker):
        time_step = 0
        update_timestep = 3

        for ep in range(1, self.max_episodes + 1):
            if not self.running:
                break
            while self.paused and self.running:
                time.sleep(0.1)

            if not self.running:
                break

            if self.step_once:
                self.paused = True
                self.step_once = False
                self.status = "paused"
                self._emit_event("session_status", self.get_state())

            self.current_episode = ep
            state, _ = env.reset()
            time_step += 1

            action = det_agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            det_agent.buffer.rewards.append(reward)
            det_agent.buffer.is_terminals.append(terminated)

            degradation = info.get('degradation_db', 0.0)
            n_failed = info.get('n_failed_lightpaths', 0)
            det_tracker.record_episode(ep, action, reward, degradation, n_failed)

            loc_reward = 0.0
            loc_f1 = 0.0
            loc_triggered = False
            localized_nodes = []
            ground_truth_faults = []
            tp, fp, tn, fn = 0, 0, 0, 0

            if action == 1:
                loc_triggered = True
                loc_env = ComponentLocalizationEnv(env.simulator)
                loc_state, loc_info = loc_env.reset()
                adj = loc_info['adjacency']

                loc_action = loc_agent.select_action(loc_state, adj)
                next_loc_state, loc_reward, loc_term, loc_trunc, l_info = loc_env.step(loc_action)

                loc_agent.buffer.rewards.append(loc_reward)
                loc_agent.buffer.is_terminals.append(True)
                loc_agent.store_ground_truth(l_info['ground_truth'])

                loc_tracker.record_episode_classifications(
                    l_info['tp'], l_info['fp'], l_info['tn'], l_info['fn'])
                loc_tracker.finalize_episode(loc_reward, l_info['f1'])
                loc_f1 = l_info['f1']
                tp, fp, tn, fn = l_info['tp'], l_info['fp'], l_info['tn'], l_info['fn']

                ground_truth = l_info['ground_truth']
                preds = loc_action
                for i in range(len(preds)):
                    if preds[i] == 1:
                        localized_nodes.append(i + 1)
                    if ground_truth[i] == 1:
                        ground_truth_faults.append(i + 1)

            # Network updates: --->
            det_a_loss, det_c_loss, det_t_loss = 0.0, 0.0, 0.0
            loc_a_loss, loc_c_loss, loc_t_loss = 0.0, 0.0, 0.0

            if time_step % update_timestep == 0:
                det_a_loss, det_c_loss, det_t_loss = det_agent.update()
                det_tracker.record_losses(time_step, det_a_loss, det_c_loss, det_t_loss)

                if len(loc_agent.buffer.states) > 0:
                    loc_a_loss, loc_c_loss, loc_t_loss = loc_agent.update()
                    loc_tracker.record_losses(time_step, loc_a_loss, loc_c_loss, loc_t_loss)

            far = self._calculate_far(det_tracker)
            mdr = self._calculate_mdr(det_tracker)
            precision, recall, f1, acc = self._calculate_loc_metrics(loc_tracker)

            # Extract richer environment details: --->
            edge_utilization = {}
            node_metrics = {}
            if hasattr(env, 'simulator'):
                sim = env.simulator
                topology = getattr(sim, 'topology', None)
                if topology and hasattr(topology, 'graph'):
                    for u, v in topology.graph.edges():
                        # Estimate spectrum utilization (mock metric based on lightpaths if actual is not easily accessible)
                        u_val = float(np.random.rand() * 100)
                        edge_id = f"{u}-{v}"
                        edge_utilization[edge_id] = u_val
                    
                    for node in topology.graph.nodes():
                        nid = str(node)
                        node_metrics[nid] = {
                            "traffic_gbps": float(np.random.randint(100, 1000)),
                            "gsnr": float(25.0 - (degradation if action == 1 else 0))
                        }

            telemetry = {
                "episode": ep,
                "total_episodes": self.max_episodes,
                "simulated_years": getattr(env, 'simulated_years', 10),
                "mode": "integrated",
                "env_state": {
                    "edge_utilization": edge_utilization,
                    "node_metrics": node_metrics
                },
                "detection": {
                    "action": action,
                    "action_name": "Isolate/Maintain" if action == 1 else "Monitor",
                    "reward": float(reward),
                    "degradation_db": float(degradation),
                    "n_failed_lightpaths": int(n_failed),
                    "actor_loss": float(det_a_loss),
                    "critic_loss": float(det_c_loss),
                    "total_loss": float(det_t_loss),
                    "far": float(far),
                    "mdr": float(mdr)
                },
                "localization": {
                    "triggered": loc_triggered,
                    "reward": float(loc_reward),
                    "f1_current": float(loc_f1),
                    "localized_nodes": localized_nodes,
                    "ground_truth_faults": ground_truth_faults,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "accuracy": float(acc),
                    "actor_loss": float(loc_a_loss),
                    "critic_loss": float(loc_c_loss),
                    "total_loss": float(loc_t_loss)
                }
            }

            self._emit_event("telemetry_frame", telemetry)
            if self.step_delay > 0:
                time.sleep(self.step_delay)

            if not self.auto_step and self.running:
                with self.lock:
                    self.paused = True
                    self.status = "paused"
                    self._emit_event("session_status", self.get_state())

    def _run_inference_loop(self, env, det_agent, loc_agent, det_tracker, loc_tracker):
        """Runs evaluation/inference with static policy (no PPO gradient updates)."""
        det_agent.policy.eval()
        loc_agent.policy.eval()

        with torch.no_grad():
            for ep in range(1, self.max_episodes + 1):
                if not self.running:
                    break
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break

                if self.step_once:
                    self.paused = True
                    self.step_once = False
                    self.status = "paused"
                    self._emit_event("session_status", self.get_state())

                self.current_episode = ep
                state, _ = env.reset()

                action = det_agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                degradation = info.get('degradation_db', 0.0)
                n_failed = info.get('n_failed_lightpaths', 0)
                det_tracker.record_episode(ep, action, reward, degradation, n_failed)

                loc_reward = 0.0
                loc_triggered = False
                localized_nodes = []
                ground_truth_faults = []
                tp, fp, tn, fn = 0, 0, 0, 0

                if action == 1:
                    loc_triggered = True
                    loc_env = ComponentLocalizationEnv(env.simulator)
                    loc_state, loc_info = loc_env.reset()
                    adj = loc_info['adjacency']

                    loc_action = loc_agent.select_action(loc_state, adj)
                    next_loc_state, loc_reward, loc_term, loc_trunc, l_info = loc_env.step(loc_action)

                    loc_tracker.record_episode_classifications(
                        l_info['tp'], l_info['fp'], l_info['tn'], l_info['fn'])
                    loc_tracker.finalize_episode(loc_reward, l_info['f1'])
                    loc_f1 = l_info['f1']
                    tp, fp, tn, fn = l_info['tp'], l_info['fp'], l_info['tn'], l_info['fn']

                    ground_truth = l_info['ground_truth']
                    for i in range(len(loc_action)):
                        if loc_action[i] == 1:
                            localized_nodes.append(i + 1)
                        if ground_truth[i] == 1:
                            ground_truth_faults.append(i + 1)

                far = self._calculate_far(det_tracker)
                mdr = self._calculate_mdr(det_tracker)
                precision, recall, f1, acc = self._calculate_loc_metrics(loc_tracker)

                # Extract richer environment details: --->
                edge_utilization = {}
                node_metrics = {}
                if hasattr(env, 'simulator'):
                    sim = env.simulator
                    topology = getattr(sim, 'topology', None)
                    if topology and hasattr(topology, 'graph'):
                        for u, v in topology.graph.edges():
                            u_val = float(np.random.rand() * 100)
                            edge_id = f"{u}-{v}"
                            edge_utilization[edge_id] = u_val
                        
                        for node in topology.graph.nodes():
                            nid = str(node)
                            node_metrics[nid] = {
                                "traffic_gbps": float(np.random.randint(100, 1000)),
                                "gsnr": float(25.0 - (degradation if action == 1 else 0))
                            }

                telemetry = {
                    "episode": ep,
                    "total_episodes": self.max_episodes,
                    "simulated_years": getattr(env, 'simulated_years', 10),
                    "mode": "inference",
                    "env_state": {
                        "edge_utilization": edge_utilization,
                        "node_metrics": node_metrics
                    },
                    "detection": {
                        "action": action,
                        "action_name": "Isolate/Maintain" if action == 1 else "Monitor",
                        "reward": float(reward),
                        "degradation_db": float(degradation),
                        "n_failed_lightpaths": int(n_failed),
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                        "total_loss": 0.0,
                        "far": float(far),
                        "mdr": float(mdr)
                    },
                    "localization": {
                        "triggered": loc_triggered,
                        "reward": float(loc_reward),
                        "f1_current": float(loc_f1) if 'loc_f1' in locals() else 0.0,
                        "localized_nodes": localized_nodes,
                        "ground_truth_faults": ground_truth_faults,
                        "tp": int(tp),
                        "fp": int(fp),
                        "tn": int(tn),
                        "fn": int(fn),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "accuracy": float(acc),
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                        "total_loss": 0.0
                    }
                }

                self._emit_event("telemetry_frame", telemetry)
                if self.step_delay > 0:
                    time.sleep(self.step_delay)

                if not self.auto_step and self.running:
                    with self.lock:
                        self.paused = True
                        self.status = "paused"
                        self._emit_event("session_status", self.get_state())

    def _calculate_far(self, det_tracker) -> float:
        if not hasattr(det_tracker, 'outcomes') or not det_tracker.outcomes:
            return 0.0
        fp = det_tracker.outcomes.count('FP')
        tn = det_tracker.outcomes.count('TN')
        denom = fp + tn
        return (fp / denom) if denom > 0 else 0.0

    def _calculate_mdr(self, det_tracker) -> float:
        if not hasattr(det_tracker, 'outcomes') or not det_tracker.outcomes:
            return 0.0
        fn = det_tracker.outcomes.count('FN')
        tp = det_tracker.outcomes.count('TP')
        denom = fn + tp
        return (fn / denom) if denom > 0 else 0.0

    def _calculate_loc_metrics(self, loc_tracker) -> tuple:
        tp = getattr(loc_tracker, 'total_tp', 0)
        fp = getattr(loc_tracker, 'total_fp', 0)
        tn = getattr(loc_tracker, 'total_tn', 0)
        fn = getattr(loc_tracker, 'total_fn', 0)

        prec_denom = tp + fp
        precision = (tp / prec_denom) if prec_denom > 0 else 0.0

        rec_denom = tp + fn
        recall = (tp / rec_denom) if rec_denom > 0 else 0.0

        f1_denom = precision + recall
        f1 = (2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0

        acc_denom = tp + fp + tn + fn
        accuracy = ((tp + tn) / acc_denom) if acc_denom > 0 else 0.0

        return precision, recall, f1, accuracy
