"""
Localization Agent Training Performance Evaluation Tracker.

Collects per-episode full-graph classification metrics during training of
the GNN-based PPO localization agent, then generates publication-quality
matplotlib figures for:
    1. Classification metrics bar chart (Precision, Recall, F1, Accuracy)
    2. Cumulative localization episode reward + moving average
    3. GNN actor-critic training loss curves

All plotting follows the same styling conventions as the demo scripts
in demo_performances/loc_agent_perform_*.py and the existing
DetectionEvalTracker in PPO/CNN_PPO/det_eval_tracker.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

_PLOT_RC = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Times", "Liberation Serif",
        "Nimbus Roman", "DejaVu Serif", "serif",
    ],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
}

_FIGSIZE = (6, 4.5)
_DPI = 300

def _apply_common_style(ax):
    """Apply shared spine, tick and font styling to an Axes object."""
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    ax.tick_params(axis = "both", colors = "black", direction = "in", top = True, right = True)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(10)

def _add_legend(ax, **kwargs):
    """Add a black-bordered, white-background legend with Times New Roman text."""
    loc = kwargs.pop("loc", "best")
    legend = ax.legend(
        loc = loc,
        frameon = True,
        facecolor = "white",
        edgecolor = "black",
        framealpha = 1.0,
        **kwargs
    )

    for text in legend.get_texts():
        text.set_fontname("Times New Roman")
        text.set_color("black")
        text.set_fontsize(10)

    return legend

class LocalizationEvalTracker:
    """
    Collects training-time full-graph classification outcomes, per-episode
    rewards, and PPO losses for the GNN-based localization agent, then
    produces publication-quality matplotlib figures.

    In the single-pass full-graph classification architecture, the agent
    classifies every component in the augmented graph simultaneously.
    The confusion matrix is computed directly from predictions vs ground truth:
        TP — node classified as faulty AND actually faulty
        FP — node classified as faulty BUT actually healthy
        TN — node classified as healthy AND actually healthy
        FN — node classified as healthy BUT actually faulty
    """

    def __init__(self):
        # ── Cumulative confusion-matrix counts (across all episodes): ──
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0

        # ── Per-episode history lists: ──
        self.episode_indices = []       # Localization episode index (1-based)
        self.episode_rewards = []       # Cumulative reward R_loc^total per episode
        self.episode_f1_scores = []     # F1-score per episode

        # ── PPO update loss history: ──
        self.loss_steps = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

        # ── Internal episode counter: ──
        self._episode_count = 0

    def record_episode_classifications(self, tp, fp, tn, fn):
        """
        Record the full confusion matrix from a single classification episode.

        Parameters
        ----------
        tp : int
            True positives (faulty components correctly classified as faulty).
        fp : int
            False positives (healthy components incorrectly classified as faulty).
        tn : int
            True negatives (healthy components correctly classified as healthy).
        fn : int
            False negatives (faulty components incorrectly classified as healthy).
        """
        self.total_tp += tp
        self.total_fp += fp
        self.total_tn += tn
        self.total_fn += fn

    def finalize_episode(self, episode_reward, episode_f1 = 0.0):
        """
        Finalize metrics for the completed localization episode.

        Parameters
        ----------
        episode_reward : float
            Total reward for this localization episode.
        episode_f1 : float
            F1-score for this specific episode.
        """
        self._episode_count += 1

        # ── Record episode-level data: ──
        self.episode_indices.append(self._episode_count)
        self.episode_rewards.append(episode_reward)
        self.episode_f1_scores.append(episode_f1)

    def record_losses(self, update_step, actor_loss, critic_loss, total_loss):
        """
        Record PPO update losses for the GNN actor-critic.

        Parameters
        ----------
        update_step : int
            PPO update index (monotonically increasing).
        actor_loss : float
            Average actor (clipped surrogate) loss.
        critic_loss : float
            Average critic (MSE value) loss.
        total_loss : float
            Average total loss (actor + critic - entropy).
        """
        self.loss_steps.append(update_step)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.total_losses.append(total_loss)

    def precision(self):
        """Precision = TP / (TP + FP)."""
        denom = self.total_tp + self.total_fp
        
        return (self.total_tp / denom) if denom > 0 else 0.0

    def recall(self):
        """Recall = TP / (TP + FN)."""
        denom = self.total_tp + self.total_fn
        
        return (self.total_tp / denom) if denom > 0 else 0.0

    def f1_score(self):
        """F1-Score = 2 * (P * R) / (P + R)."""
        p = self.precision()
        r = self.recall()
        denom = p + r
        
        return (2 * p * r / denom) if denom > 0 else 0.0

    def accuracy(self):
        """Accuracy = (TP + TN) / (TP + FP + TN + FN)."""
        denom = self.total_tp + self.total_fp + self.total_tn + self.total_fn
        
        return ((self.total_tp + self.total_tn) / denom) if denom > 0 else 0.0

    def log_to_tensorboard(self, writer, episode):
        """
        Log running localization evaluation metrics to a TensorBoard SummaryWriter.

        Parameters
        ----------
        writer : torch.utils.tensorboard.SummaryWriter
            Active TensorBoard writer.
        episode : int
            Current episode index (from the outer detection training loop).
        """
        writer.add_scalar("LocEval/Precision", self.precision(), episode)
        writer.add_scalar("LocEval/Recall", self.recall(), episode)
        writer.add_scalar("LocEval/F1_Score", self.f1_score(), episode)
        writer.add_scalar("LocEval/Accuracy", self.accuracy(), episode)
        writer.add_scalar("LocEval/TP_count", self.total_tp, episode)
        writer.add_scalar("LocEval/FP_count", self.total_fp, episode)
        writer.add_scalar("LocEval/TN_count", self.total_tn, episode)
        writer.add_scalar("LocEval/FN_count", self.total_fn, episode)

        if self.episode_rewards:
            writer.add_scalar("LocEval/Episode_Reward", self.episode_rewards[-1], episode)

        if self.episode_f1_scores:
            writer.add_scalar("LocEval/Episode_F1", self.episode_f1_scores[-1], episode)

    def generate_plots(self, save_dir = "visualizations/classification_plots"):
        """
        Generate and save all 3 localization-agent performance plots.

        Parameters
        ----------
        save_dir : str
            Directory to save PNG images into.
        """
        os.makedirs(save_dir, exist_ok = True)
        plt.rcParams.update(_PLOT_RC)

        self._plot_classification_metrics(save_dir)
        self._plot_reward_curve(save_dir)
        self._plot_loss_curves(save_dir)

        print(f"\n[LocEvalTracker] All 3 localization evaluation plots saved to: {save_dir}/")

    def _plot_classification_metrics(self, save_dir):
        """Plot 1: Classification metrics bar chart (Precision, Recall, F1, Accuracy)."""
        total = self.total_tp + self.total_fp + self.total_tn + self.total_fn

        if total == 0:
            print("[LocEvalTracker] Skipping classification metrics: no inspection data recorded.")
            return

        metrics = [r"Precision", r"Recall", r"F1-Score", r"Accuracy"]
        achieved = [self.precision(), self.recall(), self.f1_score(), self.accuracy()]

        x = np.arange(len(metrics))
        width = 0.45

        fig, ax = plt.subplots(figsize = _FIGSIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Draw achieved bars in clean gray fill with black borders: ---→
        bars = ax.bar(
            x, achieved,
            width = width,
            color = "#E0E0E0",
            edgecolor = "black",
            linewidth = 1.0,
        )

        ax.set_ylabel(
            "Metric Value",
            fontsize = 11,
            fontname = "Times New Roman",
            color = "black"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize = 11, fontname = "Times New Roman")
        ax.set_ylim(0.0, 1.05)

        # Value annotations on top of each bar: ---→
        for bar, val in zip(bars, achieved):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy = (bar.get_x() + bar.get_width() / 2, height),
                xytext = (0, 3),
                textcoords = "offset points",
                ha = "center", va = "bottom",
                fontsize = 10, fontname = "Times New Roman",
                color = "black"
            )

        _apply_common_style(ax)
        plt.tight_layout()

        out_path = os.path.join(save_dir, "loc_classification_metrics.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()

        print(f"[LocEvalTracker] Saved: {out_path}")

    def _plot_reward_curve(self, save_dir):
        """Plot 2: Cumulative reward curve with moving average."""
        Q = len(self.episode_rewards)

        if Q == 0:
            print("[LocEvalTracker] Skipping reward curve: no localization episodes recorded.")
            return

        episodes = np.array(self.episode_indices)
        rewards = np.array(self.episode_rewards)

        # Moving average with W_avg = 50: ---→
        W_avg = 50
        moving_avg = np.zeros(Q)
        for q in range(Q):
            start_idx = max(0, q - W_avg + 1)
            moving_avg[q] = np.mean(rewards[start_idx : q + 1])

        fig, ax = plt.subplots(figsize = _FIGSIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(
            episodes,
            rewards,
            color = "#CCCCCC",
            alpha = 0.7,
            linewidth = 0.8,
            label = r"$R_{\mathrm{loc}}^{\mathrm{total}}(q_{\mathrm{loc}})$"
        )
        ax.plot(
            episodes,
            moving_avg,
            color = "black",
            linewidth = 1.8,
            label = r"$\bar{R}_{\mathrm{loc}}^{\mathrm{total}}$ (window size = $50$)"
        )

        ax.set_xlabel(
            r"Localization Episode ($q_{\mathrm{loc}}$)",
            fontsize = 11,
            fontname = "Times New Roman",
            color = "black"
        )
        ax.set_ylabel(
            r"Cumulative Episode Reward",
            fontsize = 11,
            fontname = "Times New Roman",
            color = "black"
        )
        ax.set_xlim(1, Q)

        _apply_common_style(ax)
        _add_legend(ax, loc = "lower right")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "loc_reward_curve.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()

        print(f"[LocEvalTracker] Saved: {out_path}")

    def _plot_loss_curves(self, save_dir):
        """Plot 3: GNN actor-critic training loss curves."""
        if len(self.loss_steps) == 0:
            print("[LocEvalTracker] Skipping loss curves: no GNN PPO updates recorded.")
            return

        steps = np.array(self.loss_steps)
        a_losses = np.array(self.actor_losses)
        c_losses = np.array(self.critic_losses)
        t_losses = np.array(self.total_losses)

        fig, ax = plt.subplots(figsize = _FIGSIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(
            steps, a_losses,
            color = "#888888",
            linewidth = 1.2, linestyle = "--",
            label = r"$\mathcal{L}_{\mathrm{actor}}^{\mathrm{GNN}}$"
        )
        ax.plot(
            steps, c_losses,
            color = "#AAAAAA",
            linewidth = 1.2, linestyle = "-.",
            label = r"$\mathcal{L}_{\mathrm{critic}}^{\mathrm{GNN}}$"
        )
        ax.plot(
            steps, t_losses,
            color = "black",
            linewidth = 1.8, linestyle = "-",
            label = r"$\mathcal{L}_{\mathrm{total}}^{\mathrm{GNN}}$"
        )

        ax.set_xlabel(
            r"PPO Update Step",
            fontsize = 11,
            fontname = "Times New Roman",
            color = "black"
        )
        ax.set_ylabel(
            r"Loss",
            fontsize = 11,
            fontname = "Times New Roman",
            color = "black"
        )

        _apply_common_style(ax)
        _add_legend(ax, loc = "upper right")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "loc_training_loss.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()

        print(f"[LocEvalTracker] Saved: {out_path}")
