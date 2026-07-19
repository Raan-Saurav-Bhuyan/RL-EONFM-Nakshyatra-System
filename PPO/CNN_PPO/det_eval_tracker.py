"""
Detection Agent Training Performance Evaluation Tracker.

Collects per-episode metrics and PPO loss history during training,
then generates publication-quality matplotlib figures for:
    1. Cumulative episode reward + moving average
    2. Confusion matrix (TP/FP/TN/FN) with FAR and MDR
    3. CNN actor-critic training loss curves
    4. GSNR degradation vs. detection action scatter plot

All plotting follows the same styling conventions as the demo scripts
in demo_performances/det_agent_perform_*.py.
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
    legend = ax.legend(loc = loc, frameon = True, facecolor = "white", edgecolor = "black", framealpha = 1.0, **kwargs)
    
    for text in legend.get_texts():
        text.set_fontname("Times New Roman")
        text.set_color("black")
        text.set_fontsize(10)
    
    return legend

class DetectionEvalTracker:
    """
    Collects training-time episode outcomes and PPO losses for the
    CNN-based detection agent, then produces matplotlib figures.

    Ground-truth labelling follows the LaTeX specification:
        y_det = 1  if  degradation >= T_GSNR  or  n_failed > 0
        y_det = 0  otherwise
    where T_GSNR matches the reward-function threshold (0.5 dB).
    """

    # GSNR degradation threshold (must match temporal_mdp_wrapper reward): --->
    T_GSNR = 0.5

    def __init__(self):
        # Per-episode tracking lists: --->
        self.episodes = []
        self.actions = []
        self.rewards = []
        self.degradations = []
        self.n_failed_list = []
        self.ground_truths = []
        self.outcomes = []          # <--- 'TP', 'FP', 'TN', 'FN'

        # PPO update loss history: --->
        self.loss_steps = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

    def record_episode(self, episode, action, reward, degradation_db, n_failed_lightpaths):
        """
        Record a single episode's outcome.

        Parameters
        ----------
        episode : int
            Episode index (1-based).
        action : int
            Detection agent action (0 = Monitor, 1 = Localize).
        reward : float
            Episode reward R_det^(q).
        degradation_db : float
            Mean GSNR degradation delta (dB).
        n_failed_lightpaths : int
            Number of lightpaths exceeding the hard-failure BER threshold.
        """
        # Ground-truth label (Eq. 1 in performance_evaluation.tex): --->
        y_det = 1 if (degradation_db >= self.T_GSNR or n_failed_lightpaths > 0) else 0

        # Confusion-matrix classification: --->
        if action == 1 and y_det == 1:
            outcome = "TP"
        elif action == 1 and y_det == 0:
            outcome = "FP"
        elif action == 0 and y_det == 0:
            outcome = "TN"
        else:
            outcome = "FN"

        self.episodes.append(episode)
        self.actions.append(action)
        self.rewards.append(reward)
        self.degradations.append(degradation_db)
        self.n_failed_list.append(n_failed_lightpaths)
        self.ground_truths.append(y_det)
        self.outcomes.append(outcome)

    def record_losses(self, update_step, actor_loss, critic_loss, total_loss):
        """
        Record PPO update losses for the CNN actor-critic.

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

    def _confusion_counts(self):
        """Return (TP, FP, TN, FN) counts."""
        tp = self.outcomes.count("TP")
        fp = self.outcomes.count("FP")
        tn = self.outcomes.count("TN")
        fn = self.outcomes.count("FN")

        return tp, fp, tn, fn

    def far(self):
        """False Alarm Rate = FP / (FP + TN)."""
        _, fp, tn, _ = self._confusion_counts()
        denom = fp + tn

        return (fp / denom) if denom > 0 else 0.0

    def mdr(self):
        """Missed Detection Rate = FN / (FN + TP)."""
        tp, _, _, fn = self._confusion_counts()
        denom = fn + tp

        return (fn / denom) if denom > 0 else 0.0

    def log_to_tensorboard(self, writer, episode):
        """
        Log running detection evaluation metrics to a TensorBoard SummaryWriter.

        Parameters
        ----------
        writer : torch.utils.tensorboard.SummaryWriter
            Active TensorBoard writer.
        episode : int
            Current episode index.
        """
        tp, fp, tn, fn = self._confusion_counts()

        writer.add_scalar("DetEval/FAR", self.far(), episode)
        writer.add_scalar("DetEval/MDR", self.mdr(), episode)
        writer.add_scalar("DetEval/TP_count", tp, episode)
        writer.add_scalar("DetEval/FP_count", fp, episode)
        writer.add_scalar("DetEval/TN_count", tn, episode)
        writer.add_scalar("DetEval/FN_count", fn, episode)

        if self.rewards:
            writer.add_scalar("DetEval/Episode_Reward", self.rewards[-1], episode)

    def generate_plots(self, save_dir = "visualizations/detection_plots"):
        """
        Generate and save all 4 detection-agent performance plots.

        Parameters
        ----------
        save_dir : str
            Directory to save PNG images into.
        """
        os.makedirs(save_dir, exist_ok = True)
        plt.rcParams.update(_PLOT_RC)

        self._plot_reward_curve(save_dir)
        self._plot_confusion_matrix(save_dir)
        self._plot_loss_curves(save_dir)
        self._plot_degradation_vs_action(save_dir)

        print(f"\n[DetEvalTracker] All 4 detection evaluation plots saved to: {save_dir}/")

    def _plot_reward_curve(self, save_dir):
        episodes = np.array(self.episodes)
        rewards = np.array(self.rewards)
        Q = len(rewards)

        if Q == 0:
            print("[DetEvalTracker] Skipping reward curve: no episodes recorded.")
            return

        # Moving average with W_avg = 50: --->
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
            label = r"$R_{\mathrm{det}}^{(q)}$"
        )
        ax.plot(
            episodes,
            moving_avg,
            color = "black",
            linewidth = 1.8,
            label = r"$\bar{R}_{\mathrm{det}}^{(q)}$ (window size = $50$)"
        )

        ax.set_xlabel(
            r"Training Episode ($q$)",
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

        out_path = os.path.join(save_dir, "det_reward_curve.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()
        
        print(f"[DetEvalTracker] Saved: {out_path}")

    def _plot_confusion_matrix(self, save_dir):
        tp, fp, tn, fn = self._confusion_counts()
        total = tp + fp + tn + fn

        if total == 0:
            print("[DetEvalTracker] Skipping confusion matrix: no episodes recorded.")
            return

        cm = np.array([[tn, fp],
                       [fn, tp]])

        fig, ax = plt.subplots(figsize = _FIGSIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        vmax = max(cm.max(), 1)
        ax.imshow(
            cm,
            cmap = plt.cm.Greys,
            interpolation = "nearest",
            vmin = 0,
            vmax = vmax
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            [r"($a_{\mathrm{det}} = 0$)",
            r"($a_{\mathrm{det}} = 1$)"],
            fontname = "Times New Roman",
            fontsize = 11,
        )
        ax.set_yticklabels(
            [r"($y_{\mathrm{det}} = 0$)",
            r"($y_{\mathrm{det}} = 1$)"],
            fontname = "Times New Roman",
            fontsize = 11,
        )
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel(
            "Predicted Action",
            fontsize = 12,
            fontname = "Times New Roman",
            labelpad = 10
        )
        ax.set_ylabel(
            "Ground-Truth State",
            fontsize = 12,
            fontname = "Times New Roman",
            labelpad = 10
        )

        # Cell annotations: --->
        pct = lambda v: f"{v / total * 100:.1f}" if total > 0 else "0.0"
        labels = [
            [
                f"(TN)\n$N_{{\\mathrm{{TN}}}}^{{\\mathrm{{det}}}} = {tn}$\n\n({pct(tn)}%)",
                f"(FP)\n$N_{{\\mathrm{{FP}}}}^{{\\mathrm{{det}}}} = {fp}$\n\n({pct(fp)}%)",
            ],
            [
                f"(FN)\n$N_{{\\mathrm{{FN}}}}^{{\\mathrm{{det}}}} = {fn}$\n\n({pct(fn)}%)",
                f"(TP)\n$N_{{\\mathrm{{TP}}}}^{{\\mathrm{{det}}}} = {tp}$\n\n({pct(tp)}%)",
            ],
        ]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > vmax * 0.5 else "black"
                ax.text(j, i, labels[i][j], ha = "center", va = "center",
                        color = color, fontname = "Times New Roman",
                        fontsize = 10, weight = "normal")

        # Grid lines: --->
        ax.set_xticks(np.arange(-0.5, 2, 1), minor = True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor = True)
        ax.grid(which = "minor", color = "black", linestyle = "-", linewidth = 1.5)
        ax.tick_params(which = "minor", bottom = False, left = False)
        ax.tick_params(which = "major", length = 0, labelsize = 11)

        plt.tight_layout()

        out_path = os.path.join(save_dir, "det_confusion_matrix.png")
        plt.savefig(out_path, dpi = _DPI, facecolor = "white",
                    edgecolor = "none", bbox_inches = "tight")
        plt.close()

        print(f"[DetEvalTracker] Saved: {out_path}")

    def _plot_loss_curves(self, save_dir):
        if len(self.loss_steps) == 0:
            print("[DetEvalTracker] Skipping loss curves: no PPO updates recorded.")
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
            label = r"$\mathcal{L}_{\mathrm{actor}}^{\mathrm{CNN}}$"
        )
        ax.plot(
            steps, c_losses,
            color = "#AAAAAA",
            linewidth = 1.2, linestyle = "-.",
            label = r"$\mathcal{L}_{\mathrm{critic}}^{\mathrm{CNN}}$"
        )
        ax.plot(
            steps, t_losses,
            color = "black",
            linewidth = 1.8, linestyle = "-",
            label = r"$\mathcal{L}_{\mathrm{total}}^{\mathrm{CNN}}$"
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

        out_path = os.path.join(save_dir, "det_training_loss.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()
        
        print(f"[DetEvalTracker] Saved: {out_path}")

    def _plot_degradation_vs_action(self, save_dir):
        if len(self.episodes) == 0:
            print("[DetEvalTracker] Skipping degradation scatter: no episodes recorded.")
            return

        actions = np.array(self.actions)
        degradations = np.array(self.degradations)
        ground_truths = np.array(self.ground_truths)

        # Horizontal jitter for visual dispersion: --->
        rng = np.random.default_rng(seed = 42)
        jitter = rng.normal(0, 0.04, size = len(actions))
        x_positions = actions.astype(float) + jitter

        fig, ax = plt.subplots(figsize = _FIGSIZE)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Threshold line: --->
        ax.axhline(
            y = self.T_GSNR,
            color = "black",
            linestyle = "--", linewidth = 1.2,
            label = rf"Threshold $T_{{\mathrm{{GSNR}}}} = {self.T_GSNR}\ \mathrm{{dB}}$"
        )

        # Healthy episodes (y_det = 0): --->
        healthy_mask = ground_truths == 0
        ax.scatter(
            x_positions[healthy_mask], degradations[healthy_mask],
            facecolors = "#E0E0E0",
            edgecolors = "black",
            alpha = 0.6,
            s = 18, linewidths = 0.6,
            label = r"Healthy State ($y_{\mathrm{det}} = 0$)"
        )

        # Degraded episodes (y_det = 1): --->
        degraded_mask = ground_truths == 1
        ax.scatter(
            x_positions[degraded_mask], degradations[degraded_mask],
            facecolors = "black",
            edgecolors = "black",
            alpha = 0.75,
            s = 18,
            linewidths = 0.6,
            label = r"Degraded State ($y_{\mathrm{det}} = 1$)"
        )

        ax.set_xlabel(
            r"Detection Action ($a_{\mathrm{det}}^{(q)}$)",
            fontsize = 11, fontname = "Times New Roman",
            color = "black"
        )
        ax.set_ylabel(
            r"Mean GSNR Degradation $\Delta_{\mathrm{GSNR}}^{(q)}$ (dB)",
            fontsize = 11, fontname = "Times New Roman",
            color = "black"
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [r"Monitoring ($a_{\mathrm{det}} = 0$)",
             r"Localization ($a_{\mathrm{det}} = 1$)"],
            fontsize = 10, fontname = "Times New Roman",
        )

        ax.set_xlim(-0.3, 1.3)

        _apply_common_style(ax)
        _add_legend(ax, loc = "upper left")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "det_degradation_vs_action.png")
        plt.savefig(
            out_path,
            dpi = _DPI,
            facecolor = "white",
            edgecolor = "none",
            bbox_inches = "tight"
        )
        plt.close()
        
        print(f"[DetEvalTracker] Saved: {out_path}")
