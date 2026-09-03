/**
 * Chart.js Real-Time Analytics & Loss Curves Visualizer.
 */

class AnalyticsCharts {
  constructor() {
    this.charts = {};
    this.history = {
      episodes: [],
      degradation: [],
      rewardsCNN: [],
      rewardsGNN: [],
      far: [],
      mdr: [],
      precision: [],
      recall: [],
      f1: [],
      accuracy: [],
      cnnActorLoss: [],
      cnnCriticLoss: [],
      gnnActorLoss: [],
      gnnCriticLoss: []
    };
  }

  createGradient(ctx, colorStart, colorEnd) {
    if (!ctx) return colorStart;
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
  }

  init() {
    Chart.defaults.color = '#9CA3AF';
    Chart.defaults.font.family = 'Inter';

    const commonGrid = {
      color: 'rgba(255, 255, 255, 0.05)',
      borderColor: 'rgba(255, 255, 255, 0.1)'
    };

    // Helper to get Canvas Context
    const getCtx = (id) => document.getElementById(id).getContext('2d');

    // 1. Degradation vs Action Chart
    this.charts.degradation = new Chart(getCtx('chart-degradation-action'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Degradation (dB)',
          data: [],
          borderColor: '#00F2FE',
          backgroundColor: this.createGradient(getCtx('chart-degradation-action'), 'rgba(0, 242, 254, 0.4)', 'rgba(0, 242, 254, 0.0)'),
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid, min: 0 } }
      }
    });

    // 2. Classification Scores Chart
    this.charts.classification = new Chart(getCtx('chart-classification-scores'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'F1-Score', data: [], borderColor: '#00F2FE', tension: 0.2 },
          { label: 'Precision', data: [], borderColor: '#10B981', tension: 0.2 },
          { label: 'Recall', data: [], borderColor: '#6366F1', tension: 0.2 },
          { label: 'Accuracy', data: [], borderColor: '#3B82F6', tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid, min: 0, max: 1.0 } }
      }
    });

    // 3. Rewards Chart
    this.charts.rewards = new Chart(getCtx('chart-rewards'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'CNN Detection Reward', data: [], borderColor: '#00F2FE', backgroundColor: this.createGradient(getCtx('chart-rewards'), 'rgba(0, 242, 254, 0.2)', 'transparent'), fill: true, tension: 0.2 },
          { label: 'GNN Localization Reward', data: [], borderColor: '#10B981', backgroundColor: this.createGradient(getCtx('chart-rewards'), 'rgba(16, 185, 129, 0.2)', 'transparent'), fill: true, tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid } }
      }
    });

    // 4. FAR & MDR Chart
    this.charts.farMdr = new Chart(getCtx('chart-far-mdr'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'False Alarm Rate (FAR)', data: [], borderColor: '#F59E0B', backgroundColor: this.createGradient(getCtx('chart-far-mdr'), 'rgba(245, 158, 11, 0.2)', 'transparent'), fill: true, tension: 0.2 },
          { label: 'Missed Detection Rate (MDR)', data: [], borderColor: '#EF4444', backgroundColor: this.createGradient(getCtx('chart-far-mdr'), 'rgba(239, 68, 68, 0.2)', 'transparent'), fill: true, tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid, min: 0, max: 1.0 } }
      }
    });

    // 5. CNN Losses Chart
    this.charts.cnnLosses = new Chart(getCtx('chart-cnn-losses'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Actor Loss', data: [], borderColor: '#6366F1', tension: 0.2 },
          { label: 'Critic Loss', data: [], borderColor: '#3B82F6', tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid } }
      }
    });

    // 6. GNN Losses Chart
    this.charts.gnnLosses = new Chart(getCtx('chart-gnn-losses'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Actor Loss', data: [], borderColor: '#10B981', tension: 0.2 },
          { label: 'Critic Loss', data: [], borderColor: '#F59E0B', tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { grid: commonGrid }, y: { grid: commonGrid } }
      }
    });
  }

  resetCharts() {
    Object.keys(this.history).forEach(k => this.history[k] = []);
    Object.values(this.charts).forEach(chart => {
      chart.data.labels = [];
      chart.data.datasets.forEach(d => d.data = []);
      chart.update();
    });
  }

  loadHistory(frames) {
    this.resetCharts();
    if (!frames || frames.length === 0) return;
    
    // Process frames silently
    frames.forEach(frame => this._processFrameData(frame));
    this._renderCharts();
  }

  updateFrame(telemetry) {
    if (!telemetry) return;
    this._processFrameData(telemetry);
    this._renderCharts();
  }

  _processFrameData(telemetry) {
    const ep = telemetry.episode;
    const det = telemetry.detection || {};
    const loc = telemetry.localization || {};
    const maxKeep = 100;

    this.history.episodes.push(ep);
    this.history.degradation.push(det.degradation_db || 0);
    this.history.rewardsCNN.push(det.reward || 0);
    this.history.rewardsGNN.push(loc.reward || 0);
    this.history.far.push(det.far || det.live_far || 0);
    this.history.mdr.push(det.mdr || det.live_mdr || 0);

    this.history.precision.push(loc.precision || 0);
    this.history.recall.push(loc.recall || 0);
    this.history.f1.push(loc.f1_score || 0);
    this.history.accuracy.push(loc.accuracy || 0);

    this.history.cnnActorLoss.push(det.actor_loss || 0);
    this.history.cnnCriticLoss.push(det.critic_loss || 0);
    this.history.gnnActorLoss.push(loc.actor_loss || 0);
    this.history.gnnCriticLoss.push(loc.critic_loss || 0);

    if (this.history.episodes.length > maxKeep) {
      Object.keys(this.history).forEach(k => this.history[k].shift());
    }
  }

  _renderCharts() {
    const labels = this.history.episodes;

    // 1. Degradation
    this.charts.degradation.data.labels = labels;
    this.charts.degradation.data.datasets[0].data = this.history.degradation;
    this.charts.degradation.update('none');

    this.charts.classification.data.labels = labels;
    this.charts.classification.data.datasets[0].data = this.history.f1;
    this.charts.classification.data.datasets[1].data = this.history.precision;
    this.charts.classification.data.datasets[2].data = this.history.recall;
    this.charts.classification.data.datasets[3].data = this.history.accuracy;
    this.charts.classification.update('none');

    this.charts.rewards.data.labels = labels;
    this.charts.rewards.data.datasets[0].data = this.history.rewardsCNN;
    this.charts.rewards.data.datasets[1].data = this.history.rewardsGNN;
    this.charts.rewards.update('none');

    this.charts.farMdr.data.labels = labels;
    this.charts.farMdr.data.datasets[0].data = this.history.far;
    this.charts.farMdr.data.datasets[1].data = this.history.mdr;
    this.charts.farMdr.update('none');

    this.charts.cnnLosses.data.labels = labels;
    this.charts.cnnLosses.data.datasets[0].data = this.history.cnnActorLoss;
    this.charts.cnnLosses.data.datasets[1].data = this.history.cnnCriticLoss;
    this.charts.cnnLosses.update('none');

    this.charts.gnnLosses.data.labels = labels;
    this.charts.gnnLosses.data.datasets[0].data = this.history.gnnActorLoss;
    this.charts.gnnLosses.data.datasets[1].data = this.history.gnnCriticLoss;
    this.charts.gnnLosses.update('none');
  }
}

window.chartsView = new AnalyticsCharts();
