/**
 * Vis.js Network Topology & Digital Twin Graph Visualizer.
 */

class TopologyVisualizer {
  constructor() {
    this.container = null;
    this.network = null;
    this.nodesDataSet = new vis.DataSet([]);
    this.edgesDataSet = new vis.DataSet([]);
    this.currentGraph = null;
  }

  init(containerId) {
    this.container = document.getElementById(containerId);
  }

  loadTopology(graphData) {
    if (!this.container) return;
    this.currentGraph = graphData;

    const nodes = graphData.nodes.map(n => ({
      id: parseInt(n.id),
      label: n.label,
      title: this.createNodeTooltip(n.id, 'ROADM', 'Idle', 0),
      shape: 'dot',
      size: 20,
      color: {
        background: '#10B981',
        border: '#047857',
        highlight: { background: '#34D399', border: '#059669' }
      },
      font: { color: '#F3F4F6', face: 'Outfit', size: 14 },
      borderWidth: 2,
      shadow: true
    }));

    const edges = graphData.edges.map(e => ({
      id: `${e.from}-${e.to}`,
      from: parseInt(e.from),
      to: parseInt(e.to),
      label: e.label,
      title: `Link: ${e.from} ↔ ${e.to}`,
      color: { color: 'rgba(255, 255, 255, 0.15)', highlight: '#00F2FE' },
      font: { color: '#9CA3AF', face: 'JetBrains Mono', size: 10, align: 'top' },
      width: 2,
      smooth: { type: 'continuous' }
    }));

    this.nodesDataSet.clear();
    this.edgesDataSet.clear();

    this.nodesDataSet.add(nodes);
    this.edgesDataSet.add(edges);

    const data = {
      nodes: this.nodesDataSet,
      edges: this.edgesDataSet
    };

    const options = {
      physics: {
        barnesHut: {
          gravitationalConstant: -3000,
          centralGravity: 0.3,
          springLength: 120
        },
        stabilization: { iterations: 100 }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100
      }
    };

    if (this.network) {
      this.network.destroy();
    }

    this.network = new vis.Network(this.container, data, options);
  }

  createNodeTooltip(id, type, traffic, gsnr) {
    return `
      <div style="font-family: Outfit, sans-serif; background: rgba(5,8,15,0.9); padding: 10px; border-radius: 6px; border: 1px solid #1E293B; color: #F8FAFC;">
        <h4 style="margin:0 0 8px 0; color: #38BDF8; font-size:14px;">Node ${id} (${type})</h4>
        <div style="font-size:12px; margin-bottom: 4px;">Traffic: <span style="color:#A3E635">${traffic} Gbps</span></div>
        <div style="font-size:12px;">GSNR: <span style="color:#FBBF24">${gsnr} dB</span></div>
      </div>
    `;
  }

  updateFrame(telemetry) {
    if (!this.network || !telemetry || !telemetry.detection) return;

    const det = telemetry.detection;
    const loc = telemetry.localization || {};
    const envState = telemetry.env_state || { edge_utilization: {}, node_metrics: {} };
    const degradation = det.degradation_db || 0.0;
    const action = det.action;

    const localizedNodes = loc.localized_nodes || loc.pred_nodes || [];
    const gtFaults = loc.ground_truth_faults || loc.gt_nodes || [];

    // Wake up physics briefly for an organic feel
    this.network.physics.options.enabled = true;
    setTimeout(() => { if (this.network) this.network.physics.options.enabled = false; }, 500);

    // Update Nodes
    const nodeUpdates = [];
    this.nodesDataSet.forEach(node => {
      const nodeIdStr = String(node.id);
      let bg = '#10B981';
      let border = '#047857';

      if (degradation >= 0.3 && degradation < 0.5) { bg = '#F59E0B'; border = '#D97706'; }
      if (action === 1 && gtFaults.includes(node.id)) { bg = '#EF4444'; border = '#B91C1C'; }
      if (localizedNodes.includes(node.id)) { bg = '#00F2FE'; border = '#0284C7'; }

      const metrics = envState.node_metrics[nodeIdStr] || { traffic_gbps: 'N/A', gsnr: 'N/A' };
      const title = this.createNodeTooltip(node.id, 'ROADM', metrics.traffic_gbps, typeof metrics.gsnr === 'number' ? metrics.gsnr.toFixed(2) : metrics.gsnr);

      nodeUpdates.push({ id: node.id, color: { background: bg, border: border }, title: title });
    });
    this.nodesDataSet.update(nodeUpdates);

    // Update Edges
    const edgeUpdates = [];
    this.edgesDataSet.forEach(edge => {
      const util = envState.edge_utilization[edge.id] || 0;
      // map util 0-100 to width 1-6
      const width = 1 + (util / 100) * 5;
      const opacity = 0.15 + (util / 100) * 0.85; // 0.15 to 1.0
      
      let edgeColor = `rgba(255, 255, 255, ${opacity})`;
      if (util > 80) edgeColor = `rgba(244, 63, 94, ${opacity})`; // high util = red
      else if (util > 50) edgeColor = `rgba(251, 191, 36, ${opacity})`; // med = yellow

      edgeUpdates.push({
        id: edge.id,
        width: width,
        color: { color: edgeColor },
        title: `Link: ${edge.from} ↔ ${edge.to}<br>Utilization: ${util.toFixed(1)}%`
      });
    });
    this.edgesDataSet.update(edgeUpdates);
  }
}

window.topologyView = new TopologyVisualizer();
