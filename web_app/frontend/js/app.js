const { createApp, ref, reactive, onMounted, computed, nextTick } = Vue;

const app = createApp({
  setup() {
    // Session Configuration & Status
    const config = reactive({
      mode: 'integrated',
      engine_name: 'lsh',
      topology_path: 'nsfnet.json',
      max_episodes: 200,
      simulated_years: 10,
      auto_step: false,
      step_delay_ms: 100,
      cnn_checkpoint: '',
      gnn_checkpoint: ''
    });

    const sysStatus = reactive({
      status: 'idle', // idle, running, paused, completed, error
      text: 'System Idle'
    });

    const liveStats = reactive({
      episode: 0,
      total_episodes: 200,
      simulated_years: 0,
      far: 0.0,
      mdr: 0.0,
      action: 0,
      degradation_db: 0.0,
      n_failed_lightpaths: 0,
      pred_nodes: [],
      gt_nodes: [],
      tp: 0, fp: 0, tn: 0, fn: 0
    });

    // Options for dropdowns
    const topologies = ref([]);
    const cnnCheckpoints = ref([]);
    const gnnCheckpoints = ref([]);

    // UI State
    const activeTab = ref('topology');
    const activeModal = ref(null); // 'topology', 'checkpoint', null
    const isLightMode = ref(false);
    
    // File Uploads
    const topoFileName = ref('');
    const topoFileObj = ref(null);
    const ckptFileName = ref('');
    const ckptFileObj = ref(null);
    const ckptAgentType = ref('cnn');

    // Toasts & Logs
    const toasts = ref([]);
    const logs = ref([]);
    const logContent = ref(null);
    let toastIdCounter = 0;

    // Derived State
    const topologyNameBadge = computed(() => {
      const name = config.topology_path.split(/[/\\]/).pop();
      return name ? name : 'None Selected';
    });

    // Helper: Add Toast Notification
    const addToast = (message, type = 'info') => {
      const id = toastIdCounter++;
      let icon = 'fa-info-circle';
      if (type === 'success') icon = 'fa-check-circle';
      if (type === 'error') icon = 'fa-circle-exclamation';
      if (type === 'warning') icon = 'fa-triangle-exclamation';
      
      toasts.value.push({ id, message, type, icon });
      setTimeout(() => {
        toasts.value = toasts.value.filter(t => t.id !== id);
      }, 5000);
    };

    // Helper: Add Event Log
    const addLog = (message, type = 'info') => {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      logs.value.push({ time: timeStr, message, type });
      nextTick(() => {
        if (logContent.value) {
          logContent.value.scrollTop = logContent.value.scrollHeight;
        }
      });
    };

    const clearLogs = () => {
      logs.value = [];
    };

    // API Calls
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        updateSystemStatus(data);
      } catch (err) {
        console.error("Failed to fetch status:", err);
      }
    };

    const fetchDropdowns = async () => {
      try {
        const [topoRes, ckptRes] = await Promise.all([
          fetch('/api/topologies'),
          fetch('/api/checkpoints')
        ]);
        const topoData = await topoRes.json();
        const ckptData = await ckptRes.json();
        
        topologies.value = topoData.topologies || [];
        cnnCheckpoints.value = ckptData.cnn_checkpoints || [];
        gnnCheckpoints.value = ckptData.gnn_checkpoints || [];

        if (topologies.value.length > 0 && !topologies.value.find(t => t.name === config.topology_path)) {
            config.topology_path = topologies.value[0].name;
        }
        
        // Fetch graph for the selected topology
        if (config.topology_path) {
           loadSelectedTopologyGraph();
        }
      } catch (err) {
        console.error("Failed to fetch dropdowns:", err);
      }
    };

    const loadSelectedTopologyGraph = async () => {
      try {
        const res = await fetch(`/api/topology_graph?name=${config.topology_path}`);
        if (res.ok) {
          const data = await res.json();
          if (window.topologyView && data.graph) {
             window.topologyView.loadTopology(data.graph);
          }
        }
      } catch (err) {}
    };

    // Watch for dropdown changes
    Vue.watch(() => config.topology_path, () => {
       if (sysStatus.status === 'idle' || sysStatus.status === 'completed' || sysStatus.status === 'error') {
          loadSelectedTopologyGraph();
       }
    });

    // Control Actions
    const startSession = async () => {
      try {
        const res = await fetch('/api/control/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        if (res.ok) {
          addToast("Simulation Session Started", "success");
          addLog("Sent start signal to background engine.", "info");
          liveStats.episode = 0;
          liveStats.total_episodes = config.max_episodes;
          liveStats.simulated_years = config.simulated_years;
        } else {
          addToast("Failed to start session", "error");
        }
      } catch (err) { addToast("Network Error", "error"); }
    };

    const togglePause = async () => {
      try {
        const endpoint = sysStatus.status === 'paused' ? '/api/control/resume' : '/api/control/pause';
        const res = await fetch(endpoint, { method: 'POST' });
        if (res.ok) {
          addLog(sysStatus.status === 'paused' ? "Resuming simulation." : "Pausing simulation.", "info");
        }
      } catch (err) {}
    };

    const stepForward = async () => {
      try {
        const res = await fetch('/api/control/step', { method: 'POST' });
        if (res.ok) {
          addLog("Stepped forward 1 episode.", "info");
        }
      } catch (err) {}
    };

    const stopSession = async () => {
      try {
        const res = await fetch('/api/control/stop', { method: 'POST' });
        if (res.ok) {
          addToast("Session Stopped", "warning");
          addLog("Sent stop signal to background engine.", "warning");
        }
      } catch (err) {}
    };

    const updateSpeed = async () => {
      try {
        await fetch('/api/control/speed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ step_delay_ms: config.step_delay_ms })
        });
      } catch (err) {}
    };

    // Modal Handling
    const showModal = (modalName) => { activeModal.value = modalName; };
    const hideModal = () => {
      activeModal.value = null;
      topoFileName.value = ''; topoFileObj.value = null;
      ckptFileName.value = ''; ckptFileObj.value = null;
    };

    const onTopoFileChange = (e) => {
      if (e.target.files.length > 0) {
        topoFileObj.value = e.target.files[0];
        topoFileName.value = topoFileObj.value.name;
      }
    };

    const onCkptFileChange = (e) => {
      if (e.target.files.length > 0) {
        ckptFileObj.value = e.target.files[0];
        ckptFileName.value = ckptFileObj.value.name;
      }
    };

    const uploadTopology = async () => {
      if (!topoFileObj.value) return addToast("Please select a file", "error");
      const formData = new FormData();
      formData.append('file', topoFileObj.value);
      try {
        const res = await fetch('/api/upload_topology', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          addToast("Topology Uploaded Successfully", "success");
          addLog(`Topology ${data.filename} uploaded and parsed.`, "success");
          await fetchDropdowns();
          config.topology_path = data.filename;
          if (window.topologyView) window.topologyView.loadTopology(data.graph);
          hideModal();
        } else { addToast(data.detail || "Upload failed", "error"); }
      } catch (err) { addToast("Network Error", "error"); }
    };

    const uploadCheckpoint = async () => {
      if (!ckptFileObj.value) return addToast("Please select a file", "error");
      const formData = new FormData();
      formData.append('file', ckptFileObj.value);
      formData.append('agent_type', ckptAgentType.value);
      try {
        const res = await fetch('/api/upload_checkpoint', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          addToast("Checkpoint Uploaded Successfully", "success");
          addLog(`Model Checkpoint ${ckptFileObj.value.name} loaded for ${ckptAgentType.value.toUpperCase()}.`, "success");
          await fetchDropdowns();
          if (ckptAgentType.value === 'cnn') config.cnn_checkpoint = data.path;
          else config.gnn_checkpoint = data.path;
          hideModal();
        } else { addToast(data.detail || "Upload failed", "error"); }
      } catch (err) { addToast("Network Error", "error"); }
    };

    const toggleTheme = () => {
      isLightMode.value = !isLightMode.value;
      if (isLightMode.value) {
        document.documentElement.setAttribute('data-theme', 'light');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
    };

    // WebSocket Handlers (Delegated from websocket_client.js)
    const updateSystemStatus = (statusData) => {
      sysStatus.status = statusData.status || 'idle';
      
      const statMap = {
        'idle': 'System Idle',
        'running': 'Simulation Running',
        'paused': 'Simulation Paused',
        'completed': 'Training Completed',
        'error': 'System Error'
      };
      sysStatus.text = statMap[sysStatus.status] || 'Unknown';

      // Sync config block with server state if running
      if (sysStatus.status === 'running' || sysStatus.status === 'paused') {
        if (statusData.mode) config.mode = statusData.mode;
        if (statusData.engine_name) config.engine_name = statusData.engine_name;
        if (statusData.total_episodes) config.max_episodes = statusData.total_episodes;
        if (statusData.simulated_years) config.simulated_years = statusData.simulated_years;
        if (statusData.auto_step !== undefined) config.auto_step = statusData.auto_step;

        if (statusData.current_episode !== undefined) liveStats.episode = statusData.current_episode;
        if (statusData.total_episodes !== undefined) liveStats.total_episodes = statusData.total_episodes;
        if (statusData.simulated_years !== undefined) liveStats.simulated_years = statusData.simulated_years;
      }
    };

    const handleTelemetryFrame = (frame) => {
      // Update Live Stats
      liveStats.episode = frame.episode;
      liveStats.total_episodes = frame.total_episodes;
      liveStats.simulated_years = frame.simulated_years;
      
      const det = frame.detection || {};
      liveStats.action = det.action || 0;
      liveStats.degradation_db = det.degradation_db || 0;
      liveStats.n_failed_lightpaths = det.n_failed_lightpaths || 0;
      liveStats.far = det.far || det.live_far || 0;
      liveStats.mdr = det.mdr || det.live_mdr || 0;

      const loc = frame.localization || {};
      liveStats.pred_nodes = loc.localized_nodes || loc.pred_nodes || [];
      liveStats.gt_nodes = loc.ground_truth_faults || loc.gt_nodes || [];
      liveStats.tp = loc.tp || 0;
      liveStats.fp = loc.fp || 0;
      liveStats.tn = loc.tn || 0;
      liveStats.fn = loc.fn || 0;

      // Log important events
      if (det.action === 1) {
         addLog(`Ep ${frame.episode}: CNN Agent isolated degradation of ${(det.degradation_db || 0).toFixed(2)}dB`, "warning");
      }
      if (liveStats.pred_nodes.length > 0 && JSON.stringify(liveStats.pred_nodes) === JSON.stringify(liveStats.gt_nodes)) {
         addLog(`Ep ${frame.episode}: GNN perfectly localized faults at [${liveStats.gt_nodes.join(',')}]`, "success");
      } else if (liveStats.pred_nodes.length > 0) {
         addLog(`Ep ${frame.episode}: GNN Localization Mismatch (Pred: [${liveStats.pred_nodes.join(',')}], GT: [${liveStats.gt_nodes.join(',')}])`, "error");
      }

      // Dispatch to Chart & Topology Viewers
      if (window.chartsView) window.chartsView.updateFrame(frame);
      if (window.topologyView) window.topologyView.updateFrame(frame);
    };

    const handleHistorySync = (frames) => {
      addToast("Restoring chart history...", "info");
      addLog(`Restoring ${frames.length} historical telemetry frames.`, "info");
      if (window.chartsView) window.chartsView.loadHistory(frames);
      // Process last frame for UI state
      if (frames.length > 0) {
        handleTelemetryFrame(frames[frames.length - 1]);
      }
    };

    onMounted(() => {
      fetchDropdowns();
      fetchStatus();
      
      // Expose globally for websocket_client.js to push updates
      window.vueApp = {
        updateSystemStatus,
        handleTelemetryFrame,
        handleHistorySync,
        addToast,
        addLog
      };

      // Initialize Sub-Views
      if (window.topologyView) window.topologyView.init('topology-graph-container');
      if (window.chartsView) window.chartsView.init();
      
      // Init WebSocket Client
      if (window.wsClient) window.wsClient.connect();
    });

    return {
      config, sysStatus, liveStats, topologies, cnnCheckpoints, gnnCheckpoints,
      activeTab, activeModal, topoFileName, ckptFileName, ckptAgentType, toasts, logs, logContent,
      topologyNameBadge,
      showModal, hideModal, onTopoFileChange, onCkptFileChange,
      uploadTopology, uploadCheckpoint,
      startSession, togglePause, stopSession, stepForward, updateSpeed, clearLogs,
      isLightMode, toggleTheme
    };
  }
});

app.mount('#app');
