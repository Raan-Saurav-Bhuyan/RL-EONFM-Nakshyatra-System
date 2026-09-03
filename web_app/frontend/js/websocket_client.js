/**
 * Resilient WebSocket Telemetry Client for SDM-EON Digital Twin Web App.
 */

class TelemetryWebSocket {
  constructor() {
    this.socket = null;
    this.callbacks = {};
    this.reconnectTimer = null;
    this.isConnected = false;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    console.log(`[WebSocket] Connecting to ${wsUrl}...`);
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log('[WebSocket] Connected successfully!');
      this.isConnected = true;
      if (this.callbacks['connection_change']) {
        this.callbacks['connection_change']({ status: 'connected' });
      }
      if (window.vueApp && window.vueApp.addToast) {
         window.vueApp.addToast("Connected to Telemetry Server", "success");
      }
    };

    this.socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const eventName = payload.event;
        
        // Dispatch to Vue App globals if they exist
        if (eventName === 'session_status' && window.vueApp) {
            window.vueApp.updateSystemStatus(payload.data);
        } else if (eventName === 'telemetry_frame' && window.vueApp) {
            window.vueApp.handleTelemetryFrame(payload.data);
        } else if (eventName === 'history_sync' && window.vueApp) {
            window.vueApp.handleHistorySync(payload.data.frames);
        }

        if (eventName && this.callbacks[eventName]) {
          this.callbacks[eventName](payload.data, payload.timestamp);
        }
      } catch (err) {
        console.error('[WebSocket] Failed to parse message:', err);
      }
    };

    this.socket.onclose = () => {
      console.warn('[WebSocket] Connection closed. Reconnecting in 2s...');
      this.isConnected = false;
      if (this.callbacks['connection_change']) {
        this.callbacks['connection_change']({ status: 'disconnected' });
      }
      if (window.vueApp && window.vueApp.addToast) {
         window.vueApp.addToast("Connection lost. Reconnecting...", "error");
      }
      this.scheduleReconnect();
    };

    this.socket.onerror = (err) => {
      console.error('[WebSocket Error]:', err);
      this.socket.close();
    };
  }

  scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, 2000);
  }

  on(eventName, callback) {
    this.callbacks[eventName] = callback;
  }

  send(data) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }
}

window.wsClient = new TelemetryWebSocket();
