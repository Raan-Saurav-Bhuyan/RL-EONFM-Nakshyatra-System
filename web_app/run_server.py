"""
WSL Server Launcher for SDM-EON Digital Twin & RL Web Application.
Launches Uvicorn listening on 0.0.0.0:8000.
"""

import os
import sys
import uvicorn

# Ensure workspace root is in sys.path: --->
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if __name__ == '__main__':
    print("=" * 70)
    print("  LAUNCHING SDM-EON DIGITAL TWIN WEB APP SERVER (WSL -> WIN11)")
    print("=" * 70)
    print("  [Local WSL Address]: http://127.0.0.1:8000")
    print("  [Host Win11 Browser]: http://localhost:8000")
    print("=" * 70)

    uvicorn.run(
        "web_app.backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
