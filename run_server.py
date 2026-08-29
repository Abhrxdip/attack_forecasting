"""
NIDS-ML — Server & Dashboard Launcher
SIH Problem Statement #26153

Launches the Python backend REST API server on port 8000 and opens
the Neo-Brutalist Cyber Forensics Web Dashboard in the browser.
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.api import run_server

def open_browser(port: int):
    time.sleep(1.0)
    url = f"http://localhost:{port}/"
    print(f"\n[+] Opening Dashboard in browser: {url}")
    webbrowser.open(url)

def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 65)
    print(" [NIDS-ML] AI NETWORK ATTACK FORECASTING - DEFENSE CONSOLE")
    print("          SIH Problem Statement #26153 (World Models)")
    print("=" * 65)
    print(f"[*] Starting REST API Backend & Serving Frontend on port {port}...")
    
    # Launch browser in separate thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    # Run server (blocking)
    run_server(port)

if __name__ == "__main__":
    main()
