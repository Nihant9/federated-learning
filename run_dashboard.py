"""
run_dashboard.py

Convenient launcher for the Federated Diagnostics & Poisoning Defense Dashboard.
Starts the FastAPI server and automatically launches Google Chrome at http://127.0.0.1:8000.
"""

import os
import sys
import time
import subprocess
import webbrowser

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")

    venv_python = os.path.join(backend_dir, "venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    print("\n=======================================================")
    print("  Federated Healthcare AI & Poisoning Defense Dashboard")
    print("=======================================================\n")
    print(f"[*] Python Interpreter: {python_exe}")
    print("[*] Starting FastAPI Server on http://127.0.0.1:8000 ...")

    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir

    server_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=backend_dir,
        env=env
    )

    url = "http://127.0.0.1:8000"
    time.sleep(2)

    # Standard Google Chrome paths on Windows
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
    ]
    opened = False
    for cp in chrome_paths:
        if os.path.exists(cp):
            try:
                subprocess.Popen([cp, url])
                opened = True
                print(f"[+] Interface opened in Google Chrome: {url}")
                break
            except Exception:
                pass

    if not opened:
        print(f"[+] Opening default browser at: {url}")
        webbrowser.open(url)

    print("[*] Server is running. Press Ctrl + C in this terminal to stop.\n")
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n[!] Stopping server...")
        server_process.terminate()

if __name__ == "__main__":
    main()
