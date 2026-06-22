#!/usr/bin/env python3
"""
Run Dashboard Script

Starts the Streamlit dashboard for the AI Appliance Assessment Platform.
"""

import os
import sys
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run the Streamlit dashboard"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dashboard_path = os.path.join(project_root, "dashboard", "app.py")

    # Create output directories
    os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

    print("=" * 60)
    print("Starting AI Appliance Assessment Dashboard")
    print("=" * 60)
    print(f"Dashboard path: {dashboard_path}")
    print("")
    print("Open your browser to: http://localhost:8501")
    print("=" * 60)

    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
