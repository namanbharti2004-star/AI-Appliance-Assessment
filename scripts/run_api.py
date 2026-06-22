#!/usr/bin/env python3
"""
Run API Server Script

Starts the FastAPI server for the AI Appliance Assessment Platform.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run the FastAPI server"""
    import argparse
    from api import run_server

    parser = argparse.ArgumentParser(description="Run API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print("=" * 60)
    print("Starting AI Appliance Assessment API Server")
    print("=" * 60)
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/health")
    print("=" * 60)

    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
