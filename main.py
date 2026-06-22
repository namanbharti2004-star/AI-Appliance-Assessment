"""
AI-Powered Appliance Damage Assessment and Insurance Fraud Detection Platform

Main entry point for the application.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from configs.config import (
    PROJECT_NAME,
    VERSION,
    LOG_CONFIG,
    PATHS,
    get_device
)
from utils import setup_logging


def print_banner():
    """Print application banner"""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║          🔍 {PROJECT_NAME} 🔍                         ║
║                                                                               ║
║          Version {VERSION}                                                           ║
║                                                                               ║
║          AI-Powered Inspection Platform for:                                 ║
║          • Appliance Detection                                                ║
║          • Damage Assessment                                                  ║
║          • Fraud Detection                                                    ║
║          • Risk Scoring                                                       ║
║          • Report Generation                                                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def initialize_app():
    """Initialize the application"""
    print_banner()

    # Setup logging
    setup_logging(log_dir=PATHS["logs"], level=LOG_CONFIG["level"])
    logger.info(f"Starting {PROJECT_NAME} v{VERSION}")

    # Log device info
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create necessary directories
    for path_key, path_value in PATHS.items():
        os.makedirs(path_value, exist_ok=True)

    logger.info("Application initialized successfully")

    return True


def main():
    """Main application entry point"""
    if not initialize_app():
        logger.error("Failed to initialize application")
        return 1

    logger.info("Application is ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
