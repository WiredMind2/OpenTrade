"""
Shared logger for all scripts in the scripts/ folder.

This module provides a standardized logger configuration that can be imported
by all scripts to ensure consistent logging behavior.
"""

import logging
import sys
import os

# Set up logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create the shared logger instance
logger = logging.getLogger('scripts')

# Set the logger level (can be overridden by individual scripts if needed)
logger.setLevel(logging.INFO)