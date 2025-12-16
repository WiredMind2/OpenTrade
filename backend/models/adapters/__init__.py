"""
Model adapters for different model formats.

This module contains adapters that wrap various model formats
to provide a unified interface.
"""

from .base_adapter import BaseModelAdapter
from .joblib_adapter import JoblibModelAdapter

__all__ = ['BaseModelAdapter', 'JoblibModelAdapter']