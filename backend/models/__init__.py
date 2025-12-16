"""
Model management system for the trading backtester.

This module provides a unified interface for loading, managing, and using
machine learning models in the trading system.
"""

from .base import BaseModel
from .registry import ModelRegistry

__all__ = ['BaseModel', 'ModelRegistry']