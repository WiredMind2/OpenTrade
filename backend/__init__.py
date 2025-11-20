"""
Backend package initializer for the backtesting project.

This file makes the `backend` directory a Python package so tests and other
imports can reference backend modules as `backend.<module>`.
"""
__all__ = [
    'auth_utils', 'config', 'data_validation', 'error_handling',
    'feature_engineering', 'logging_config', 'main'
]
