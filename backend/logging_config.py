"""
Production logging framework for trading backtesting system.

This module provides structured logging with different levels, file rotation,
JSON logging, and integration with monitoring systems.
"""
import os
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import traceback


@dataclass
class LogEntry:
    """Structured log entry for JSON logging."""
    timestamp: str
    level: str
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trading_strategy: Optional[str] = None
    ticker: Optional[str] = None
    position_id: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    system_metrics: Optional[Dict[str, float]] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract standard fields
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info else None
            }
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                if value is not None:
                    log_entry[key] = value
        
        # Add system metrics for critical log levels
        if record.levelno >= logging.ERROR:
            try:
                process = psutil.Process()
                log_entry["system_metrics"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "open_files": len(process.open_files()),
                    "threads": process.num_threads()
                }
            except Exception:
                pass
        
        return json.dumps(log_entry, default=str)


class TradingLogger:
    """Enhanced logger with trading-specific context and structured logging."""
    
    def __init__(self, name: str, config=None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Context storage for thread-local data
        self._context = {}
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        if self.logger.handlers:
            return  # Already configured
        
        self.logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.config.get('structured_logging', True):
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ))
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.config.get('file_path', 'logs/app.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.get('timed_rotation', False):
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.get('backup_count', 5),
                encoding='utf-8'
            )
        else:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_file_size', 10*1024*1024),  # 10MB
                backupCount=self.config.get('backup_count', 5),
                encoding='utf-8'
            )
        
        if self.config.get('structured_logging', True):
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ))
        
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with context."""
        # Merge context with additional fields
        log_data = {**self._context, **kwargs}
        
        # Create log record with extra data
        extra = {}
        for key, value in log_data.items():
            if value is not None:
                extra[key] = value
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Alias for warning."""
        self.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exception'] = True
        self._log(logging.ERROR, message, **kwargs)
    
    # Trading-specific logging methods
    def trade_execution(self, action: str, ticker: str, quantity: int, price: float, 
                       strategy: str, **kwargs):
        """Log trade execution details."""
        self.info(
            f"Trade {action}: {quantity} shares of {ticker} at ${price:.2f}",
            event_type="trade_execution",
            action=action,
            ticker=ticker,
            quantity=quantity,
            price=price,
            strategy=strategy,
            **kwargs
        )
    
    def model_prediction(self, ticker: str, prediction: float, confidence: float,
                        horizon: str, strategy: str, **kwargs):
        """Log model prediction details."""
        self.info(
            f"Model prediction for {ticker}: {prediction:.4f} (confidence: {confidence:.2f})",
            event_type="model_prediction",
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            horizon=horizon,
            strategy=strategy,
            **kwargs
        )
    
    def data_ingestion(self, source: str, records_processed: int, 
                      records_failed: int = 0, **kwargs):
        """Log data ingestion results."""
        level = logging.WARNING if records_failed > 0 else logging.INFO
        self._log(
            level,
            f"Data ingestion from {source}: {records_processed} records processed, "
            f"{records_failed} failed",
            event_type="data_ingestion",
            source=source,
            records_processed=records_processed,
            records_failed=records_failed,
            **kwargs
        )
    
    def backtest_result(self, strategy: str, total_return: float, sharpe_ratio: float,
                       max_drawdown: float, **kwargs):
        """Log backtest results."""
        self.info(
            f"Backtest complete - Strategy: {strategy}, Return: {total_return:.2%}, "
            f"Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}",
            event_type="backtest_result",
            strategy=strategy,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            **kwargs
        )
    
    def performance_metric(self, metric_name: str, value: float, 
                          strategy: str | None = None, **kwargs):
        """Log performance metric."""
        self.info(
            f"Performance metric {metric_name}: {value:.4f}",
            event_type="performance_metric",
            metric_name=metric_name,
            value=value,
            strategy=strategy,
            **kwargs
        )
    
    def system_alert(self, alert_type: str, message: str, severity: str = "warning",
                    **kwargs):
        """Log system alert."""
        level = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO
        }.get(severity.lower(), logging.WARNING)
        
        self._log(
            level,
            f"SYSTEM ALERT [{alert_type}]: {message}",
            event_type="system_alert",
            alert_type=alert_type,
            severity=severity,
            **kwargs
        )
    
    @contextmanager
    def context(self, **context_vars):
        """Context manager for setting temporary context variables."""
        old_context = self._context.copy()
        self._context.update(context_vars)
        try:
            yield
        finally:
            self._context = old_context
    
    def set_context(self, **context_vars):
        """Set context variables for subsequent log entries."""
        self._context.update(context_vars)
    
    def clear_context(self):
        """Clear all context variables."""
        self._context.clear()


# Global logger instances
_loggers: Dict[str, TradingLogger] = {}


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> TradingLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        from config import config as global_config
        
        # Merge global logging config with local config
        log_config = {
            'level': getattr(global_config.logging, 'level', 'INFO'),
            'format': getattr(global_config.logging, 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'file_path': getattr(global_config.logging, 'file_path', 'logs/app.log'),
            'max_file_size': getattr(global_config.logging, 'max_file_size', 10*1024*1024),
            'backup_count': getattr(global_config.logging, 'backup_count', 5),
            'structured_logging': getattr(global_config.logging, 'structured_logging', True),
            'timed_rotation': False  # Default to size-based rotation
        }
        
        if config:
            log_config.update(config)
        
        _loggers[name] = TradingLogger(name, log_config)
    
    return _loggers[name]


# Convenience function for getting the main application logger
def get_app_logger() -> TradingLogger:
    """Get the main application logger."""
    return get_logger("trading_backtester")


# Specialized logger functions
def log_trade_execution(logger: TradingLogger, action: str, ticker: str, 
                       quantity: int, price: float, strategy: str, **kwargs):
    """Log trade execution with standardized format."""
    logger.trade_execution(action, ticker, quantity, price, strategy, **kwargs)


def log_model_prediction(logger: TradingLogger, ticker: str, prediction: float,
                        confidence: float, horizon: str, strategy: str, **kwargs):
    """Log model prediction with standardized format."""
    logger.model_prediction(ticker, prediction, confidence, horizon, strategy, **kwargs)


def log_data_ingestion(logger: TradingLogger, source: str, records_processed: int,
                      records_failed: int = 0, **kwargs):
    """Log data ingestion results with standardized format."""
    logger.data_ingestion(source, records_processed, records_failed, **kwargs)


def log_backtest_result(logger: TradingLogger, strategy: str, total_return: float,
                       sharpe_ratio: float, max_drawdown: float, **kwargs):
    """Log backtest results with standardized format."""
    logger.backtest_result(strategy, total_return, sharpe_ratio, max_drawdown, **kwargs)


# Logging decorators for function execution tracking
def log_function_execution(logger: TradingLogger):
    """Decorator to log function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module = func.__module__
            
            logger.debug(f"Starting {func_name}", function=func_name, module=module)
            start_time = datetime.utcnow()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.debug(
                    f"Completed {func_name} in {execution_time:.3f}s",
                    function=func_name,
                    module=module,
                    execution_time=execution_time,
                    status="success"
                )
                return result
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(
                    f"Failed {func_name} after {execution_time:.3f}s: {str(e)}",
                    function=func_name,
                    module=module,
                    execution_time=execution_time,
                    status="error",
                    exception=True
                )
                raise
        
        return wrapper
    return decorator


def setup_logging():
    """Setup global logging configuration."""
    try:
        from backend.config import config as global_config
        
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(global_config.logging, 'level', 'INFO'))
        
        # Add main application logger
        get_app_logger()
    except ImportError:
        # Config not available (e.g., during testing), use default logging
        root_logger = logging.getLogger()
        root_logger.setLevel('INFO')
        get_app_logger()