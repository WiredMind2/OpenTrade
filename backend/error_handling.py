"""
Error handling and recovery mechanisms for trading backtesting system.

This module provides comprehensive error handling, custom exceptions,
retry mechanisms, circuit breakers, and graceful degradation strategies.
"""
import time
import functools
import asyncio
from typing import Any, Callable, Optional, Dict, List, Type, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
import sqlite3
import requests
import pandas as pd


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    timestamp: datetime
    severity: ErrorSeverity
    recoverable: bool
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 1.0
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    additional_info: Dict[str, Any] | None = None


class TradingBacktesterError(Exception):
    """Base exception for trading backtester system."""
    
    def __init__(self, message: str, error_code: str = None, 
                 context: ErrorContext = None, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context
        self.original_error = original_error
        self.timestamp = datetime.utcnow()


class DataIngestionError(TradingBacktesterError):
    """Error during data ingestion."""
    pass


class ModelError(TradingBacktesterError):
    """Error during model operations."""
    pass


class DatabaseError(TradingBacktesterError):
    """Error during database operations."""
    pass


class APIError(TradingBacktesterError):
    """Error during API operations."""
    pass


class ConfigurationError(TradingBacktesterError):
    """Error during configuration operations."""
    pass


class BacktestError(TradingBacktesterError):
    """Error during backtesting operations."""
    pass


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker functionality."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Cannot execute {func.__name__}"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class CircuitBreakerOpenError(TradingBacktesterError):
    """Raised when circuit breaker is open."""
    pass


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry functionality."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        break
                    
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
            
            # All retries failed
            raise last_exception
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay


class DatabaseRetryHandler(RetryHandler):
    """Specialized retry handler for database operations."""
    
    def __init__(self, max_retries: int = 3, retry_on: List[Type] | None = None):
        if retry_on is None:
            retry_on = [sqlite3.OperationalError, sqlite3.IntegrityError]
        
        super().__init__(max_retries=max_retries)
        self.retry_on = retry_on
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for database retry functionality."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not any(isinstance(e, exc_type) for exc_type in self.retry_on):
                        # Don't retry on non-retryable exceptions
                        raise
                    
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        break
                    
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Database operation attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
            
            # All retries failed
            raise DatabaseError(
                f"Database operation failed after {self.max_retries} retries",
                original_error=last_exception
            )
        
        return wrapper


class APITimeoutHandler(RetryHandler):
    """Specialized retry handler for API operations."""
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        super().__init__(max_retries=max_retries)
        self.timeout = timeout
        self.retry_on = [
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        ]
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for API retry functionality."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs, timeout=self.timeout)
                except Exception as e:
                    # Check if this is a retryable error
                    if not any(isinstance(e, exc_type) for exc_type in self.retry_on):
                        # Don't retry on non-retryable errors
                        if isinstance(e, requests.exceptions.HTTPError):
                            # Only retry on 5xx errors, not 4xx
                            if e.response.status_code < 500:
                                raise APIError(
                                    f"API request failed with non-retryable error: {e}",
                                    original_error=e
                                )
                    
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        break
                    
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"API request attempt {attempt + 1} failed, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)
            
            # All retries failed
            raise APIError(
                f"API request failed after {self.max_retries} retries",
                original_error=last_exception
            )
        
        return wrapper


class ErrorHandler:
    """Central error handler for the application."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
        self.handlers = {}
    
    def register_handler(self, error_type: Type[Exception], handler: Callable):
        """Register custom error handler."""
        self.handlers[error_type] = handler
    
    def handle_error(self, error: Exception, context: ErrorContext = None) -> bool:
        """Handle an error using registered handlers or default logic."""
        error_type = type(error)
        
        # Use custom handler if registered
        if error_type in self.handlers:
            try:
                return self.handlers[error_type](error, context)
            except Exception as handler_error:
                logging.error(f"Error handler failed: {handler_error}")
        
        # Default error handling
        return self._default_error_handling(error, context)
    
    def _default_error_handling(self, error: Exception, context: ErrorContext = None) -> bool:
        """Default error handling logic."""
        error_key = f"{type(error).__name__}:{context.component if context else 'unknown'}"
        
        # Track error counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.utcnow()
        
        # Determine if error is recoverable
        recoverable = self._is_recoverable_error(error, context)
        
        if not recoverable:
            logging.critical(
                f"Non-recoverable error in {context.component}:{context.operation} - {str(error)}",
                extra={"error_type": type(error).__name__, "context": context}
            )
            return False
        
        # Log recoverable error
        logging.warning(
            f"Recoverable error in {context.component}:{context.operation} - {str(error)}",
            extra={"error_type": type(error).__name__, "context": context}
        )
        
        return True
    
    def _is_recoverable_error(self, error: Exception, context: ErrorContext = None) -> bool:
        """Determine if an error is recoverable."""
        if context and not context.recoverable:
            return False
        
        # Network errors are typically recoverable
        if isinstance(error, (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        )):
            return True
        
        # Database connection errors are recoverable
        if isinstance(error, (sqlite3.OperationalError, sqlite3.DatabaseError)):
            return True
        
        # Data validation errors are not recoverable
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        # Configuration errors are not recoverable
        if isinstance(error, ConfigurationError):
            return False
        
        # Default to recoverable for unknown errors
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "unique_error_types": len(self.error_counts),
            "last_errors": self.last_errors.copy()
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(component: str, operation: str, recoverable: bool = True,
                 max_retries: int = 3, recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                timestamp=datetime.utcnow(),
                severity=ErrorSeverity.MEDIUM,
                recoverable=recoverable,
                max_retries=max_retries,
                recovery_strategy=recovery_strategy
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context with additional info
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    timestamp=datetime.utcnow(),
                    severity=_determine_severity(e, context),
                    recoverable=recoverable,
                    max_retries=max_retries,
                    recovery_strategy=recovery_strategy
                )
                
                # Handle the error
                handled = error_handler.handle_error(e, error_context)
                
                if not handled:
                    # Error couldn't be handled, raise it
                    raise
                
                # If this is a TradingBacktesterError, re-raise it
                if isinstance(e, TradingBacktesterError):
                    raise
                
                # Wrap unknown errors
                raise TradingBacktesterError(
                    f"Error in {component}:{operation} - {str(e)}",
                    original_error=e,
                    context=error_context
                )
        
        return wrapper
    return decorator


def _determine_severity(error: Exception, context: ErrorContext) -> ErrorSeverity:
    """Determine error severity based on error type and context."""
    if isinstance(error, (
        ConfigurationError,
        DatabaseError,
        BacktestError
    )):
        return ErrorSeverity.HIGH
    elif isinstance(error, (
        DataIngestionError,
        ModelError,
        APIError
    )):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW


# Database-specific error handling
def handle_database_errors(func: Callable) -> Callable:
    """Decorator specifically for database error handling."""
    return DatabaseRetryHandler(max_retries=3)(handle_errors(
        component="database",
        operation=func.__name__,
        recoverable=True,
        recovery_strategy=RecoveryStrategy.RETRY
    )(func))


# API-specific error handling
def handle_api_errors(func: Callable) -> Callable:
    """Decorator specifically for API error handling."""
    return APITimeoutHandler(max_retries=3, timeout=30.0)(handle_errors(
        component="api",
        operation=func.__name__,
        recoverable=True,
        recovery_strategy=RecoveryStrategy.RETRY
    )(func))


# Data processing error handling
def handle_data_errors(func: Callable) -> Callable:
    """Decorator specifically for data processing error handling."""
    return handle_errors(
        component="data_processing",
        operation=func.__name__,
        recoverable=True,
        recovery_strategy=RecoveryStrategy.FALLBACK
    )(func)


# Model error handling
def handle_model_errors(func: Callable) -> Callable:
    """Decorator specifically for model error handling."""
    return handle_errors(
        component="model",
        operation=func.__name__,
        recoverable=False,
        recovery_strategy=RecoveryStrategy.FAIL_FAST
    )(func)


# Circuit breaker for external services
external_api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)


def with_circuit_breaker(func: Callable) -> Callable:
    """Apply circuit breaker to function calls."""
    return external_api_circuit_breaker(func)


# Graceful degradation utilities
class GracefulDegradation:
    """Utilities for graceful degradation of system functionality."""
    
    @staticmethod
    def provide_fallback(func: Callable, fallback_value: Any = None):
        """Provide fallback value when function fails."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_value
        return wrapper
    
    @staticmethod
    def cache_fallback(cache_key: str, fallback_func: Callable, max_age: int = 300):
        """Use cached data with fallback to fresh computation."""
        # This would integrate with a caching system
        # For now, just a placeholder
        try:
            return fallback_func()
        except Exception as e:
            logging.error(f"Fallback computation failed: {e}")
            return None


# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and system stabilization."""
    
    @staticmethod
    def reset_circuit_breakers():
        """Reset all circuit breakers."""
        external_api_circuit_breaker.state = "CLOSED"
        external_api_circuit_breaker.failure_count = 0
        logging.info("All circuit breakers reset")
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get overall system health based on error rates."""
        stats = error_handler.get_error_statistics()
        
        # Calculate health score (0-100)
        total_errors = stats["total_errors"]
        unique_errors = stats["unique_error_types"]
        
        # Simple health calculation
        if total_errors == 0:
            health_score = 100
        elif total_errors < 10:
            health_score = 80
        elif total_errors < 50:
            health_score = 60
        else:
            health_score = 30
        
        return {
            "health_score": health_score,
            "total_errors": total_errors,
            "unique_error_types": unique_errors,
            "circuit_breaker_state": external_api_circuit_breaker.state,
            "timestamp": datetime.utcnow().isoformat()
        }


# Auto-setup error handlers
def setup_error_handlers():
    """Setup default error handlers."""
    # Register custom handlers for specific error types
    error_handler.register_handler(
        sqlite3.OperationalError,
        lambda e, ctx: logging.error(f"Database operational error: {e}")
    )
    
    error_handler.register_handler(
        requests.exceptions.RequestException,
        lambda e, ctx: logging.warning(f"API request error: {e}")
    )


# Initialize error handlers
setup_error_handlers()