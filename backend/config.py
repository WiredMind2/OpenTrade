"""
Production configuration management for trading backtesting system.

This module provides centralized configuration management with environment-specific settings,
validation, and secure handling of secrets.
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "data/backtest.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = True
    rate_limit_per_minute: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 7
    structured_logging: bool = True
    timed_rotation: bool = True


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    model_dir: str = "models"
    experiment_tracking: bool = True
    model_registry: str = "mlflow"
    auto_retrain: bool = False
    retrain_schedule: str = "weekly"
    validation_split: float = 0.2
    test_split: float = 0.1
    hyperparameter_tuning: bool = False


@dataclass
class DataConfig:
    """Data ingestion and processing configuration."""
    data_dir: str = "data"
    price_update_interval: int = 3600  # seconds
    news_update_interval: int = 1800   # seconds
    max_retries: int = 3
    retry_delay: int = 5
    batch_size: int = 1000
    cache_ttl: int = 300  # seconds


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    initial_capital: float = 100000.0
    commission_per_share: float = 0.005
    slippage_pct: float = 0.0002
    max_position_pct: float = 0.1
    max_total_exposure: float = 0.5
    risk_free_rate: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    train_optimizer_engine: str = "backtrader"


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-here"
    jwt_secret_key: str = "your-jwt-secret-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    encrypt_sensitive_data: bool = True
    session_timeout: int = 1800  # 30 minutes


@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "error_rate": 5.0
    })


@dataclass
class Config:
    """Main application configuration."""
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # External services
    newsapi_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    alerts_webhook_url: Optional[str] = None


class ConfigManager:
    """Configuration manager with environment overrides and validation."""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        self.config_path = config_path
        self.env_file = env_file
        self._config: Optional[Config] = None
        
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
    
    @property
    def config(self) -> Config:
        """Get current configuration, loading and validating if necessary."""
        if self._config is None:
            self._config = self._load_config()
            self._validate_config()
        return self._config
    
    def _load_config(self) -> Config:
        """Load configuration from files and environment variables."""
        config = Config()
        
        # Load from YAML config file if provided
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._update_config_from_dict(config, file_config)
        
        # Override with environment variables
        self._update_config_from_env(config)
        
        return config
    
    def _update_config_from_dict(self, config: Config, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        # Database config
        if 'database' in config_dict:
            db_config = config_dict['database']
            for key, value in db_config.items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)
        
        # API config
        if 'api' in config_dict:
            api_config = config_dict['api']
            for key, value in api_config.items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        # Logging config
        if 'logging' in config_dict:
            log_config = config_dict['logging']
            for key, value in log_config.items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        # Model config
        if 'model' in config_dict:
            model_config = config_dict['model']
            for key, value in model_config.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Data config
        if 'data' in config_dict:
            data_config = config_dict['data']
            for key, value in data_config.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # Trading config
        if 'trading' in config_dict:
            trading_config = config_dict['trading']
            for key, value in trading_config.items():
                if hasattr(config.trading, key):
                    setattr(config.trading, key, value)
        
        # Security config
        if 'security' in config_dict:
            security_config = config_dict['security']
            for key, value in security_config.items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        # Monitoring config
        if 'monitoring' in config_dict:
            monitoring_config = config_dict['monitoring']
            for key, value in monitoring_config.items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        # General config
        for key in ['environment', 'debug', 'version']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # External services
        for key in ['newsapi_key', 'mlflow_tracking_uri', 'alerts_webhook_url']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
    
    def _update_config_from_env(self, config: Config):
        """Update configuration from environment variables."""
        # Database config
        db_path = os.getenv('DB_PATH')
        if db_path:
            config.database.path = db_path
        db_echo = os.getenv('DB_ECHO')
        if db_echo:
            config.database.echo = db_echo.lower() == 'true'
        
        # API config
        api_host = os.getenv('API_HOST')
        if api_host:
            config.api.host = api_host
        api_port = os.getenv('API_PORT')
        if api_port:
            config.api.port = int(api_port)
        api_workers = os.getenv('API_WORKERS')
        if api_workers:
            config.api.workers = int(api_workers)
        
        # Trading config
        initial_capital = os.getenv('INITIAL_CAPITAL')
        if initial_capital:
            config.trading.initial_capital = float(initial_capital)
        commission_per_share = os.getenv('COMMISSION_PER_SHARE')
        if commission_per_share:
            config.trading.commission_per_share = float(commission_per_share)
        slippage_pct = os.getenv('SLIPPAGE_PCT')
        if slippage_pct:
            config.trading.slippage_pct = float(slippage_pct)
        train_optimizer_engine = os.getenv('TRAIN_OPT_ENGINE')
        if train_optimizer_engine:
            config.trading.train_optimizer_engine = train_optimizer_engine.strip().lower()
        
        # External services
        config.newsapi_key = os.getenv('NEWSAPI_KEY')
        config.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        config.alerts_webhook_url = os.getenv('ALERTS_WEBHOOK_URL')
        
        # General config
        config.environment = os.getenv('ENVIRONMENT', 'development')
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        global_config = self._config
        
        # Validate trading parameters
        if global_config.trading.max_position_pct <= 0 or global_config.trading.max_position_pct > 1:
            errors.append("max_position_pct must be between 0 and 1")
        
        if global_config.trading.max_total_exposure <= 0 or global_config.trading.max_total_exposure > 1:
            errors.append("max_total_exposure must be between 0 and 1")
        
        if global_config.trading.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        
        # Validate database path
        if global_config.database.path:
            db_dir = Path(global_config.database.path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create database directory: {e}")
        
        # Validate API configuration
        if global_config.api.port <= 0 or global_config.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        # Validate external API keys for production
        if global_config.environment == 'production':
            if not global_config.newsapi_key or global_config.newsapi_key == 'your_newsapi_key_here':
                errors.append("NEWSAPI_KEY must be set for production")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def save_config(self, filepath: str):
        """Save current configuration to file."""
        config_dict = self._config_to_dict(self.config)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        result = {
            'environment': config.environment,
            'debug': config.debug,
            'version': config.version,
            'database': {
                'path': config.database.path,
                'echo': config.database.echo,
                'pool_size': config.database.pool_size,
                'max_overflow': config.database.max_overflow,
                'pool_timeout': config.database.pool_timeout,
                'pool_recycle': config.database.pool_recycle
            },
            'api': {
                'host': config.api.host,
                'port': config.api.port,
                'workers': config.api.workers,
                'reload': config.api.reload,
                'cors_origins': config.api.cors_origins,
                'api_key_required': config.api.api_key_required,
                'rate_limit_per_minute': config.api.rate_limit_per_minute
            },
            'logging': {
                'level': config.logging.level,
                'format': config.logging.format,
                'file_path': config.logging.file_path,
                'max_file_size': config.logging.max_file_size,
                'backup_count': config.logging.backup_count,
                'structured_logging': config.logging.structured_logging,
                'timed_rotation': config.logging.timed_rotation
            },
            'model': {
                'model_dir': config.model.model_dir,
                'experiment_tracking': config.model.experiment_tracking,
                'model_registry': config.model.model_registry,
                'auto_retrain': config.model.auto_retrain,
                'retrain_schedule': config.model.retrain_schedule,
                'validation_split': config.model.validation_split,
                'test_split': config.model.test_split,
                'hyperparameter_tuning': config.model.hyperparameter_tuning
            },
            'data': {
                'data_dir': config.data.data_dir,
                'price_update_interval': config.data.price_update_interval,
                'news_update_interval': config.data.news_update_interval,
                'max_retries': config.data.max_retries,
                'retry_delay': config.data.retry_delay,
                'batch_size': config.data.batch_size,
                'cache_ttl': config.data.cache_ttl
            },
            'trading': {
                'initial_capital': config.trading.initial_capital,
                'commission_per_share': config.trading.commission_per_share,
                'slippage_pct': config.trading.slippage_pct,
                'max_position_pct': config.trading.max_position_pct,
                'max_total_exposure': config.trading.max_total_exposure,
                'risk_free_rate': config.trading.risk_free_rate,
                'stop_loss_pct': config.trading.stop_loss_pct,
                'take_profit_pct': config.trading.take_profit_pct,
                'train_optimizer_engine': config.trading.train_optimizer_engine
            },
            'security': {
                'secret_key': config.security.secret_key,
                'jwt_secret_key': config.security.jwt_secret_key,
                'jwt_algorithm': config.security.jwt_algorithm,
                'jwt_expiration': config.security.jwt_expiration,
                'encrypt_sensitive_data': config.security.encrypt_sensitive_data,
                'session_timeout': config.security.session_timeout
            },
            'monitoring': {
                'enable_metrics': config.monitoring.enable_metrics,
                'metrics_port': config.monitoring.metrics_port,
                'health_check_interval': config.monitoring.health_check_interval,
                'alert_thresholds': config.monitoring.alert_thresholds
            },
            'mlflow_tracking_uri': config.mlflow_tracking_uri,
            'alerts_webhook_url': config.alerts_webhook_url
        }
        return result


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.config


def get_config() -> Config:
    """Get the current configuration."""
    return config


def reload_config():
    """Reload configuration from files."""
    global config_manager, config
    config_manager._config = None
    config = config_manager.config
    return config


# Global API key for external services
API_KEY = os.getenv('API_KEY')