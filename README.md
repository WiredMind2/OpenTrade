# Trading Backtester Production System

A comprehensive, production-ready algorithmic trading backtesting system with sentiment analysis, machine learning models, and real-time monitoring capabilities.

I had multiple goals when starting this project. First of all, I've wanted to create an "autonomous trading bot" for a very long time, as I've always thought that it would be very interesting for passive income.
The other main goal was to try to see how far was it possible to push "vibe coding"; how easy would it be to create a new project with high complexity almost completely with agent coding?
The short answer is, it's very hard ;) . Coding agents are incredible early on in the project, to define the project's boundary a create a code basis with dummy features.
However, it is incredibly difficult and time consuming to debug an AI-introduced bug, as they often happen in very unnatural places (due to an unclear prompt for example). The colossal amount of code constantly changing is also very challenging to track what is going on in the project.
Still, agent coding is just another powerful tool, that has its pros and cons. But this project did reassure for one thing, that is that I probably won't be replaced by an AI anytime soon. They are incredibly quick and efficient, but that goes for creating both good and bad code, and we will still need someone to find the mistakes and correct them, at least for the near future ;)

## 🚀 Features

### Core Functionality
- **News-based Sentiment Analysis**: Process financial news to predict market sentiment
- **Machine Learning Models**: LightGBM-based prediction models for 1d, 3d, and 7d horizons
- **Intraday Backtesting**: Realistic backtesting with slippage and commission modeling
- **Multi-horizon Predictions**: Support for multiple prediction timeframes
- **Interactive Charts**: Real-time OHLC charts with AI prediction overlays
- **Advanced Visualization**: Volume histograms, confidence bands, and prediction aggregation
- **Data Caching**: Optimized performance with intelligent data caching

### Production-Ready Infrastructure
- **Configuration Management**: Centralized config with environment variable support
- **Comprehensive Logging**: Structured logging with different levels and output destinations
- **Error Handling & Recovery**: Circuit breakers, retry mechanisms, and graceful degradation
- **Database Migrations**: Version-controlled schema evolution
- **Data Validation**: Real-time data quality monitoring and anomaly detection
- **Feature Engineering**: Automated feature extraction and selection pipeline
- **Model Versioning**: Model lifecycle management and A/B testing framework
- **REST API**: FastAPI-based endpoints for external integration

### Monitoring & Observability
- **Performance Metrics**: System and application performance monitoring
- **Health Checks**: Comprehensive health check endpoints
- **Alerting System**: Real-time alerting for critical issues
- **Quality Monitoring**: Data quality dashboards and trend analysis

### Security & Deployment
- **Authentication**: API key and JWT-based authentication
- **CI/CD Pipeline**: Automated testing, building, and deployment

   _Note: This repository does not include a `.github/workflows` directory by default. If you want automated CI, add your GitHub Actions workflows under `.github/workflows/`._

## 🤖 Model System

### Overview

The system features a new modular model architecture that enables seamless integration of various machine learning models for trading predictions. This design supports both pre-trained joblib models and custom Python-based models, providing flexibility for different use cases.

### Model Bundle Canonical Format

All models are stored in a standardized bundle format:

```json
{
  "meta": {
    "name": "str",
    "type": "str",
    "version": "str",
    "description": "str",
    "config_schema": {}
  },
  "model": "estimator",
  "extras": {}
}
```

### Adding a New Model

#### For Joblib Models
Save your trained model in the canonical format using `joblib.dump()`.

#### For Python Models
Create a new class in `backend/models/` that inherits from `BaseModel` and implements the required methods (e.g., `predict`, `train`).

### Developer Workflow

- **Model Discovery**: Use the model registry to list available models
- **API Integration**: Access models through dedicated API endpoints
- **Version Management**: Track model versions and performance metrics

### Running the Application and Tests

1. Activate the virtual environment: `& .venv\Scripts\Activate.ps1`
2. Start the backend: `python main.py`
3. Start the frontend: `cd frontend && npm run dev`
4. Run tests: `pytest`

## 📁 Project Structure

```
trading-backtesting/
├── main.py                    # Top-level import shim that re-exports backend app
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── schemas/               # Pydantic models (e.g., `schemas/udf.py`)
│   ├── routes/                # API route modules (health, predictions, backtests, scripts, websocket, ...)
│   ├── config.py              # Configuration management
│   ├── logging_config.py      # Comprehensive logging setup
│   ├── error_handling.py      # Error handling and recovery
│   ├── data_processing.py     # ETL / data processing utilities
│   ├── data_validation.py     # Data quality monitoring
│   ├── routes/monitoring.py  # Performance metrics and monitoring
│   ├── requirements.txt           # Backend Python dependencies
│   └── scripts/                   # Original trading and data ingestion scripts
├── db/                        # Database files & schema
│   └── schema.sql              # Database schema
├── frontend/                  # React/TypeScript frontend
│   ├── package.json
│   ├── src/
│   └── README.md
├── models/                    # Trained model artifacts (.joblib files)
├── tests/                     # Comprehensive test suite (API, integration, unit tests)
├── htmlcov/                   # Generated coverage report
├── .venv/                     # Local development virtual environment (not committed by policy)
└── README.md                  # This README
```

## 🖥 Running the Frontend

```powershell
# In one terminal: start backend (ensure .venv is activated)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# In a second terminal: start the frontend
cd frontend
npm run dev
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10+
- SQLite (or PostgreSQL for production)

### Quick Start

1. **Clone and Setup**:
   ```powershell
   git clone <repository-url>
   cd trading-backtesting
   python -m venv .venv
   # PowerShell
   & .venv\Scripts\Activate.ps1
   # Or use the cross-platform-activation for bash/macOS:
   # source .venv/bin/activate
   # Install backend Python requirements
   pip install -r backend/requirements.txt
   # Install frontend dependencies (optional, if you will run the frontend)
   cd frontend
   npm install
   cd ..
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Initialize Database**:
   ```powershell
   # Run schema migration to create the database schema
   python backend/scripts/apply_schema.py

   # Optionally run the ingestion & pipeline scripts to populate sample data
   python backend/scripts/run_pipeline.py
   ```

4. **Start API Server**:
   ```powershell
   # From repo root (after activating .venv):
   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

   # Or run the module directly from backend:
   cd backend
   python main.py
   ```

## 🔧 Configuration

The system uses a hierarchical configuration system:

1. **Default Configuration** (config.py)
2. **Environment Variables** (.env file)
3. **Runtime Configuration** (API calls)

Key configuration sections:
- **Database**: Connection settings, pool sizes, timeouts
- **API**: Server settings, CORS, authentication
- **Trading**: Capital, commissions, slippage, exposure limits
- **Models**: Model paths, training parameters
- **Logging**: Log levels, output formats, destinations
- **Monitoring**: Alert thresholds, performance metrics

## 📊 API Documentation

### Core Endpoints

#### Health & Monitoring
- `GET /health` - System health check
- `GET /metrics` - Performance metrics
- `GET /docs` - Interactive API documentation

#### Predictions
- `POST /predict` - Make trading predictions
- `GET /predictions/recent` - Get recent predictions
- `GET /models` - List available models

#### Backtesting
- `POST /backtest` - Run backtest
- `GET /backtest/{id}` - Get backtest results

#### Data Access
- `GET /data/prices/{ticker}` - Get price data
- `GET /portfolio/current` - Get current portfolio

#### Script Execution
- `POST /scripts/execute` - Execute data processing or ML script
- `GET /scripts/status/{execution_id}` - Get script execution status
- `GET /scripts/executions` - List all script executions
- `POST /scripts/pipeline/run` - Run the full data processing pipeline
- `GET /scripts/pipeline/status/{execution_id}` - Get pipeline execution status

### Authentication

API endpoints support authentication via:
- API Key (header: `Authorization: Bearer <key>`)
- JWT Tokens (for advanced use cases)

## 🧪 Testing

### Test Status
This repository includes a comprehensive test suite that covers API endpoints, the backtesting engine, data processing and integrations. Run the tests locally to verify the current status and coverage.

### Running Tests
- API endpoints (health, predictions, backtests, data, portfolio, scripts, monitoring, websockets)
- Backtesting engine functionality
- Script execution and pipeline management
- Integration workflows

### Running Tests

```powershell
# Run all tests (make sure .venv is activated)
pytest tests/

# Run with coverage
pytest --cov=backend --cov-report=html

# Run a specific tests file
pytest tests/test_backtesting.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Individual component testing (API endpoints, data validation, utilities)
- **Integration Tests**: End-to-end workflow testing (pipeline execution, backtest flows)
- **WebSocket Tests**: Real-time communication testing
- **Script Execution Tests**: Background task and pipeline validation

## 📈 Usage Examples

### Making Predictions

```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'ticker': 'AAPL',
    'horizon': '1d',
    'context': {'market_conditions': 'normal'}
})

prediction = response.json()
print(f"Predicted return: {prediction['predicted_return']:.4f}")
```


### Running Backtests

```python
response = requests.post('http://localhost:8000/backtest', json={
    'strategy_name': 'sentiment_momentum',
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 100000,
    'parameters': {'sentiment_threshold': 0.02}
})

backtest_id = response.json()['id']
```


### Data Quality Monitoring

```python
from data_validation import create_data_quality_monitor

monitor = create_data_quality_monitor()
reports = monitor.run_quality_checks(['price_daily', 'sentiment_predictions'])

for table, report in reports.items():
    print(f"{table}: {report.quality_level.value} ({report.quality_score:.2%})")
```

## 🔍 Monitoring & Alerting

### Health Checks

The system provides multiple health check endpoints:
- Database connectivity
- Model availability
- Data freshness
- System resources
- API response times

### Performance Monitoring

Track key metrics:
- Prediction latency
- Model accuracy over time
- Data quality scores
- System resource usage
- Error rates

### Alerting

Configure alerts for:
- Model accuracy degradation
- Data quality issues
- System resource constraints
- API performance degradation
- Prediction confidence thresholds

### Code Quality Standards

- **Type Hints**: All functions should have type annotations
- **Documentation**: Comprehensive docstrings for all public APIs
- **Testing**: Minimum 80% test coverage
- **Logging**: Appropriate logging for debugging and monitoring
- **Error Handling**: Comprehensive error handling and recovery

## 🙏 Acknowledgments

- Built with FastAPI, pandas, scikit-learn, and LightGBM
- Uses TA-Lib for technical analysis indicators
- Inspired by modern MLOps best practices
- Designed for production financial trading systems

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading involves risk of financial loss. Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.