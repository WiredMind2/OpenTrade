"""Top-level backend launcher and compatibility exports.

From repo root:
    python main.py

This module also preserves `from main import app, app_state` imports used by
tests and route fallbacks by re-exporting from `app_shim`.
"""
from app_shim import app, app_state

__all__ = ["app", "app_state"]


if __name__ == "__main__":
    import uvicorn
    from backend.config import get_config

    config = get_config()
    uvicorn.run(
        "backend.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers if config.environment == "production" else 1,
        reload=config.api.reload if config.environment == "development" else False,
        log_level=config.logging.level.lower(),
    )
