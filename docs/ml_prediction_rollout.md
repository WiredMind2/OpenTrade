# ML Prediction Rollout Guide

## Phase 1: Shadow Mode
- Keep legacy prediction outputs active for UI.
- Run new `PredictionService` in parallel and store outputs to `sentiment_predictions`.
- Compare metrics daily: directional accuracy, MAE, calibration.

## Phase 2: Controlled Cutover
- Enable realtime path with config gate (`ML_PREDICTION_V2_ENABLED=true`).
- Monitor p95 latency and error rate by horizon.
- Validate confidence interval coverage over 3 trading sessions.

## Phase 3: Full Production
- Route `POST /predict` fully to shared service.
- Keep rollback flag available for one release cycle.
- Archive old model artifacts and keep last known good bundle.

## Acceptance Thresholds
- Directional accuracy >= 53% on each horizon.
- MAE improves by at least 5% vs legacy baseline.
- p95 latency < 300ms for cache-hot requests.
- Error rate < 1% over rolling 24h.

## Rollback Procedure
1. Toggle `ML_PREDICTION_V2_ENABLED=false`.
2. Restart API workers.
3. Confirm `/predict` success and recent predictions read path.
4. Open incident note and preserve failing model artifacts.
