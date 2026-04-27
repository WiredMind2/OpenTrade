"""
Forecast and trading-oriented metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.ml.forecasting.contracts import HorizonMetrics


class Evaluator:
    def compute_horizon_metrics(self, y_true_by_h: Dict[int, List[float]], y_pred_by_h: Dict[int, List[float]]) -> List[HorizonMetrics]:
        rows: List[HorizonMetrics] = []
        for h in sorted(y_true_by_h.keys()):
            yt = np.asarray(y_true_by_h[h], dtype=float)
            yp = np.asarray(y_pred_by_h[h], dtype=float)
            if len(yt) == 0 or len(yp) == 0:
                continue
            rmse = float(mean_squared_error(yt, yp) ** 0.5)
            mae = float(mean_absolute_error(yt, yp))
            da = float(np.mean(np.sign(yt) == np.sign(yp)))
            corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else 0.0
            r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
            mape = None
            if np.all(np.abs(yt) > 1e-12):
                mape = float(np.mean(np.abs((yt - yp) / yt)))
            rows.append(HorizonMetrics(horizon=h, rmse=rmse, mae=mae, directional_accuracy=da, correlation=corr, r2_oos=r2, mape=mape))
        return rows
