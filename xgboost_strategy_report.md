# XGBoost intraday trading strategy implementation report

This report is written as an engineering specification for implementing an XGBoost-based trading strategy inside an existing backtesting and bot project. It is intentionally concrete and opinionated so an implementation agent can translate it directly into code.

## Objective

Build a reproducible, leakage-safe, walk-forward XGBoost strategy for intraday stock or ETF trading. The model predicts a short-horizon target from tabular market features, converts predictions into trading signals, sizes positions under risk constraints, and is evaluated with realistic costs, slippage, and retraining schedules. Walk-forward validation is the correct evaluation approach for time series because temporal order must be respected and randomized cross-validation is inappropriate.[web:26][web:27]

## What the model should predict

The first implementation should predict future return, not raw price. Predicting returns is usually more stable and directly maps into trading decisions, while price prediction can look numerically accurate without producing tradable edge.[page:1]

Recommended primary target:
- `y_t = log(close_{t+h} / close_t)` for horizon `h` in bars, where `h` is typically 1, 3, or 5 bars for intraday work.

Recommended secondary targets for later experiments:
- Binary direction: `sign(y_t) > 0`
- Volatility-adjusted return: `y_t / realized_vol_t`
- Triple-barrier label for meta-labeling after the first version

Implementation rule: start with regression using `XGBRegressor` and `objective='reg:squarederror'`, then derive trade decisions from predicted return.[page:1][page:2]

## Trading universe

The first production-like version should focus on highly liquid instruments because execution modeling is simpler and slippage is lower. Suitable choices are SPY, QQQ, or a small basket of large-cap US stocks or liquid ETFs.[cite:24][web:23]

Implementation defaults:
- Universe size: 1 to 10 liquid symbols initially
- Bar frequency: 1-minute or 5-minute bars
- Session: regular market hours only unless the project explicitly supports premarket and after-hours logic
- Timezone handling: normalize all timestamps to exchange time and ensure session boundaries are explicit

## Data requirements

The system must train on historical intraday OHLCV data plus any derived features. Long-range intraday research depends heavily on data quality; free sources are usually limited, while broker or market-data APIs are more suitable for serious testing.[cite:2][web:23]

Required raw fields per bar:
- timestamp
- open
- high
- low
- close
- volume
- vwap if available, otherwise compute an approximation if the framework supports it
- bid/ask or spread fields if available; if not, backtest must inject a spread/slippage model

Strong recommendation:
- Store data in a columnar format like parquet
- Precompute session id, bar index within session, and market calendar metadata
- Never compute features with future information

## Core architecture

The implementation should be split into modules with clean interfaces:

1. `data/loader.py`
   - Loads historical bars
   - Aligns sessions and handles missing bars
   - Returns a canonical dataframe indexed by timestamp and symbol

2. `features/builder.py`
   - Computes all lagged and rolling features using only past data
   - Enforces NaN trimming and feature validity windows

3. `labels/targets.py`
   - Builds forward returns for one or several horizons
   - Optionally builds classification labels

4. `models/xgb_model.py`
   - Creates XGBoost model objects
   - Fits, predicts, saves, loads, exposes feature importance and SHAP hooks

5. `validation/walk_forward.py`
   - Runs rolling or expanding walk-forward splits
   - Retrains on schedule
   - Produces out-of-sample predictions only

6. `strategy/signal.py`
   - Converts predictions into discrete actions: flat, long, short
   - Applies thresholds, cooldowns, regime filters, and risk gates

7. `portfolio/execution.py`
   - Simulates fills, slippage, commissions, spread costs, and position sizing

8. `evaluation/metrics.py`
   - Forecast metrics and trading metrics

9. `configs/xgb_strategy.yaml`
   - Contains all feature, model, threshold, and risk parameters

## Label construction

The agent should implement labels exactly and avoid any ambiguity.

Primary regression target:
\[
y_t^{(h)} = \log\left(\frac{C_{t+h}}{C_t}\right)
\]

where `C_t` is the close of the current bar and `h` is the forecast horizon in bars. Use a shift of `-h` and drop the final `h` rows per symbol after label generation.[page:1]

Optional classification label:
- `label_up = 1 if y_t > 0 else 0`

Optional thresholded label for class balance:
- `label_up = 1 if y_t > tau else 0`, `0` otherwise, where `tau` is a small return threshold chosen to exceed estimated costs

Implementation requirements:
- Labels must be computed separately per symbol
- No forward-filled labels across missing market periods
- If trading only during regular hours, horizon must not cross session close unless that behavior is explicitly intended

## Feature set

The first feature set should be tabular, compact, and interpretable. XGBoost works well on engineered tabular features and often outperforms more complex models in noisy financial settings when validation is strict.[page:1]

### Price and return features

Per symbol, compute:
- `ret_1`, `ret_2`, `ret_3`, `ret_5`, `ret_10`: lagged log returns over various horizons
- `gap_from_open`: `log(close_t / session_open)`
- `range_1`: `(high - low) / close`
- `body_frac`: `(close - open) / max(high - low, eps)`
- `close_to_high`, `close_to_low`
- `rolling_mean_ret_n` for `n in {5, 10, 20}`
- `rolling_std_ret_n` for `n in {5, 10, 20}`
- z-score of current return relative to rolling mean and std

### Trend and momentum features

- EMA distances: `(close - ema_n) / ema_n` for `n in {5, 10, 20, 50}`
- EMA slope approximations over last `k` bars
- RSI 14 and RSI 5
- Stochastic oscillator if already available in project utilities
- MACD line, signal, histogram if desired, but do not bloat the first version

### Volume and activity features

- `vol_ratio_n = volume / rolling_mean_volume_n`
- rolling volume std
- signed price move times volume proxy
- intraday cumulative volume percentile within session if available

### VWAP and intraday structure features

- distance to session VWAP: `(close - vwap_session) / vwap_session`
- signed distance to intraday high/low
- minutes since open, minutes to close
- bar index in session and normalized time-of-day encodings
- morning/late-session indicator

### Market context features

If trading multiple symbols:
- same features for SPY or sector ETF as context
- relative strength versus SPY over recent windows
- symbol return minus benchmark return over `n` bars

### Volatility and risk features

- realized volatility over 5, 10, 20 bars
- ATR-like range estimate over 14 bars
- rolling max adverse excursion proxy
- rolling spread estimate if bid/ask unavailable

### Regime features

These are important because many strategies only work in certain micro-regimes.
- volatility regime bucket
- trend regime bucket from longer EMA slope
- opening-auction regime flag for first `m` bars
- lunch regime flag
- event filter flags if the project has earnings or macro calendar support

Implementation rules:
- Every rolling statistic must be shifted so it uses only data known at prediction time
- Feature calculations must be grouped by symbol and session where needed
- All features should be stored with a deterministic feature list so training and inference use identical columns

## Feature hygiene

The implementation agent must enforce the following safeguards:
- Drop rows with insufficient lookback after feature creation
- Replace infinite values with NaN, then drop or impute by training-only rules
- For tree models, scaling is usually unnecessary, but deterministic missing-value handling is still required
- Track the exact feature names and order in the saved model artifact
- Prevent duplicated features or features derived from future bars

## Model choice

Use tree boosting with XGBoost and start with the `hist` tree method because it is fast and generally preferred for squared-error objectives; the XGBoost documentation notes that `hist` should be preferred for objectives like `reg:squarederror` with constant hessian, while `approx` may sometimes help at greater computational cost.[page:2]

Default model class:
- `xgboost.XGBRegressor`

Default initial parameters:
- `objective='reg:squarederror'`
- `tree_method='hist'`
- `booster='gbtree'`
- `n_estimators=400`
- `learning_rate=0.03`
- `max_depth=4`
- `min_child_weight=8`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `gamma=0.0`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `max_bin=256`
- `random_state=<fixed>`
- `n_jobs=-1`

Why these defaults:
- Small learning rate and moderate tree depth reduce overfitting on noisy returns
- Subsampling and column subsampling add regularization
- Mild L1/L2 regularization helps stability
- Hist tree method keeps training fast enough for walk-forward retraining[page:1][page:2]

## Hyperparameter tuning

Tuning must happen only on the training segment of each research phase, never on the final test region, and temporal order must be respected.[page:1][web:26][web:27]

Recommended search space, adapted from practical XGBoost tuning ranges reported in finance/time-series examples:
- `n_estimators`: 200 to 1200
- `max_depth`: 3 to 8
- `learning_rate`: 0.01 to 0.15 on log scale
- `subsample`: 0.6 to 1.0
- `colsample_bytree`: 0.6 to 1.0
- `min_child_weight`: 1 to 15
- `gamma`: 0.0 to 5.0
- `reg_alpha`: 0.0 to 2.0
- `reg_lambda`: 0.1 to 5.0
- `max_bin`: 128 to 512[page:1]

Search method recommendation:
- Use Optuna or a similar Bayesian optimizer
- Optimize on a time-series split within the training set only
- Target metric should be trading-aligned, not just MSE

Good optimization targets, in descending preference:
1. Out-of-sample information ratio after estimated costs on validation slices
2. Directional accuracy on trades that exceed the action threshold
3. Rank correlation between prediction and future return
4. RMSE or MAE on return prediction[page:1]

The report on NEPSE used time-series split with hyperparameter tuning on the initial training segment and then evaluated with walk-forward validation, which is the correct structure to emulate here.[page:1]

## Validation protocol

The strategy must use walk-forward validation or rolling-origin evaluation, not shuffled k-fold cross-validation, because the latter leaks temporal structure and produces unrealistic estimates.[web:26][web:27]

### Recommended split design

- Chronological split the full dataset into:
  - train-development segment: first 60%
  - validation-development segment: next 20%
  - final untouched test segment: last 20%
- Use time-series tuning only on the first 80%
- Reserve the last 20% for one final walk-forward evaluation[page:1]

### Walk-forward variants

1. Expanding window
   - Train on all data up to time `t`
   - Predict on the next block or next bar
   - Expand the training set forward

2. Rolling window
   - Train on a fixed-length recent history window
   - Predict on the next block or next bar
   - Slide forward

Expanding and rolling windows are both valid; expanding windows often benefit from more data, while rolling windows adapt faster to regime changes.[page:1][web:26]

### Retraining schedule

Intraday implementation should not retrain every bar unless computationally cheap and deliberately chosen. Recommended initial schedule:
- Retrain once per day before market open for 1-minute or 5-minute strategies
- Optionally retrain every hour in research mode and compare performance

### Prediction timing contract

At bar close `t`:
- compute features using data up to and including bar `t`
- predict `y_t` for horizon `h`
- place order for execution on bar `t+1` open or with explicit next-bar fill logic

This timing contract must be hard-coded and audited because many trading backtests accidentally trade on the same bar that generated the signal.

## Signal conversion

Predicted returns must not be traded raw without thresholds. The model may have weak average edge, and only tail predictions should generate trades.

Recommended basic rule:
- Long if `pred_return > long_threshold`
- Short if `pred_return < -short_threshold`
- Flat otherwise

Threshold design:
- Start with symmetric thresholds
- Set threshold above estimated total round-trip cost plus a margin
- Optimize thresholds on validation data, not on the final test

Example:
- If expected one-way spread plus slippage plus fees is 4 bps, a 1-bar strategy should not trade at thresholds near 0 bps
- Start testing thresholds like 6, 8, 10, 12 bps in predicted return units

Alternative ranking design for multi-asset version:
- Rank symbols by prediction each bar
- Long top `k`, short bottom `k`, subject to confidence and liquidity filters

## Meta-filters and trade gating

The first version should include simple gates that usually improve robustness:
- No trade in first `n` minutes if open is too noisy, unless strategy is explicitly open-driven
- No trade in last `m` minutes unless holding through close is supported
- Skip if rolling spread estimate exceeds threshold
- Skip if current realized volatility exceeds a circuit-breaker threshold
- Skip if benchmark regime filter says trend strategy is off or mean-reversion strategy is off
- Cooldown after stop-out or after a sequence of losses

## Position sizing

Do not start with all-in sizing. Use volatility-aware or fixed-risk sizing.

Recommended initial rule:
- Max gross exposure per symbol fixed as a fraction of equity
- Position size scales with confidence and inverse volatility

Example formula:
\[
size_t = clip\left(k \cdot \frac{|\hat y_t|}{\sigma_t + \epsilon},\ 0,\ size_{max}\right)
\]

where `sigma_t` is recent realized volatility and `k` is calibrated on the validation set.

Safer implementation defaults:
- Start with fixed one-unit sizing in research mode
- Once signal quality is proven, move to capped volatility scaling
- Always enforce max leverage, max symbol concentration, and max daily turnover

## Risk management

Risk rules are part of the strategy, not optional wrappers.

Minimum required controls:
- Per-trade stop loss in return or ATR units
- Per-trade take profit if strategy horizon benefits from it, though many short-horizon models work better with time-based exits
- Maximum holding period in bars, such as exit after `h` or `2h` bars regardless of signal decay
- Maximum concurrent positions
- Maximum daily loss, after which the strategy stops trading for the day
- Kill switch on abnormal spread, missing data, or stale predictions

Recommended first exit policy:
- Enter on next bar after signal
- Exit after `h` bars, or earlier on stop loss / take profit
- Re-entry only after a cooldown of `c` bars if stopped out

## Execution model

Backtesting without execution realism is one of the most common ways to fool yourself. Slippage and liquidity must be modeled explicitly.[web:22][page:1]

Required cost components:
- Fixed commissions if applicable
- Half-spread or full-spread cost depending on order model
- Slippage proportional to volatility, spread, participation, or bar range
- Optional market-impact proxy for larger sizes

Recommended simple fill model for first implementation:
- Signal generated at close of bar `t`
- Order executed at open of bar `t+1`
- Fill price adjusted by:
  - `+ spread/2 + slippage` for buys
  - `- spread/2 - slippage` for sells

If bid/ask is unavailable, approximate spread using a rolling estimator or a fixed bps assumption by symbol and time of day.

Recommended slippage baseline:
- `slippage_bps = a + b * normalized_volatility + c * participation_rate`
- For the first version, use a fixed slippage bps grid and stress test results under multiple values

## Evaluation metrics

The implementation agent must report both forecast quality and trading performance. Forecast metrics alone are insufficient.[page:1][web:22]

### Forecast metrics
- RMSE on returns
- MAE on returns
- Sign accuracy
- Correlation between predicted and realized returns
- Rank IC if cross-sectional

### Trading metrics
- Net return
- Sharpe ratio or intraday annualized Sharpe if the framework already standardizes it
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Win rate
- Profit factor
- Average trade return
- Average holding time
- Turnover
- Exposure
- Stability across symbols and time slices

### Robustness diagnostics
- Performance by month and regime
- Performance by hour-of-day bucket
- Long-only, short-only, and combined decomposition
- Performance before and after cost stress
- Sensitivity to threshold and retraining frequency

## Leakage checklist

The implementation agent must validate these points before trusting any result:
- No feature uses future bars
- Rolling statistics are computed with past-only windows
- Label is shifted forward correctly
- Train/validation/test boundaries are strictly chronological
- Hyperparameter tuning excludes the final test window
- Threshold optimization excludes the final test window
- Execution occurs after signal generation, not on the same bar unless the data truly supports that assumption
- Universe membership changes are handled without survivorship bias if using many stocks

This is critical because rigorous walk-forward evaluation is specifically recommended to avoid lookahead bias in financial forecasting.[web:26][web:27][page:1]

## Explainability and diagnostics

One reason to use XGBoost is that it remains much more interpretable than many deep models in tabular financial problems.[page:1]

The implementation agent should support:
- Gain-based feature importance
- Permutation importance on validation windows
- SHAP values for local and global explanations
- Prediction distribution monitoring over time
- Drift checks for feature distributions versus training history

Practical use:
- Verify that a small subset of sensible features dominates importance
- If one suspicious feature dominates, audit it for leakage
- Check whether feature importance regime-shifts over time

## Recommended experiment sequence

The implementation should follow this sequence rather than jumping to a huge search.

### Phase 1: Minimal viable model
- One symbol, e.g. SPY
- 5-minute bars
- Horizon `h = 3`
- 20 to 40 core features
- Daily retraining
- Regression target
- Next-bar open execution
- Fixed fees and fixed slippage

Success criteria:
- Stable positive validation performance after costs
- Not dominated by one month or one regime

### Phase 2: Threshold and exit optimization
- Tune long/short thresholds
- Tune stop loss, take profit, and max holding bars
- Test fixed-size versus volatility-scaled positions

### Phase 3: Robustness expansion
- Add QQQ or a handful of liquid stocks
- Add benchmark context features
- Compare expanding versus rolling retraining windows
- Stress test with higher costs and slippage

### Phase 4: Advanced improvements
- Multi-horizon predictions
- Meta-labeling to decide whether to take model signals
- Regime-specific models
- Ensemble of XGBoost with linear baseline
- Separate long and short models if asymmetry is strong

## Concrete pseudocode

```python
for symbol in symbols:
    df = load_bars(symbol)
    df = build_features(df)
    df = build_labels(df, horizon=h)
    store(df)

all_data = concat_symbols(symbol_frames)

for split in walk_forward_splits(all_data, mode="expanding", retrain="1D"):
    train_df = split.train
    test_df = split.test

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_df["pred"] = preds
    test_df["signal"] = 0
    test_df.loc[test_df.pred > long_threshold, "signal"] = 1
    test_df.loc[test_df.pred < -short_threshold, "signal"] = -1

    test_df = apply_trade_gates(test_df)
    trades = simulate_execution(test_df, fill_model="next_open_plus_costs")
    record_predictions_and_trades(trades)

report = evaluate(all_predictions, all_trades)
```

## Suggested configuration file

```yaml
strategy_name: xgb_intraday_v1
symbols: [SPY]
bar_size: 5min
session: regular
horizon_bars: 3
retrain_frequency: 1D
walk_forward_mode: expanding
train_window_bars: null
prediction_timing: bar_close
execution_timing: next_bar_open
objective: reg:squarederror
model:
  tree_method: hist
  booster: gbtree
  n_estimators: 400
  learning_rate: 0.03
  max_depth: 4
  min_child_weight: 8
  subsample: 0.8
  colsample_bytree: 0.8
  gamma: 0.0
  reg_alpha: 0.1
  reg_lambda: 1.0
  max_bin: 256
thresholds:
  long: 0.0008
  short: 0.0008
risk:
  max_position_per_symbol: 0.10
  max_gross_exposure: 0.50
  max_daily_loss: 0.02
  stop_loss: 0.003
  take_profit: 0.004
  max_holding_bars: 3
costs:
  commission_bps: 0.0
  half_spread_bps: 1.0
  slippage_bps: 2.0
features:
  include:
    - ret_1
    - ret_2
    - ret_3
    - ret_5
    - ret_10
    - range_1
    - body_frac
    - rolling_mean_ret_5
    - rolling_mean_ret_10
    - rolling_std_ret_5
    - rolling_std_ret_10
    - ema_dist_5
    - ema_dist_10
    - ema_dist_20
    - rsi_5
    - rsi_14
    - vol_ratio_5
    - vol_ratio_20
    - vwap_dist
    - minutes_since_open
    - minutes_to_close
```

## Implementation notes specific to XGBoost

- Prefer `gbtree` booster initially; `dart` adds stochastic dropout complexity and can make inference behavior less straightforward for a first trading build.[web:34]
- Prefer `tree_method='hist'` initially because it is the fastest built-in method and is specifically recommended for many common regression objectives.[page:2]
- Keep depth moderate; deeper trees often memorize noise in financial returns
- Use fixed random seeds and deterministic data ordering for reproducibility
- Save the fitted model, feature list, hyperparameters, train period, and training metrics together as one versioned artifact

## Failure modes to watch

The agent should explicitly test for these failure modes:
- Great RMSE but unprofitable trading after costs
- Profits concentrated in one regime only
- Performance collapses under slightly higher slippage assumptions
- Signal imbalance, e.g. nearly all long or nearly all flat
- Feature leakage hidden in rolling computations or session-boundary logic
- Hyperparameter overfitting from too many trials on too little data
- Retraining too frequently and effectively fitting noise

## Deliverables the implementation agent should produce

1. A feature builder module with deterministic column outputs
2. A target builder module for forward-return labels
3. An XGBoost trainer with versioned config support
4. A walk-forward runner that emits out-of-sample predictions only
5. A signal engine translating predictions to orders
6. A cost-aware execution simulator
7. A metrics report generator
8. A feature-importance and SHAP diagnostic module
9. Unit tests covering label alignment, feature leakage checks, and execution timing
10. A reproducible experiment config for the first baseline strategy

## Minimal acceptance test

The first implementation can be considered correct if all of the following are true:
- The pipeline runs end to end on one symbol with no leakage errors
- Walk-forward predictions are generated strictly out of sample
- The model can be retrained on schedule and reused for inference
- Trades are executed one bar after signal generation under explicit cost assumptions
- The backtest report includes both forecast and trading metrics
- Results can be reproduced from a saved config and seed

## Final recommendation

The best first XGBoost strategy is not a giant feature soup or a black-box optimizer. It is a small, disciplined, leakage-safe return-prediction system with walk-forward validation, thresholded trade selection, and realistic execution. In time-series trading work, the validation design and cost model matter as much as the model itself, and walk-forward retraining on chronologically separated data is the standard approach for realistic assessment.[web:26][web:27][page:1]
