Time‑series-based trading today is dominated by a mix of factor/stat‑arb methods, deep sequence models (Transformers, TCNs, RNNs), and reinforcement learning agents that act directly on price streams.  Below is a structured “design space” you can hand to an AI agent: for each strategy family I spell out model classes, data, features, training targets, evaluation, and implementation notes. [simplechart](https://simplechart.in/quantitative-trading-strategies-2025/)

***

## 1. Core time series strategy families

These are the main quantitative paradigms that map naturally to time series models. [wemastertrade](https://wemastertrade.com/what-is-quantitative-trading/)

### 1.1 Trend / momentum models

Goal: exploit serial correlation in returns and medium‑term trends.

Typical approaches:  
- Classical:
  - Moving‑average crossovers, time‑series momentum (e.g., 12‑1 momentum), breakout rules. [simplechart](https://simplechart.in/quantitative-trading-strategies-2025/)
- Time‑series ML:
  - Sequence models (LSTM / GRU / TCN / Transformer) that forecast:
    - Next‑period return \(r_{t+1}\) (regression).  
    - Direction \(\mathrm{sign}(r_{t+1})\) (classification).  
  - Trading rule: go long if prediction > threshold, short if < −threshold, flat in between.

Model choices (state of the art):  
- Transformer encoder over price/volume sequences (very strong in comparative tests for financial series). [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Temporal Convolutional Networks (TCNs) with causal dilated convolutions (almost as accurate as Transformers, much cheaper). [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- LSTM/GRU as lighter baselines where compute is constrained. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

### 1.2 Mean reversion / statistical arbitrage

Goal: exploit temporary deviations from “fair value.”

Variants:  
- Single‑asset mean reversion:
  - Predict z‑score of price vs. a rolling mean or a learned equilibrium; trade toward reversion. [simplechart](https://simplechart.in/quantitative-trading-strategies-2025/)
- Pairs / basket trading:
  - Learn cointegrated relationships between assets; trade the spread. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)
  - Modern approach: time‑series deep learning to model spread dynamics and forecast spread moves rather than assuming linear relationships. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)

Modern twist:  
- Use TCN / LSTM / Transformer on the spread time series (or on both legs) to forecast:
  - Future spread value.  
  - Probability of spread reverting within a horizon.  
- Use probabilistic models (e.g., DeepAR) to get a full predictive distribution for spreads and derive optimal position sizing from expected value and variance. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

### 1.3 Volatility and risk‑targeting strategies

Goal: forecast volatility and use it for position sizing or volatility trading.

Approaches:  
- Replace (or augment) GARCH with deep time series models:
  - TCN / LSTM to forecast realized volatility over the next \(k\) bars. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
  - DeepAR (probabilistic forecasting) to output a distribution of volatility. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Strategies:
  - Volatility‑scaled trend/mean‑reversion: target constant risk by setting position size \(\propto 1/\hat{\sigma}_{t+1}\).  
  - Vol‑carry or vol‑arbitrage if you can trade options or volatility products (more complex).

### 1.4 Cross‑sectional factor models with time‑series backbones

Goal: rank many assets each day and build long/short portfolios.

Approach:  
- For each asset, run a time‑series model to extract dynamic features (e.g., last hidden state of an LSTM) and/or to produce a forecast of future alpha. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)
- Combine with cross‑sectional inputs (fundamentals, sector, size, value, quality, etc.) in a second‑stage model to produce a score or expected return, then build a portfolio by sorting scores. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

State of the art:  
- Many recent “deep alpha” papers use combinations of:
  - Temporal encoders (TCN/Transformer) over each stock’s history. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)
  - Attention across assets or factors to model cross‑sectional dependencies. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

***

## 2. Deep time series models you can deploy

A 2024 comparative study finds that Transformers and TCNs often outperform LSTM/GRU and DeepAR on financial forecasting benchmarks, with TCNs being the most computationally efficient and Transformers the most accurate but heavy. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

### 2.1 Model zoo (with roles)

For each symbol \(i\) and time \(t\), you can define a sliding window \(X_{t-L+1:t}^{(i)}\) and target \(y_t^{(i)}\). Models below act on these sequences:

- LSTM / GRU:
  - Pros: simple, robust, fast.  
  - Use cases: next‑bar return classification, low‑latency models. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Temporal Convolutional Network (TCN):
  - 1D causal dilated convolutions, residual blocks.  
  - Pros: best speed/accuracy trade‑off; handles long context well. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
  - Use cases: price/volatility forecasting, spread dynamics in pairs trading. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)
- Transformer encoder:
  - Self‑attention over time, possibly with learnable temporal embeddings.  
  - Pros: best accuracy in several financial tests, good at long‑range dependencies. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
  - Cons: heavy; must be regularized aggressively. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- DeepAR / other probabilistic forecasters:
  - Autoregressive RNN with likelihood (Gaussian, Student‑t, etc.) gives full predictive distribution instead of just mean. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
  - Use cases: volatility forecasting, risk‑aware spread forecasting. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

Targets you can train on:  
- Regression:
  - \(y_t = r_{t+1}\) or \(k\)-step cumulative return, volatility, spread change.  
- Classification:
  - \(y_t = \mathbb{1}[r_{t+1} > 0]\) or multi‑class (strong up, flat, strong down).  
- Distributional:
  - Parameters of a distribution (mean, scale) or quantiles for VaR / CVaR.

Losses:  
- MSE or MAE for continuous targets; cross‑entropy / focal loss for classification.  
- Negative log likelihood for probabilistic models (e.g., Gaussian NLL in DeepAR). [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

***

## 3. Reinforcement learning trading agents

Instead of predicting prices and then hand‑coding rules, RL agents directly learn a mapping from state (time‑series features) to actions (buy/sell/hold, position size). [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)

### 3.1 Environment and reward

Core design (consistent across recent frameworks):

- State \(s_t\):  
  - Recent price/volume windows, indicators, model embeddings.  
  - Portfolio state (current positions, cash, unrealized P&L).  
- Actions \(a_t\):  
  - Discrete: {buy, sell, hold} or {−1, 0, +1}. [arxiv](https://arxiv.org/html/2411.07585v1)
  - Continuous: position size or target weight \(\in [-1,1]\) (DDPG / PPO style). [blog.mlq](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)
- Reward \(r_t\):  
  - Wealth change or log‑return of portfolio, often with risk penalties. [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)
  - Examples:
    - \(r_t = \Delta \text{Wealth}_t - \lambda \cdot \text{TransactionCosts}_t\). [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)
    - Sharpe‑like reward: \(r_t = R_t - \lambda \cdot \sigma_t\). [arxiv](https://arxiv.org/html/2411.07585v1)

Recent work (ProfitPulse, 2025; RL frameworks review) shows RL trading strategies outperform buy‑and‑hold and simple technical baselines under backtest, using Q‑learning or Deep Q‑Networks with simulated trading environments. [arxiv](https://arxiv.org/html/2411.07585v1)

### 3.2 Algorithms

- Value‑based:
  - DQN with CNN/TCN/LSTM over price windows, predicting Q(s,a) for discrete actions. [blog.mlq](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)
- Policy‑gradient:
  - PPO, A2C, TRPO acting on state embeddings; often more stable for continuous action spaces. [blog.mlq](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)
- Actor‑critic with continuous control:
  - DDPG, TD3 for directly setting position sizes. [arxiv](https://arxiv.org/html/2411.07585v1)

Hybrid setups:  
- Use a supervised time‑series model to produce features (e.g., hidden states of a Transformer), and let the RL agent operate on these features rather than raw prices, improving sample efficiency. [blog.mlq](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)

Implementation notes for your AI agent:  
- Create a gym‑like environment with methods step, reset, render. [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)
- Expose transaction costs, slippage, and position limits as environment parameters. [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)
- Allow training on historical data first, then live paper‑trading with online fine‑tuning.

***

## 4. Data, features, labels, and evaluation

### 4.1 Data sources and frequency

- Inputs:
  - OHLCV time series; corporate actions (splits/dividends) adjusted prices.  
  - Optional: fundamentals, macro variables, VIX or other volatility indices, and even text‑derived signals (e.g., sentiment scores) for richer models. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)
- Frequencies:
  - Daily / 60‑min for research and swing strategies (easier to get clean data).  
  - Intraday (1–5 min) requires more careful treatment of microstructure effects.

A 10‑year daily dataset over 100 S&P 500 stocks (2014–2023) is a typical scale used to compare deep models, with train/val/test splits like 70/15/15, test being the most recent data. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

### 4.2 Feature engineering

Even with deep models, careful pre‑processing is crucial: [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)

- Raw price transforms:
  - Log returns, high–low / close–open ranges, volume changes, rolling volatility.  
- Technical summaries:
  - Moving averages, RSI, MACD, Bollinger‑band distance, etc. [wemastertrade](https://wemastertrade.com/what-is-quantitative-trading/)
- Cross‑asset features:
  - For pairs/triples, include the spread, z‑score, and relative strength vs. benchmarks. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024)
- Normalization:
  - Per‑asset z‑scaling or robust scaling over a rolling window to stabilize training.

For cross‑sectional models, you can stack features as \([N_{\text{assets}}, T_{\text{lookback}}, d_{\text{features}}]\) and then pool across assets or run attention layers to learn relationships. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

### 4.3 Labeling and training setups

Common labeling schemes:  
- Next‑period return / direction (single‑step supervised). [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Multi‑step horizon: sum of returns across next \(k\) periods.  
- For RL, labels are implicit via rewards; no supervised label needed. [arxiv](https://arxiv.org/html/2411.07585v1)

Backtesting and evaluation:  
- Out‑of‑sample on the latest segment only; avoid information leakage. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Metrics:
  - Annualized return, Sharpe ratio, max drawdown, Calmar, hit ratio, turnover.  
  - For pure forecasting: RMSE, MAE, accuracy, directional accuracy. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
  - Many studies report that Transformers and TCNs provide higher directional accuracy and better robustness across low/medium/high volatility regimes than other deep models. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)

Robustness checks:  
- Evaluate across volatility regimes defined through VIX or realized volatility buckets. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
- Walk‑forward validation: retrain periodically and roll test window.

***

## 5. Implementation blueprint for your AI agent

Below is a concrete way to structure an “AI quant agent” that can implement these strategies.

### 5.1 Modular architecture

Components for your system:

- Data module:
  - Interfaces to fetch and cache OHLCV and optional fundamentals.  
  - Resampling and alignment for different frequencies.  
- Feature/label module:
  - Configurable pipelines for:
    - Trend features.  
    - Mean‑reversion / pairs features.  
    - Volatility features.  
  - Label functions (next‑day return, spread move, etc.).  
- Model zoo:
  - Implement reusable wrappers for:
    - LSTM/GRU/TCN/Transformer forecasters. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
    - DeepAR‑style probabilistic models. [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf)
    - RL agents (DQN, PPO, DDPG) over a gym environment. [blog.mlq](https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/)
- Strategy layer:
  - Deterministic mapping from model outputs to positions:
    - Threshold‑based for supervised models.  
    - Direct actions for RL models.  
- Backtester:
  - Order execution simulation with:
    - Fixed or dynamic transaction costs and slippage.  
    - Portfolio constraints (max leverage, position limits, sector caps).  
- Evaluator:
  - Compute risk/return metrics and generate reports.

### 5.2 Example “menu” of implementable strategies

You can instruct the agent to instantiate any of these as configurations:

| Strategy type | Model | Target | Action rule |
| --- | --- | --- | --- |
| Time‑series momentum | TCN / Transformer  [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf) | Next‑day return | Long if \(p(r_{t+1}>0) > 0.55\), short if < 0.45 |
| Single‑asset mean reversion | LSTM / TCN | Next‑day price deviation from mean | Trade toward mean when deviation exceeds threshold |
| Pairs trading | TCN / Transformer on spread  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405918826000024) | Spread move over horizon | Long spread on predicted mean reversion, exit at z‑score near 0 |
| Volatility targeting | DeepAR / TCN  [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf) | Next‑day volatility | Position size set to target risk, direction from parallel alpha model |
| Cross‑sectional long/short | Transformer + factor MLP  [isrgpublishers](https://isrgpublishers.com/wp-content/uploads/2024/07/ISRGJAHSS5512024.pdf) | Ranking score | Long top decile, short bottom decile, market‑neutral |
| RL directional agent | DQN / PPO  [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf) | Reward = P&L with penalties | Agent directly chooses buy/flat/sell each bar |
| RL portfolio allocator | DDPG / PPO | Reward = portfolio return minus risk | Agent outputs target weights for multiple assets |

Each row can be turned into a JSON “strategy spec” for your AI agent, including: input universes, lookback window, features, model architecture, loss, optimizer, trading frequency, and risk rules.

### 5.3 Practical constraints and safeguards

Recent surveys emphasize that many academic strategies fail in live trading due to overfitting and ignoring frictions. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

Key guards to bake into your agent:

- Strong regularization:
  - Dropout, weight decay, early stopping, and limiting model size, especially for Transformers. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)
- Proper train/val/test splits and walk‑forward validation. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590005625000177)
- Realistic transaction costs and slippage in backtests. [cureusjournals](https://www.cureusjournals.com/articles/994-profitpulse-reinforcement-learning-driven-trading-strategy.pdf)
- Risk controls:
  - Stop‑loss, max position and leverage constraints, daily loss limits, risk parity / volatility scaling.

***

If you tell me your preferred horizon (intraday vs swing/daily) I can next outline one or two fully specified strategies (with concrete hyperparameters, state spaces for RL, and pseudocode) that your AI agent can implement directly.