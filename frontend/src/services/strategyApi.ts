import api from './api'

interface StrategyMetadata {
  name: string;
  description: string;
  type: string;
  parameters_schema: Record<string, any>;
  can_train: boolean;
}

interface ProjectionPoint {
  time: string;
  price: number;
}

export interface StrategyForecastResponse {
  symbol: string;
  horizon_days: number;
  predicted_return: number;
  confidence: number;
  predicted_path: ProjectionPoint[];
  metadata: Record<string, any>;
}

export interface StrategySignalPoint {
  ticker: string;
  target_pct: number;
  reason: string;
  confidence: number;
  timestamp: string;
  metadata: Record<string, any>;
}

export async function getStrategies(): Promise<StrategyMetadata[]> {
  const response = await api.get('/api/strategies');
  return response.data;
}

export async function projectStrategy(
  strategyName: string,
  symbol: string,
  startTime: string,
  startPrice: number,
  params: object,
  horizon: number
): Promise<ProjectionPoint[]> {
  const response = await api.post(`/api/strategies/${strategyName}/project`, {
    symbol,
    startTime,
    startPrice,
    params,
    horizon,
  });

  const { data } = response;
  if (Array.isArray(data)) {
    return data;
  }

  // Backend strategies endpoint can return summary metrics. Build a deterministic
  // projection path so the interactive chart flow still renders.
  const projectedReturn = Number(data?.projected_return ?? 0);
  const safeHorizon = Math.max(1, horizon);
  const dailyReturn = projectedReturn / safeHorizon;
  const start = new Date(startTime).getTime();
  let price = startPrice;
  const points: ProjectionPoint[] = [];
  for (let day = 0; day < safeHorizon; day += 1) {
    price = Math.max(0.01, price * (1 + dailyReturn));
    points.push({
      time: new Date(start + day * 24 * 60 * 60 * 1000).toISOString(),
      price,
    });
  }
  return points;
}

export async function forecastStrategy(
  strategyName: string,
  symbol: string,
  params: Record<string, any> = {},
  horizon_days: number = 5
): Promise<StrategyForecastResponse> {
  const response = await api.post(`/api/strategies/${strategyName}/forecast`, {
    symbol,
    params,
    horizon_days,
  });
  return response.data;
}

export async function generateStrategySignals(
  strategyName: string,
  symbols: string[],
  params: Record<string, any> = {},
  current_prices: Record<string, number> = {}
): Promise<{ strategy: string; as_of: string; signals: StrategySignalPoint[] }> {
  const response = await api.post(`/api/strategies/${strategyName}/signals`, {
    symbols,
    params,
    current_prices,
  });
  return response.data;
}