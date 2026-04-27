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