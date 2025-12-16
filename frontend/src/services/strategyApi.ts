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
  const response = await fetch('/api/strategies');
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function projectStrategy(
  strategyName: string,
  symbol: string,
  startTime: string,
  startPrice: number,
  params: object,
  horizon: number
): Promise<ProjectionPoint[]> {
  const response = await fetch(`/api/strategies/${strategyName}/project`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbol,
      startTime,
      startPrice,
      params,
      horizon,
    }),
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}