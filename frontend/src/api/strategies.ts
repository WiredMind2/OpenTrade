import instance from '../services/api'

const TRAINING_REQUEST_TIMEOUT_MS = 5 * 60 * 1000

export interface StrategyMetadata {
  name: string
  description: string
  type: string
  can_train: boolean
  parameters_schema: any
}

export interface StrategyTrainRequest {
  ticker: string
  start_date: string
  end_date: string
  initial_capital?: number
  objective?: 'sharpe' | 'return' | 'drawdown' | 'balanced'
  max_evals?: number
  /** grid = deterministic search order; random = shuffled subset of the same candidate grid */
  optimizer_mode?: 'grid' | 'random'
  random_seed?: number | null
}

export interface StrategyTrainResponse {
  strategy: string
  ticker: string
  start_date: string
  end_date: string
  objective: string
  optimizer_mode?: string
  experiment_id?: string
  evaluations_run: number
  best_params: Record<string, any>
  best_metrics: {
    total_return: number
    sharpe_ratio: number
    max_drawdown: number
    volatility: number
    total_trades: number
  }
  top_candidates: Array<{
    params: Record<string, any>
    metrics: Record<string, number>
    score: number
  }>
}

interface ProjectionPoint {
  time: string
  price: number
}

export interface StrategyForecastResponse {
  symbol: string
  horizon_days: number
  predicted_return: number
  confidence: number
  predicted_path: ProjectionPoint[]
  metadata: Record<string, any>
}

export interface StrategySignalPoint {
  ticker: string
  target_pct: number
  reason: string
  confidence: number
  timestamp: string
  metadata: Record<string, any>
}

export interface StrategyPreflightIssue {
  code: string
  severity: string
  message: string
  details: Record<string, any>
}

export interface StrategyPreflightResponse {
  ready: boolean
  issues: StrategyPreflightIssue[]
  warnings: StrategyPreflightIssue[]
  suggestions: string[]
  diagnostics: Record<string, any>
}

export const listStrategies = async (): Promise<StrategyMetadata[]> => {
  const response = await instance.get('/api/strategies')
  return response.data
}

export const getStrategies = listStrategies

export const getStrategy = async (name: string): Promise<StrategyMetadata> => {
  const response = await instance.get(`/api/strategies/${name}`)
  return response.data
}

export const preflightStrategy = async (
  strategyName: string,
  payload: { ticker: string; start_date: string; end_date: string }
): Promise<StrategyPreflightResponse> => {
  const response = await instance.post(`/api/strategies/${strategyName}/preflight`, payload)
  return response.data
}

export const trainStrategy = async (
  strategyName: string,
  config: StrategyTrainRequest | Record<string, any>
): Promise<StrategyTrainResponse | Record<string, any>> => {
  const response = await instance.post(`/api/strategies/${strategyName}/train`, config, {
    timeout: TRAINING_REQUEST_TIMEOUT_MS,
    timeoutErrorMessage:
      'Training is still running after 5 minutes. Try fewer max evaluations or a shorter date range.',
  })
  return response.data
}

export async function projectStrategy(
  strategyName: string,
  symbol: string,
  startTime: string,
  startPrice: number,
  params: object,
  horizon: number
): Promise<ProjectionPoint[]> {
  const response = await instance.post(`/api/strategies/${strategyName}/project`, {
    symbol,
    startTime,
    startPrice,
    params,
    horizon,
  })

  const { data } = response
  if (Array.isArray(data)) {
    return data
  }

  // Fallback for backends returning aggregate projection info.
  const projectedReturn = Number(data?.projected_return ?? 0)
  const safeHorizon = Math.max(1, horizon)
  const dailyReturn = projectedReturn / safeHorizon
  const start = new Date(startTime).getTime()
  let price = startPrice
  const points: ProjectionPoint[] = []
  for (let day = 0; day < safeHorizon; day += 1) {
    price = Math.max(0.01, price * (1 + dailyReturn))
    points.push({
      time: new Date(start + day * 24 * 60 * 60 * 1000).toISOString(),
      price,
    })
  }
  return points
}

export async function forecastStrategy(
  strategyName: string,
  symbol: string,
  params: Record<string, any> = {},
  horizon_days: number = 5
): Promise<StrategyForecastResponse> {
  const response = await instance.post(`/api/strategies/${strategyName}/forecast`, {
    symbol,
    params,
    horizon_days,
  })
  return response.data
}

export async function generateStrategySignals(
  strategyName: string,
  symbols: string[],
  params: Record<string, any> = {},
  current_prices: Record<string, number> = {}
): Promise<{ strategy: string; as_of: string; signals: StrategySignalPoint[] }> {
  const response = await instance.post(`/api/strategies/${strategyName}/signals`, {
    symbols,
    params,
    current_prices,
  })
  return response.data
}

export interface MonteCarloRequest {
  strategy_name: string
  ticker: string
  start_date: string
  end_date: string
  initial_capital?: number
  strategy_params?: Record<string, any>
  num_simulations?: number
  time_horizon_days?: number
}

export interface MonteCarloResult {
  simulation_id: string
  strategy_name: string
  ticker: string
  num_simulations: number
  time_horizon_days: number
  aggregated_results: {
    mean_final_value: number
    std_final_value: number
    mean_total_return: number
    std_total_return: number
    confidence_lower_return: number
    confidence_upper_return: number
    worst_case_return: number
    best_case_return: number
    probability_positive_return: number
  }
  risk_metrics: {
    value_at_risk_95: number
    expected_shortfall_95: number
    volatility: number
    probability_positive_return: number
  }
  created_at: string
}

export async function runMonteCarloSimulation(request: MonteCarloRequest): Promise<MonteCarloResult> {
  // Mock implementation for testing - replace with actual API call when backend is running
  console.log('Monte Carlo request:', request)

  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 2000))

  // Generate mock Monte Carlo results
  const mockResults = {
    simulation_id: `mc_mock_${Date.now()}`,
    strategy_name: request.strategy_name,
    ticker: request.ticker,
    num_simulations: request.num_simulations || 1000,
    time_horizon_days: request.time_horizon_days || 252,
    aggregated_results: {
      mean_final_value: (request.initial_capital || 100000) * 1.08,
      std_final_value: (request.initial_capital || 100000) * 0.15,
      mean_total_return: 0.08,
      std_total_return: 0.15,
      confidence_lower_return: -0.05,
      confidence_upper_return: 0.21,
      worst_case_return: -0.12,
      best_case_return: 0.35,
      probability_positive_return: 0.68
    },
    risk_metrics: {
      value_at_risk_95: -0.08,
      expected_shortfall_95: -0.12,
      volatility: 0.15,
      probability_positive_return: 0.68
    },
    created_at: new Date().toISOString()
  }

  return mockResults
}
