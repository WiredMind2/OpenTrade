import instance from '../services/api'

export interface StrategyMetadata {
  name: string
  description: string
  type: string
  can_train: boolean
  parameters_schema: any
  model_info?: any
}

export interface StrategyTrainRequest {
  ticker: string
  start_date: string
  end_date: string
  initial_capital?: number
  objective?: 'sharpe' | 'return' | 'drawdown' | 'balanced'
  max_evals?: number
}

export interface StrategyTrainResponse {
  strategy: string
  ticker: string
  start_date: string
  end_date: string
  objective: string
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
  const response = await instance.post(`/api/strategies/${strategyName}/train`, config)
  return response.data
}