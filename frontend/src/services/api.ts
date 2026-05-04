import axios from 'axios'
import type {
  StrategyAnalyticsFilters,
  StrategyVariantSummary,
  StrategyVariantTimeseriesResponse,
  StrategyDistributionResponse,
  TickerStrategyLeaderboardResponse,
} from '../types'

// Jest (CommonJS) cannot parse `import.meta`. Keep baseURL flexible without relying on it.
//
// Defaulting to `window.location.origin` makes chart UDF calls deterministic in more setups:
// - Vite dev server + proxy (same-origin requests still proxy)
// - Frontend served by backend (same origin is the backend)
// - Static hosting where the backend is co-hosted behind the same origin
const API_BASE =
  (typeof window !== 'undefined' && (window as any).__API_BASE__) ||
  (typeof process !== 'undefined' && (process as any).env?.VITE_API_BASE) ||
  (typeof window !== 'undefined' && window.location?.origin) ||
  ''

const instance = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})


// Add response interceptor for logging
instance.interceptors.response.use(
  (response) => {
    console.log(`API ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`)
    return response
  },
  (error) => {
    console.error(`API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} - ${error.response?.status || 'Network Error'}`, error)
    return Promise.reject(error)
  }
)

export default instance

// Script execution API functions
export const executeScript = async (scriptName: string, parameters: Record<string, any> = {}) => {
  const response = await instance.post('/scripts/execute', {
    script_name: scriptName,
    parameters
  })
  return response.data
}

export const getScriptStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/status/${executionId}`)
  return response.data
}

export const listScriptExecutions = async () => {
  const response = await instance.get('/scripts/executions')
  return response.data
}

export const runPipeline = async (steps?: string[]) => {
  const data = steps ? { steps } : undefined
  const response = await instance.post('/scripts/pipeline/run', data)
  return response.data
}

export const getPipelineStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/pipeline/status/${executionId}`)
  return response.data
}

export const runBatchStrategyTraining = async (data: {
  ticker: string
  start_date: string
  end_date: string
  initial_capital?: number
  objective?: string
  max_evals?: number
  optimizer_mode?: string
  random_seed?: number | null
  pair_ticker?: string | null
  universe_limit?: number
  stop_on_error?: boolean
}) => {
  const response = await instance.post('/scripts/batch-strategy-training', data)
  return response.data
}

// Predictions API functions
export const getPredictions = async () => {
  const response = await instance.get('/predictions/recent')
  return response.data
}

export const getTickers = async () => {
  const response = await instance.get('/predictions/tickers')
  return response.data
}

export interface UdfSearchSymbol {
  symbol: string
  full_name: string
  description: string
  exchange: string
  ticker: string
  type: string
}

export const searchUdfSymbols = async (
  query: string,
  exchange: string = '',
  limit: number = 20
): Promise<UdfSearchSymbol[]> => {
  const response = await instance.get('/udf/search', {
    params: {
      q: query,
      type: 'stock',
      exchange,
      limit,
    },
  })
  if (response.data?.s === 'error') {
    return []
  }
  return Array.isArray(response.data) ? response.data : []
}

export interface LatestPriceAnchor {
  latestPrice: number
  latestTime: number
}

export type PriceDailyRow = {
  date: string
  open?: number | null
  high?: number | null
  low?: number | null
  close?: number | null
  adjusted_close?: number | null
  volume?: number | null
}

/** Daily OHLC rows for a ticker in [startDate, endDate] (YYYY-MM-DD), ascending by date in DB but returned newest-first from API. */
export const getTickerPricesForRange = async (
  ticker: string,
  startDate: string,
  endDate: string,
  limit = 1000,
): Promise<PriceDailyRow[]> => {
  const response = await instance.get(`/data/prices/${ticker.toUpperCase()}`, {
    params: {
      start_date: `${startDate.slice(0, 10)}T00:00:00`,
      end_date: `${endDate.slice(0, 10)}T23:59:59`,
      limit,
    },
  })
  const rows = response.data?.data
  return Array.isArray(rows) ? rows : []
}

export interface PriceHistoryRow {
  date: string
  open: number
  high: number
  low: number
  close: number
  adjusted_close: number
  volume: number
}

export const getPriceHistory = async (ticker: string, limit: number = 2): Promise<PriceHistoryRow[]> => {
  const response = await instance.get(`/data/prices/${ticker.toUpperCase()}`, {
    params: { limit },
  })
  return Array.isArray(response.data?.data) ? response.data.data : []
}

export const getLatestPriceAnchor = async (ticker: string): Promise<LatestPriceAnchor | null> => {
  const response = await instance.get(`/data/prices/${ticker.toUpperCase()}`, {
    params: { limit: 1 },
  })
  const rows = response.data?.data
  if (!Array.isArray(rows) || rows.length === 0) {
    return null
  }

  const latest = rows[0]
  if (typeof latest?.close !== 'number' || typeof latest?.date !== 'string') {
    return null
  }

  const timestampMs = Date.parse(`${latest.date}T00:00:00Z`)
  if (!Number.isFinite(timestampMs)) {
    return null
  }

  return {
    latestPrice: latest.close,
    latestTime: Math.floor(timestampMs / 1000),
  }
}

export type CreatePredictionOptions = {
  as_of?: string
  persist_prediction?: boolean
  include_forward_actuals?: boolean
}

export const createPrediction = async (
  ticker: string,
  horizon: string,
  strategyName?: string,
  strategyParams?: Record<string, any>,
  options?: CreatePredictionOptions
) => {
  const response = await instance.post('/predict', {
    ticker: ticker.toUpperCase(),
    horizon,
    strategy_name: strategyName,
    strategy_params: strategyParams,
    ...(options?.as_of != null && options.as_of !== ''
      ? {
          as_of: options.as_of,
          persist_prediction: options.persist_prediction,
          include_forward_actuals: options.include_forward_actuals ?? true,
        }
      : {}),
  })
  return response.data
}

export interface PredictionProjectionRequest {
  symbol: string
  anchor_time: string
  anchor_price: number
  horizon_days: number
  strategy_names?: string[]
  params_by_strategy?: Record<string, Record<string, any>>
}

export const getPredictionProjections = async (payload: PredictionProjectionRequest) => {
  const response = await instance.post('/api/predictions/projections', payload)
  return response.data
}


// Portfolio API functions
export const getPortfolio = async () => {
  const response = await instance.get('/portfolio/current')
  return response.data
}

// Health API functions
export const getHealth = async () => {
  const response = await instance.get('/health')
  return response.data
}

// Backtests API functions
export const getBacktests = async () => {
  const response = await instance.get('/trading/backtest')
  return response.data
}

export const runBacktest = async (data: {
  strategy_name: string
  start_date: string
  end_date: string
  initial_capital: number
  parameters?: Record<string, any>
}) => {
  const response = await instance.post('/backtest', data)
  return response.data
}

export const getBacktest = async (backtestId: string) => {
  const response = await instance.get(`/backtest/${encodeURIComponent(backtestId)}`)
  return response.data
}

// MA Predictions API functions
export const generateMAPredictions = async (data: {
  start_date: string
  end_date: string
  short_ma_range?: number[]
  medium_ma_range?: number[]
  long_ma_range?: number[]
  skip_optimization?: boolean
  fixed_short?: number
  fixed_medium?: number
  fixed_long?: number
}) => {
  const response = await instance.post('/scripts/generate-ma-predictions', data)
  return response.data
}

export const getMAPredictionStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/generate-ma-predictions/status/${executionId}`)
  return response.data
}

// Strategy analytics API functions
export const getStrategyAnalyticsFilters = async (): Promise<StrategyAnalyticsFilters> => {
  const response = await instance.get('/api/strategy-analytics/filters')
  return response.data
}

export const getTickerStrategyLeaderboard = async (query: {
  objective: string
  top_n: number
  ticker?: string
}): Promise<TickerStrategyLeaderboardResponse> => {
  const response = await instance.get('/api/strategy-analytics/tickers/leaderboard', { params: query })
  return response.data
}

export const getStrategyVariantSummary = async (query: {
  strategy: string
  objective?: string
  top_n?: number
  ticker?: string
}): Promise<StrategyVariantSummary> => {
  const response = await instance.get('/api/strategy-analytics/variants/summary', { params: query })
  return response.data
}

export const getStrategyVariantTimeseries = async (query: {
  strategy: string
  params_hashes: string
  benchmark_ticker?: string
  preset?: string
  granularity?: 'daily' | 'weekly' | 'monthly'
  rolling_window?: number
  objective?: string
  ticker?: string
}): Promise<StrategyVariantTimeseriesResponse> => {
  const response = await instance.get('/api/strategy-analytics/variants/timeseries', { params: query })
  return response.data
}

export const getStrategyVariantDistribution = async (
  strategy: string,
  params_hash: string,
  objective?: string,
  ticker?: string
): Promise<StrategyDistributionResponse> => {
  const response = await instance.get(`/api/strategy-analytics/variants/distributions/${strategy}`, {
    params: { params_hash, objective: objective ?? 'balanced', ticker },
  })
  return response.data
}

// Mock data generators (for when API is unavailable)
export const generateMockChartData = (totalReturn: number, points: number = 20) => {
  const data = []
  for (let i = 0; i <= points; i++) {
    data.push({
      day: i,
      value: 100000 * (1 + (totalReturn / 100) * (i / points))
    })
  }
  return data
}
