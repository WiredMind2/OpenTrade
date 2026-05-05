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

const LONG_RUNNING_REQUEST_TIMEOUT_MS = 5 * 60 * 1000


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

export interface UdfSearchSymbol {
  symbol: string
  full_name: string
  description: string
  exchange: string
  ticker: string
  type: string
  currency_code?: string
  pricescale?: number
}

export const searchUdfSymbols = async (
  query: string,
  exchange: string = '',
  limit: number = 20
): Promise<UdfSearchSymbol[]> => {
  const response = await instance.get('/udf/search', {
    params: {
      q: query,
      type: '',
      exchange,
      limit,
    },
  })
  if (response.data?.s === 'error') {
    return []
  }
  return Array.isArray(response.data) ? response.data : []
}

export interface UdfSymbolInfo {
  name: string
  ticker?: string
  description: string
  type: string
  exchange: string
  listed_exchange?: string
  currency_code?: string
  original_currency_code?: string
  pricescale?: number
}

export interface UdfQuote {
  s: 'ok' | 'error'
  n: string
  v: {
    lp?: number
    open_price?: number
    high_price?: number
    low_price?: number
    volume?: number
    ch?: number
    chp?: number
    prev_close_price?: number
  }
}

export const getUdfSymbolInfo = async (symbol: string): Promise<UdfSymbolInfo | null> => {
  const response = await instance.get('/udf/symbols', {
    params: { symbol: symbol.toUpperCase() },
  })
  if (response.data?.s === 'error') return null
  return response.data
}

export const getUdfQuotes = async (symbols: string[]): Promise<UdfQuote[]> => {
  const cleaned = symbols.map((s) => s.trim().toUpperCase()).filter(Boolean)
  if (cleaned.length === 0) return []
  const response = await instance.get('/udf/quotes', {
    params: { symbols: cleaned.join(',') },
  })
  return Array.isArray(response.data) ? response.data : []
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

/** Same-key concurrent calls share one HTTP request (e.g. many equity charts on Backtests). */
const tickerPricesForRangeInflight = new Map<string, Promise<PriceDailyRow[]>>()

function tickerPricesForRangeKey(ticker: string, startDate: string, endDate: string, limit: number): string {
  return `${ticker.toUpperCase()}\0${startDate.slice(0, 10)}\0${endDate.slice(0, 10)}\0${limit}`
}

/** Daily OHLC rows for a ticker in [startDate, endDate] (YYYY-MM-DD), ascending by date in DB but returned newest-first from API. */
export const getTickerPricesForRange = async (
  ticker: string,
  startDate: string,
  endDate: string,
  limit = 1000,
): Promise<PriceDailyRow[]> => {
  const key = tickerPricesForRangeKey(ticker, startDate, endDate, limit)
  const inflight = tickerPricesForRangeInflight.get(key)
  if (inflight) return inflight

  const promise = (async () => {
    try {
      const response = await instance.get(`/data/prices/${ticker.toUpperCase()}`, {
        params: {
          start_date: `${startDate.slice(0, 10)}T00:00:00`,
          end_date: `${endDate.slice(0, 10)}T23:59:59`,
          limit,
        },
      })
      const rows = response.data?.data
      return Array.isArray(rows) ? rows : []
    } finally {
      tickerPricesForRangeInflight.delete(key)
    }
  })()

  tickerPricesForRangeInflight.set(key, promise)
  return promise
}

export const getTickerPriceOnDate = async (
  ticker: string,
  asOfDate?: string,
): Promise<PriceDailyRow | null> => {
  const response = await instance.get(`/data/prices/${ticker.toUpperCase()}`, {
    params: {
      ...(asOfDate ? { end_date: `${asOfDate.slice(0, 10)}T23:59:59` } : {}),
      limit: 1,
    },
  })
  const rows = response.data?.data
  return Array.isArray(rows) && rows.length > 0 ? rows[0] : null
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

export interface SavedModel {
  id: number
  name: string
  strategy_name: string
  ticker: string
  params: Record<string, any>
  params_hash: string
  objective: string
  baseline_metrics?: Record<string, any> | null
  latest_metrics?: Record<string, any> | null
  latest_equity_curve?: unknown[] | null
  degrade_status?: string
  degrade_reason?: string | null
  last_evaluated_at?: string | null
  created_at?: string | null
  updated_at?: string | null
  is_active?: boolean
}

export interface SavedModelEvaluation {
  model_id: number
  name?: string | null
  params?: Record<string, any>
  status: string
  strategy_name: string
  ticker: string
  params_hash: string
  objective: string
  metrics: Record<string, any>
  equity_curve: Array<Record<string, any>>
  degrade_status: string
  degrade_reason?: string | null
  evaluated_at: string
  error?: string | null
}

export type SavedModelSignalAction = 'buy' | 'sell' | 'hold'

export interface SavedModelSignal {
  model_id: number
  name?: string | null
  strategy_name: string
  ticker: string
  params: Record<string, any>
  params_hash: string
  as_of: string
  last_price: number
  action: SavedModelSignalAction
  target_pct: number
  confidence: number
  reason: string
  degrade_status: string
  degrade_reason?: string | null
  error?: string | null
}

export const listSavedModels = async (query?: {
  ticker?: string
  active?: boolean
  strategy?: string
}): Promise<SavedModel[]> => {
  const response = await instance.get('/api/models/saved', { params: query })
  return response.data
}

export const evaluateSavedModelsBatch = async (body: {
  ticker: string
  start_date: string
  end_date: string
  initial_capital?: number
  objective?: string
  top_n?: number
  rank_after_evaluation?: boolean
  max_evaluate?: number
  include_model_ids?: number[]
  exclude_model_ids?: number[]
  drift_thresholds?: Record<string, number>
}): Promise<SavedModelEvaluation[]> => {
  const response = await instance.post('/api/models/saved/evaluate-batch', body)
  return response.data
}

export const signalsSavedModelsBatch = async (body: {
  ticker: string
  objective?: string
  top_n?: number
  include_model_ids?: number[]
  exclude_model_ids?: number[]
  /** YYYY-MM-DD — latest bar on or before this date */
  as_of_date?: string | null
}): Promise<SavedModelSignal[]> => {
  const response = await instance.post('/api/models/saved/signals-batch', body)
  return response.data
}

export const createSavedModel = async (body: {
  name: string
  strategy_name: string
  ticker: string
  params?: Record<string, any>
  objective?: string
  is_active?: boolean
}): Promise<SavedModel> => {
  const response = await instance.post('/api/models/saved', {
    name: body.name,
    strategy_name: body.strategy_name,
    ticker: body.ticker.trim().toUpperCase(),
    params: body.params ?? {},
    objective: body.objective ?? 'balanced',
    is_active: body.is_active ?? true,
  })
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

// Backtests API functions (single endpoint: list with pagination, or one row via backtestId)
export const getBacktests = async (opts?: {
  page?: number
  limit?: number
  /** When set, returns a one-element array (or caller should handle 404 from axios). */
  backtestId?: string
}) => {
  const id = opts?.backtestId?.trim()
  const params: Record<string, string | number> =
    id && id.length > 0
      ? { backtest_id: id }
      : { page: opts?.page ?? 1, limit: opts?.limit ?? 50 }
  const response = await instance.get('/trading/backtest', { params })
  return response.data
}


export const runMonteCarloBacktest = async (data: {
  strategy_name: string
  start_date: string
  end_date: string
  initial_capital: number
  parameters?: Record<string, any>
  monte_carlo: {
    num_simulations: number
    time_horizon: number
    confidence_level: number
  }
}) => {
  const response = await instance.post('/monte-carlo-backtest', data)
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

export type TraderStyle = 'auto' | 'short' | 'swing' | 'long'

export interface TradePlanRequest {
  ticker: string
  style: TraderStyle
  account_size: number
  risk_percent: number
  as_of_date?: string
  signal_action?: string
  signal_confidence?: number
  signal_reason?: string
  strategy_name?: string
  backtest_metrics?: Record<string, number>
}

export interface TradePlanResponse {
  ticker: string
  style: 'short' | 'swing' | 'long'
  trader_type: string
  direction: 'long' | 'short' | 'wait' | 'exit'
  confidence: number
  entry: number | null
  stop_loss: number | null
  take_profit_1: number | null
  take_profit_2: number | null
  trailing_stop: number | null
  invalidation: string
  time_exit: string
  risk_reward: number | null
  risk_amount: number
  position_size: number
  latest_close: number
  price_date: string
  strategy: string
  reasons: string[]
  warnings: string[]
  indicators: Record<string, number | null>
  style_scores: Record<'short' | 'swing' | 'long', number>
}

export const createTradePlan = async (payload: TradePlanRequest): Promise<TradePlanResponse> => {
  const response = await instance.post('/api/trade-plan', payload, {
    timeout: LONG_RUNNING_REQUEST_TIMEOUT_MS,
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
