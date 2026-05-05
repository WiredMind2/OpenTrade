import React, { useEffect, useRef, useState } from 'react'
import { getBacktests } from '../services/api'
import {
  preflightStrategy,
  trainStrategy,
  type StrategyPreflightResponse,
  type StrategyTrainResponse,
} from '../api/strategies'
import websocketService from '../services/websocket'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { BacktestResult } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import { Skeleton } from '../components/ui/skeleton'
import {
  BarChart3,
  Calendar,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Loader2,
  X,
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import StrategySelector from '../components/StrategySelector'
import TickerSearch from '../components/TickerSearch'
import BacktestEquityCompareChart from '../components/BacktestEquityCompareChart'
import { buildBacktestEquitySeries } from '../utils/backtestChart'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'
import { runMonteCarloSimulation, MonteCarloResult } from '../api/strategies'

type BacktestListItem = BacktestResult & {
  id?: string | number
  ticker?: string | null
  status?: string
  error?: string
  chart_data?: Array<{ day: number; value: number }>
  execution_engine?: string
  signals_emitted?: number
  order_intents?: number
  order_fills?: number
}

type ServerWaitPhase = 'idle' | 'preflight' | 'training'

function backtestCorrelationKey(b: BacktestListItem): string {
  const k = b.metrics?.backtest_id ?? b.id
  return k !== undefined && k !== null && String(k) !== '' ? String(k) : ''
}

/**
 * Apply the first page from the API: server order wins for rows present in `incoming`;
 * keep extra client rows (e.g. "load more") whose key is not on that page.
 */
function reconcileBacktestsFirstPage(incoming: BacktestListItem[], previous: BacktestListItem[]): BacktestListItem[] {
  const incomingKeys = new Set(
    incoming.map(backtestCorrelationKey).filter((k) => k !== ''),
  )
  const leftover = previous.filter((b) => {
    const key = backtestCorrelationKey(b)
    if (!key) return true
    return !incomingKeys.has(key)
  })
  return [...incoming, ...leftover]
}

const BACKTEST_RESULTS_PAGE_SIZE = 10

function isBacktestInFlight(b: BacktestListItem): boolean {
  const st = (b.status ?? b.metrics?.status) as string | undefined
  if (st === 'running') return true
  if (st === 'failed' || st === 'completed') return false
  const ph = b.metrics?.phase as string | undefined
  if (typeof ph === 'string' && ph !== 'completed' && ph !== 'failed') return true
  return false
}

/** Stable fingerprint so we can skip setState (and chart remounts) when the API returns the same logical rows. */
function backtestRowStableSig(b: BacktestListItem): string {
  const k = backtestCorrelationKey(b)
  const st = (b.status ?? b.metrics?.status) ?? ''
  const ph = (b.metrics?.phase as string | undefined) ?? ''
  const tr = typeof b.total_return === 'number' && Number.isFinite(b.total_return) ? b.total_return : ''
  const fv = typeof b.final_value === 'number' && Number.isFinite(b.final_value) ? b.final_value : ''
  const eq = Array.isArray(b.equity_curve) ? b.equity_curve : []
  const n = eq.length
  let lastV = ''
  if (n > 0) {
    const tail = eq[n - 1]
    if (tail && typeof tail === 'object' && 'value' in tail) {
      lastV = String((tail as Record<string, unknown>).value)
    }
  }
  const rid = b.id !== undefined && b.id !== null ? String(b.id) : ''
  // Omit wall-clock `timestamp` — it can differ between WS payloads and GET rows for the same run.
  return [k, rid, st, ph, tr, fv, n, lastV].join('\u001f')
}

function backtestsListStableSig(list: BacktestListItem[]): string {
  return list.map(backtestRowStableSig).join('\u0002')
}

function appendUniqueByBacktestKey(
  previous: BacktestListItem[],
  nextSlice: BacktestListItem[],
): BacktestListItem[] {
  const keys = new Set(previous.map(backtestCorrelationKey).filter((k) => k !== ''))
  const extra = nextSlice.filter((b) => {
    const k = backtestCorrelationKey(b)
    if (!k) return true
    if (keys.has(k)) return false
    keys.add(k)
    return true
  })
  return [...previous, ...extra]
}

function serverWaitPhaseLabel(phase: ServerWaitPhase): string {
  switch (phase) {
    case 'preflight':
      return 'Contacting the server: validating data for your ticker and dates…'
    case 'training':
      return 'Running parameter optimization on the server (this can take several minutes)…'
    default:
      return ''
  }
}

export default function Backtests() {
  const [detailBacktest, setDetailBacktest] = useState<BacktestListItem | null>(null)
  const [backtests, setBacktests] = useState<BacktestListItem[]>([])
  const [strategy, setStrategy] = useState('')
  const [startDate, setStartDate] = useState(() => `${new Date().getFullYear() - 1}-01-01`)
  const [endDate, setEndDate] = useState(() => `${new Date().getFullYear() - 1}-12-31`)
  const [training, setTraining] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [ticker, setTicker] = useState(() => getStoredTicker())
  const [pairTicker, setPairTicker] = useState('')
  const [trainObjective, setTrainObjective] = useState<'sharpe' | 'return' | 'drawdown' | 'balanced'>('balanced')
  const [maxEvals, setMaxEvals] = useState(8)
  const [optimizerMode, setOptimizerMode] = useState<'grid' | 'random'>('grid')
  const [randomSeed, setRandomSeed] = useState<string>('')
  const [trainedParams, setTrainedParams] = useState<Record<string, any> | null>(null)
  const [trainResult, setTrainResult] = useState<StrategyTrainResponse | null>(null)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [preflight, setPreflight] = useState<StrategyPreflightResponse | null>(null)
  const [serverWaitPhase, setServerWaitPhase] = useState<ServerWaitPhase>('idle')
  const [monteCarloRunning, setMonteCarloRunning] = useState(false)
  const [monteCarloResult, setMonteCarloResult] = useState<MonteCarloResult | null>(null)
  const [monteCarloError, setMonteCarloError] = useState<string | null>(null)
  const [numSimulations, setNumSimulations] = useState(1000)
  const [timeHorizonDays, setTimeHorizonDays] = useState(252)
  const [resultsHasMore, setResultsHasMore] = useState(false)
  const [resultsNextPage, setResultsNextPage] = useState(2)
  const [loadingMoreResults, setLoadingMoreResults] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null)
  const backtestsRef = useRef<BacktestListItem[]>([])
  backtestsRef.current = backtests

  useEffect(() => {
    if (!detailBacktest) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDetailBacktest(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [detailBacktest])

  const fetchBacktests = async () => {
    setLoading(true)
    setError(null)
    setLoadMoreError(null)
    try {
      const data = await getBacktests({ page: 1, limit: BACKTEST_RESULTS_PAGE_SIZE })
      const rows = data as BacktestListItem[]
      setBacktests((prev) => {
        const next = reconcileBacktestsFirstPage(rows, prev)
        if (backtestsListStableSig(next) === backtestsListStableSig(prev)) return prev
        return next
      })
      setResultsHasMore(rows.length >= BACKTEST_RESULTS_PAGE_SIZE)
      setResultsNextPage(2)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch backtests')
    } finally {
      setLoading(false)
    }
  }

  const loadMoreBacktestResults = async () => {
    if (loadingMoreResults || !resultsHasMore) return
    setLoadingMoreResults(true)
    setLoadMoreError(null)
    try {
      const data = await getBacktests({ page: resultsNextPage, limit: BACKTEST_RESULTS_PAGE_SIZE })
      const rows = data as BacktestListItem[]
      setBacktests((prev) => appendUniqueByBacktestKey(prev, rows))
      setResultsHasMore(rows.length >= BACKTEST_RESULTS_PAGE_SIZE)
      setResultsNextPage((p) => p + 1)
    } catch (e: any) {
      setLoadMoreError(e.message || 'Failed to load more backtests')
    } finally {
      setLoadingMoreResults(false)
    }
  }

  useEffect(() => {
    fetchBacktests()
  }, [])

  const hasValidStrategySelection = strategy.trim().length > 0

  /** Poll only while a run is in-flight; skip setState when data unchanged (avoids chart re-fetch / animation loops). */
  useEffect(() => {
    const POLL_MS = 4000
    let inFlight = false

    const refreshQuiet = () => {
      if (document.visibilityState !== 'visible') return
      if (!backtestsRef.current.some(isBacktestInFlight)) return
      if (inFlight) return
      inFlight = true
      getBacktests({ page: 1, limit: BACKTEST_RESULTS_PAGE_SIZE })
        .then((data) => {
          const rows = data as BacktestListItem[]
          setBacktests((prev) => {
            const next = reconcileBacktestsFirstPage(rows, prev)
            if (backtestsListStableSig(next) === backtestsListStableSig(prev)) return prev
            return next
          })
          setLoading(false)
        })
        .catch(() => {
          /* ignore */
        })
        .finally(() => {
          inFlight = false
        })
    }

    const id = window.setInterval(refreshQuiet, POLL_MS)
    const onVis = () => {
      if (document.visibilityState === 'visible' && backtestsRef.current.some(isBacktestInFlight)) {
        refreshQuiet()
      }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      window.clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  useEffect(() => {
    const parseMetrics = (raw: unknown): Record<string, unknown> | undefined => {
      if (!raw) return undefined
      if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) return raw as Record<string, unknown>
      if (typeof raw === 'string') {
        try {
          const p = JSON.parse(raw) as unknown
          if (p && typeof p === 'object' && !Array.isArray(p)) return p as Record<string, unknown>
        } catch {
          return undefined
        }
      }
      return undefined
    }

    const handleBacktestStatus = (message: any) => {
      const payload = message?.data
      if (!payload || typeof payload !== 'object') return

      const metricsObj = parseMetrics((payload as BacktestListItem).metrics) ?? {}
      const backtestResult: BacktestListItem = {
        ...(payload as BacktestListItem),
        metrics: metricsObj as BacktestListItem['metrics'],
      }

      let chartData: BacktestListItem['chart_data']
      try {
        chartData = buildBacktestEquitySeries(backtestResult).map((p) => ({
          day: p.day,
          value: p.value,
          ...(p.dateKey ? { date: p.dateKey } : {}),
        }))
      } catch {
        chartData = []
      }

      const normalizedResult: BacktestListItem = {
        ...backtestResult,
        chart_data: chartData,
        status: (metricsObj.status as string | undefined) ?? backtestResult.status,
        error: (metricsObj.error as string | undefined) ?? backtestResult.error,
      }

      const backtestId = metricsObj.backtest_id

      setLoading(false)
      setError(null)

      setBacktests((prev) => {
        const existingIndex = prev.findIndex((b) => {
          const existingId = b.metrics?.backtest_id ?? b.id
          return (
            backtestId != null &&
            backtestId !== '' &&
            existingId != null &&
            String(existingId) === String(backtestId)
          )
        })

        let next: BacktestListItem[]
        if (existingIndex >= 0) {
          const updated = [...prev]
          updated[existingIndex] = normalizedResult
          next = updated
        } else {
          next = [normalizedResult, ...prev]
        }
        if (backtestsListStableSig(next) === backtestsListStableSig(prev)) return prev
        return next
      })
    }

    // Register the listener
    // Cleanup on unmount
    return websocketService.registerListener('backtest_status', handleBacktestStatus)
  }, [])

  const runTraining = async () => {
    if (!hasValidStrategySelection) {
      alert('Please select a strategy')
      return
    }
    if (!ticker.trim()) {
      alert('Please provide a ticker for training')
      return
    }
    const pt = pairTicker.trim().toUpperCase() || undefined
    if (strategy === 'pairs_trading' && !pt) {
      alert('pairs_trading requires a second ticker — fill in the "Second ticker" field next to Ticker')
      return
    }
    setTraining(true)
    setTrainError(null)
    setServerWaitPhase('preflight')
    try {
      const normalizedTicker = rememberTicker(ticker)
      const check = await preflightStrategy(strategy, {
        ticker: normalizedTicker,
        start_date: startDate,
        end_date: endDate,
        ...(pt ? { pair_ticker: pt } : {}),
      })
      setPreflight(check)
      if (!check.ready) {
        setTrainError(check.issues[0]?.message || 'Preflight failed')
        return
      }
      setServerWaitPhase('training')
      const seedNum = randomSeed.trim() === '' ? undefined : Number(randomSeed)
      const response = await trainStrategy(strategy, {
        ticker: normalizedTicker,
        start_date: startDate,
        end_date: endDate,
        initial_capital: 100000,
        objective: trainObjective,
        max_evals: maxEvals,
        optimizer_mode: optimizerMode,
        ...(Number.isFinite(seedNum as number) ? { random_seed: seedNum } : {}),
        ...(pt ? { pair_ticker: pt } : {}),
      })
      if (
        response &&
        typeof response === 'object' &&
        'best_params' in response &&
        'best_metrics' in response &&
        'evaluations_run' in response
      ) {
        const typed = response as StrategyTrainResponse
        setTrainResult(typed)
        setTrainedParams(typed.best_params)
      } else {
        setTrainResult(null)
        setTrainedParams(null)
        setTrainError('Strategy returned non-optimization training response.')
      }
    } catch (e: any) {
      setTrainError(e.message || 'Failed to train strategy parameters')
    } finally {
      setTraining(false)
      setServerWaitPhase('idle')
      fetchBacktests()
    }
  }

  const runMonteCarlo = async () => {
    if (!hasValidStrategySelection) {
      alert('Please select a strategy first')
      return
    }
    if (!ticker.trim()) {
      alert('Please provide a ticker for Monte Carlo simulation')
      return
    }
    if (!trainedParams) {
      alert('Please train strategy parameters before running Monte Carlo simulation')
      return
    }

    setMonteCarloRunning(true)
    setMonteCarloError(null)
    setMonteCarloResult(null)

    try {
      const result = await runMonteCarloSimulation({
        strategy_name: strategy,
        ticker: ticker,
        start_date: startDate,
        end_date: endDate,
        initial_capital: 100000,
        strategy_params: trainedParams,
        num_simulations: numSimulations,
        time_horizon_days: timeHorizonDays,
      })
      setMonteCarloResult(result)
    } catch (e: any) {
      setMonteCarloError(e.message || 'Failed to run Monte Carlo simulation')
    } finally {
      setMonteCarloRunning(false)
    }
  }

  const getReturnColor = (returnVal: number) => {
    if (returnVal > 0) return 'text-success'
    if (returnVal < 0) return 'text-destructive'
    return 'text-muted-foreground'
  }

  const getReturnBadge = (returnVal: number) => {
    if (returnVal > 10) return { variant: 'success' as const, label: 'Excellent' }
    if (returnVal > 0) return { variant: 'default' as const, label: 'Positive' }
    if (returnVal > -10) return { variant: 'warning' as const, label: 'Negative' }
    return { variant: 'destructive' as const, label: 'Poor' }
  }

  return (
    <>
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Backtests</h2>
        <p className="text-muted-foreground">
          Test and analyze your trading strategies
        </p>
      </div>

      {/* Backtest Form */}
      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Train strategy parameters
          </CardTitle>
          <CardDescription>
            Optimize parameters for your ticker and date range. Run historical backtests from the Predictions tab
            (chart) to validate signals against past prices.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Strategy Name</label>
              <StrategySelector
                onStrategyChange={(selectedStrategy) => {
                  setStrategy(selectedStrategy)
                  setTrainResult(null)
                  setTrainedParams(null)
                  setTrainError(null)
                  setPreflight(null)
                }}
              />
            </div>

            <Separator />

            <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4 shrink-0" />
                  Start Date
                </label>
                <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4 shrink-0" />
                  End Date
                </label>
                <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Ticker</label>
                <TickerSearch
                  value={ticker}
                  onChange={(t) => setTicker(rememberTicker(t))}
                  placeholder="Search a ticker"
                />
              </div>

              {strategy === 'pairs_trading' && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    Second ticker <span className="text-destructive">*</span>
                  </label>
                  <TickerSearch
                    value={pairTicker}
                    onChange={(t) => setPairTicker(t.toUpperCase())}
                    placeholder="e.g. MSFT"
                  />
                </div>
              )}

              <div className="space-y-2">
                <label className="text-sm font-medium">Objective</label>
                <select
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainObjective}
                  onChange={(e) => setTrainObjective(e.target.value as typeof trainObjective)}
                >
                  <option value="balanced">Balanced</option>
                  <option value="sharpe">Sharpe</option>
                  <option value="return">Return</option>
                  <option value="drawdown">Drawdown</option>
                </select>
              </div>

              <div className="space-y-2 sm:col-span-2 lg:col-span-1">
                <label className="text-sm font-medium">Max Evaluations</label>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  className="h-10"
                  value={maxEvals}
                  onChange={(e) => setMaxEvals(Math.max(1, Math.min(50, Number(e.target.value || 8))))}
                />
                <p className="text-xs leading-snug text-muted-foreground">
                  Higher values improve search quality but take longer.
                </p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Optimizer mode</label>
                <select
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  value={optimizerMode}
                  onChange={(e) => setOptimizerMode(e.target.value as 'grid' | 'random')}
                >
                  <option value="grid">Grid (deterministic)</option>
                  <option value="random">Random (shuffled grid)</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Random seed (optional)</label>
                <Input
                  type="number"
                  className="h-10"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(e.target.value)}
                  placeholder="e.g. 42"
                  disabled={optimizerMode !== 'random'}
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <Button
                onClick={runTraining}
                disabled={training || !hasValidStrategySelection}
                variant="secondary"
                className="w-full md:w-auto"
                size="lg"
              >
                {training ? (
                  <>
                    <Activity className="mr-2 h-4 w-4 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Target className="mr-2 h-4 w-4" />
                    Train Strategy
                  </>
                )}
              </Button>
            </div>
            {serverWaitPhase !== 'idle' && (
              <div
                className="rounded-lg border border-primary/25 bg-primary/5 p-3 space-y-2"
                role="status"
                aria-live="polite"
                aria-busy="true"
              >
                <div className="flex items-center gap-2 text-sm text-foreground">
                  <Activity className="h-4 w-4 shrink-0 animate-spin text-primary" aria-hidden />
                  <span>{serverWaitPhaseLabel(serverWaitPhase)}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-700"
                    style={{ width: serverWaitPhase === 'preflight' ? '25%' : '70%' }}
                  />
                </div>
              </div>
            )}
            {trainError && <p className="text-sm text-destructive">{trainError}</p>}
            {preflight && !preflight.ready && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm space-y-1">
                <p className="font-medium text-destructive">Preflight blocked execution</p>
                {preflight.issues.map((issue, idx) => (
                  <p key={`${issue.code}-${idx}`}>- {issue.message}</p>
                ))}
                {preflight.suggestions.length > 0 && (
                  <p className="text-muted-foreground">Suggestion: {preflight.suggestions[0]}</p>
                )}
              </div>
            )}
            {preflight && preflight.ready && preflight.warnings.length > 0 && (
              <div className="rounded-lg border border-amber-400/40 bg-amber-100/20 p-3 text-sm space-y-1">
                <p className="font-medium">Preflight warnings</p>
                {preflight.warnings.map((warning, idx) => (
                  <p key={`${warning.code}-${idx}`}>- {warning.message}</p>
                ))}
              </div>
            )}
            {trainResult && (
              <div className="rounded-lg border bg-muted/40 p-3 text-sm space-y-1">
                <p className="font-medium">Training result ({trainResult.strategy})</p>
                <p>
                  Best params: <code>{JSON.stringify(trainResult.best_params)}</code>
                </p>
                <p>
                  Metrics: Sharpe {trainResult.best_metrics.sharpe_ratio.toFixed(3)} | Return {(trainResult.best_metrics.total_return * 100).toFixed(2)}% | Max DD {(trainResult.best_metrics.max_drawdown * 100).toFixed(2)}% | Trades {trainResult.best_metrics.total_trades}
                </p>
                <p>Evaluations: {trainResult.evaluations_run}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Monte Carlo Risk Assessment */}
      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Monte Carlo Risk Assessment
          </CardTitle>
          <CardDescription>
            Run Monte Carlo simulations to assess risk and potential outcomes for the trained strategy parameters.
            This requires strategy training to be completed first.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Number of Simulations</label>
                <Input
                  type="number"
                  min={100}
                  max={5000}
                  value={numSimulations}
                  onChange={(e) => setNumSimulations(Math.max(100, Math.min(5000, Number(e.target.value || 1000))))}
                />
                <p className="text-xs text-muted-foreground">
                  Higher values provide more accurate results but take longer.
                </p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Time Horizon (Days)</label>
                <Input
                  type="number"
                  min={30}
                  max={1000}
                  value={timeHorizonDays}
                  onChange={(e) => setTimeHorizonDays(Math.max(30, Math.min(1000, Number(e.target.value || 252))))}
                />
                <p className="text-xs text-muted-foreground">
                  Trading days to simulate (252 ≈ 1 year).
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <Button
                onClick={runMonteCarlo}
                disabled={monteCarloRunning || !hasValidStrategySelection || !trainedParams}
                variant="outline"
                className="w-full md:w-auto"
                size="lg"
              >
                {monteCarloRunning ? (
                  <>
                    <Activity className="mr-2 h-4 w-4 animate-spin" />
                    Running Simulations...
                  </>
                ) : (
                  <>
                    <Activity className="mr-2 h-4 w-4" />
                    Run Monte Carlo Analysis
                  </>
                )}
              </Button>
            </div>

            {monteCarloError && <p className="text-sm text-destructive">{monteCarloError}</p>}

            {monteCarloResult && (
              <div className="rounded-lg border bg-muted/40 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">Monte Carlo Results</h4>
                  <Badge variant="secondary">{monteCarloResult.num_simulations} simulations</Badge>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <h5 className="text-sm font-medium">Return Statistics</h5>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground">Mean Return</p>
                        <p className="font-semibold">{(monteCarloResult.aggregated_results.mean_total_return * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Volatility</p>
                        <p className="font-semibold">{(monteCarloResult.aggregated_results.std_total_return * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">95% CI Lower</p>
                        <p className="font-semibold">{(monteCarloResult.aggregated_results.confidence_lower_return * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">95% CI Upper</p>
                        <p className="font-semibold">{(monteCarloResult.aggregated_results.confidence_upper_return * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Best Case</p>
                        <p className="font-semibold text-success">{(monteCarloResult.aggregated_results.best_case_return * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Worst Case</p>
                        <p className="font-semibold text-destructive">{(monteCarloResult.aggregated_results.worst_case_return * 100).toFixed(2)}%</p>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h5 className="text-sm font-medium">Risk Metrics</h5>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground">VaR (95%)</p>
                        <p className="font-semibold">{(monteCarloResult.risk_metrics.value_at_risk_95 * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Expected Shortfall</p>
                        <p className="font-semibold">{(monteCarloResult.risk_metrics.expected_shortfall_95 * 100).toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Prob. Positive</p>
                        <p className="font-semibold">{(monteCarloResult.aggregated_results.probability_positive_return * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Strategy</p>
                        <p className="font-semibold">{monteCarloResult.strategy_name}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Backtests List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Backtest Results</h3>
          <Badge variant="secondary">
            {backtests.length} {resultsHasMore ? 'loaded' : 'total'}
          </Badge>
        </div>

        {loading ? (
          <div className="space-y-4">
            {[1, 2].map(i => (
              <Card key={i}>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <div className="flex gap-4">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-4 w-32" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : error ? (
          <ErrorMessage message={error} onRetry={fetchBacktests} />
        ) : backtests.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground text-center">
                No backtests yet. Run a historical strategy simulation from Predictions (Chart tab).
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {backtests.map((b, i) => {
              const rowKey = backtestCorrelationKey(b) || `idx-${i}`
              const returnPercent = b.total_return * 100
              const returnColor = getReturnColor(returnPercent)
              const returnBadge = getReturnBadge(returnPercent)
              const isPositive = returnPercent > 0
              const status = b.status ?? b.metrics?.status ?? 'completed'
              const isFailed = status === 'failed'
              const initialCap = b.initial_capital ?? 100000
              return (
                <Card
                  key={rowKey}
                  role="button"
                  tabIndex={0}
                  onClick={() => setDetailBacktest(b)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setDetailBacktest(b)
                    }
                  }}
                  className="hover:shadow-lg transition-all border-muted hover:border-primary/50 cursor-pointer text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <CardHeader>
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5 text-primary" />
                          {b.strategy_name}
                        </CardTitle>
                        <CardDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-2">
                          {b.ticker && (
                            <span className="font-semibold text-foreground">{b.ticker}</span>
                          )}
                          <span className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {b.start_date} → {b.end_date}
                          </span>
                        </CardDescription>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <Badge variant={isFailed ? 'destructive' : status === 'running' ? 'secondary' : 'outline'}>
                          {status}
                        </Badge>
                        <Badge variant={returnBadge.variant}>
                          {returnBadge.label}
                        </Badge>
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${returnColor} flex items-center gap-1`}>
                            {isPositive ? (
                              <TrendingUp className="h-5 w-5" />
                            ) : (
                              <TrendingDown className="h-5 w-5" />
                            )}
                            {returnPercent.toFixed(2)}%
                          </div>
                          <p className="text-xs text-muted-foreground">Total Return</p>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <Separator />
                  
                  <CardContent className="pt-6">
                    {isFailed && (
                      <p className="mb-3 text-sm text-destructive">
                        Backtest failed: {b.error || b.metrics?.error || 'Unknown error'}
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground mb-4">
                      Click this card for an interactive chart and full run details (opens in a dialog).
                    </p>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <DollarSign className="h-8 w-8 text-primary" />
                        <div>
                          <p className="text-xs text-muted-foreground">Initial Capital</p>
                          <p className="font-semibold">${initialCap.toLocaleString()}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Activity className="h-8 w-8 text-blue-500" />
                        <div>
                          <p className="text-xs text-muted-foreground">Final Value</p>
                          <p className="font-semibold">
                            ${(initialCap * (1 + b.total_return)).toFixed(0)}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 col-span-2 md:col-span-1">
                        <Calendar className="h-8 w-8 text-orange-500" />
                        <div>
                          <p className="text-xs text-muted-foreground">Executed</p>
                          <p className="text-sm font-medium">
                            {new Date(b.timestamp).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 text-xs text-muted-foreground">
                      Engine: {b.execution_engine ?? b.metrics?.execution_summary?.engine ?? 'backtrader'} | Signals: {b.signals_emitted ?? b.metrics?.execution_summary?.signals_emitted ?? 0} | Intents: {b.order_intents ?? b.metrics?.execution_summary?.order_intents ?? 0} | Fills: {b.order_fills ?? b.metrics?.execution_summary?.order_fills ?? 0}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
            {resultsHasMore && (
              <div className="flex flex-col items-center gap-2 pt-2">
                {loadMoreError && (
                  <p className="text-sm text-destructive text-center max-w-md">{loadMoreError}</p>
                )}
                <Button
                  type="button"
                  variant="outline"
                  disabled={loadingMoreResults}
                  onClick={loadMoreBacktestResults}
                >
                  {loadingMoreResults ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
                      Loading…
                    </>
                  ) : (
                    'Load more'
                  )}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>

    {detailBacktest ? (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        role="presentation"
        onClick={() => setDetailBacktest(null)}
      >
        <div
          role="dialog"
          aria-modal
          aria-labelledby="backtest-detail-title"
          className="relative flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="absolute right-3 top-3 z-10 rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Close"
            onClick={() => setDetailBacktest(null)}
          >
            <X className="h-5 w-5" />
          </button>
          <div className="overflow-y-auto overscroll-contain p-6 pt-14">
            {(() => {
              const d = detailBacktest
              const dReturn = d.total_return * 100
              const dPositive = dReturn > 0
              const dStatus = d.status ?? d.metrics?.status ?? 'completed'
              const dFailed = dStatus === 'failed'
              const dKey = backtestCorrelationKey(d) || String(d.timestamp)
              const dIc = d.initial_capital ?? 100000
              return (
                <>
                  <h2 id="backtest-detail-title" className="text-xl font-semibold pr-10">
                    {d.strategy_name}
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {d.ticker ? <span className="font-medium text-foreground">{d.ticker}</span> : null}
                    {d.ticker ? ' · ' : null}
                    {d.start_date} → {d.end_date}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge variant={dFailed ? 'destructive' : dStatus === 'running' ? 'secondary' : 'outline'}>
                      {dStatus}
                    </Badge>
                    <Badge variant={getReturnBadge(dReturn).variant}>{getReturnBadge(dReturn).label}</Badge>
                    <span className={`text-lg font-bold ${getReturnColor(dReturn)}`}>{dReturn.toFixed(2)}% return</span>
                  </div>
                  {dFailed && (
                    <p className="mt-3 text-sm text-destructive">
                      {d.error || d.metrics?.error || 'Unknown error'}
                    </p>
                  )}
                  <div className="mt-5 min-h-[420px] w-full">
                    <BacktestEquityCompareChart
                      key={dKey}
                      backtest={d}
                      isPositive={dPositive}
                      isFailed={dFailed}
                      height={420}
                    />
                  </div>
                  <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Sharpe</p>
                      <p className="font-mono text-sm font-semibold">{Number(d.sharpe_ratio).toFixed(3)}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Max drawdown</p>
                      <p className="font-mono text-sm font-semibold">{(Number(d.max_drawdown) * 100).toFixed(2)}%</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Win rate</p>
                      <p className="font-mono text-sm font-semibold">{(Number(d.win_rate) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Trades</p>
                      <p className="font-mono text-sm font-semibold">{d.total_trades}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Volatility</p>
                      <p className="font-mono text-sm font-semibold">{(Number(d.volatility) * 100).toFixed(2)}%</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Avg trade return</p>
                      <p className="font-mono text-sm font-semibold">${Number(d.avg_trade_return).toFixed(2)}</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Annualized</p>
                      <p className="font-mono text-sm font-semibold">{(Number(d.annualized_return) * 100).toFixed(2)}%</p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-xs text-muted-foreground">Run at</p>
                      <p className="text-sm font-medium">{new Date(d.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                  <p className="mt-4 text-xs text-muted-foreground">
                    Engine: {d.execution_engine ?? d.metrics?.execution_summary?.engine ?? 'backtrader'} | Signals:{' '}
                    {d.signals_emitted ?? d.metrics?.execution_summary?.signals_emitted ?? 0} | Intents:{' '}
                    {d.order_intents ?? d.metrics?.execution_summary?.order_intents ?? 0} | Fills:{' '}
                    {d.order_fills ?? d.metrics?.execution_summary?.order_fills ?? 0}
                  </p>
                </>
              )
            })()}
          </div>
        </div>
      </div>
    ) : null}
    </>
  )
}
