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
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import StrategySelector from '../components/StrategySelector'
import TickerSearch from '../components/TickerSearch'
import BacktestEquityCompareChart from '../components/BacktestEquityCompareChart'
import { buildBacktestEquitySeries } from '../utils/backtestChart'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'

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
  const [backtests, setBacktests] = useState<BacktestListItem[]>([])
  const [strategy, setStrategy] = useState('')
  const [strategyParams, setStrategyParams] = useState<Record<string, any>>({})
  const [startDate, setStartDate] = useState('2025-01-01')
  const [endDate, setEndDate] = useState('2025-12-31')
  const [training, setTraining] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [ticker, setTicker] = useState(() => getStoredTicker())
  const [trainObjective, setTrainObjective] = useState<'sharpe' | 'return' | 'drawdown' | 'balanced'>('balanced')
  const [maxEvals, setMaxEvals] = useState(8)
  const [optimizerMode, setOptimizerMode] = useState<'grid' | 'random'>('grid')
  const [randomSeed, setRandomSeed] = useState<string>('')
  const [trainedParams, setTrainedParams] = useState<Record<string, any> | null>(null)
  const [trainResult, setTrainResult] = useState<StrategyTrainResponse | null>(null)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [preflight, setPreflight] = useState<StrategyPreflightResponse | null>(null)
  const [serverWaitPhase, setServerWaitPhase] = useState<ServerWaitPhase>('idle')
  const [resultsHasMore, setResultsHasMore] = useState(false)
  const [resultsNextPage, setResultsNextPage] = useState(2)
  const [loadingMoreResults, setLoadingMoreResults] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null)
  const backtestsRef = useRef<BacktestListItem[]>([])
  backtestsRef.current = backtests

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
    setTraining(true)
    setTrainError(null)
    setServerWaitPhase('preflight')
    try {
      const normalizedTicker = rememberTicker(ticker)
      const check = await preflightStrategy(strategy, {
        ticker: normalizedTicker,
        start_date: startDate,
        end_date: endDate,
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
                onStrategyChange={(selectedStrategy, params) => {
                  setStrategy(selectedStrategy)
                  setStrategyParams(params)
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
                <label className="text-sm font-medium">Training Ticker</label>
                <TickerSearch
                  value={ticker}
                  onChange={(t) => setTicker(rememberTicker(t))}
                  placeholder="AAPL"
                />
              </div>

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
                  <div className="h-full w-2/5 rounded-full bg-primary server-wait-bar" />
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
                    <Skeleton className="h-32 w-full" />
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
              return (
                <Card 
                  key={rowKey}
                  className="hover:shadow-lg transition-all border-muted hover:border-primary/50"
                >
                  <CardHeader>
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5 text-primary" />
                          {b.strategy_name}
                        </CardTitle>
                        <CardDescription className="flex items-center gap-2 mt-2">
                          <Calendar className="h-3 w-3" />
                          {b.start_date} → {b.end_date}
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
                    <div className="mb-4">
                      <BacktestEquityCompareChart
                        backtest={b}
                        isPositive={isPositive}
                        isFailed={isFailed}
                        height={170}
                      />
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <DollarSign className="h-8 w-8 text-primary" />
                        <div>
                          <p className="text-xs text-muted-foreground">Initial Capital</p>
                          <p className="font-semibold">$100,000</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Activity className="h-8 w-8 text-blue-500" />
                        <div>
                          <p className="text-xs text-muted-foreground">Final Value</p>
                          <p className="font-semibold">
                            ${(100000 * (1 + b.total_return)).toFixed(0)}
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
  )
}
