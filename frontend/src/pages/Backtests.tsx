import React, { useEffect, useState } from 'react'
import { getBacktests, runBacktest } from '../services/api'
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
  Play,
  Calendar,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Target
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts'
import StrategySelector from '../components/StrategySelector'

type BacktestListItem = BacktestResult & {
  id?: string | number
  status?: string
  error?: string
  chart_data?: Array<{ day: number; value: number }>
  execution_engine?: string
  signals_emitted?: number
  order_intents?: number
  order_fills?: number
}

function toChartNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v)
  return null
}

/** Prefer chart_data; fall back to equity_curve so completed runs still plot after WS/JSON quirks. */
function buildBacktestEquityChartData(b: BacktestListItem): Array<{ day: number; value: number }> {
  const fromChart = Array.isArray(b.chart_data) ? b.chart_data : []
  const fromChartPoints = fromChart
    .map((p: Record<string, unknown>, idx: number) => {
      const value = toChartNumber(p?.value)
      if (value == null) return null
      const dayRaw = p?.day
      const day = typeof dayRaw === 'number' && Number.isFinite(dayRaw) ? dayRaw : idx
      return { day, value }
    })
    .filter((p): p is { day: number; value: number } => p != null)

  if (fromChartPoints.length > 0) return fromChartPoints

  const eq = Array.isArray(b.equity_curve) ? b.equity_curve : []
  return eq
    .map((p: Record<string, unknown>, idx: number) => {
      const value = toChartNumber(p?.value)
      if (value == null) return null
      return { day: idx, value }
    })
    .filter((p): p is { day: number; value: number } => p != null)
}

function equityChartYDomain(chartData: Array<{ value: number }>): [number, number] | undefined {
  if (chartData.length === 0) return undefined
  const values = chartData.map(d => d.value)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const span = maxV - minV
  const pad = span > 0 ? span * 0.05 : Math.max(Math.abs(minV) * 0.01, 1)
  return [minV - pad, maxV + pad]
}

type ServerWaitPhase = 'idle' | 'preflight' | 'backtest' | 'training'

function serverWaitPhaseLabel(phase: ServerWaitPhase): string {
  switch (phase) {
    case 'preflight':
      return 'Contacting the server: validating data for your ticker and dates…'
    case 'backtest':
      return 'Starting backtest job…'
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
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2023-12-31')
  const [running, setRunning] = useState(false)
  const [training, setTraining] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [ticker, setTicker] = useState('AAPL')
  const [trainObjective, setTrainObjective] = useState<'sharpe' | 'return' | 'drawdown' | 'balanced'>('balanced')
  const [maxEvals, setMaxEvals] = useState(24)
  const [optimizerMode, setOptimizerMode] = useState<'grid' | 'random'>('grid')
  const [randomSeed, setRandomSeed] = useState<string>('')
  const [trainedParams, setTrainedParams] = useState<Record<string, any> | null>(null)
  const [trainResult, setTrainResult] = useState<StrategyTrainResponse | null>(null)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [preflight, setPreflight] = useState<StrategyPreflightResponse | null>(null)
  const [serverWaitPhase, setServerWaitPhase] = useState<ServerWaitPhase>('idle')

  const fetchBacktests = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getBacktests()
      setBacktests(data as BacktestListItem[])
    } catch (e: any) {
      setError(e.message || 'Failed to fetch backtests')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchBacktests()
  }, [])

  const hasValidStrategySelection = strategy.trim().length > 0

  useEffect(() => {
    const handleBacktestStatus = (message: any) => {
      const backtestResult = message.data as BacktestListItem
      const chartData = buildBacktestEquityChartData(backtestResult)
      const normalizedResult: BacktestListItem = {
        ...backtestResult,
        chart_data: chartData,
        status: backtestResult.metrics?.status ?? backtestResult.status,
        error: backtestResult.metrics?.error ?? backtestResult.error,
      }

      // Update the backtests list with the new status
      setBacktests(prev => {
        const backtestId = normalizedResult.metrics?.backtest_id
        const existingIndex = prev.findIndex(b => {
          const existingId = b.metrics?.backtest_id ?? b.id
          return (
            backtestId != null &&
            backtestId !== '' &&
            existingId != null &&
            String(existingId) === String(backtestId)
          )
        })

        if (existingIndex >= 0) {
          // Update existing backtest
          const updated = [...prev]
          updated[existingIndex] = normalizedResult
          return updated
        } else {
          // Add new backtest at the beginning
          return [normalizedResult, ...prev]
        }
      })
    }

    // Register the listener
    // Cleanup on unmount
    return websocketService.registerListener('backtest_status', handleBacktestStatus)
  }, [])

  const startBacktest = async () => {
    if (!hasValidStrategySelection) {
      alert('Please select a strategy')
      return
    }
    setRunning(true)
    setServerWaitPhase('preflight')
    try {
      const check = await preflightStrategy(strategy, {
        ticker: ticker.trim().toUpperCase(),
        start_date: startDate,
        end_date: endDate,
      })
      setPreflight(check)
      if (!check.ready) {
        const topError = check.issues[0]?.message || 'Preflight failed'
        alert(topError)
        return
      }
      setServerWaitPhase('backtest')
      const data = await runBacktest({
        strategy_name: strategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: 100000,
        parameters: { ...strategyParams, ...(trainedParams ?? {}), ticker: ticker.trim().toUpperCase() },
      })

      setBacktests(prev => [{ ...data, id: data.metrics?.backtest_id, status: 'running', chart_data: [] }, ...prev])
    } catch (e: any) {
      alert('Failed to start backtest: ' + (e.message || 'Unknown error'))
    } finally {
      setRunning(false)
      setServerWaitPhase('idle')
    }
  }

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
      const check = await preflightStrategy(strategy, {
        ticker: ticker.trim().toUpperCase(),
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
        ticker: ticker.trim().toUpperCase(),
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
        (strategy === 'moving_average' || strategy === 'recursive_forecast')
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
            <Play className="h-5 w-5 text-primary" />
            Configure Backtest
          </CardTitle>
          <CardDescription>
            Set up a new backtest with your preferred parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Strategy Name</label>
                <StrategySelector
                  onStrategyChange={(selectedStrategy, params) => {
                    setStrategy(selectedStrategy)
                    setStrategyParams(params)
                  }}
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Start Date
                </label>
                <Input 
                  type="date" 
                  value={startDate} 
                  onChange={e => setStartDate(e.target.value)} 
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  End Date
                </label>
                <Input 
                  type="date" 
                  value={endDate} 
                  onChange={e => setEndDate(e.target.value)} 
                />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Training Ticker</label>
                <Input
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="AAPL"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Objective</label>
                <select
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                  value={trainObjective}
                  onChange={(e) => setTrainObjective(e.target.value as typeof trainObjective)}
                >
                  <option value="balanced">Balanced</option>
                  <option value="sharpe">Sharpe</option>
                  <option value="return">Return</option>
                  <option value="drawdown">Drawdown</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Evaluations</label>
                <Input
                  type="number"
                  min={1}
                  max={200}
                  value={maxEvals}
                  onChange={(e) => setMaxEvals(Number(e.target.value || 24))}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Optimizer mode</label>
                <select
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm"
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
              <Button
                onClick={startBacktest}
                disabled={running || !hasValidStrategySelection}
                className="w-full md:w-auto"
                size="lg"
              >
                {running ? (
                  <>
                    <Activity className="mr-2 h-4 w-4 animate-spin" />
                    Running Backtest...
                  </>
                ) : (
                  <>
                    <Target className="mr-2 h-4 w-4" />
                    Start Backtest
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
          <Badge variant="secondary">{backtests.length} Total</Badge>
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
                No backtests yet. Configure and run your first backtest above!
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {backtests.map((b, i) => {
              const returnPercent = b.total_return * 100
              const returnColor = getReturnColor(returnPercent)
              const returnBadge = getReturnBadge(returnPercent)
              const isPositive = returnPercent > 0
              const status = b.status ?? b.metrics?.status ?? 'completed'
              const isFailed = status === 'failed'
              const chartData = buildBacktestEquityChartData(b)
              const yDomain = equityChartYDomain(chartData)
              
              return (
                <Card 
                  key={i}
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
                    {/* Mini Chart */}
                    <div className="mb-4">
                      <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis 
                            dataKey="day" 
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                          />
                          <YAxis 
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                            domain={yDomain}
                            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                          />
                          <RechartsTooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '0.5rem'
                            }}
                            formatter={(value: any) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={isPositive ? 'hsl(142, 76%, 36%)' : 'hsl(0, 84.2%, 60.2%)'} 
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
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
          </div>
        )}
      </div>
    </div>
  )
}
