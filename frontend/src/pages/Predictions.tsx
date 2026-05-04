import { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import {
  getPredictions,
  getPredictionProjections,
  getLatestPriceAnchor,
  runBacktest,
  getBacktest,
} from '../services/api'
import websocketService from '../services/websocket'
import { BacktestResult } from '../types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { PredictionResponse, PredictionProjection } from '../types'
import ErrorMessage from '../components/ErrorMessage'
import { Skeleton } from '../components/ui/skeleton'
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Target,
  Activity,
  Eye,
  Calendar,
  BarChart3,
  DollarSign,
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import OHLCChart from '../components/OHLCChart'
import StrategySelector from '../components/StrategySelector'
import { resolveProjectionAnchor } from '../utils/projectionAnchor'
import { NewsSidebar } from '../components/NewsSidebar'
import BacktestEquityCompareChart from '../components/BacktestEquityCompareChart'
import { buildBacktestEquitySeries } from '../utils/backtestChart'

type PredictionBacktestRow = BacktestResult & {
  id?: string | number
  ticker?: string
  status?: string
  error?: string
  chart_data?: Array<{ day: number; value: number; date?: string }>
}

function snapshotChartDataFromRow(b: PredictionBacktestRow): Array<{ day: number; value: number; date?: string }> {
  return buildBacktestEquitySeries(b).map((p) => ({
    day: p.day,
    value: p.value,
    ...(p.dateKey ? { date: p.dateKey } : {}),
  }))
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function backtestPhaseLabel(phase: string | undefined): string {
  switch (phase) {
    case 'queued':
      return 'Job queued on the server…'
    case 'preflight':
      return 'Validating price history and strategy requirements…'
    case 'loading_data':
      return 'Preparing backtest run…'
    case 'loading_prices':
      return 'Loading market data…'
    case 'executing':
      return 'Running simulation (this may take a while)…'
    default:
      return 'Waiting for the server…'
  }
}

export default function Predictions() {
  const [preds, setPreds] = useState<PredictionResponse[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [activeTab, setActiveTab] = useState('chart')

  // Projection controls state
  const [projectionStrategy, setProjectionStrategy] = useState('')
  const [projectionParams, setProjectionParams] = useState<Record<string, any>>({})
  const [projectionHorizon, setProjectionHorizon] = useState(30)

  // Prediction projections state
  const [showPredictionProjections, setShowPredictionProjections] = useState(false)
  const [predictionProjections, setPredictionProjections] = useState<PredictionProjection[]>([])
  const [projectionAnchorWarning, setProjectionAnchorWarning] = useState<string | null>(null)

  const [backtestStartDate, setBacktestStartDate] = useState('2025-01-01')
  const [backtestEndDate, setBacktestEndDate] = useState('2025-12-31')
  const [backtestRunning, setBacktestRunning] = useState(false)
  const [backtestPollPhase, setBacktestPollPhase] = useState<string | undefined>(undefined)
  const [backtestError, setBacktestError] = useState<string | null>(null)
  const [predictionBacktest, setPredictionBacktest] = useState<PredictionBacktestRow | null>(null)

  const chartRef = useRef<any>(null)

  const fetchPredictions = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getPredictions()
      setPreds(data)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch predictions')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPredictions()
  }, [])

  useEffect(() => {
    const handleBacktestStatus = (message: { data: PredictionBacktestRow }) => {
      const row = message.data
      const bid = row.metrics?.backtest_id
      setPredictionBacktest(prev => {
        if (!prev || !bid) return prev
        const prevId = prev.metrics?.backtest_id ?? prev.id
        if (prevId == null || String(prevId) !== String(bid)) return prev
        const chartData = snapshotChartDataFromRow(row as PredictionBacktestRow)
        return {
          ...row,
          chart_data: chartData,
          status: row.metrics?.status ?? row.status,
          error: row.metrics?.error ?? row.error,
        }
      })
    }
    return websocketService.registerListener('backtest_status', handleBacktestStatus)
  }, [])

  const startHistoricalBacktest = async () => {
    if (!projectionStrategy.trim()) {
      alert('Choose a strategy in Projection Controls above before running a backtest.')
      return
    }
    const sym = selectedTicker.trim().toUpperCase()
    if (!sym) {
      alert('Select a chart ticker first.')
      return
    }
    setBacktestRunning(true)
    setBacktestError(null)
    setBacktestPollPhase('queued')
    try {
      const data = await runBacktest({
        strategy_name: projectionStrategy,
        start_date: backtestStartDate,
        end_date: backtestEndDate,
        initial_capital: 100000,
        parameters: { ...projectionParams, ticker: sym },
      })
      const backtestId = data.metrics?.backtest_id as string | undefined
      if (!backtestId) {
        throw new Error('Server did not return a backtest id')
      }
      setPredictionBacktest({
        ...data,
        id: backtestId,
        ticker: sym,
        status: (data.metrics?.status as string | undefined) ?? 'running',
        chart_data: [],
      })

      const deadline = Date.now() + 15 * 60_000
      while (Date.now() < deadline) {
        await sleep(1200)
        const row = await getBacktest(backtestId)
        const metrics = row.metrics as Record<string, unknown> | undefined
        const phase = typeof metrics?.phase === 'string' ? metrics.phase : undefined
        setBacktestPollPhase(phase)
        const st = typeof metrics?.status === 'string' ? metrics.status : undefined
        setPredictionBacktest({
          ...row,
          id: backtestId,
          ticker: sym,
          status: st ?? row.metrics?.status,
          chart_data: snapshotChartDataFromRow({ ...row, id: backtestId } as PredictionBacktestRow),
        })
        if (st === 'completed' || st === 'failed') {
          if (st === 'failed' && typeof metrics?.error === 'string') {
            setBacktestError(metrics.error)
          }
          break
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setBacktestError(msg)
      alert('Failed to run backtest: ' + msg)
    } finally {
      setBacktestRunning(false)
      setBacktestPollPhase(undefined)
    }
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return { variant: 'success' as const, label: 'High' }
    if (confidence >= 0.6) return { variant: 'warning' as const, label: 'Medium' }
    return { variant: 'secondary' as const, label: 'Low' }
  }

  const getReturnColor = (returnVal: number) => {
    if (returnVal > 0) return 'text-success'
    if (returnVal < 0) return 'text-destructive'
    return 'text-muted-foreground'
  }


  const handleTickerClick = (tickerSymbol: string) => {
    setSelectedTicker(tickerSymbol)
    setActiveTab('chart')
  }

  // Generate prediction projections for the selected ticker
  const generatePredictionProjections = async (ticker: string): Promise<boolean> => {
    const anchor = await resolveProjectionAnchor(
      () => chartRef.current?.getLatestPrice?.() ?? null,
      () => chartRef.current?.getLatestTime?.() ?? null,
      {
        fallbackAnchor: () => getLatestPriceAnchor(ticker),
      }
    )

    if (!anchor) {
      setProjectionAnchorWarning('Projection overlay is waiting for latest chart data.')
      return false
    }

    setProjectionAnchorWarning(null)
    try {
      const strategiesToUse = projectionStrategy ? [projectionStrategy] : undefined
      const paramsByStrategy = projectionStrategy ? { [projectionStrategy]: projectionParams } : undefined

      const projections = await getPredictionProjections({
        symbol: ticker,
        anchor_time: new Date(anchor.latestTime * 1000).toISOString(),
        anchor_price: anchor.latestPrice,
        horizon_days: projectionHorizon,
        strategy_names: strategiesToUse,
        params_by_strategy: paramsByStrategy,
      })
      setPredictionProjections(projections)
      return true
    } catch (e: any) {
      console.error('Failed to generate prediction projections:', e)
      setProjectionAnchorWarning(e.message || 'Projection overlay failed to load.')
      return false
    }
  }

  // Handle prediction projections toggle
  const handlePredictionProjectionsToggle = (enabled: boolean) => {
    setShowPredictionProjections(enabled)
    if (!enabled) {
      setProjectionAnchorWarning(null)
      setPredictionProjections([])
      return
    }

    void generatePredictionProjections(selectedTicker)
  }

  useEffect(() => {
    if (!showPredictionProjections) return
    void generatePredictionProjections(selectedTicker)
  }, [selectedTicker, showPredictionProjections, projectionHorizon, projectionStrategy])

  const handleStrategyChange = (strategy: string, params: Record<string, any>) => {
    setProjectionStrategy(strategy)
    setProjectionParams(params)
    if (chartRef.current) {
      chartRef.current.setProjectionStrategy(strategy, params, projectionHorizon)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Predictions</h2>
        <p className="text-muted-foreground">
          Choose a symbol on the chart, configure projection controls, then run a historical strategy backtest to
          simulate the strategy over your date range. ML sentiment rows (if any) still appear under Recent Predictions;
          parameter training lives on the Backtests page.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="chart">Chart</TabsTrigger>
          <TabsTrigger value="predictions">Recent Predictions</TabsTrigger>
        </TabsList>

        <TabsContent value="chart">
          <div className="space-y-4">
            {/* Projection Controls */}
            <Card className="border-muted shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  Projection Controls
                </CardTitle>
                <CardDescription>
                  Pick a strategy and parameters for the chart; this drives projection overlays and the historical
                  backtest below.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <StrategySelector onStrategyChange={handleStrategyChange} />

                  <div className="flex items-end gap-2">
                      <Button
                        type="button"
                        variant={showPredictionProjections ? 'default' : 'outline'}
                        className="w-full md:w-auto"
                        onClick={() => handlePredictionProjectionsToggle(!showPredictionProjections)}
                      >
                        <Eye className="mr-2 h-4 w-4" />
                        Prediction Projections
                      </Button>
                  </div>

                  {projectionAnchorWarning && (
                    <p className="text-xs text-amber-500">{projectionAnchorWarning}</p>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="border-muted shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  Historical strategy backtest
                </CardTitle>
                <CardDescription>
                  Simulates the strategy selected in Projection Controls on the chart symbol ({' '}
                  <span className="font-mono text-foreground">{selectedTicker || '—'}</span>
                  ) over your date range. Results stream below; full history stays on the{' '}
                  <Link to="/backtests" className="text-primary underline-offset-4 hover:underline">
                    Backtests
                  </Link>{' '}
                  page.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      Start date
                    </label>
                    <Input
                      type="date"
                      value={backtestStartDate}
                      onChange={e => setBacktestStartDate(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      End date
                    </label>
                    <Input
                      type="date"
                      value={backtestEndDate}
                      onChange={e => setBacktestEndDate(e.target.value)}
                    />
                  </div>
                </div>
                <Button
                  type="button"
                  onClick={() => void startHistoricalBacktest()}
                  disabled={backtestRunning || !projectionStrategy.trim() || !selectedTicker.trim()}
                  className="w-full sm:w-auto"
                >
                  {backtestRunning ? (
                    <>
                      <Activity className="mr-2 h-4 w-4 animate-spin" />
                      Starting…
                    </>
                  ) : (
                    <>
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Run historical backtest
                    </>
                  )}
                </Button>
                {!projectionStrategy.trim() && (
                  <p className="text-xs text-muted-foreground">
                    Pick a strategy in Projection Controls to enable this button.
                  </p>
                )}
                {backtestRunning && (
                  <p className="text-xs text-muted-foreground" role="status" aria-live="polite">
                    {backtestPhaseLabel(backtestPollPhase)}
                  </p>
                )}
                {backtestError && <p className="text-sm text-destructive">{backtestError}</p>}
                {predictionBacktest && (() => {
                  const b = predictionBacktest
                  const returnPercent = b.total_return * 100
                  const isPositive = returnPercent > 0
                  const status = b.status ?? b.metrics?.status ?? 'completed'
                  const isFailed = status === 'failed'
                  const curvePreview = snapshotChartDataFromRow(b)
                  const hasCurve =
                    curvePreview.length > 0 || (Array.isArray(b.equity_curve) && b.equity_curve.length > 0)
                  return (
                    <div className="rounded-lg border bg-muted/20 p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <p className="font-medium text-sm">
                          {b.strategy_name}{' '}
                          <span className="text-muted-foreground font-normal">
                            ({b.start_date} → {b.end_date})
                          </span>
                        </p>
                        <Badge variant={isFailed ? 'destructive' : status === 'running' ? 'secondary' : 'outline'}>
                          {status}
                        </Badge>
                      </div>
                      {isFailed && (
                        <p className="text-sm text-destructive">
                          {b.error || b.metrics?.error || 'Backtest failed'}
                        </p>
                      )}
                      {hasCurve && (
                        <BacktestEquityCompareChart
                          backtest={{
                            ...b,
                            ticker:
                              (typeof b.ticker === 'string' && b.ticker.trim()
                                ? b.ticker.trim().toUpperCase()
                                : selectedTicker.trim().toUpperCase()) || undefined,
                          }}
                          isPositive={isPositive}
                          isFailed={isFailed}
                          tickerOverride={selectedTicker.trim().toUpperCase() || undefined}
                          height={170}
                        />
                      )}
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                        <div className="flex items-center gap-2 rounded-md bg-background/80 p-2">
                          <DollarSign className="h-5 w-5 text-primary shrink-0" />
                          <div>
                            <p className="text-xs text-muted-foreground">Return</p>
                            <p className={`font-semibold ${isPositive ? 'text-success' : 'text-destructive'}`}>
                              {returnPercent.toFixed(2)}%
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 rounded-md bg-background/80 p-2">
                          <Activity className="h-5 w-5 text-blue-500 shrink-0" />
                          <div>
                            <p className="text-xs text-muted-foreground">Sharpe</p>
                            <p className="font-semibold">{b.sharpe_ratio?.toFixed?.(3) ?? '—'}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 rounded-md bg-background/80 p-2">
                          <Target className="h-5 w-5 text-orange-500 shrink-0" />
                          <div>
                            <p className="text-xs text-muted-foreground">Trades</p>
                            <p className="font-semibold">{b.total_trades ?? '—'}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </CardContent>
            </Card>

            {/* Chart */}
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1 min-w-0">
                <OHLCChart 
                  
                  ref={chartRef}
                  symbol={selectedTicker}
                  height="600px"
                  strategyName={projectionStrategy}
                  params={projectionParams}
                  horizon={projectionHorizon}
                  showPredictionProjections={showPredictionProjections}
                  predictionProjections={predictionProjections}
                  onSymbolChange={(symbol) => {
                    const normalized = symbol.trim().toUpperCase()
                    if (normalized && normalized !== selectedTicker) {
                      setSelectedTicker(normalized)
                    }
                  }}
                />
              </div>
              <div className="w-full lg:w-80 xl:w-96 flex-shrink-0">
                <NewsSidebar ticker={selectedTicker} />
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="predictions">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Recent Predictions</h3>
              <Badge variant="secondary">{preds.length} Total</Badge>
            </div>

            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map(i => (
                  <Card key={i}>
                    <CardContent className="p-6">
                      <div className="space-y-3">
                        <Skeleton className="h-5 w-32" />
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-2/3" />
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : error ? (
              <ErrorMessage message={error} onRetry={fetchPredictions} />
            ) : preds.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <TrendingUp className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-center max-w-md">
                    No ML prediction rows in the database yet. Open the Chart tab, choose a symbol, then use
                    Historical strategy backtest to run a simulation; pipeline-generated predictions will appear here
                    when available.
                  </p>
                </CardContent>
              </Card>
            ) : (() => {
              // Group predictions by ticker, preserving insertion order (most recent first)
              const groups = preds.reduce<Record<string, typeof preds>>((acc, p) => {
                if (!acc[p.ticker]) acc[p.ticker] = []
                acc[p.ticker].push(p)
                return acc
              }, {})
              const horizonOrder = ['1d', '3d', '7d']
              return (
                <div className="grid gap-4">
                  {Object.entries(groups).map(([tickerKey, tickerPreds]) => {
                    const byHorizon: Record<string, typeof preds[0]> = {}
                    tickerPreds.forEach(p => { byHorizon[p.horizon] = p })
                    const latest = tickerPreds[0]
                    return (
                      <Card key={tickerKey} className="border-muted">
                        <CardContent className="p-6 space-y-4">
                          {/* Header */}
                          <div className="flex items-center justify-between">
                            <button
                              onClick={() => handleTickerClick(tickerKey)}
                              className="text-2xl font-bold font-mono hover:text-primary transition-colors cursor-pointer"
                            >
                              {tickerKey}
                            </button>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              {new Date(latest.timestamp).toLocaleString()}
                            </div>
                          </div>

                          <Separator />

                          {/* 3 horizons side by side */}
                          <div className="grid grid-cols-3 gap-4">
                            {horizonOrder.map(h => {
                              const p = byHorizon[h]
                              if (!p) return (
                                <div key={h} className="flex flex-col items-center gap-1 text-muted-foreground text-sm">
                                  <span className="font-medium">{h}</span>
                                  <span>—</span>
                                </div>
                              )
                              const isPositive = p.predicted_return > 0
                              const returnColor = getReturnColor(p.predicted_return)
                              const confidenceBadge = getConfidenceBadge(p.confidence)
                              return (
                                <div key={h} className="flex flex-col items-center gap-2">
                                  <Badge variant="outline" className="font-mono text-xs">{h}</Badge>
                                  <div className="flex items-center gap-1">
                                    {isPositive
                                      ? <TrendingUp className="h-4 w-4 text-success" />
                                      : <TrendingDown className="h-4 w-4 text-destructive" />}
                                    <span className={`text-lg font-bold ${returnColor}`}>
                                      {(p.predicted_return * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <Badge variant={confidenceBadge.variant} className="text-xs">
                                    {confidenceBadge.label} ({(p.confidence * 100).toFixed(0)}%)
                                  </Badge>
                                  {(p.interval_lower !== undefined && p.interval_upper !== undefined) && (
                                    <span className="text-xs text-muted-foreground text-center">
                                      {(p.interval_lower! * 100).toFixed(1)}% → {(p.interval_upper! * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                              )
                            })}
                          </div>

                          {(() => {
                            const sim = horizonOrder
                              .map(h => byHorizon[h]?.metadata?.simulation_as_of)
                              .find(Boolean) as string | undefined
                            const rowsByH = Object.fromEntries(
                              horizonOrder.map(h => {
                                const p = byHorizon[h]
                                const rows = p?.metadata?.forward_actual_closes as
                                  | { date: string; close: number }[]
                                  | undefined
                                return [h, rows]
                              })
                            ) as Record<string, { date: string; close: number }[] | undefined>
                            if (!sim && !horizonOrder.some(h => (rowsByH[h] ?? []).length > 0)) return null
                            return (
                              <div className="rounded-md border bg-muted/30 p-3 text-xs space-y-2">
                                {sim && (
                                  <p className="text-muted-foreground">
                                    Simulated as of: <span className="font-mono text-foreground">{sim}</span>
                                  </p>
                                )}
                                {horizonOrder.some(h => (rowsByH[h] ?? []).length > 0) && (
                                  <>
                                    <p className="font-medium text-muted-foreground">Realized daily closes (forward)</p>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                                      {horizonOrder.map(h => {
                                        const rows = rowsByH[h] ?? []
                                        if (!rows.length) return <div key={h} />
                                        return (
                                          <div key={h}>
                                            <span className="font-mono text-[10px] text-muted-foreground">{h}</span>
                                            <ul className="mt-1 space-y-0.5 font-mono">
                                              {rows.map((r) => (
                                                <li key={r.date}>
                                                  {r.date}: {r.close.toFixed(2)}
                                                </li>
                                              ))}
                                            </ul>
                                          </div>
                                        )
                                      })}
                                    </div>
                                  </>
                                )}
                              </div>
                            )
                          })()}
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              )
            })()}
          </div>
        </TabsContent>

      </Tabs>
    </div>
  )
}
