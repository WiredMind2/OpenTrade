import React, { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Activity, BarChart3, Gauge, Layers3, Target, TrendingUp } from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select'
import { Skeleton } from '../components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import {
  getStrategyAnalyticsFilters,
  getStrategyAnalyticsSummary,
  getStrategyDistributions,
  getStrategyTimeseries,
} from '../services/api'
import type {
  StrategyAnalyticsFilters,
  StrategyComparisonSummary,
  StrategyDistributionResponse,
  StrategyMetricPoint,
  StrategyTimeseriesResponse,
} from '../types'

const COLOR_PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4']

const fmtPct = (value: number) => `${(value * 100).toFixed(2)}%`
const fmtNum = (value: number) => value.toFixed(2)

export default function StrategyPerformance() {
  const [filters, setFilters] = useState<StrategyAnalyticsFilters | null>(null)
  const [summary, setSummary] = useState<StrategyComparisonSummary | null>(null)
  const [seriesMap, setSeriesMap] = useState<Record<string, StrategyTimeseriesResponse>>({})
  const [distMap, setDistMap] = useState<Record<string, StrategyDistributionResponse>>({})
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [selectedBenchmark, setSelectedBenchmark] = useState('SPY')
  const [selectedPreset, setSelectedPreset] = useState('MAX')
  const [selectedGranularity, setSelectedGranularity] = useState<'daily' | 'weekly' | 'monthly'>('daily')
  const [selectedRolling, setSelectedRolling] = useState(30)
  const [activeStrategy, setActiveStrategy] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadDashboard = async (seedStrategies?: string[]) => {
    setLoading(true)
    setError(null)
    try {
      const allFilters = filters ?? (await getStrategyAnalyticsFilters())
      if (!filters) {
        setFilters(allFilters)
      }
      const strategies = (seedStrategies && seedStrategies.length > 0 ? seedStrategies : selectedStrategies).slice(0, 6)
      const summaryData = await getStrategyAnalyticsSummary({
        strategies,
        benchmark_ticker: selectedBenchmark,
        preset: selectedPreset,
        granularity: selectedGranularity,
        rolling_window: selectedRolling,
      })
      setSummary(summaryData)

      const seriesEntries = await Promise.all(
        summaryData.metrics.map(async (metric) => {
          const [series, distributions] = await Promise.all([
            getStrategyTimeseries(metric.strategy, {
              benchmark_ticker: selectedBenchmark,
              preset: selectedPreset,
              granularity: selectedGranularity,
              rolling_window: selectedRolling,
            }),
            getStrategyDistributions(metric.strategy),
          ])
          return [metric.strategy, { series, distributions }] as const
        })
      )
      const nextSeries: Record<string, StrategyTimeseriesResponse> = {}
      const nextDist: Record<string, StrategyDistributionResponse> = {}
      seriesEntries.forEach(([strategy, payload]) => {
        nextSeries[strategy] = payload.series
        nextDist[strategy] = payload.distributions
      })
      setSeriesMap(nextSeries)
      setDistMap(nextDist)
    } catch (e: any) {
      setError(e.message || 'Failed to load strategy analytics dashboard')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const initialFilters = await getStrategyAnalyticsFilters()
        if (!mounted) return
        setFilters(initialFilters)
        const defaults = initialFilters.strategies.slice(0, 3)
        setSelectedStrategies(defaults)
        if (initialFilters.benchmarks.includes('SPY')) {
          setSelectedBenchmark('SPY')
        } else if (initialFilters.benchmarks.length > 0) {
          setSelectedBenchmark(initialFilters.benchmarks[0])
        }
        await loadDashboard(defaults)
      } catch (e: any) {
        if (mounted) {
          setError(e.message || 'Failed to initialize strategy analytics')
          setLoading(false)
        }
      }
    })()
    return () => {
      mounted = false
    }
  }, [])

  const comparisonCurve = useMemo(() => {
    const byDate: Record<string, any> = {}
    Object.values(seriesMap).forEach((series, i) => {
      series.points.forEach((p) => {
        if (!byDate[p.date]) byDate[p.date] = { date: p.date }
        byDate[p.date][series.strategy] = p.normalized_equity
      })
      if (series.benchmark_points.length > 0) {
        series.benchmark_points.forEach((bp) => {
          if (!byDate[bp.date]) byDate[bp.date] = { date: bp.date }
          byDate[bp.date][`${selectedBenchmark}_benchmark`] = bp.normalized_equity
        })
      }
    })
    return Object.values(byDate).sort((a: any, b: any) => a.date.localeCompare(b.date))
  }, [seriesMap, selectedBenchmark])

  const riskReturnScatter = useMemo(
    () =>
      (summary?.metrics ?? []).map((m) => ({
        strategy: m.strategy,
        risk: m.volatility,
        return: m.cagr,
      })),
    [summary]
  )

  const rankingBars = useMemo(
    () =>
      (summary?.metrics ?? []).map((m) => ({
        strategy: m.strategy,
        cagr: m.cagr,
        sharpe: m.sharpe,
        maxDrawdown: m.max_drawdown,
      })),
    [summary]
  )

  useEffect(() => {
    const firstStrategy = summary?.metrics[0]?.strategy ?? null
    if (!firstStrategy) {
      setActiveStrategy(null)
      return
    }
    if (!activeStrategy || !summary?.metrics.some((m) => m.strategy === activeStrategy)) {
      setActiveStrategy(firstStrategy)
    }
  }, [summary, activeStrategy])

  const activeMetric: StrategyMetricPoint | undefined =
    summary?.metrics.find((m) => m.strategy === activeStrategy) ?? summary?.metrics[0]
  const activeDist = activeMetric ? distMap[activeMetric.strategy] : undefined
  const activeSeries = activeMetric ? seriesMap[activeMetric.strategy] : undefined

  const toggleStrategy = (strategy: string) => {
    setSelectedStrategies((prev) =>
      prev.includes(strategy) ? prev.filter((s) => s !== strategy) : [...prev, strategy].slice(0, 6)
    )
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Strategy Performance</h2>
        <p className="text-muted-foreground">
          Advanced cross-strategy analytics with risk, return, trade quality, and benchmark-relative insights.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers3 className="h-5 w-5 text-primary" />
            Comparison Controls
          </CardTitle>
          <CardDescription>Choose strategy set, benchmark, preset range, granularity, and rolling window.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <Select value={selectedBenchmark} onValueChange={setSelectedBenchmark}>
              <SelectTrigger><SelectValue placeholder="Benchmark" /></SelectTrigger>
              <SelectContent>
                {(filters?.benchmarks ?? []).map((b) => (
                  <SelectItem key={b} value={b}>{b}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedPreset} onValueChange={setSelectedPreset}>
              <SelectTrigger><SelectValue placeholder="Preset" /></SelectTrigger>
              <SelectContent>
                {(filters?.available_presets ?? []).map((p) => (
                  <SelectItem key={p} value={p}>{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedGranularity} onValueChange={(v) => setSelectedGranularity(v as any)}>
              <SelectTrigger><SelectValue placeholder="Granularity" /></SelectTrigger>
              <SelectContent>
                {(filters?.available_granularities ?? []).map((g) => (
                  <SelectItem key={g} value={g}>{g}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={String(selectedRolling)} onValueChange={(v) => setSelectedRolling(Number(v))}>
              <SelectTrigger><SelectValue placeholder="Rolling window" /></SelectTrigger>
              <SelectContent>
                {(filters?.rolling_windows ?? []).map((r) => (
                  <SelectItem key={r} value={String(r)}>{r} periods</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-wrap gap-2">
            {(filters?.strategies ?? []).map((strategy) => (
              <Button
                key={strategy}
                variant={selectedStrategies.includes(strategy) ? 'default' : 'outline'}
                size="sm"
                onClick={() => toggleStrategy(strategy)}
              >
                {strategy}
              </Button>
            ))}
          </div>
          <Button onClick={() => loadDashboard()} disabled={loading || selectedStrategies.length === 0}>
            {loading ? 'Refreshing...' : 'Refresh Analytics'}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card>
          <CardContent className="pt-6 text-destructive">{error}</CardContent>
        </Card>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((id) => <Skeleton key={id} className="h-56 w-full" />)}
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Active Strategy</CardTitle>
              <CardDescription>Select which strategy drives the detailed panels below.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {(summary?.metrics ?? []).map((metric) => (
                <Button
                  key={metric.strategy}
                  size="sm"
                  variant={activeMetric?.strategy === metric.strategy ? 'default' : 'outline'}
                  onClick={() => setActiveStrategy(metric.strategy)}
                >
                  {metric.strategy}
                </Button>
              ))}
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card><CardHeader><CardDescription>Total Return</CardDescription><CardTitle className="text-2xl">{activeMetric ? fmtPct(activeMetric.total_return) : '-'}</CardTitle></CardHeader></Card>
            <Card><CardHeader><CardDescription>Sharpe / Sortino</CardDescription><CardTitle className="text-2xl">{activeMetric ? `${fmtNum(activeMetric.sharpe)} / ${fmtNum(activeMetric.sortino)}` : '-'}</CardTitle></CardHeader></Card>
            <Card><CardHeader><CardDescription>Max Drawdown</CardDescription><CardTitle className="text-2xl">{activeMetric ? fmtPct(activeMetric.max_drawdown) : '-'}</CardTitle></CardHeader></Card>
            <Card><CardHeader><CardDescription>Win Rate</CardDescription><CardTitle className="text-2xl">{activeMetric ? fmtPct(activeMetric.win_rate) : '-'}</CardTitle></CardHeader></Card>
          </div>

          <Tabs defaultValue="overview">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="risk">Risk</TabsTrigger>
              <TabsTrigger value="distributions">Distributions</TabsTrigger>
              <TabsTrigger value="monthly">Monthly</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-4">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5 text-primary" />Normalized Equity Comparison</CardTitle>
                  </CardHeader>
                  <CardContent className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={comparisonCurve}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {(summary?.metrics ?? []).map((m, idx) => (
                          <Line key={m.strategy} dataKey={m.strategy} stroke={COLOR_PALETTE[idx % COLOR_PALETTE.length]} dot={false} strokeWidth={2} />
                        ))}
                        <Line dataKey={`${selectedBenchmark}_benchmark`} stroke="#94a3b8" strokeDasharray="6 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Gauge className="h-5 w-5 text-primary" />Risk vs Return Scatter</CardTitle>
                  </CardHeader>
                  <CardContent className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid />
                        <XAxis type="number" dataKey="risk" name="Volatility" />
                        <YAxis type="number" dataKey="return" name="CAGR" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter data={riskReturnScatter} fill="#3b82f6" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="risk" className="mt-4">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5 text-primary" />Rolling Sharpe & Sortino</CardTitle>
                  </CardHeader>
                  <CardContent className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={activeSeries?.points ?? []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" hide />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="rolling_sharpe" stroke="#3b82f6" dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="rolling_sortino" stroke="#10b981" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-primary" />Metric Ranking</CardTitle>
                  </CardHeader>
                  <CardContent className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={rankingBars}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="strategy" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="cagr" fill="#3b82f6" />
                        <Bar dataKey="sharpe" fill="#10b981" />
                        <Line dataKey="maxDrawdown" stroke="#ef4444" dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="distributions" className="mt-4">
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <Card>
                  <CardHeader><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-primary" />Returns Distribution</CardTitle></CardHeader>
                  <CardContent className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activeDist?.returns_histogram ?? []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="bucket" hide />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#6366f1" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Trade PnL Distribution</CardTitle></CardHeader>
                  <CardContent className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activeDist?.trade_pnl_histogram ?? []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="bucket" hide />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#0ea5e9" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>PnL Contribution by Symbol</CardTitle></CardHeader>
                  <CardContent className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activeDist?.pnl_by_symbol ?? []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="bucket" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value">
                          {(activeDist?.pnl_by_symbol ?? []).map((_, idx) => (
                            <Cell key={idx} fill={COLOR_PALETTE[idx % COLOR_PALETTE.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="monthly" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Monthly Returns Matrix</CardTitle>
                  <CardDescription>Heatmap-style table for rapid monthly seasonality comparison.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 pr-3">Year</th>
                          {Array.from({ length: 12 }, (_, i) => (
                            <th key={i} className="text-right py-2 px-2">{String(i + 1).padStart(2, '0')}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(activeSeries?.monthly_returns ?? {}).map(([year, months]) => (
                          <tr key={year} className="border-b border-border/50">
                            <td className="py-2 pr-3 font-medium">{year}</td>
                            {Array.from({ length: 12 }, (_, i) => {
                              const key = String(i + 1).padStart(2, '0')
                              const value = months[key]
                              return (
                                <td key={key} className="text-right py-2 px-2">
                                  {value === undefined ? (
                                    <Badge variant="outline">-</Badge>
                                  ) : (
                                    <Badge variant={value >= 0 ? 'success' : 'destructive'}>{fmtPct(value)}</Badge>
                                  )}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
}
