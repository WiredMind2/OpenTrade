import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
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
import { Activity, BarChart3, BookmarkPlus, Gauge, Layers3, Target, TrendingUp } from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select'
import { Skeleton } from '../components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import {
  getStrategyAnalyticsFilters,
  getTickerStrategyLeaderboard,
  getStrategyVariantDistribution,
  getStrategyVariantSummary,
  getStrategyVariantTimeseries,
  createSavedModel,
} from '../services/api'
import type {
  StrategyAnalyticsFilters,
  StrategyDistributionResponse,
  TickerStrategyLeaderboardResponse,
  TickerStrategyRow,
  StrategyTimeseriesPoint,
  StrategyVariantRow,
  StrategyVariantSummary,
  StrategyVariantTimeseriesResponse,
} from '../types'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'

const COLOR_PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4']

const fmtPct = (value: number) => `${(value * 100).toFixed(2)}%`
const fmtNum = (value: number) => value.toFixed(2)
const safeNumber = (value: unknown, fallback: number = 0) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback

/** Tight Y domain so normalized equity curves (often near 1.0) stay visually separated. */
function normalizedEquityComparisonYDomain(
  rows: Array<Record<string, unknown>>,
  seriesKeys: string[],
): [number, number] | undefined {
  const values: number[] = []
  for (const row of rows) {
    for (const k of seriesKeys) {
      const v = row[k]
      if (typeof v === 'number' && Number.isFinite(v)) values.push(v)
    }
  }
  if (values.length === 0) return undefined
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const span = maxV - minV
  const pad = span > 0 ? span * 0.1 : Math.max(Math.abs(minV) * 0.01, 0.01)
  return [minV - pad, maxV + pad]
}

function shortHash(h: string) {
  return h.length > 10 ? `${h.slice(0, 6)}…` : h
}

function leaderboardRowKey(row: Pick<TickerStrategyRow, 'ticker' | 'strategy' | 'params_hash'>) {
  return `${row.ticker}|${row.strategy}|${row.params_hash}`
}

function defaultSavedModelName(row: Pick<TickerStrategyRow, 'strategy' | 'ticker' | 'params_hash'>) {
  return `Perf ${row.strategy} ${shortHash(row.params_hash)} @ ${row.ticker}`
}

function monthlyReturnsFromPoints(points: StrategyTimeseriesPoint[]): Record<string, Record<string, number>> {
  if (!points?.length) return {}
  const byYm = new Map<string, number>()
  for (const p of points) {
    const ym = p.date.slice(0, 7)
    byYm.set(ym, p.normalized_equity)
  }
  const sortedYm = [...byYm.keys()].sort()
  const heat: Record<string, Record<string, number>> = {}
  for (let i = 1; i < sortedYm.length; i++) {
    const prevYm = sortedYm[i - 1]
    const curYm = sortedYm[i]
    const prev = byYm.get(prevYm)!
    const curr = byYm.get(curYm)!
    const r = (curr - prev) / Math.max(prev, 1e-9)
    const [y, mo] = curYm.split('-')
    if (!heat[y]) heat[y] = {}
    heat[y][mo.padStart(2, '0')] = r
  }
  return heat
}

export default function StrategyPerformance() {
  const [filters, setFilters] = useState<StrategyAnalyticsFilters | null>(null)
  const [strategy, setStrategy] = useState('')
  const [variantSummary, setVariantSummary] = useState<StrategyVariantSummary | null>(null)
  const [variantTs, setVariantTs] = useState<StrategyVariantTimeseriesResponse | null>(null)
  const [dist, setDist] = useState<StrategyDistributionResponse | null>(null)
  const [selectedPreset, setSelectedPreset] = useState('MAX')
  const [selectedGranularity, setSelectedGranularity] = useState<'daily' | 'weekly' | 'monthly'>('daily')
  const [selectedRolling, setSelectedRolling] = useState(30)
  const [objective, setObjective] = useState<'sharpe' | 'return' | 'drawdown' | 'balanced'>('balanced')
  const [topN, setTopN] = useState(8)
  const [selectedTicker, setSelectedTicker] = useState(() => {
    const t = getStoredTicker()?.trim().toUpperCase()
    return t || 'ALL'
  })
  const [activeParamsHash, setActiveParamsHash] = useState<string | null>(null)
  const [analyticsTab, setAnalyticsTab] = useState<'overview' | 'risk' | 'distributions' | 'monthly'>('overview')
  const [tickerLeaderboard, setTickerLeaderboard] = useState<TickerStrategyLeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedLeaderboardKeys, setSelectedLeaderboardKeys] = useState<Set<string>>(() => new Set())
  const [saveBusy, setSaveBusy] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string | null>(null)
  const strategyRef = useRef(strategy)
  strategyRef.current = strategy
  const activeParamsHashRef = useRef<string | null>(null)
  activeParamsHashRef.current = activeParamsHash

  const effectiveBenchmark = useMemo(() => {
    const list = filters?.benchmarks ?? []
    if (list.includes('SPY')) return 'SPY'
    return list[0] ?? 'SPY'
  }, [filters])

  /** Omit ticker on variant APIs when leaderboard scope is all symbols. */
  const variantTickerApi = useMemo(
    () => (selectedTicker !== 'ALL' ? selectedTicker : undefined),
    [selectedTicker],
  )

  const loadDashboard = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const tickerSummary = await getTickerStrategyLeaderboard({
        objective,
        top_n: topN,
        ...(selectedTicker !== 'ALL' ? { ticker: selectedTicker } : {}),
      })
      setTickerLeaderboard(tickerSummary)

      const flatRows = tickerSummary.tickers.flatMap((bucket) => bucket.strategies)
      const hashNow = activeParamsHashRef.current
      const strategyNow = strategyRef.current

      let strategyToLoad = ''
      let preferredHash: string | null = null

      if (flatRows.length === 0) {
        strategyToLoad = ''
        preferredHash = null
      } else {
        const exact = flatRows.find((r) => r.strategy === strategyNow && r.params_hash === hashNow)
        if (exact) {
          strategyToLoad = exact.strategy
          preferredHash = exact.params_hash
        } else if (strategyNow && hashNow && flatRows.some((r) => r.strategy === strategyNow)) {
          strategyToLoad = strategyNow
          preferredHash = hashNow
        } else {
          strategyToLoad = flatRows[0].strategy
          preferredHash = flatRows[0].params_hash
        }
      }

      if (!strategyToLoad) {
        setStrategy('')
        setVariantSummary(null)
        setVariantTs(null)
        setActiveParamsHash(null)
        setDist(null)
        return
      }

      const summary = await getStrategyVariantSummary({
        strategy: strategyToLoad,
        objective,
        top_n: topN,
        ...(variantTickerApi ? { ticker: variantTickerApi } : {}),
      })
      setVariantSummary(summary)
      const hashes = summary.variants.map((v) => v.params_hash).join(',')
      if (!hashes) {
        setVariantTs(null)
        setActiveParamsHash(null)
        setStrategy(strategyToLoad)
        return
      }
      const ts = await getStrategyVariantTimeseries({
        strategy: strategyToLoad,
        params_hashes: hashes,
        benchmark_ticker: effectiveBenchmark,
        preset: selectedPreset,
        granularity: selectedGranularity,
        rolling_window: selectedRolling,
        objective,
        ...(variantTickerApi ? { ticker: variantTickerApi } : {}),
      })
      setVariantTs(ts)
      setActiveParamsHash((prev) => {
        if (preferredHash && summary.variants.some((v) => v.params_hash === preferredHash)) {
          return preferredHash
        }
        if (prev && summary.variants.some((v) => v.params_hash === prev)) return prev
        return summary.variants[0]?.params_hash ?? null
      })
      setStrategy(strategyToLoad)
    } catch (e: any) {
      setError(e.message || 'Failed to load variant analytics')
      setVariantSummary(null)
      setVariantTs(null)
      setTickerLeaderboard(null)
    } finally {
      setLoading(false)
    }
  }, [
    objective,
    topN,
    selectedTicker,
    variantTickerApi,
    effectiveBenchmark,
    selectedPreset,
    selectedGranularity,
    selectedRolling,
  ])

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const f = await getStrategyAnalyticsFilters()
        if (!mounted) return
        setFilters(f)
      } catch (e: any) {
        if (mounted) setError(e.message || 'Failed to initialize')
      } finally {
        if (mounted) setLoading(false)
      }
    })()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!filters) return
    void loadDashboard()
  }, [
    filters,
    objective,
    topN,
    selectedTicker,
    variantTickerApi,
    selectedPreset,
    selectedGranularity,
    selectedRolling,
    loadDashboard,
  ])

  useEffect(() => {
    if (analyticsTab !== 'distributions' || !strategy || !activeParamsHash) return
    let cancelled = false
    ;(async () => {
      try {
        const d = await getStrategyVariantDistribution(
          strategy,
          activeParamsHash,
          objective,
          variantTickerApi,
        )
        if (!cancelled) setDist(d)
      } catch {
        if (!cancelled) setDist(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [analyticsTab, strategy, activeParamsHash, objective, variantTickerApi])

  const activeVariant: StrategyVariantRow | undefined = useMemo(
    () => variantSummary?.variants.find((v) => v.params_hash === activeParamsHash),
    [variantSummary, activeParamsHash]
  )

  const activeSeries = useMemo(() => {
    if (!variantTs || !activeParamsHash) return undefined
    return variantTs.variant_series.find((s) => s.params_hash === activeParamsHash)
  }, [variantTs, activeParamsHash])

  const variantComparisonData = useMemo(() => {
    if (!variantTs?.variant_series?.length) return []
    const byDate: Record<string, any> = {}
    variantTs.variant_series.forEach((vs, idx) => {
      const key = `v_${idx}_${shortHash(vs.params_hash)}`
      for (const p of vs.points) {
        if (!byDate[p.date]) byDate[p.date] = { date: p.date }
        byDate[p.date][key] = safeNumber(p.normalized_equity, 0)
      }
    })
    if (variantTs.benchmark_points?.length) {
      const bmKey = `${effectiveBenchmark}_benchmark`
      for (const bp of variantTs.benchmark_points) {
        if (byDate[bp.date]) {
          byDate[bp.date][bmKey] = safeNumber(bp.normalized_equity, 0)
        }
      }
    }
    return Object.values(byDate).sort((a: any, b: any) => a.date.localeCompare(b.date))
  }, [variantTs, effectiveBenchmark])

  const lineKeys = useMemo(() => {
    if (!variantTs?.variant_series?.length) return [] as string[]
    return variantTs.variant_series.map((_, idx) => `v_${idx}_${shortHash(variantTs.variant_series[idx].params_hash)}`)
  }, [variantTs])

  const variantEquityYDomain = useMemo(() => {
    const bmKey = `${effectiveBenchmark}_benchmark`
    const keys = lineKeys.length ? [...lineKeys, bmKey] : []
    return normalizedEquityComparisonYDomain(variantComparisonData, keys)
  }, [variantComparisonData, lineKeys, effectiveBenchmark])

  const riskReturnScatter = useMemo(
    () =>
      (variantSummary?.variants ?? []).map((v) => ({
        name: shortHash(v.params_hash),
        risk: safeNumber(v.volatility, 0),
        return: safeNumber(v.total_return, 0),
      })),
    [variantSummary]
  )

  const rankingBars = useMemo(
    () =>
      (variantSummary?.variants ?? []).map((v, idx) => ({
        name: shortHash(v.params_hash),
        sharpe: safeNumber(v.sharpe_ratio, 0),
        total_return: safeNumber(v.total_return, 0),
        maxDrawdown: safeNumber(v.max_drawdown, 0),
        color: COLOR_PALETTE[idx % COLOR_PALETTE.length],
      })),
    [variantSummary]
  )

  const rollingSeries = useMemo(
    () =>
      (activeSeries?.points ?? []).map((p) => ({
        ...p,
        rolling_sharpe: safeNumber(p.rolling_sharpe, 0),
        rolling_sortino: safeNumber(p.rolling_sortino, 0),
      })),
    [activeSeries]
  )

  const monthlyMatrix = useMemo(() => monthlyReturnsFromPoints(activeSeries?.points ?? []), [activeSeries])

  const tickerOptions = useMemo(() => {
    const discovered = [...new Set((tickerLeaderboard?.tickers ?? []).map((row) => row.ticker))]
    if (selectedTicker !== 'ALL' && !discovered.includes(selectedTicker)) discovered.unshift(selectedTicker)
    return ['ALL', ...discovered]
  }, [tickerLeaderboard, selectedTicker])

  const tickerStrategyRows = useMemo(
    () =>
      (tickerLeaderboard?.tickers ?? []).flatMap((bucket) =>
        bucket.strategies.map((row, idx) => ({
          ...row,
          rank: idx + 1,
        }))
      ),
    [tickerLeaderboard]
  )

  const allLeaderboardSelected = useMemo(() => {
    if (!tickerStrategyRows.length) return false
    return tickerStrategyRows.every((r) => selectedLeaderboardKeys.has(leaderboardRowKey(r)))
  }, [tickerStrategyRows, selectedLeaderboardKeys])

  const toggleSelectAllLeaderboard = useCallback(() => {
    setSelectedLeaderboardKeys((prev) => {
      const keys = tickerStrategyRows.map(leaderboardRowKey)
      if (!keys.length) return new Set()
      const allOn = keys.every((k) => prev.has(k))
      if (allOn) return new Set()
      return new Set(keys)
    })
  }, [tickerStrategyRows])

  const toggleLeaderboardRowSelected = useCallback((row: TickerStrategyRow) => {
    const k = leaderboardRowKey(row)
    setSelectedLeaderboardKeys((prev) => {
      const next = new Set(prev)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      return next
    })
  }, [])

  const formatSaveError = (e: unknown) => {
    const err = e as { response?: { data?: { detail?: unknown } }; message?: string }
    const d = err.response?.data?.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join('; ')
    return err.message || 'Request failed'
  }

  const saveSelectedLeaderboardModels = useCallback(async () => {
    const rows = tickerStrategyRows.filter((r) => selectedLeaderboardKeys.has(leaderboardRowKey(r)))
    if (!rows.length) return
    setSaveBusy(true)
    setSaveMessage(null)
    let ok = 0
    const failures: string[] = []
    for (const row of rows) {
      try {
        await createSavedModel({
          name: defaultSavedModelName(row),
          strategy_name: row.strategy,
          ticker: row.ticker.trim().toUpperCase(),
          params: row.params ?? {},
          objective,
        })
        ok += 1
      } catch (e: unknown) {
        failures.push(`${row.ticker} ${row.strategy}: ${formatSaveError(e)}`)
      }
    }
    setSaveMessage(
      failures.length
        ? `Saved ${ok} of ${rows.length}. ${failures.slice(0, 3).join(' · ')}${failures.length > 3 ? '…' : ''}`
        : `Saved ${ok} model(s) to Predictions / saved models.`,
    )
    if (!failures.length) setSelectedLeaderboardKeys(new Set())
    setSaveBusy(false)
  }, [tickerStrategyRows, selectedLeaderboardKeys, objective])

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Strategy Performance</h2>
        <p className="text-muted-foreground">
          Each leaderboard row is a model: a strategy script plus one fixed parameter set. Compare models per ticker
          across strategy families and drill in from the table.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers3 className="h-5 w-5 text-primary" />
            Variant comparison controls
          </CardTitle>
          <CardDescription>
            One ticker control: filtering the leaderboard to a symbol, and driving variant charts for that symbol. Choose
            &quot;All tickers&quot; to list every symbol&apos;s leaders; charts then use pooled backtest rows with no ticker
            filter. Set ranking objective, top-N, preset, granularity, and rolling window. Benchmark defaults to SPY. Pick a
            leaderboard row (or use the default top row) to choose the strategy family.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Ticker</label>
              <Select
                value={selectedTicker}
                onValueChange={(v) => {
                  setSelectedTicker(v)
                  if (v !== 'ALL') rememberTicker(v)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Ticker" />
                </SelectTrigger>
                <SelectContent>
                  {tickerOptions.map((sym) => (
                    <SelectItem key={sym} value={sym}>
                      {sym === 'ALL' ? 'All tickers' : sym}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Objective (ranking)</label>
              <Select value={objective} onValueChange={(v) => setObjective(v as typeof objective)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="sharpe">Sharpe</SelectItem>
                  <SelectItem value="return">Return</SelectItem>
                  <SelectItem value="drawdown">Drawdown</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Top N variants</label>
              <Select value={String(topN)} onValueChange={(v) => setTopN(Number(v))}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[3, 5, 8, 10, 15].map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Select value={selectedPreset} onValueChange={setSelectedPreset}>
              <SelectTrigger>
                <SelectValue placeholder="Preset" />
              </SelectTrigger>
              <SelectContent>
                {(filters?.available_presets ?? []).map((p) => (
                  <SelectItem key={p} value={p}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedGranularity} onValueChange={(v) => setSelectedGranularity(v as any)}>
              <SelectTrigger>
                <SelectValue placeholder="Granularity" />
              </SelectTrigger>
              <SelectContent>
                {(filters?.available_granularities ?? []).map((g) => (
                  <SelectItem key={g} value={g}>
                    {g}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={String(selectedRolling)} onValueChange={(v) => setSelectedRolling(Number(v))}>
              <SelectTrigger>
                <SelectValue placeholder="Rolling window" />
              </SelectTrigger>
              <SelectContent>
                {(filters?.rolling_windows ?? []).map((r) => (
                  <SelectItem key={r} value={String(r)}>
                    {r} periods
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button variant="outline" onClick={() => void loadDashboard()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
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
          {[1, 2, 3, 4].map((id) => (
            <Skeleton key={id} className="h-56 w-full" />
          ))}
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Ticker model leaderboard</CardTitle>
              <CardDescription>
                Best models per ticker using objective {objective}. Click any row to drill into that strategy family and
                parameter variant. Select rows and save them as persisted models for the Predictions page and runtime
                evaluation.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  disabled={selectedLeaderboardKeys.size === 0 || saveBusy}
                  onClick={() => void saveSelectedLeaderboardModels()}
                >
                  <BookmarkPlus className="mr-2 h-4 w-4" />
                  Save selected ({selectedLeaderboardKeys.size})
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={selectedLeaderboardKeys.size === 0 || saveBusy}
                  onClick={() => setSelectedLeaderboardKeys(new Set())}
                >
                  Clear selection
                </Button>
                {saveMessage && (
                  <span className={`text-sm ${saveMessage.startsWith('Saved ') && !saveMessage.includes('of') ? 'text-emerald-600' : 'text-muted-foreground'}`}>
                    {saveMessage}
                  </span>
                )}
              </div>
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-center py-2 px-1 w-10">
                        <input
                          type="checkbox"
                          checked={allLeaderboardSelected}
                          onChange={() => toggleSelectAllLeaderboard()}
                          disabled={!tickerStrategyRows.length || saveBusy}
                          aria-label="Select all leaderboard rows"
                        />
                      </th>
                      <th className="text-left py-2 pr-3">Ticker</th>
                      <th className="text-right py-2 px-2">Rank</th>
                      <th className="text-left py-2 px-2">Strategy</th>
                      <th className="text-left py-2 px-2">Variant</th>
                      <th className="text-right py-2 px-2">Runs</th>
                      <th className="text-right py-2 px-2">Trades</th>
                      <th className="text-right py-2 px-2">Return</th>
                      <th className="text-right py-2 px-2">Ann. return</th>
                      <th className="text-right py-2 px-2">Sharpe</th>
                      <th className="text-right py-2 px-2">Vol</th>
                      <th className="text-right py-2 px-2">Max DD</th>
                      <th className="text-right py-2 px-2">Win rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tickerStrategyRows.map((row) => (
                      <tr
                        key={`${row.ticker}-${row.strategy}-${row.params_hash}`}
                        className={`border-b border-border/50 ${
                          strategy === row.strategy && activeParamsHash === row.params_hash ? 'bg-muted/30' : ''
                        }`}
                        onClick={() => {
                          strategyRef.current = row.strategy
                          activeParamsHashRef.current = row.params_hash
                          setStrategy(row.strategy)
                          setActiveParamsHash(row.params_hash)
                          void loadDashboard()
                        }}
                      >
                        <td
                          className="text-center py-2 px-1"
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            checked={selectedLeaderboardKeys.has(leaderboardRowKey(row))}
                            onChange={() => toggleLeaderboardRowSelected(row)}
                            disabled={saveBusy}
                            aria-label={`Select ${row.ticker} ${row.strategy}`}
                          />
                        </td>
                        <td className="py-2 pr-3 font-medium">{row.ticker}</td>
                        <td className="text-right py-2 px-2">{row.rank}</td>
                        <td className="py-2 px-2">{row.strategy}</td>
                        <td className="py-2 px-2 font-mono text-xs">{shortHash(row.params_hash)}</td>
                        <td className="text-right py-2 px-2">{row.run_count}</td>
                        <td className="text-right py-2 px-2">{row.total_trades}</td>
                        <td className="text-right py-2 px-2">{fmtPct(row.total_return)}</td>
                        <td className="text-right py-2 px-2">{fmtPct(row.annualized_return)}</td>
                        <td className="text-right py-2 px-2">{fmtNum(row.sharpe_ratio)}</td>
                        <td className="text-right py-2 px-2">{fmtPct(row.volatility)}</td>
                        <td className="text-right py-2 px-2">{fmtPct(row.max_drawdown)}</td>
                        <td className="text-right py-2 px-2">{fmtPct(row.win_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Tabs value={analyticsTab} onValueChange={(v) => setAnalyticsTab(v as typeof analyticsTab)}>
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="risk">Risk</TabsTrigger>
              <TabsTrigger value="distributions">Distributions</TabsTrigger>
              <TabsTrigger value="monthly">Monthly</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-4">
              {analyticsTab === 'overview' && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-primary" />
                        Normalized equity (variants + benchmark)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-72 min-w-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={variantComparisonData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" hide />
                          <YAxis
                            domain={variantEquityYDomain}
                            tickFormatter={(v: number | string) => {
                              const n = Number(v)
                              if (!Number.isFinite(n)) return ''
                              return n.toFixed(Number.isInteger(n) ? 0 : 3)
                            }}
                            width={52}
                          />
                          <Tooltip />
                          <Legend />
                          {lineKeys.map((k, idx) => (
                            <Line
                              key={k}
                              dataKey={k}
                              stroke={COLOR_PALETTE[idx % COLOR_PALETTE.length]}
                              dot={false}
                              strokeWidth={2}
                            />
                          ))}
                          <Line
                            dataKey={`${effectiveBenchmark}_benchmark`}
                            stroke="#94a3b8"
                            strokeDasharray="6 3"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Gauge className="h-5 w-5 text-primary" />
                        Risk vs return (variants)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-72 min-w-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                          <CartesianGrid />
                          <XAxis type="number" dataKey="risk" name="Volatility" />
                          <YAxis type="number" dataKey="return" name="Total return" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter data={riskReturnScatter} fill="#3b82f6" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            <TabsContent value="risk" className="mt-4">
              {analyticsTab === 'risk' && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="h-5 w-5 text-primary" />
                        Rolling Sharpe & Sortino (active variant)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-72 min-w-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rollingSeries}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" hide />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="rolling_sharpe" stroke="#3b82f6" dot={false} strokeWidth={2} />
                          <Line
                            type="monotone"
                            dataKey="rolling_sortino"
                            stroke="#10b981"
                            dot={false}
                            strokeWidth={2}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Target className="h-5 w-5 text-primary" />
                        Variant ranking snapshot
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-72 min-w-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={rankingBars}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="sharpe" fill="#3b82f6" />
                          <Bar dataKey="total_return" fill="#10b981" />
                          <Line type="monotone" dataKey="maxDrawdown" stroke="#ef4444" dot={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            <TabsContent value="distributions" className="mt-4">
              {analyticsTab === 'distributions' && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-primary" />
                        Returns distribution (active variant)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-64 min-w-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dist?.returns_histogram ?? []}>
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
                    <CardHeader>
                      <CardTitle>Params (active)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-xs overflow-auto max-h-64 bg-muted/40 p-3 rounded-md">
                        {JSON.stringify(activeVariant?.params ?? {}, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            <TabsContent value="monthly" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Monthly returns (active variant)</CardTitle>
                  <CardDescription>Approximated from normalized equity curve.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 pr-3">Year</th>
                          {Array.from({ length: 12 }, (_, i) => (
                            <th key={i} className="text-right py-2 px-2">
                              {String(i + 1).padStart(2, '0')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(monthlyMatrix).map(([year, months]) => (
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
