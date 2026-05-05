import React, { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Activity, AlertTriangle, ArrowDown, ArrowRight, ArrowUp, BarChart3, Calculator, Shield, Target } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Badge } from './ui/badge'
import {
  createTradePlan,
  getStrategyAnalyticsFilters,
  getTickerStrategyLeaderboard,
  type TradePlanResponse,
  type TraderStyle,
} from '../services/api'
import type { TickerStrategyRow } from '../types'
import { rememberTicker } from '../utils/tickerMemory'

export type TradePlanObjective = 'balanced' | 'sharpe' | 'return' | 'drawdown'

function money(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '-'
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: value >= 100 ? 2 : 4 })}`
}

function pct(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '-'
  return `${(value * 100).toFixed(0)}%`
}

function directionBadge(direction: TradePlanResponse['direction']) {
  if (direction === 'long') return { label: 'Long setup', variant: 'success' as const, icon: ArrowUp }
  if (direction === 'short') return { label: 'Short setup', variant: 'destructive' as const, icon: ArrowDown }
  if (direction === 'exit') return { label: 'Exit / reduce', variant: 'warning' as const, icon: AlertTriangle }
  return { label: 'Wait', variant: 'secondary' as const, icon: ArrowRight }
}

const styles: Array<{ value: TraderStyle; label: string }> = [
  { value: 'auto', label: 'Auto' },
  { value: 'short', label: 'Short-term' },
  { value: 'swing', label: 'Swing' },
  { value: 'long', label: 'Long-term' },
]

function apiErrorMessage(e: unknown, fallback: string) {
  const err = e as { response?: { data?: { detail?: unknown } }; message?: string }
  const detail = err?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item: { msg?: string; message?: string }) => item?.msg || item?.message || JSON.stringify(item)).filter(Boolean).join('; ')
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail)
  return err?.message || fallback
}

export interface TradePlanBuilderPanelProps {
  ticker: string
  priceDate: string
  onPriceDateChange: (value: string) => void
  objective: TradePlanObjective
  onObjectiveChange: (value: TradePlanObjective) => void
  traderStyle: TraderStyle
  onTraderStyleChange: (value: TraderStyle) => void
  accountSize: number
  onAccountSizeChange: (value: number) => void
  riskPercent: number
  onRiskPercentChange: (value: number) => void
  selectedStrategy: string
  onSelectedStrategyChange: (value: string) => void
  autoRefresh: boolean
  onAutoRefreshChange: (value: boolean) => void
}

export function TradePlanBuilderPanel({
  ticker,
  priceDate,
  onPriceDateChange,
  objective,
  onObjectiveChange,
  traderStyle,
  onTraderStyleChange,
  accountSize,
  onAccountSizeChange,
  riskPercent,
  onRiskPercentChange,
  selectedStrategy,
  onSelectedStrategyChange,
  autoRefresh,
  onAutoRefreshChange,
}: TradePlanBuilderPanelProps) {
  const [availableStrategies, setAvailableStrategies] = useState<string[]>([])
  const [bestBacktestStrategy, setBestBacktestStrategy] = useState<TickerStrategyRow | null>(null)
  const [plan, setPlan] = useState<TradePlanResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [ranking, setRanking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rankError, setRankError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void getStrategyAnalyticsFilters()
      .then((filters) => {
        if (!cancelled) setAvailableStrategies(filters.strategies || [])
      })
      .catch(() => {
        if (!cancelled) setAvailableStrategies([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    setPlan(null)
    setBestBacktestStrategy(null)
    setRankError(null)
    setError(null)
  }, [ticker, priceDate, objective])

  const findBestBacktestStrategy = useCallback(async () => {
    const symbol = rememberTicker(ticker)
    if (!symbol) return
    setRanking(true)
    setRankError(null)
    setPlan(null)
    try {
      const leaderboard = await getTickerStrategyLeaderboard({
        ticker: symbol,
        objective,
        top_n: 1,
      })
      const best = leaderboard.tickers.find((bucket) => bucket.ticker === symbol)?.strategies[0] ?? null
      setBestBacktestStrategy(best)
      if (best) onSelectedStrategyChange(best.strategy)
      if (!best) {
        setRankError('No completed backtest found for this ticker yet. Run or train strategies first.')
      }
    } catch (e: unknown) {
      setBestBacktestStrategy(null)
      setRankError(e instanceof Error ? e.message : 'Failed to rank completed backtests')
    } finally {
      setRanking(false)
    }
  }, [ticker, objective, onSelectedStrategyChange])

  const loadPlan = useCallback(async () => {
    const symbol = rememberTicker(ticker)
    if (!symbol) return
    const activeStrategy = bestBacktestStrategy?.strategy || selectedStrategy
    setLoading(true)
    setError(null)
    try {
      const result = await createTradePlan({
        ticker: symbol,
        style: traderStyle,
        account_size: accountSize,
        risk_percent: riskPercent,
        ...(priceDate.trim() ? { as_of_date: `${priceDate.trim()}T23:59:59` } : {}),
        ...(activeStrategy ? { strategy_name: activeStrategy } : {}),
        ...(bestBacktestStrategy && activeStrategy === bestBacktestStrategy.strategy
          ? {
              backtest_metrics: {
                total_return: bestBacktestStrategy.total_return,
                sharpe_ratio: bestBacktestStrategy.sharpe_ratio,
                max_drawdown: bestBacktestStrategy.max_drawdown,
                volatility: bestBacktestStrategy.volatility,
                total_trades: bestBacktestStrategy.total_trades,
              },
            }
          : {}),
      })
      setPlan(result)
    } catch (e: unknown) {
      setPlan(null)
      setError(apiErrorMessage(e, 'Failed to build trade plan'))
    } finally {
      setLoading(false)
    }
  }, [
    ticker,
    traderStyle,
    accountSize,
    riskPercent,
    priceDate,
    bestBacktestStrategy,
    selectedStrategy,
  ])

  useEffect(() => {
    if (!autoRefresh || priceDate.trim() || !plan || loading) return
    const interval = window.setInterval(() => {
      void loadPlan()
    }, 60000)
    return () => window.clearInterval(interval)
  }, [autoRefresh, priceDate, plan, loading, loadPlan])

  const active = plan ? directionBadge(plan.direction) : null
  const ActiveIcon = active?.icon
  const sym = ticker.trim().toUpperCase()

  return (
    <div className="space-y-4">
      <Card className="border-muted shadow-md">
        <CardHeader>
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                Plan builder
              </CardTitle>
              <CardDescription>
                Anchor date and ranking objective apply here and to saved-model signals below. Choose strategy and risk,
                then build entry, stop, and size. Ticker is set in the panel at the bottom.
              </CardDescription>
            </div>
            {selectedStrategy ? (
              <Badge variant={bestBacktestStrategy ? 'default' : 'outline'} className="w-fit">
                {bestBacktestStrategy ? 'Best backtest selected' : 'Manual strategy'}
              </Badge>
            ) : (
              <Badge variant="secondary" className="w-fit">
                Price action only
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted-foreground">Ticker:</span>
            <span className="font-mono font-semibold">{sym || '—'}</span>
            <Link to={`/strategy-performance${sym ? `?ticker=${encodeURIComponent(sym)}` : ''}`} className="text-primary underline-offset-4 hover:underline ml-2">
              Compare strategies
            </Link>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Price date (anchor)</label>
              <Input
                type="date"
                value={priceDate}
                onChange={(e) => onPriceDateChange(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">Leave empty for latest bar (live signals refresh faster).</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Trader style</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={traderStyle}
                onChange={(e) => {
                  onTraderStyleChange(e.target.value as TraderStyle)
                  setPlan(null)
                }}
              >
                {styles.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Account size</label>
              <Input type="number" value={accountSize} onChange={(e) => onAccountSizeChange(Number(e.target.value) || 0)} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Risk %</label>
              <Input
                type="number"
                min={0.1}
                max={10}
                step={0.1}
                value={riskPercent}
                onChange={(e) => onRiskPercentChange(Number(e.target.value) || 1)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Ranking objective</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={objective}
                onChange={(e) => onObjectiveChange(e.target.value as TradePlanObjective)}
              >
                <option value="balanced">Balanced</option>
                <option value="sharpe">Sharpe</option>
                <option value="return">Return</option>
                <option value="drawdown">Low drawdown</option>
              </select>
              <p className="text-xs text-muted-foreground">Used for “Use best” and for ordering saved-model signals.</p>
            </div>
          </div>

          <div className="grid gap-3 lg:grid-cols-[minmax(260px,1fr)_auto_auto] lg:items-end">
            <div className="space-y-2">
              <label className="text-sm font-medium">Strategy</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={selectedStrategy}
                onChange={(e) => {
                  onSelectedStrategyChange(e.target.value)
                  setBestBacktestStrategy((prev) => (prev?.strategy === e.target.value ? prev : null))
                  setRankError(null)
                  setPlan(null)
                }}
              >
                <option value="">Price action only</option>
                {availableStrategies.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <Button type="button" variant="outline" onClick={() => void findBestBacktestStrategy()} disabled={ranking} className="w-full lg:w-auto">
              {ranking ? (
                <>
                  <Activity className="mr-2 h-4 w-4 animate-spin" />
                  Ranking...
                </>
              ) : (
                <>
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Use best
                </>
              )}
            </Button>
            <Button className="w-full lg:w-auto" onClick={() => void loadPlan()} disabled={loading}>
              <Calculator className="mr-2 h-4 w-4" />
              {loading ? 'Building...' : 'Build plan'}
            </Button>
          </div>

          {bestBacktestStrategy && (
            <div className="grid gap-2 rounded-md border bg-muted/20 p-3 text-sm md:grid-cols-5">
              <Metric label="Best strategy" value={bestBacktestStrategy.strategy} />
              <Metric label="Return" value={`${(bestBacktestStrategy.total_return * 100).toFixed(2)}%`} />
              <Metric label="Sharpe" value={bestBacktestStrategy.sharpe_ratio.toFixed(2)} />
              <Metric label="Max DD" value={`${(bestBacktestStrategy.max_drawdown * 100).toFixed(2)}%`} />
              <Metric label="Win rate" value={`${(bestBacktestStrategy.win_rate * 100).toFixed(0)}%`} />
            </div>
          )}

          {rankError && <p className="text-sm text-destructive">{rankError}</p>}

          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={autoRefresh}
              disabled={Boolean(priceDate.trim())}
              onChange={(e) => onAutoRefreshChange(e.target.checked)}
            />
            Auto-refresh latest plan every 60s while no historical price date is selected.
          </label>
          {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {plan && active && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="border-muted shadow-md lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2">
                {ActiveIcon && <ActiveIcon className="h-5 w-5 text-primary" />}
                {plan.ticker} trade plan
                <Badge variant={active.variant}>{active.label}</Badge>
              </CardTitle>
              <CardDescription>
                {plan.trader_type} using {plan.price_date} close {money(plan.latest_close)}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Metric label="Entry" value={money(plan.entry)} />
                <Metric label="Stop" value={money(plan.stop_loss)} />
                <Metric label="Target 1" value={money(plan.take_profit_1)} />
                <Metric label="Target 2" value={money(plan.take_profit_2)} />
                <Metric label="Risk/Reward" value={plan.risk_reward?.toFixed(2) ?? '-'} />
                <Metric label="Shares" value={plan.position_size.toString()} />
                <Metric label="Risk amount" value={money(plan.risk_amount)} />
                <Metric label="Confidence" value={pct(plan.confidence)} />
              </div>
              <div className="rounded-md border bg-muted/30 p-3">
                <p className="text-sm font-medium">Invalidation</p>
                <p className="text-sm text-muted-foreground">{plan.invalidation}</p>
              </div>
              <div className="rounded-md border bg-muted/30 p-3">
                <p className="text-sm font-medium">Time exit</p>
                <p className="text-sm text-muted-foreground">{plan.time_exit}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-muted shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                Evidence
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-sm font-medium">Reasons</p>
                <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                  {plan.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </div>
              {plan.warnings.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-amber-500">Warnings</p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    {plan.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(plan.indicators).map(([key, value]) => (
                  <Metric key={key} label={key} value={value == null ? '-' : String(value)} />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-semibold tabular-nums">{value}</p>
    </div>
  )
}
