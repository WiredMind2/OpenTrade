import React, { useState } from 'react'
import { AlertTriangle, ArrowDown, ArrowRight, ArrowUp, Calculator, Shield, Target } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import {
  createTradePlan,
  type TradePlanResponse,
  type TraderStyle,
} from '../services/api'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'

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

function apiErrorMessage(e: any, fallback: string) {
  const detail = e?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg || item?.message || JSON.stringify(item)).filter(Boolean).join('; ')
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail)
  return e?.message || fallback
}

export default function TradePlan() {
  const [ticker, setTicker] = useState(() => getStoredTicker())
  const [style, setStyle] = useState<TraderStyle>('auto')
  const [asOfDate, setAsOfDate] = useState('')
  const [accountSize, setAccountSize] = useState(10000)
  const [riskPercent, setRiskPercent] = useState(1)
  const [plan, setPlan] = useState<TradePlanResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadPlan = async () => {
    const symbol = rememberTicker(ticker)
    if (!symbol) return
    setLoading(true)
    setError(null)
    try {
      const result = await createTradePlan({
        ticker: symbol,
        style,
        account_size: accountSize,
        risk_percent: riskPercent,
        ...(asOfDate.trim() ? { as_of_date: `${asOfDate.trim()}T23:59:59` } : {}),
      })
      setPlan(result)
    } catch (e: any) {
      setPlan(null)
      setError(apiErrorMessage(e, 'Failed to build trade plan'))
    } finally {
      setLoading(false)
    }
  }

  const active = plan ? directionBadge(plan.direction) : null
  const ActiveIcon = active?.icon

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Trade Plan</h2>
        <p className="text-muted-foreground">
          Build an entry, stop, target, and position-size plan from the selected ticker price history.
        </p>
      </div>

      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Plan Builder
          </CardTitle>
          <CardDescription>Choose ticker, price date, style, and risk budget before taking a signal seriously.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Ticker</label>
              <Input
                value={ticker}
                onChange={(e) => {
                  setTicker(e.target.value.toUpperCase())
                  setPlan(null)
                }}
                placeholder="AAPL"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Price date</label>
              <Input
                type="date"
                value={asOfDate}
                onChange={(e) => {
                  setAsOfDate(e.target.value)
                  setPlan(null)
                }}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Trader Style</label>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={style}
                onChange={(e) => {
                  setStyle(e.target.value as TraderStyle)
                  setPlan(null)
                }}
              >
                {styles.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Account Size</label>
              <Input type="number" value={accountSize} onChange={(e) => setAccountSize(Number(e.target.value) || 0)} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Risk %</label>
              <Input type="number" min={0.1} max={10} step={0.1} value={riskPercent} onChange={(e) => setRiskPercent(Number(e.target.value) || 1)} />
            </div>
            <div className="flex items-end">
              <Button className="w-full" onClick={() => void loadPlan()} disabled={loading}>
                <Calculator className="mr-2 h-4 w-4" />
                {loading ? 'Building...' : 'Build Plan'}
              </Button>
            </div>
          </div>
          {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {plan && active && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="border-muted shadow-md lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex flex-wrap items-center gap-2">
                {ActiveIcon && <ActiveIcon className="h-5 w-5 text-primary" />}
                {plan.ticker} Trade Plan
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
                <Metric label="Risk Amount" value={money(plan.risk_amount)} />
                <Metric label="Confidence" value={pct(plan.confidence)} />
              </div>
              <div className="rounded-md border bg-muted/30 p-3">
                <p className="text-sm font-medium">Invalidation</p>
                <p className="text-sm text-muted-foreground">{plan.invalidation}</p>
              </div>
              <div className="rounded-md border bg-muted/30 p-3">
                <p className="text-sm font-medium">Time Exit</p>
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
                  {plan.reasons.map((r) => <li key={r}>{r}</li>)}
                </ul>
              </div>
              {plan.warnings.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-amber-500">Warnings</p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    {plan.warnings.map((w) => <li key={w}>{w}</li>)}
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
