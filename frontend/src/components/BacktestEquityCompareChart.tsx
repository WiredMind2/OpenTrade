import { useEffect, useMemo, useState } from 'react'
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { getTickerPricesForRange } from '../services/api'
import {
  type BacktestChartPoint,
  attachDecisionMarkers,
  buildBacktestEquitySeries,
  combinedYDomain,
  formatChartTooltipTimestamp,
  mergeBuyHoldOntoSeries,
  resolveSimulationDateRange,
} from '../utils/backtestChart'

export type BacktestEquityCompareSource = {
  equity_curve?: unknown[]
  chart_data?: unknown[]
  metrics?: Record<string, unknown>
  start_date?: string
  end_date?: string
  initial_capital?: number
  ticker?: string | null
}

type TooltipProps = {
  active?: boolean
  payload?: Array<{
    dataKey?: string | number
    name?: string
    value?: number
    color?: string
    payload?: BacktestChartPoint
  }>
}

function EquityCompareTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload as BacktestChartPoint | undefined
  if (!row) return null
  const ts = formatChartTooltipTimestamp(row)
  const lines = payload.filter(
    (e) => e.dataKey === 'value' || e.dataKey === 'tickerValue',
  )
  const hasSignal =
    (typeof row.buyMarker === 'number' && Number.isFinite(row.buyMarker)) ||
    (typeof row.sellMarker === 'number' && Number.isFinite(row.sellMarker))
  return (
    <div
      className="rounded-md border border-border bg-card p-2 text-sm shadow-md"
      style={{ minWidth: 200 }}
    >
      <p className="text-xs text-muted-foreground border-b border-border pb-1 mb-1.5">{ts}</p>
      {hasSignal && (
        <p className="text-xs font-medium text-foreground mb-1.5">
          {typeof row.buyMarker === 'number' && Number.isFinite(row.buyMarker) ? '▲ Buy signal' : '▼ Sell signal'}
          {row.signalTicker ? ` · ${row.signalTicker}` : ''}
          {row.signalReason ? (
            <span className="text-muted-foreground font-normal"> ({row.signalReason})</span>
          ) : null}
        </p>
      )}
      <ul className="space-y-1">
        {lines.map((entry, i) => {
          const v = entry.value
          if (typeof v !== 'number' || !Number.isFinite(v)) return null
          return (
            <li key={i} className="flex justify-between gap-4">
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full shrink-0" style={{ background: entry.color }} />
                {entry.name ?? '—'}
              </span>
              <span className="font-mono tabular-nums">${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

type Props = {
  backtest: BacktestEquityCompareSource
  isPositive: boolean
  isFailed: boolean
  tickerOverride?: string | null
  height?: number
}

function equitySeriesSignature(equityCurve: unknown, chartData: unknown): string {
  const eq = Array.isArray(equityCurve) ? equityCurve : []
  if (eq.length === 0) {
    const ch = Array.isArray(chartData) ? chartData : []
    return `c:${ch.length}`
  }
  const tail = eq[eq.length - 1]
  const v =
    tail && typeof tail === 'object' && tail !== null && 'value' in tail
      ? String((tail as Record<string, unknown>).value)
      : ''
  return `e:${eq.length}:${v}`
}

function BuyTriangle(props: { cx?: number; cy?: number }) {
  const { cx, cy } = props
  if (typeof cx !== 'number' || typeof cy !== 'number' || !Number.isFinite(cx) || !Number.isFinite(cy)) return <g />
  return (
    <path
      d={`M ${cx} ${cy - 7} L ${cx - 6} ${cy + 5} L ${cx + 6} ${cy + 5} Z`}
      fill="hsl(142, 76%, 38%)"
      stroke="hsl(142, 76%, 22%)"
      strokeWidth={1}
    />
  )
}

function SellTriangle(props: { cx?: number; cy?: number }) {
  const { cx, cy } = props
  if (typeof cx !== 'number' || typeof cy !== 'number' || !Number.isFinite(cx) || !Number.isFinite(cy)) return <g />
  return (
    <path
      d={`M ${cx} ${cy + 7} L ${cx - 6} ${cy - 5} L ${cx + 6} ${cy - 5} Z`}
      fill="hsl(0, 72%, 48%)"
      stroke="hsl(0, 84%, 28%)"
      strokeWidth={1}
    />
  )
}

export default function BacktestEquityCompareChart({
  backtest,
  isPositive,
  isFailed,
  tickerOverride,
  height = 160,
}: Props) {
  const [series, setSeries] = useState<BacktestChartPoint[]>([])

  const equityCurve = backtest.equity_curve
  const chartData = backtest.chart_data
  const m = backtest.metrics
  const ms = typeof m?.start_date === 'string' ? m.start_date : ''
  const me = typeof m?.end_date === 'string' ? m.end_date : ''
  const sig = equitySeriesSignature(equityCurve, chartData)

  const baseSeries = useMemo(
    () =>
      buildBacktestEquitySeries({
        equity_curve: Array.isArray(equityCurve) ? equityCurve : undefined,
        chart_data: Array.isArray(chartData) ? chartData : undefined,
        metrics: m,
        start_date: backtest.start_date,
        end_date: backtest.end_date,
      }),
    [sig, ms, me, backtest.start_date, backtest.end_date, equityCurve, chartData, m?.phase, m?.status],
  )

  const ticker =
    (tickerOverride && tickerOverride.trim().toUpperCase()) ||
    (typeof backtest.ticker === 'string' && backtest.ticker.trim()
      ? backtest.ticker.trim().toUpperCase()
      : '')
  const initialCap = Number(backtest.initial_capital ?? 100000)

  useEffect(() => {
    let cancelled = false
    if (isFailed || baseSeries.length === 0) {
      setSeries([])
      return
    }

    const range = resolveSimulationDateRange({
      metrics: m,
      equity_curve: equityCurve as unknown[],
      start_date: backtest.start_date,
      end_date: backtest.end_date,
    })
    if (!ticker || !range) {
      setSeries(baseSeries.map((p) => ({ ...p, tickerValue: null })))
      return
    }

    ;(async () => {
      try {
        const prices = await getTickerPricesForRange(ticker, range.start, range.end, 1000)
        if (cancelled) return
        setSeries(mergeBuyHoldOntoSeries(baseSeries, prices, initialCap))
      } catch {
        if (cancelled) return
        setSeries(baseSeries.map((p) => ({ ...p, tickerValue: null })))
      }
    })()

    return () => {
      cancelled = true
    }
  }, [isFailed, ticker, initialCap, baseSeries, ms, me, equityCurve, backtest.start_date, backtest.end_date])

  const rawData = series.length > 0 ? series : baseSeries
  const data = attachDecisionMarkers(rawData, m?.decision_markers)
  const yDomain = combinedYDomain(data)
  const strategyStroke = isPositive ? 'hsl(142, 76%, 36%)' : 'hsl(0, 84.2%, 60.2%)'
  const tickerStroke = 'hsl(217, 91%, 60%)'
  const hasOverlay = data.some((p) => typeof p.tickerValue === 'number' && Number.isFinite(p.tickerValue))
  const hasBuyMarkers = data.some((p) => typeof p.buyMarker === 'number' && Number.isFinite(p.buyMarker))
  const hasSellMarkers = data.some((p) => typeof p.sellMarker === 'number' && Number.isFinite(p.sellMarker))

  if (data.length === 0) return null

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis dataKey="day" className="text-xs" tick={{ fill: 'hsl(var(--muted-foreground))' }} />
          <YAxis
            className="text-xs"
            tick={{ fill: 'hsl(var(--muted-foreground))' }}
            domain={yDomain}
            tickFormatter={(v) => `$${(Number(v) / 1000).toFixed(0)}k`}
          />
          <Tooltip content={<EquityCompareTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          {(hasOverlay && ticker) || hasBuyMarkers || hasSellMarkers ? (
            <Legend wrapperStyle={{ fontSize: 12 }} />
          ) : null}
          <Line
            type="monotone"
            dataKey="value"
            name="Strategy equity"
            stroke={strategyStroke}
            strokeWidth={2}
            dot={false}
          />
          {hasOverlay && ticker ? (
            <Line
              type="monotone"
              dataKey="tickerValue"
              name={`${ticker} (buy & hold)`}
              stroke={tickerStroke}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          ) : null}
          {hasBuyMarkers ? (
            <Scatter name="Buy signal" dataKey="buyMarker" fill="hsl(142, 76%, 38%)" shape={BuyTriangle} />
          ) : null}
          {hasSellMarkers ? (
            <Scatter name="Sell signal" dataKey="sellMarker" fill="hsl(0, 72%, 48%)" shape={SellTriangle} />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
