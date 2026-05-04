/** Points for Recharts: strategy equity plus optional buy-and-hold overlay. */

import type { PriceDailyRow } from '../services/api'

export type BacktestChartPoint = {
  day: number
  value: number
  /** Calendar day YYYY-MM-DD when known (from engine equity_curve). */
  dateKey?: string
  /** Same capital as strategy, fully invested at first close (for comparison). */
  tickerValue?: number | null
  /** Y-position for scatter marker (strategy equity at signal day). */
  buyMarker?: number | null
  sellMarker?: number | null
  signalReason?: string
  signalTicker?: string
}

function toChartNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v)
  return null
}

function normalizeDateKey(raw: unknown): string | undefined {
  if (typeof raw !== 'string' || !raw.trim()) return undefined
  const s = raw.trim()
  if (s.length >= 10) return s.slice(0, 10)
  return undefined
}

function parseIsoDayOnly(iso: unknown): string | undefined {
  if (typeof iso !== 'string') return undefined
  const s = iso.trim()
  if (s.length >= 10) return s.slice(0, 10)
  return undefined
}

function addCalendarDays(ymd: string, deltaDays: number): string | undefined {
  const t = Date.parse(`${ymd}T12:00:00`)
  if (!Number.isFinite(t)) return undefined
  const d = new Date(t)
  d.setUTCDate(d.getUTCDate() + deltaDays)
  return d.toISOString().slice(0, 10)
}

export function simulationStartYmd(backtest: {
  metrics?: Record<string, unknown>
  start_date?: string
}): string | undefined {
  const m = backtest.metrics
  if (m && typeof m.start_date === 'string') {
    const d = parseIsoDayOnly(m.start_date)
    if (d) return d
  }
  return parseIsoDayOnly(backtest.start_date)
}

export function resolveSimulationDateRange(backtest: {
  metrics?: Record<string, unknown>
  equity_curve?: unknown[]
  start_date?: string
  end_date?: string
}): { start: string; end: string } | null {
  const m = backtest.metrics
  const ms = m && typeof m.start_date === 'string' ? parseIsoDayOnly(m.start_date) : undefined
  const me = m && typeof m.end_date === 'string' ? parseIsoDayOnly(m.end_date) : undefined
  if (ms && me) return { start: ms, end: me }

  const eq = Array.isArray(backtest.equity_curve) ? backtest.equity_curve : []
  const keys: string[] = []
  for (const p of eq) {
    if (p && typeof p === 'object' && 'date' in p) {
      const k = normalizeDateKey((p as Record<string, unknown>).date)
      if (k) keys.push(k)
    }
  }
  if (keys.length > 0) return { start: keys[0], end: keys[keys.length - 1] }

  const a = parseIsoDayOnly(backtest.start_date)
  const b = parseIsoDayOnly(backtest.end_date)
  if (a && b) return { start: a, end: b }
  return null
}

/**
 * Build equity series with optional calendar keys for tooltips and price alignment.
 * Prefers `equity_curve` (has dates) over `chart_data` (often index-only).
 */
export function buildBacktestEquitySeries(backtest: {
  equity_curve?: unknown[]
  chart_data?: unknown[]
  metrics?: Record<string, unknown>
  start_date?: string
  end_date?: string
}): BacktestChartPoint[] {
  const eq = Array.isArray(backtest.equity_curve) ? backtest.equity_curve : []
  const simStart = simulationStartYmd(backtest)

  const fromEq: BacktestChartPoint[] = []
  for (let idx = 0; idx < eq.length; idx++) {
    const p = eq[idx]
    if (!p || typeof p !== 'object') continue
    const row = p as Record<string, unknown>
    const value = toChartNumber(row.value)
    if (value == null) continue
    const dateKey = normalizeDateKey(row.date) ?? (simStart ? addCalendarDays(simStart, idx) : undefined)
    fromEq.push({ day: idx, value, dateKey })
  }
  if (fromEq.length > 0) return fromEq

  const fromChart = Array.isArray(backtest.chart_data) ? backtest.chart_data : []
  const out: BacktestChartPoint[] = []
  for (let idx = 0; idx < fromChart.length; idx++) {
    const p = fromChart[idx]
    if (!p || typeof p !== 'object') continue
    const row = p as Record<string, unknown>
    const value = toChartNumber(row.value)
    if (value == null) continue
    const dayRaw = row.day
    const day = typeof dayRaw === 'number' && Number.isFinite(dayRaw) ? dayRaw : idx
    const dateKey =
      normalizeDateKey(row.date) ?? (simStart ? addCalendarDays(simStart, day) : undefined)
    out.push({ day, value, dateKey })
  }
  return out
}

function rowClosePx(r: PriceDailyRow): number | null {
  const adj = r.adjusted_close
  const cl = r.close
  if (typeof adj === 'number' && Number.isFinite(adj) && adj > 0) return adj
  if (typeof cl === 'number' && Number.isFinite(cl) && cl > 0) return cl
  return null
}

/** API returns newest first; sort ascending by date for lookups. */
export function sortPricesAscending(rows: PriceDailyRow[]): Array<{ date: string; close: number }> {
  const parsed: Array<{ date: string; close: number }> = []
  for (const r of rows) {
    const d = typeof r.date === 'string' ? r.date.slice(0, 10) : ''
    const c = rowClosePx(r)
    if (d && c != null) parsed.push({ date: d, close: c })
  }
  parsed.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))
  return parsed
}

/** Last close on or before `ymd` (inclusive), using trading-day series. */
export function closeOnOrBefore(asc: Array<{ date: string; close: number }>, ymd: string): number | undefined {
  if (asc.length === 0) return undefined
  let lo = 0
  let hi = asc.length - 1
  let best = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (asc[mid].date <= ymd) {
      best = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return best >= 0 ? asc[best].close : undefined
}

export function mergeBuyHoldOntoSeries(
  points: BacktestChartPoint[],
  priceRows: PriceDailyRow[],
  initialCapital: number,
): BacktestChartPoint[] {
  const ic = Number(initialCapital)
  if (!points.length || !Number.isFinite(ic) || ic <= 0) return points

  const asc = sortPricesAscending(priceRows)
  if (asc.length === 0) return points.map((p) => ({ ...p, tickerValue: null }))

  const firstKey = points.map((p) => p.dateKey).find(Boolean)
  if (!firstKey) return points.map((p) => ({ ...p, tickerValue: null }))

  const firstClose = closeOnOrBefore(asc, firstKey)
  if (firstClose == null || firstClose <= 0) return points.map((p) => ({ ...p, tickerValue: null }))

  return points.map((p) => {
    if (!p.dateKey) return { ...p, tickerValue: null }
    const c = closeOnOrBefore(asc, p.dateKey)
    if (c == null || c <= 0) return { ...p, tickerValue: null }
    return { ...p, tickerValue: ic * (c / firstClose) }
  })
}

export function combinedYDomain(points: BacktestChartPoint[]): [number, number] | undefined {
  const values: number[] = []
  for (const p of points) {
    if (Number.isFinite(p.value)) values.push(p.value)
    const t = p.tickerValue
    if (typeof t === 'number' && Number.isFinite(t)) values.push(t)
  }
  if (values.length === 0) return undefined
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const span = maxV - minV
  const pad = span > 0 ? span * 0.05 : Math.max(Math.abs(minV) * 0.01, 1)
  return [minV - pad, maxV + pad]
}

/** Map API ``metrics.decision_markers`` onto equity points by calendar date. */
export function attachDecisionMarkers(
  points: BacktestChartPoint[],
  markers: unknown,
): BacktestChartPoint[] {
  if (!Array.isArray(markers) || markers.length === 0) return points

  const byDate = new Map<string, { side: string; reason?: string; ticker?: string }>()
  for (const raw of markers) {
    if (!raw || typeof raw !== 'object') continue
    const m = raw as Record<string, unknown>
    if (typeof m.date !== 'string') continue
    const date = m.date.slice(0, 10)
    const side = String(m.side || '').toLowerCase()
    if (side !== 'buy' && side !== 'sell') continue
    const reason = typeof m.reason === 'string' ? m.reason : undefined
    const ticker = typeof m.ticker === 'string' ? m.ticker : undefined
    byDate.set(date, { side, reason, ticker })
  }

  return points.map((p) => {
    if (!p.dateKey) return p
    const hit = byDate.get(p.dateKey)
    if (!hit) return p
    const next: BacktestChartPoint = {
      ...p,
      signalReason: hit.reason,
      signalTicker: hit.ticker,
    }
    if (hit.side === 'buy') {
      next.buyMarker = p.value
      next.sellMarker = null
    } else {
      next.sellMarker = p.value
      next.buyMarker = null
    }
    return next
  })
}

/** Human-readable timestamp for tooltip (daily bars). */
export function formatChartTooltipTimestamp(point: BacktestChartPoint): string {
  if (point.dateKey) {
    const ms = Date.parse(`${point.dateKey}T12:00:00`)
    if (Number.isFinite(ms)) {
      return new Date(ms).toLocaleDateString(undefined, {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    }
    return point.dateKey
  }
  return `Bar ${point.day}`
}
