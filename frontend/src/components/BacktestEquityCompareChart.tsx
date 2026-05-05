import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  ChartingLibraryWidgetOptions,
  IChartingLibraryWidget,
} from '../../public/charting_library/charting_library.d.ts'
import { useTheme } from './ThemeProvider'
import TradingViewUDFDatafeed from '../services/tradingViewUDF'
import { getTickerPricesForRange } from '../services/api'
import {
  type BacktestChartPoint,
  attachDecisionMarkers,
  buildBacktestEquitySeries,
  closeOnOrAfter,
  closeOnOrBefore,
  dateKeyToUnixMsUtc,
  dateKeyToUnixSecondsUtc,
  lastTradeBarIndex,
  mergeBuyHoldOntoSeries,
  resolveSimulationDateRange,
  sortPricesAscending,
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

type OverlayPayload = {
  data: BacktestChartPoint[]
  anchorClose: number | null
  initialCap: number
  isPositive: boolean
  ticker: string
  markers: unknown
}

function overlayPrice(anchorClose: number, initialCap: number, equity: number): number {
  return anchorClose * (equity / initialCap)
}

export default function BacktestEquityCompareChart({
  backtest,
  isPositive,
  isFailed,
  tickerOverride,
  height = 200,
}: Props) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const [series, setSeries] = useState<BacktestChartPoint[]>([])
  const [anchorClose, setAnchorClose] = useState<number | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const widgetRef = useRef<IChartingLibraryWidget | null>(null)
  const overlayEntityIdsRef = useRef<Array<string | number>>([])
  const containerIdRef = useRef(`backtest_tv_${Math.random().toString(36).slice(2, 11)}`)
  const payloadRef = useRef<OverlayPayload | null>(null)

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
    [sig, ms, me, backtest.start_date, backtest.end_date, m?.phase, m?.status],
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
      setAnchorClose(null)
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
      setAnchorClose(null)
      return
    }

    ;(async () => {
      try {
        const prices = await getTickerPricesForRange(ticker, range.start, range.end, 1000)
        if (cancelled) return
        const asc = sortPricesAscending(prices)
        const firstKey = baseSeries.map((p) => p.dateKey).find(Boolean)
        const fc = firstKey
          ? closeOnOrBefore(asc, firstKey) ?? closeOnOrAfter(asc, firstKey)
          : undefined
        setAnchorClose(typeof fc === 'number' && Number.isFinite(fc) && fc > 0 ? fc : null)
        setSeries(mergeBuyHoldOntoSeries(baseSeries, prices, initialCap))
      } catch {
        if (cancelled) return
        setSeries(baseSeries.map((p) => ({ ...p, tickerValue: null })))
        setAnchorClose(null)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [isFailed, ticker, initialCap, baseSeries, ms, me, sig, backtest.start_date, backtest.end_date])

  const rawData = series.length > 0 ? series : baseSeries
  const data = attachDecisionMarkers(rawData, m?.decision_markers)

  payloadRef.current = {
    data,
    anchorClose,
    initialCap,
    isPositive,
    ticker,
    markers: m?.decision_markers,
  }

  const clearOverlayEntities = useCallback((chart: { removeEntity: (id: string | number) => void }) => {
    for (const id of overlayEntityIdsRef.current) {
      try {
        chart.removeEntity(id)
      } catch {
        /* ignore */
      }
    }
    overlayEntityIdsRef.current = []
  }, [])

  const redrawOverlays = useCallback(() => {
    const widget = widgetRef.current
    const payload = payloadRef.current
    if (!widget || !payload?.data.length) return

    let chart: ReturnType<IChartingLibraryWidget['activeChart']>
    try {
      chart = widget.activeChart()
    } catch {
      return
    }

    clearOverlayEntities(chart)

    const { data: pts, anchorClose: ac, initialCap: ic, isPositive: pos, markers } = payload
    /** Portfolio (strategy equity) line — high-contrast vs candles. */
    const portfolioLineColor = pos ? '#facc15' : '#fb923c'
    const benchColor = '#60a5fa'

    const pricedPoints = (p: BacktestChartPoint): { timeSec: number; timeMs: number; price: number } | null => {
      if (!p.dateKey) return null
      const timeSec = dateKeyToUnixSecondsUtc(p.dateKey)
      const timeMs = dateKeyToUnixMsUtc(p.dateKey)
      if (timeSec == null || timeMs == null) return null
      if (typeof ac === 'number' && Number.isFinite(ac) && ac > 0 && Number.isFinite(p.value)) {
        return { timeSec, timeMs, price: overlayPrice(ac, ic, p.value) }
      }
      return null
    }

    for (let i = 1; i < pts.length; i++) {
      const a = pricedPoints(pts[i - 1])
      const b = pricedPoints(pts[i])
      if (!a || !b) continue
      let id = chart.createMultipointShape([{ time: a.timeSec, price: a.price }, { time: b.timeSec, price: b.price }], {
        shape: 'trend_line',
        lock: true,
        disableSelection: true,
        disableSave: true,
        disableUndo: true,
        overrides: {
          linestyle: 0,
          linewidth: 3,
          linecolor: portfolioLineColor,
          transparency: 0,
        },
      })
      if (!id) {
        id = chart.createMultipointShape([{ time: a.timeMs, price: a.price }, { time: b.timeMs, price: b.price }], {
          shape: 'trend_line',
          lock: true,
          disableSelection: true,
          disableSave: true,
          disableUndo: true,
          overrides: {
            linestyle: 0,
            linewidth: 3,
            linecolor: portfolioLineColor,
            transparency: 0,
          },
        })
      }
      if (id) overlayEntityIdsRef.current.push(id)
    }

    const hasBench = pts.some((p) => typeof p.tickerValue === 'number' && Number.isFinite(p.tickerValue))
    if (hasBench && typeof ac === 'number' && Number.isFinite(ac) && ac > 0) {
      for (let i = 1; i < pts.length; i++) {
        const tv0 = pts[i - 1].tickerValue
        const tv1 = pts[i].tickerValue
        if (typeof tv0 !== 'number' || typeof tv1 !== 'number' || !Number.isFinite(tv0) || !Number.isFinite(tv1)) {
          continue
        }
        const a = pricedPoints(pts[i - 1])
        const b = pricedPoints(pts[i])
        if (!a || !b) continue
        const paSec = { time: a.timeSec, price: overlayPrice(ac, ic, tv0) }
        const pbSec = { time: b.timeSec, price: overlayPrice(ac, ic, tv1) }
        const paMs = { time: a.timeMs, price: overlayPrice(ac, ic, tv0) }
        const pbMs = { time: b.timeMs, price: overlayPrice(ac, ic, tv1) }
        let id = chart.createMultipointShape([paSec, pbSec], {
          shape: 'trend_line',
          lock: true,
          disableSelection: true,
          disableSave: true,
          disableUndo: true,
          overrides: {
            linestyle: 2,
            linewidth: 1,
            linecolor: benchColor,
            transparency: 35,
          },
        })
        if (!id) {
          id = chart.createMultipointShape([paMs, pbMs], {
            shape: 'trend_line',
            lock: true,
            disableSelection: true,
            disableSave: true,
            disableUndo: true,
            overrides: {
              linestyle: 2,
              linewidth: 1,
              linecolor: benchColor,
              transparency: 35,
            },
          })
        }
        if (id) overlayEntityIdsRef.current.push(id)
      }
    }

    for (const p of pts) {
      if (!p.dateKey) continue
      const tSec = dateKeyToUnixSecondsUtc(p.dateKey)
      const tMs = dateKeyToUnixMsUtc(p.dateKey)
      if (tSec == null || tMs == null) continue
      const buy = typeof p.buyMarker === 'number' && Number.isFinite(p.buyMarker)
      const sell = typeof p.sellMarker === 'number' && Number.isFinite(p.sellMarker)
      if (!buy && !sell) continue
      const price = typeof ac === 'number' && Number.isFinite(ac) && ac > 0 && Number.isFinite(p.value)
        ? overlayPrice(ac, ic, p.value)
        : null
      if (price == null || !Number.isFinite(price) || price <= 0) continue
      const markColor = buy ? '#22c55e' : '#ef4444'

      const arrowOverrides = buy
        ? {
            color: markColor,
            transparency: 0,
            size: 3,
            'linetoolarrowmarkup.arrowColor': markColor,
            'linetoolarrowmarkup.color': markColor,
            'linetoolarrowmarkup.fontsize': 28,
            'linetoolarrowmarkup.bold': true,
            'linetoolarrowmarkup.showLabel': false,
          }
        : {
            color: markColor,
            transparency: 0,
            size: 3,
            'linetoolarrowmarkdown.arrowColor': markColor,
            'linetoolarrowmarkdown.color': markColor,
            'linetoolarrowmarkdown.fontsize': 28,
            'linetoolarrowmarkdown.bold': true,
            'linetoolarrowmarkdown.showLabel': false,
          }

      let arrowId = chart.createShape(
        { time: tSec, price },
        {
          shape: buy ? 'arrow_up' : 'arrow_down',
          lock: true,
          disableSelection: true,
          disableSave: true,
          disableUndo: true,
          overrides: arrowOverrides as Record<string, string | number | boolean>,
        },
      )
      if (!arrowId) {
        arrowId = chart.createShape(
          { time: tMs, price },
          {
            shape: buy ? 'arrow_up' : 'arrow_down',
            lock: true,
            disableSelection: true,
            disableSave: true,
            disableUndo: true,
            overrides: arrowOverrides as Record<string, string | number | boolean>,
          },
        )
      }
      if (arrowId) {
        overlayEntityIdsRef.current.push(arrowId)
      }
    }

    const anchorIdx = lastTradeBarIndex(pts, markers)
    const anchorKey =
      anchorIdx != null && pts[anchorIdx]?.dateKey
        ? pts[anchorIdx].dateKey
        : pts[pts.length - 1]?.dateKey
    const anchorSec = anchorKey ? dateKeyToUnixSecondsUtc(anchorKey) : null
    if (anchorSec != null) {
      const windowSec = 140 * 24 * 60 * 60
      const padSec = 6 * 24 * 60 * 60
      chart
        .setVisibleRange(
          { from: anchorSec - windowSec, to: anchorSec + padSec },
          { percentRightMargin: 8 },
        )
        .catch(() => {
          /* ignore */
        })
    }
  }, [clearOverlayEntities])

  useEffect(() => {
    if (isFailed || !ticker || data.length === 0 || !containerRef.current) return

    let cancelled = false

    const run = async () => {
      const TradingView = await import('../../public/charting_library/charting_library.esm.js')
      if (cancelled) return
      if (widgetRef.current) {
        try {
          widgetRef.current.remove()
        } catch {
          /* ignore */
        }
        widgetRef.current = null
      }
      if (!containerRef.current || cancelled) return

      const datafeed = new TradingViewUDFDatafeed()

      const disabled = [
        'header_widget',
        'left_toolbar',
        'timeframes_toolbar',
        'control_bar',
        'context_menus',
        'header_symbol_search',
        'header_compare',
        'header_indicators',
        'header_resolutions',
        'header_undo_redo',
        'header_screenshot',
        'header_fullscreen_button',
        'header_settings',
        'header_quick_search',
        'symbol_search_hot_key',
        'edit_buttons_in_legend',
        'legend_context_menu',
        'pane_context_menu',
        'scales_context_menu',
        'save_chart_properties_to_local_storage',
        'use_localstorage_for_settings',
        'create_volume_indicator_by_default',
      ] as const

      const widgetOptions: ChartingLibraryWidgetOptions = {
        symbol: ticker,
        datafeed,
        interval: '1D' as ChartingLibraryWidgetOptions['interval'],
        container: containerIdRef.current,
        library_path: '/charting_library/',
        locale: 'en',
        disabled_features: [...disabled] as ChartingLibraryWidgetOptions['disabled_features'],
        enabled_features: [],
        custom_css_url: 'tv-theme.css',
        client_id: 'tradingview.com',
        user_id: 'public_user_id',
        fullscreen: false,
        autosize: true,
        studies_overrides: {},
        theme: isDark ? 'dark' : 'light',
        timezone: 'Etc/UTC',
        toolbar_bg: isDark ? '#171717' : '#F5F5F5',
        loading_screen: {
          backgroundColor: isDark ? '#171717' : '#FFFFFF',
          foregroundColor: isDark ? '#a3a3a3' : '#555555',
        },
        overrides: {
          'paneProperties.background': isDark ? '#171717' : '#FFFFFF',
          'paneProperties.backgroundType': 'solid',
          'paneProperties.vertGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
          'paneProperties.horzGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
          'mainSeriesProperties.style': 1,
          'mainSeriesProperties.candleStyle.upColor': '#22c55e',
          'mainSeriesProperties.candleStyle.downColor': '#ef4444',
          'mainSeriesProperties.candleStyle.borderUpColor': '#22c55e',
          'mainSeriesProperties.candleStyle.borderDownColor': '#ef4444',
          'mainSeriesProperties.candleStyle.wickUpColor': '#22c55e',
          'mainSeriesProperties.candleStyle.wickDownColor': '#ef4444',
          'scalesProperties.textColor': isDark ? '#787B86' : '#555555',
          'scalesProperties.lineColor': isDark ? '#2A2E39' : '#E0E3EB',
        },
      }

      widgetRef.current = new TradingView.widget(widgetOptions)

      widgetRef.current.onChartReady(() => {
        if (cancelled || !widgetRef.current) return
        try {
          const chart = widgetRef.current.activeChart()
          widgetRef.current.applyOverrides({
            'paneProperties.background': isDark ? '#171717' : '#FFFFFF',
            'paneProperties.backgroundType': 'solid',
            'mainSeriesProperties.style': 1,
          })
        } catch {
          /* ignore */
        }

        try {
          widgetRef.current.activeChart().onDataLoaded().subscribe(
            null,
            () => {
              redrawOverlays()
            },
            true,
          )
        } catch {
          /* ignore */
        }

        setTimeout(() => redrawOverlays(), 400)
      })
    }

    void run()

    return () => {
      cancelled = true
      if (widgetRef.current) {
        try {
          const chart = widgetRef.current.activeChart()
          clearOverlayEntities(chart)
        } catch {
          /* ignore */
        }
        try {
          widgetRef.current.remove()
        } catch {
          /* ignore */
        }
        widgetRef.current = null
      }
    }
  }, [ticker, isDark, isFailed, sig, height, clearOverlayEntities, redrawOverlays])

  const overlayEpoch = useMemo(() => {
    const arr = Array.isArray(m?.decision_markers) ? m.decision_markers : []
    return `${sig}|${data.length}|${anchorClose ?? 'na'}|${arr.length}|${isPositive ? 1 : 0}`
  }, [sig, data.length, anchorClose, m?.decision_markers, isPositive])

  useEffect(() => {
    if (!widgetRef.current || isFailed) return
    const t = window.setTimeout(() => redrawOverlays(), 200)
    return () => clearTimeout(t)
  }, [overlayEpoch, redrawOverlays, isFailed])

  if (isFailed) {
    return (
      <div
        className="flex items-center justify-center rounded-md border border-destructive/30 bg-destructive/5 text-xs text-destructive px-2"
        style={{ height }}
      >
        Chart unavailable for failed backtest
      </div>
    )
  }

  if (baseSeries.length === 0) return null

  if (!ticker) {
    return (
      <div
        className="flex items-center justify-center rounded-md border border-border bg-muted/30 text-xs text-muted-foreground px-2"
        style={{ height }}
      >
        Add a ticker to this backtest to load the TradingView chart
      </div>
    )
  }

  return (
    <div className="flex w-full flex-col gap-1" style={{ height }}>
      <p className="text-[10px] text-muted-foreground leading-tight px-0.5">
        Candles: underlying · Thick line: portfolio (equity vs first close) · Dashed: buy &amp; hold · Flags / arrows:
        model trades
      </p>
      <div ref={containerRef} id={containerIdRef.current} className="min-h-0 w-full flex-1 rounded-md border border-border overflow-hidden" />
    </div>
  )
}
