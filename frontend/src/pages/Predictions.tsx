import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  getTickerPriceOnDate,
  getUdfQuotes,
  getUdfSymbolInfo,
  listSavedModels,
  signalsSavedModelsBatch,
  type PriceDailyRow,
  type SavedModel,
  type SavedModelSignal,
  type TraderStyle,
  type UdfQuote,
  type UdfSymbolInfo,
} from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Activity, BarChart3, Target } from 'lucide-react'
import OHLCChart from '../components/OHLCChart'
import { NewsSidebar } from '../components/NewsSidebar'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'
import { TradePlanBuilderPanel, type TradePlanObjective } from '../components/TradePlanBuilderPanel'

const popularSymbols = [
  { symbol: 'AAPL', label: 'Apple' },
  { symbol: 'MSFT', label: 'Microsoft' },
  { symbol: 'NVDA', label: 'Nvidia' },
  { symbol: 'TSLA', label: 'Tesla' },
  { symbol: 'SPY', label: 'S&P 500 ETF' },
  { symbol: 'QQQ', label: 'Nasdaq ETF' },
  { symbol: 'BTC-USD', label: 'Bitcoin' },
  { symbol: 'ETH-USD', label: 'Ethereum' },
  { symbol: 'XAUUSD=X', label: 'Gold spot' },
  { symbol: 'GC=F', label: 'Gold futures' },
  { symbol: 'CL=F', label: 'Crude oil' },
]

/** Matches OHLC (TradingView) height so the news column aligns and scrolls inside the same visual block. */
const PREDICTIONS_CHART_HEIGHT = '600px'

function formatQuotePrice(value: number | null | undefined, currency = 'USD') {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  const decimals = Math.abs(value) >= 100 ? 2 : 4
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value)
  } catch {
    return `${value.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })} ${currency}`
  }
}

function formatSigned(value: number | undefined, digits = 2) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`
}

function degradeVariant(status: string | undefined): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'degraded') return 'destructive'
  if (status === 'watch') return 'secondary'
  return 'outline'
}

function signalActionBadgeVariant(action: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (action === 'buy') return 'default'
  if (action === 'sell') return 'destructive'
  return 'secondary'
}

function signalActionLabel(action: string): string {
  if (action === 'buy') return 'Buy'
  if (action === 'sell') return 'Sell'
  return 'Hold'
}

export default function Predictions() {
  const [searchParams] = useSearchParams()
  const [selectedTicker, setSelectedTicker] = useState(() => getStoredTicker())
  const [symbolInfo, setSymbolInfo] = useState<UdfSymbolInfo | null>(null)
  const [quote, setQuote] = useState<UdfQuote | null>(null)
  const [priceOnDate, setPriceOnDate] = useState<PriceDailyRow | null>(null)
  /** Anchor date for header price, saved-model signals, and trade plan (single control lives in plan builder). */
  const [priceDate, setPriceDate] = useState('')

  const [objective, setObjective] = useState<TradePlanObjective>('balanced')
  const [topN, setTopN] = useState(5)
  const [savedModels, setSavedModels] = useState<SavedModel[]>([])
  const [topSignals, setTopSignals] = useState<SavedModelSignal[]>([])
  const [excludedIds, setExcludedIds] = useState<Set<number>>(() => new Set())
  const [pinnedIds, setPinnedIds] = useState<Set<number>>(() => new Set())
  const [modelsLoading, setModelsLoading] = useState(false)
  const [signalLoading, setSignalLoading] = useState(false)
  const [signalError, setSignalError] = useState<string | null>(null)

  const [traderStyle, setTraderStyle] = useState<TraderStyle>('auto')
  const [accountSize, setAccountSize] = useState(10000)
  const [riskPercent, setRiskPercent] = useState(1)
  const [selectedStrategy, setSelectedStrategy] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(false)

  const excludeKey = useMemo(() => [...excludedIds].sort((a, b) => a - b).join(','), [excludedIds])
  const pinKey = useMemo(() => [...pinnedIds].sort((a, b) => a - b).join(','), [pinnedIds])

  const chartStrategyParams = useMemo(() => ({}), [])
  const sym = selectedTicker.trim().toUpperCase()

  useEffect(() => {
    const t = searchParams.get('ticker')
    if (t) {
      const normalized = rememberTicker(t)
      if (normalized) setSelectedTicker(normalized)
    }
    if (searchParams.has('date')) {
      setPriceDate(searchParams.get('date') ?? '')
    }
  }, [searchParams])

  useEffect(() => {
    setExcludedIds(new Set())
    setPinnedIds(new Set())
  }, [sym])

  useEffect(() => {
    if (!sym) return
    let cancelled = false
    const loadHeader = async () => {
      try {
        const [info, quotes, asOfPrice] = await Promise.all([
          getUdfSymbolInfo(sym),
          getUdfQuotes([sym]),
          getTickerPriceOnDate(sym, priceDate.trim() || undefined),
        ])
        if (cancelled) return
        setSymbolInfo(info)
        setQuote(quotes[0] ?? null)
        setPriceOnDate(asOfPrice)
      } catch {
        if (!cancelled) {
          setSymbolInfo(null)
          setQuote(null)
          setPriceOnDate(null)
        }
      }
    }
    void loadHeader()
    const interval = window.setInterval(() => void loadHeader(), priceDate.trim() ? 60000 : 15000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [sym, priceDate])

  useEffect(() => {
    if (!sym) {
      setSavedModels([])
      return
    }
    let cancelled = false
    setModelsLoading(true)
    setSavedModels([])
    void listSavedModels({ ticker: sym, active: true })
      .then((rows) => {
        if (!cancelled) setSavedModels(rows)
      })
      .catch(() => {
        if (!cancelled) setSavedModels([])
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [sym])

  const runTopSignals = useCallback(async () => {
    if (!sym) {
      setTopSignals([])
      return
    }
    setSignalLoading(true)
    setSignalError(null)
    try {
      const trimmedAsOf = priceDate.trim()
      const rows = await signalsSavedModelsBatch({
        ticker: sym,
        objective,
        top_n: topN,
        include_model_ids: pinKey ? pinKey.split(',').map((s) => parseInt(s, 10)) : [],
        exclude_model_ids: excludeKey ? excludeKey.split(',').map((s) => parseInt(s, 10)) : [],
        ...(trimmedAsOf ? { as_of_date: trimmedAsOf } : {}),
      })
      setTopSignals(rows)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Signal request failed'
      setSignalError(msg)
      setTopSignals([])
    } finally {
      setSignalLoading(false)
    }
  }, [sym, priceDate, objective, topN, excludeKey, pinKey])

  useEffect(() => {
    if (!sym || savedModels.length === 0) {
      setTopSignals([])
      return
    }
    const t = window.setTimeout(() => {
      void runTopSignals()
    }, 500)
    return () => window.clearTimeout(t)
  }, [sym, savedModels.length, priceDate, objective, topN, excludeKey, pinKey, runTopSignals])

  const toggleExclude = (id: number) => {
    setExcludedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const togglePin = (id: number) => {
    setPinnedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Predictions</h2>
        <p className="text-muted-foreground">
          Chart and news at the top, then build a trade plan. Saved models (ranked with the same objective as in the plan
          builder) produce buy / sell / hold signals at the anchor date—set in the plan builder—or latest when empty. Save
          models from Strategy Performance or Backtests.
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex flex-col lg:flex-row gap-4 lg:items-stretch">
          <div className="flex-1 min-w-0">
            <OHLCChart
              symbol={selectedTicker}
              height={PREDICTIONS_CHART_HEIGHT}
              strategyName=""
              params={chartStrategyParams}
              horizon={30}
              onSymbolChange={(symbol) => {
                const normalized = rememberTicker(symbol)
                if (normalized && normalized !== selectedTicker) {
                  setSelectedTicker(normalized)
                }
              }}
            />
          </div>
          <div
            className="w-full lg:w-80 xl:w-96 flex-shrink-0 min-h-0 flex flex-col"
            style={{ height: PREDICTIONS_CHART_HEIGHT }}
          >
            <NewsSidebar ticker={selectedTicker} />
          </div>
        </div>

        <TradePlanBuilderPanel
          ticker={selectedTicker}
          priceDate={priceDate}
          onPriceDateChange={setPriceDate}
          objective={objective}
          onObjectiveChange={setObjective}
          traderStyle={traderStyle}
          onTraderStyleChange={setTraderStyle}
          accountSize={accountSize}
          onAccountSizeChange={setAccountSize}
          riskPercent={riskPercent}
          onRiskPercentChange={setRiskPercent}
          selectedStrategy={selectedStrategy}
          onSelectedStrategyChange={setSelectedStrategy}
          autoRefresh={autoRefresh}
          onAutoRefreshChange={setAutoRefresh}
        />

        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Saved models for <span className="font-mono">{sym || '—'}</span>
            </CardTitle>
            <CardDescription>
              Ranking objective is set in the plan builder above. Exclude models from the candidate pool or pin
              favourites. Top N picks how many ranked models receive a signal at the anchor date (or latest when empty).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Top N signals</label>
                <Input
                  type="number"
                  min={1}
                  max={25}
                  value={topN}
                  onChange={(e) => setTopN(Math.min(25, Math.max(1, parseInt(e.target.value, 10) || 1)))}
                />
              </div>
              <div className="space-y-2 rounded-md border bg-muted/20 p-3 text-sm">
                <p className="font-medium">Current ranking</p>
                <p className="text-muted-foreground capitalize">
                  Objective: <span className="font-mono text-foreground">{objective}</span>
                </p>
                <p className="text-muted-foreground">
                  Anchor date:{' '}
                  <span className="font-mono text-foreground">{priceDate.trim() || 'latest bar'}</span>
                </p>
              </div>
            </div>

            {modelsLoading && (
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Activity className="h-3 w-3 animate-spin" />
                Loading saved models…
              </p>
            )}
            {!modelsLoading && sym && savedModels.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No active saved models for this ticker yet. Save a strategy configuration from Strategy Performance /
                Backtests or the API.
              </p>
            )}

            {savedModels.length > 0 && (
              <div className="overflow-x-auto rounded-md border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/40 text-left">
                      <th className="p-2 font-medium">Model</th>
                      <th className="p-2 font-medium">Strategy</th>
                      <th className="p-2 font-medium">Exclude</th>
                      <th className="p-2 font-medium">Pin</th>
                    </tr>
                  </thead>
                  <tbody>
                    {savedModels.map((m) => (
                      <tr key={m.id} className="border-b last:border-0">
                        <td className="p-2">
                          <span className="font-medium">{m.name}</span>
                          <span className="text-muted-foreground ml-1">#{m.id}</span>
                        </td>
                        <td className="p-2 font-mono text-xs">{m.strategy_name}</td>
                        <td className="p-2">
                          <input
                            type="checkbox"
                            checked={excludedIds.has(m.id)}
                            onChange={() => toggleExclude(m.id)}
                            aria-label={`Exclude ${m.name}`}
                          />
                        </td>
                        <td className="p-2">
                          <input
                            type="checkbox"
                            checked={pinnedIds.has(m.id)}
                            onChange={() => togglePin(m.id)}
                            aria-label={`Pin ${m.name}`}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {signalLoading && (
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Activity className="h-3 w-3 animate-spin" />
                Computing signals…
              </p>
            )}
            {signalError && <p className="text-sm text-destructive">{signalError}</p>}

            {topSignals.length > 0 && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-baseline gap-2">
                  <p className="text-sm font-medium">Today&apos;s stance (anchor bar)</p>
                  <span className="text-xs text-muted-foreground">
                    As of {topSignals[0]?.as_of} · last close {(topSignals[0]?.last_price ?? 0).toFixed(4)}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {topSignals.map((s) => (
                    <Badge
                      key={s.model_id}
                      variant={signalActionBadgeVariant(s.action)}
                      className="gap-1 font-medium"
                      title={s.reason}
                    >
                      <Target className="h-3 w-3" />
                      {s.name || `Model ${s.model_id}`}: {signalActionLabel(s.action)}
                    </Badge>
                  ))}
                </div>

                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/40 text-left">
                        <th className="p-2 font-medium">Model</th>
                        <th className="p-2 font-medium">Signal</th>
                        <th className="text-right p-2 font-medium">Target %</th>
                        <th className="text-right p-2 font-medium">Conf.</th>
                        <th className="p-2 font-medium">Reason</th>
                        <th className="p-2 font-medium">Drift</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topSignals.map((s) => (
                        <tr key={s.model_id} className="border-b last:border-0">
                          <td className="p-2">
                            <span className="font-medium">{s.name || `#${s.model_id}`}</span>
                            <div className="text-xs font-mono text-muted-foreground">{s.strategy_name}</div>
                          </td>
                          <td className="p-2">
                            <Badge variant={signalActionBadgeVariant(s.action)}>{signalActionLabel(s.action)}</Badge>
                          </td>
                          <td className="p-2 text-right font-mono">{(s.target_pct * 100).toFixed(2)}%</td>
                          <td className="p-2 text-right font-mono">
                            {(s.confidence <= 1 ? s.confidence * 100 : s.confidence).toFixed(0)}
                            {s.confidence <= 1 ? '%' : ''}
                          </td>
                          <td className="p-2 max-w-xs">
                            <span className="text-xs break-words">{s.reason}</span>
                            {s.error && <span className="block text-xs text-destructive mt-1">{s.error}</span>}
                          </td>
                          <td className="p-2">
                            <Badge variant={degradeVariant(s.degrade_status)} className="text-xs">
                              {s.degrade_status}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle>Ticker</CardTitle>
            <CardDescription>
              Set the symbol for the chart, news, saved-model signals, and trade plan. Pick a preset or type a ticker.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2 max-w-md">
              <label className="text-sm font-medium" htmlFor="predictions-ticker-input">
                Symbol
              </label>
              <Input
                id="predictions-ticker-input"
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value.toUpperCase())}
                onBlur={() => {
                  const normalized = rememberTicker(selectedTicker)
                  setSelectedTicker(normalized)
                }}
                placeholder="e.g. AAPL"
              />
            </div>

            <div className="flex flex-col gap-3 border rounded-md px-3 py-3 md:flex-row md:items-center md:justify-between bg-muted/10">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-lg font-semibold leading-tight">{sym || 'Select ticker'}</h3>
                  <Badge variant="outline">{symbolInfo?.exchange || 'MARKET'}</Badge>
                  <Badge variant="secondary">{symbolInfo?.currency_code || 'USD'}</Badge>
                </div>
                <p className="truncate text-sm text-muted-foreground">
                  {symbolInfo?.description || 'Loading symbol details'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-semibold tabular-nums">
                  {formatQuotePrice(priceOnDate?.close ?? quote?.v?.lp, symbolInfo?.currency_code || 'USD')}
                </p>
                <p className="text-xs text-muted-foreground">
                  Price date: {priceOnDate?.date || 'latest available'}
                </p>
                {!priceDate.trim() && (
                  <p
                    className={`text-sm tabular-nums ${
                      Number(quote?.v?.ch ?? 0) >= 0 ? 'text-success' : 'text-destructive'
                    }`}
                  >
                    {formatSigned(quote?.v?.ch)} ({formatSigned(quote?.v?.chp)}%)
                  </p>
                )}
              </div>
            </div>

            <div className="flex gap-2 overflow-x-auto pb-1">
              {popularSymbols.map((item) => (
                <Button
                  key={item.symbol}
                  type="button"
                  size="sm"
                  variant={sym === item.symbol ? 'default' : 'outline'}
                  className="shrink-0"
                  title={item.label}
                  onClick={() => {
                    const normalized = rememberTicker(item.symbol)
                    setSelectedTicker(normalized)
                  }}
                >
                  {item.symbol}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
