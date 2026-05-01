import { useEffect, useState, useRef } from 'react'
import { getPredictions, createPrediction, getTickers, getPredictionProjections, getLatestPriceAnchor, searchUdfSymbols } from '../services/api'
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
  Sparkles,
  Clock,
  Target,
  Activity,
  Eye
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import OHLCChart from '../components/OHLCChart'
import StrategySelector from '../components/StrategySelector'
import { resolveProjectionAnchor } from '../utils/projectionAnchor'
import { NewsSidebar } from '../components/NewsSidebar'

export default function Predictions() {
  const [preds, setPreds] = useState<PredictionResponse[]>([])
  const [ticker, setTicker] = useState('AAPL')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [activeTab, setActiveTab] = useState('chart')
  const COMMON_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'SPY']

  const [availableTickers, setAvailableTickers] = useState<string[]>(COMMON_TICKERS)
  const [tickerSearch, setTickerSearch] = useState('')
  const [searchingTickers, setSearchingTickers] = useState(false)
  const [showTickerSuggestions, setShowTickerSuggestions] = useState(true)
  const [highlightedIndex, setHighlightedIndex] = useState(-1)
  const suggestionsRef = useRef<HTMLDivElement>(null)

  // Projection controls state
  const [projectionStrategy, setProjectionStrategy] = useState('')
  const [projectionParams, setProjectionParams] = useState<Record<string, any>>({})
  const [projectionHorizon, setProjectionHorizon] = useState(30)

  // Prediction projections state
  const [showPredictionProjections, setShowPredictionProjections] = useState(false)
  const [predictionProjections, setPredictionProjections] = useState<PredictionProjection[]>([])
  const [projectionAnchorWarning, setProjectionAnchorWarning] = useState<string | null>(null)

  const chartRef = useRef<any>(null)
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  const fetchTickers = async () => {
    try {
      const data: string[] = await getTickers()
      if (data?.length) mergeTickers(data)
    } catch (e: any) {
      console.error('Failed to fetch tickers:', e)
    }
  }

  const filterTickers = (val: string, tickers: string[]) => {
    if (!val) return tickers
    const upper = val.toUpperCase()
    const startsWith = tickers.filter(t => t.startsWith(upper))
    const contains = tickers.filter(t => !t.startsWith(upper) && t.includes(upper))
    return [...startsWith, ...contains]
  }

  const shownTickers = filterTickers(tickerSearch, availableTickers)

  const mergeTickers = (symbols: string[]) => {
    setAvailableTickers((prev) => {
      const merged = new Set<string>(prev)
      symbols.forEach((s) => {
        if (s?.trim()) {
          merged.add(s.trim().toUpperCase())
        }
      })
      return Array.from(merged).sort()
    })
  }

  const searchAndAddTicker = async (queryOverride?: string) => {
    const query = (queryOverride ?? tickerSearch).trim().toUpperCase()
    if (!query) return
    setSearchingTickers(true)
    try {
      const results = await searchUdfSymbols(query, '', 30)
      const foundTickers = results
        .map((item) => (item.ticker || item.symbol || '').toUpperCase())
        .filter(Boolean)
      mergeTickers(foundTickers.length > 0 ? foundTickers : [query])
    } catch (e) {
      console.error('Ticker search failed:', e)
    } finally {
      setSearchingTickers(false)
    }
  }

  const handleTickerSelect = (symbol: string) => {
    const normalized = symbol.trim().toUpperCase()
    if (!normalized) return
    mergeTickers([normalized])
    setTickerSearch(normalized)
    setTicker(normalized)
    setSelectedTicker(normalized)
    setShowTickerSuggestions(false)
  }

  useEffect(() => {
    fetchPredictions()
    fetchTickers()
  }, [])

  const submit = async () => {
    if (!ticker.trim()) {
      alert('Please enter a ticker symbol')
      return
    }
    setSubmitting(true)
    try {
      const normalizedTicker = ticker.trim().toUpperCase()
      const results = await Promise.all(
        ['1d', '3d', '7d'].map(h =>
          createPrediction(
            normalizedTicker,
            h,
            projectionStrategy || undefined,
            projectionStrategy ? projectionParams : undefined
          )
        )
      )
      setPreds(prev => [...results, ...prev])
      setSelectedTicker(normalizedTicker)
      setTicker(normalizedTicker)
      setActiveTab('predictions')
    } catch (e: any) {
      alert('Failed to make prediction: ' + (e.message || 'Unknown error'))
    } finally {
      setSubmitting(false)
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
          Generate ad-hoc forecasts for individual stocks using trained models
        </p>
      </div>

      {/* Prediction Form — always visible */}
      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Generate New Prediction
          </CardTitle>
          <CardDescription>
            Search a ticker, then click Predict to generate 1-day, 3-day and 7-day forecasts
          </CardDescription>
          {projectionStrategy && (
            <p className="text-xs text-muted-foreground">
              Active prediction strategy: <span className="font-mono">{projectionStrategy}</span>
            </p>
          )}
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative w-full sm:flex-1">
              <Input
                value={tickerSearch}
                onChange={(e) => {
                  const val = e.target.value.toUpperCase()
                  setTickerSearch(val)
                  setHighlightedIndex(-1)
                  setShowTickerSuggestions(true)
                  if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
                  if (val.length >= 1) {
                    searchDebounceRef.current = setTimeout(() => {
                      void searchAndAddTicker(val)
                    }, 400)
                  }
                }}
                onFocus={() => {
                  setShowTickerSuggestions(true)
                  setHighlightedIndex(-1)
                }}
                onKeyDown={(e) => {
                  if (!showTickerSuggestions) return
                  if (e.key === 'ArrowDown') {
                    e.preventDefault()
                    setHighlightedIndex(i => {
                      const next = Math.min(i + 1, shownTickers.length - 1)
                      suggestionsRef.current?.children[next]?.scrollIntoView({ block: 'nearest' })
                      return next
                    })
                  } else if (e.key === 'ArrowUp') {
                    e.preventDefault()
                    setHighlightedIndex(i => {
                      const next = Math.max(i - 1, 0)
                      suggestionsRef.current?.children[next]?.scrollIntoView({ block: 'nearest' })
                      return next
                    })
                  } else if (e.key === 'Enter') {
                    e.preventDefault()
                    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
                    if (highlightedIndex >= 0 && shownTickers[highlightedIndex]) {
                      handleTickerSelect(shownTickers[highlightedIndex])
                    } else if (tickerSearch.trim()) {
                      void searchAndAddTicker()
                    }
                  } else if (e.key === 'Escape') {
                    setShowTickerSuggestions(false)
                  }
                }}
                placeholder="Rechercher un ticker (ex: AAPL, GOOGL…)"
                className="font-mono"
              />
              {showTickerSuggestions && (
                <div className="absolute z-20 mt-1 w-full rounded-md border bg-popover shadow-lg">
                  <div className="flex items-center justify-between px-3 py-1.5 border-b text-xs text-muted-foreground">
                    <span>
                      {shownTickers.length === 0
                        ? 'Aucun résultat'
                        : `${shownTickers.length} résultat${shownTickers.length > 1 ? 's' : ''}`}
                    </span>
                    {searchingTickers && <span className="text-primary animate-pulse">Recherche…</span>}
                  </div>
                  <div ref={suggestionsRef} className="max-h-72 overflow-y-auto py-1">
                    {shownTickers.map((symbol, idx) => (
                      <button
                        key={symbol}
                        type="button"
                        className={`w-full px-3 py-2 text-left text-sm font-mono transition-colors ${
                          idx === highlightedIndex ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/60'
                        }`}
                        onMouseEnter={() => setHighlightedIndex(idx)}
                        onMouseDown={(e) => {
                          e.preventDefault()
                          handleTickerSelect(symbol)
                        }}
                      >
                        {tickerSearch
                          ? (() => {
                              const upper = tickerSearch.toUpperCase()
                              const start = symbol.indexOf(upper)
                              if (start === -1) return symbol
                              return (
                                <>
                                  {symbol.slice(0, start)}
                                  <span className="font-bold text-primary">{symbol.slice(start, start + upper.length)}</span>
                                  {symbol.slice(start + upper.length)}
                                </>
                              )
                            })()
                          : symbol}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <Button
              onClick={submit}
              disabled={submitting || !ticker.trim()}
              className="w-full sm:w-auto min-w-[120px]"
              size="default"
            >
              {submitting ? (
                <>
                  <Activity className="mr-2 h-4 w-4 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Target className="mr-2 h-4 w-4" />
                  Predict
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

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
                  Configure strategy projections and prediction overlays for the chart
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
                      setTickerSearch(normalized)
                      setTicker(normalized)
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
                  <p className="text-muted-foreground text-center">
                    No predictions yet. Generate your first prediction above!
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
