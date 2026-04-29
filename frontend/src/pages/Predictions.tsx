import { useEffect, useState, useRef } from 'react'
import { getPredictions, createPrediction, getTickers, getPredictionProjections, getLatestPriceAnchor, searchUdfSymbols } from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select'
import { Switch } from '../components/ui/switch'
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
  Trash2,
  Eye
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import OHLCChart from '../components/OHLCChart'
import StrategySelector from '../components/StrategySelector'
import { getStrategies } from '../services/strategyApi'
import { resolveProjectionAnchor } from '../utils/projectionAnchor'

export default function Predictions() {
  const [preds, setPreds] = useState<PredictionResponse[]>([])
  const [ticker, setTicker] = useState('AAPL')
  const [horizon, setHorizon] = useState('1d')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [activeTab, setActiveTab] = useState('chart')
  const [availableTickers, setAvailableTickers] = useState<string[]>([])
  const [tickerSearch, setTickerSearch] = useState('')
  const [searchingTickers, setSearchingTickers] = useState(false)

  // Projection controls state
  const [projectionStrategy, setProjectionStrategy] = useState('')
  const [projectionParams, setProjectionParams] = useState<Record<string, any>>({})
  const [projectionHorizon, setProjectionHorizon] = useState(30)

  // Prediction projections state
  const [showPredictionProjections, setShowPredictionProjections] = useState(false)
  const [predictionProjections, setPredictionProjections] = useState<PredictionProjection[]>([])
  const [projectionAnchorWarning, setProjectionAnchorWarning] = useState<string | null>(null)
  const [registeredStrategies, setRegisteredStrategies] = useState<string[]>([])

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

  const fetchTickers = async () => {
    try {
      const data = await getTickers()
      setAvailableTickers(data)
    } catch (e: any) {
      console.error('Failed to fetch tickers:', e)
      // Fallback to hardcoded list if API fails
      setAvailableTickers(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD'])
    }
  }

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

  const searchAndAddTicker = async () => {
    const query = tickerSearch.trim().toUpperCase()
    if (!query) return
    setSearchingTickers(true)
    try {
      const results = await searchUdfSymbols(query, '', 20)
      const foundTickers = results
        .map((item) => (item.ticker || item.symbol || '').toUpperCase())
        .filter(Boolean)
      if (foundTickers.length > 0) {
        mergeTickers(foundTickers)
        setTicker(foundTickers[0])
        setSelectedTicker(foundTickers[0])
      } else {
        // Keep manual-symbol workflows possible even if provider returns no suggestions.
        mergeTickers([query])
        setTicker(query)
        setSelectedTicker(query)
      }
    } catch (e) {
      console.error('Ticker search failed:', e)
    } finally {
      setSearchingTickers(false)
    }
  }

  useEffect(() => {
    fetchPredictions()
    fetchTickers()
    getStrategies()
      .then((items) => setRegisteredStrategies(items.map((item) => item.name)))
      .catch((e) => console.error('Failed to fetch strategies for prediction overlays:', e))
  }, [])

  const submit = async () => {
    if (!ticker.trim()) {
      alert('Please enter a ticker symbol')
      return
    }
    setSubmitting(true)
    try {
      const normalizedTicker = ticker.trim().toUpperCase()
      const data = await createPrediction(normalizedTicker, horizon)
      setPreds(prev => [data, ...prev])
      setSelectedTicker(normalizedTicker)
      setTicker(normalizedTicker)
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
      const strategiesToUse = registeredStrategies.length > 0
        ? registeredStrategies
        : projectionStrategy
          ? [projectionStrategy]
          : undefined
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
      return
    }

    void generatePredictionProjections(selectedTicker)
  }

  useEffect(() => {
    if (!showPredictionProjections) return
    void generatePredictionProjections(selectedTicker)
  }, [selectedTicker, showPredictionProjections, projectionHorizon, projectionStrategy, registeredStrategies])

  const handleStrategyChange = (strategy: string, params: Record<string, any>) => {
    setProjectionStrategy(strategy)
    setProjectionParams(params)
    if (chartRef.current) {
      chartRef.current.setProjectionStrategy(strategy, params, projectionHorizon)
    }
  }

  const handleHorizonChange = (value: number) => {
    setProjectionHorizon(value)
    if (chartRef.current && projectionStrategy) {
      chartRef.current.setProjectionStrategy(projectionStrategy, projectionParams, value)
    }
  }

  const clearProjections = () => {
    setProjectionStrategy('')
    setProjectionParams({})
    if (chartRef.current) {
      chartRef.current.clearProjections()
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
            Select a ticker and time horizon, then click Predict
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="flex w-full sm:flex-1 gap-2">
              <Input
                value={tickerSearch}
                onChange={(e) => setTickerSearch(e.target.value.toUpperCase())}
                placeholder="Search ticker (e.g. GOOGL)"
                className="font-mono"
              />
              <Button
                type="button"
                variant="outline"
                onClick={searchAndAddTicker}
                disabled={searchingTickers || !tickerSearch.trim()}
              >
                {searchingTickers ? 'Searching...' : 'Search'}
              </Button>
            </div>
            <Select value={ticker} onValueChange={setTicker}>
              <SelectTrigger className="w-full sm:flex-1">
                <SelectValue placeholder="Select ticker" />
              </SelectTrigger>
              <SelectContent>
                {availableTickers.map((symbol) => (
                  <SelectItem key={symbol} value={symbol}>
                    {symbol}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={horizon} onValueChange={setHorizon}>
              <SelectTrigger className="w-full sm:w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1d">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    1 Day
                  </div>
                </SelectItem>
                <SelectItem value="3d">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    3 Days
                  </div>
                </SelectItem>
                <SelectItem value="7d">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    7 Days
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
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

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Symbol</label>
                      <Select value={selectedTicker} onValueChange={setSelectedTicker}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {availableTickers.map((symbol) => (
                            <SelectItem key={symbol} value={symbol}>
                              {symbol}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Horizon (days)</label>
                      <Input
                        type="number"
                        value={projectionHorizon}
                        onChange={e => handleHorizonChange(parseInt(e.target.value) || 30)}
                        min="1"
                        max="365"
                        className="font-mono"
                      />
                    </div>

                    <div className="flex items-end">
                      <Button
                        onClick={clearProjections}
                        variant="outline"
                        className="w-full"
                        disabled={!projectionStrategy}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Clear Projections
                      </Button>
                    </div>
                  </div>

                  <Separator />

                  {/* Prediction Projections Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <Eye className="h-4 w-4 text-primary" />
                        <label className="text-sm font-medium">Prediction Projections</label>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Show AI model prediction overlays on the chart
                      </p>
                    </div>
                    <Switch
                      checked={showPredictionProjections}
                      onCheckedChange={handlePredictionProjectionsToggle}
                    />
                  </div>
                  {projectionAnchorWarning && (
                    <p className="text-xs text-amber-500">{projectionAnchorWarning}</p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Chart */}
            <OHLCChart
              ref={chartRef}
              symbol={selectedTicker}
              height="600px"
              strategyName={projectionStrategy}
              params={projectionParams}
              horizon={projectionHorizon}
              showPredictionProjections={showPredictionProjections}
              predictionProjections={predictionProjections}
            />
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
            ) : (
              <div className="grid gap-4">
                {preds.map((p, i) => {
                  const confidenceBadge = getConfidenceBadge(p.confidence)
                  const returnColor = getReturnColor(p.predicted_return)
                  const isPositive = p.predicted_return > 0

                  return (
                    <Card key={i} className="border-muted">
                      <CardContent className="p-6">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                          {/* Left Section */}
                          <div className="space-y-3">
                            <div className="flex items-center gap-3">
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleTickerClick(p.ticker)}
                                  className="text-2xl font-bold font-mono hover:text-primary transition-colors cursor-pointer"
                                >
                                  {p.ticker}
                                </button>
                                <Badge variant="outline" className="font-normal">
                                  {p.horizon}
                                </Badge>
                              </div>
                            </div>

                            <div className="flex items-center gap-2">
                              {isPositive ? (
                                <TrendingUp className="h-4 w-4 text-success" />
                              ) : (
                                <TrendingDown className="h-4 w-4 text-destructive" />
                              )}
                              <span className={`text-lg font-semibold ${returnColor}`}>
                                {(p.predicted_return * 100).toFixed(2)}%
                              </span>
                              <span className="text-sm text-muted-foreground">
                                predicted return
                              </span>
                            </div>
                          </div>

                          <Separator orientation="vertical" className="hidden sm:block h-16" />

                          {/* Right Section */}
                          <div className="flex flex-col items-start sm:items-end gap-2">
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-muted-foreground">Confidence:</span>
                              <Badge variant={confidenceBadge.variant}>
                                {confidenceBadge.label} ({(p.confidence * 100).toFixed(0)}%)
                              </Badge>
                            </div>

                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              {new Date(p.timestamp).toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              model: {p.model_version}
                            </div>
                            {(p.interval_lower !== undefined && p.interval_upper !== undefined) && (
                              <div className="text-xs text-muted-foreground">
                                range: {(p.interval_lower! * 100).toFixed(2)}% to {(p.interval_upper! * 100).toFixed(2)}%
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            )}
          </div>
        </TabsContent>

      </Tabs>
    </div>
  )
}
