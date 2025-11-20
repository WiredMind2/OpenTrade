import React, { useEffect, useState, useMemo } from 'react'
import { getPredictions, createPrediction, getTickers } from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select'
import { PredictionResponse } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import { Skeleton } from '../components/ui/skeleton'
import {
  TrendingUp,
  TrendingDown,
  Sparkles,
  Clock,
  Target,
  Activity
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import OHLCChart from '../components/OHLCChart'

export default function Predictions() {
  const [preds, setPreds] = useState<PredictionResponse[]>([])
  const [ticker, setTicker] = useState('AAPL')
  const [horizon, setHorizon] = useState('1d')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [activeTab, setActiveTab] = useState('predictions')
  const [availableTickers, setAvailableTickers] = useState<string[]>([])

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
      const data = await createPrediction(ticker, horizon)
      setPreds(prev => [data, ...prev])
      setTicker('') // Clear input after successful prediction
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Predictions</h2>
        <p className="text-muted-foreground">
          Generate and track AI-powered stock predictions with interactive visualizations
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="chart">Chart</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions">
          <div className="space-y-6">
            {/* Prediction Form */}
            <Card className="border-muted shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Generate New Prediction
                </CardTitle>
                <CardDescription>
                  Enter a ticker symbol and select a time horizon
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col sm:flex-row gap-3">
                  <div className="flex-1">
                    <Input
                      value={ticker}
                      onChange={e => setTicker(e.target.value.toUpperCase())}
                      placeholder="e.g., AAPL, MSFT, GOOGL"
                      className="font-mono"
                      onKeyDown={e => e.key === 'Enter' && submit()}
                    />
                  </div>
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

            {/* Predictions List */}
            <div>
              <div className="flex items-center justify-between mb-4">
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
                      <Card
                        key={i}
                        className="border-muted"
                      >
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
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="chart">
          <OHLCChart symbol={selectedTicker} height="600px" />
        </TabsContent>
      </Tabs>
    </div>
  )
}
