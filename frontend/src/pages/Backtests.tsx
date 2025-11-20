import React, { useEffect, useState } from 'react'
import { getBacktests, runBacktest } from '../services/api'
import websocketService from '../services/websocket'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { BacktestResult } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import { Skeleton } from '../components/ui/skeleton'
import { 
  BarChart3,
  Play,
  Calendar,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Target
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts'

export default function Backtests() {
  const [backtests, setBacktests] = useState<BacktestResult[]>([])
  const [strategy, setStrategy] = useState('sentiment_momentum')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2023-12-31')
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchBacktests = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getBacktests()
      setBacktests(data)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch backtests')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchBacktests()
  }, [])

  useEffect(() => {
    const handleBacktestStatus = (message: any) => {
      const backtestResult = message.data as BacktestResult

      // Update the backtests list with the new status
      setBacktests(prev => {
        // Find existing backtest by strategy_name and timestamp, or add new one
        const existingIndex = prev.findIndex(b =>
          b.strategy_name === backtestResult.strategy_name &&
          b.timestamp === backtestResult.timestamp
        )

        if (existingIndex >= 0) {
          // Update existing backtest
          const updated = [...prev]
          updated[existingIndex] = backtestResult
          return updated
        } else {
          // Add new backtest at the beginning
          return [backtestResult, ...prev]
        }
      })
    }

    // Register the listener
    const unsubscribe = websocketService.registerListener('backtest_status', handleBacktestStatus)

    // Cleanup on unmount
    return unsubscribe
  }, [])

  const startBacktest = async () => {
    if (!strategy.trim()) {
      alert('Please enter a strategy name')
      return
    }
    setRunning(true)
    try {
      const data = await runBacktest({
        strategy_name: strategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: 100000
      })

      setBacktests(prev => [{ ...data, id: data.metrics?.backtest_id }, ...prev])
    } catch (e: any) {
      alert('Failed to start backtest: ' + (e.message || 'Unknown error'))
    } finally {
      setRunning(false)
    }
  }

  const getReturnColor = (returnVal: number) => {
    if (returnVal > 0) return 'text-success'
    if (returnVal < 0) return 'text-destructive'
    return 'text-muted-foreground'
  }

  const getReturnBadge = (returnVal: number) => {
    if (returnVal > 10) return { variant: 'success' as const, label: 'Excellent' }
    if (returnVal > 0) return { variant: 'default' as const, label: 'Positive' }
    if (returnVal > -10) return { variant: 'warning' as const, label: 'Negative' }
    return { variant: 'destructive' as const, label: 'Poor' }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Backtests</h2>
        <p className="text-muted-foreground">
          Test and analyze your trading strategies
        </p>
      </div>

      {/* Backtest Form */}
      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5 text-primary" />
            Configure Backtest
          </CardTitle>
          <CardDescription>
            Set up a new backtest with your preferred parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Strategy Name</label>
                <Input 
                  value={strategy} 
                  onChange={e => setStrategy(e.target.value)} 
                  placeholder="e.g., sentiment_momentum"
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Start Date
                </label>
                <Input 
                  type="date" 
                  value={startDate} 
                  onChange={e => setStartDate(e.target.value)} 
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  End Date
                </label>
                <Input 
                  type="date" 
                  value={endDate} 
                  onChange={e => setEndDate(e.target.value)} 
                />
              </div>
            </div>

            <Button 
              onClick={startBacktest} 
              disabled={running || !strategy.trim()}
              className="w-full md:w-auto"
              size="lg"
            >
              {running ? (
                <>
                  <Activity className="mr-2 h-4 w-4 animate-spin" />
                  Running Backtest...
                </>
              ) : (
                <>
                  <Target className="mr-2 h-4 w-4" />
                  Start Backtest
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Backtests List */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Backtest Results</h3>
          <Badge variant="secondary">{backtests.length} Total</Badge>
        </div>

        {loading ? (
          <div className="space-y-4">
            {[1, 2].map(i => (
              <Card key={i}>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-32 w-full" />
                    <div className="flex gap-4">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-4 w-32" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : error ? (
          <ErrorMessage message={error} onRetry={fetchBacktests} />
        ) : backtests.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground text-center">
                No backtests yet. Configure and run your first backtest above!
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {backtests.map((b, i) => {
              const returnPercent = b.total_return * 100
              const returnColor = getReturnColor(returnPercent)
              const returnBadge = getReturnBadge(returnPercent)
              const isPositive = returnPercent > 0
              
              return (
                <Card 
                  key={i}
                  className="hover:shadow-lg transition-all border-muted hover:border-primary/50"
                >
                  <CardHeader>
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5 text-primary" />
                          {b.strategy_name}
                        </CardTitle>
                        <CardDescription className="flex items-center gap-2 mt-2">
                          <Calendar className="h-3 w-3" />
                          {b.start_date} → {b.end_date}
                        </CardDescription>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <Badge variant={returnBadge.variant}>
                          {returnBadge.label}
                        </Badge>
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${returnColor} flex items-center gap-1`}>
                            {isPositive ? (
                              <TrendingUp className="h-5 w-5" />
                            ) : (
                              <TrendingDown className="h-5 w-5" />
                            )}
                            {returnPercent.toFixed(2)}%
                          </div>
                          <p className="text-xs text-muted-foreground">Total Return</p>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <Separator />
                  
                  <CardContent className="pt-6">
                    {/* Mini Chart */}
                    <div className="mb-4">
                      <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={b.chart_data}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis 
                            dataKey="day" 
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                          />
                          <YAxis 
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                          />
                          <RechartsTooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '0.5rem'
                            }}
                            formatter={(value: any) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={isPositive ? 'hsl(142, 76%, 36%)' : 'hsl(0, 84.2%, 60.2%)'} 
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <DollarSign className="h-8 w-8 text-primary" />
                        <div>
                          <p className="text-xs text-muted-foreground">Initial Capital</p>
                          <p className="font-semibold">$100,000</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                        <Activity className="h-8 w-8 text-blue-500" />
                        <div>
                          <p className="text-xs text-muted-foreground">Final Value</p>
                          <p className="font-semibold">
                            ${(100000 * (1 + b.total_return)).toFixed(0)}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 col-span-2 md:col-span-1">
                        <Calendar className="h-8 w-8 text-orange-500" />
                        <div>
                          <p className="text-xs text-muted-foreground">Executed</p>
                          <p className="text-sm font-medium">
                            {new Date(b.timestamp).toLocaleDateString()}
                          </p>
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
  )
}
