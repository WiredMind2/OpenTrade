import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Separator } from '../components/ui/separator'
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Zap
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { MonteCarloResult } from '../types'

interface MonteCarloResultsProps {
  results: MonteCarloResult[]
}

export default function MonteCarloResults({ results }: MonteCarloResultsProps) {
  if (results.length === 0) return null

  const latestResult = results[0]

  // Generate distribution data for visualization
  const generateDistributionData = () => {
    if (!latestResult.simulations || latestResult.simulations.length === 0) return []

    const returns = latestResult.simulations.map(s => s.total_return * 100)
    const min = Math.min(...returns)
    const max = Math.max(...returns)
    const bins = 20
    const binSize = (max - min) / bins

    const distribution = []
    for (let i = 0; i < bins; i++) {
      const binStart = min + i * binSize
      const binEnd = min + (i + 1) * binSize
      const count = returns.filter(r => r >= binStart && r < binEnd).length
      distribution.push({
        bin: `${binStart.toFixed(1)}%`,
        count,
        percentage: (count / returns.length) * 100
      })
    }

    return distribution
  }

  const distributionData = generateDistributionData()

  const formatCurrency = (value: number) => `$${(value / 1000).toFixed(0)}k`
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`

  return (
    <div className="space-y-4">
      {results.map((result, index) => (
        <Card key={index} className="border-primary/20 shadow-lg">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Zap className="h-6 w-6 text-primary" />
                <div>
                  <CardTitle className="flex items-center gap-2">
                    Monte Carlo Results: {result.strategy_name}
                    <Badge variant="secondary" className="bg-primary/10">
                      {result.num_simulations} Simulations
                    </Badge>
                  </CardTitle>
                </div>
              </div>
              <Badge variant="outline">
                {new Date(result.completed_at).toLocaleDateString()}
              </Badge>
            </div>
          </CardHeader>

          <Separator />

          <CardContent className="pt-6">
            {/* Key Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <TrendingUp className="h-8 w-8 text-green-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Mean Return</p>
                  <p className="text-lg font-semibold text-green-600">
                    {formatPercent(result.mean_total_return)}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <Activity className="h-8 w-8 text-blue-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Confidence Interval</p>
                  <p className="text-sm font-semibold">
                    {formatPercent(result.confidence_lower_return)} to {formatPercent(result.confidence_upper_return)}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <Target className="h-8 w-8 text-orange-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Win Probability</p>
                  <p className="text-lg font-semibold">
                    {formatPercent(result.probability_positive_return)}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <BarChart3 className="h-8 w-8 text-purple-500" />
                <div>
                  <p className="text-xs text-muted-foreground">Best/Worst Case</p>
                  <p className="text-xs font-semibold text-green-600">
                    Best: {formatPercent(result.best_case_return)}
                  </p>
                  <p className="text-xs font-semibold text-red-600">
                    Worst: {formatPercent(result.worst_case_return)}
                  </p>
                </div>
              </div>
            </div>

            {/* Distribution Chart */}
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3">Return Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="bin"
                    className="text-xs"
                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    className="text-xs"
                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                    label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
                  />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '0.5rem'
                    }}
                    formatter={(value: any) => [`${value} simulations`, 'Count']}
                  />
                  <Bar
                    dataKey="count"
                    fill="hsl(var(--primary))"
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Risk Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="border-muted">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Value at Risk (95%)</p>
                      <p className="text-lg font-semibold text-red-600">
                        -{formatPercent(Math.abs(result.confidence_lower_return - result.mean_total_return))}
                      </p>
                    </div>
                    <Activity className="h-5 w-5 text-red-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-muted">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Expected Shortfall</p>
                      <p className="text-lg font-semibold text-red-600">
                        -{formatPercent(Math.abs(result.worst_case_return))}
                      </p>
                    </div>
                    <TrendingDown className="h-5 w-5 text-red-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-muted">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Standard Deviation</p>
                      <p className="text-lg font-semibold">
                        {formatPercent(result.std_total_return)}
                      </p>
                    </div>
                    <BarChart3 className="h-5 w-5 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}