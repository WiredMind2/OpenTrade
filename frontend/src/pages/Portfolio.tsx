import React, { useEffect, useState } from 'react'
import { getPortfolio } from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { PortfolioResponse } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import { 
  Wallet,
  DollarSign,
  TrendingUp,
  PieChart,
  Briefcase,
  Target
} from 'lucide-react'
import { Separator } from '../components/ui/separator'
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartsTooltip } from 'recharts'

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPortfolio = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getPortfolio()
      setPortfolio(data)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch portfolio')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPortfolio()
    const interval = setInterval(fetchPortfolio, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} onRetry={fetchPortfolio} />

  if (!portfolio) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Wallet className="h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-muted-foreground text-center">
            No portfolio data available
          </p>
        </CardContent>
      </Card>
    )
  }

  const investedValue = portfolio.total_value - portfolio.cash
  const colors = ['hsl(217, 91%, 60%)', 'hsl(142, 76%, 36%)', 'hsl(38, 92%, 50%)', 'hsl(0, 84%, 60%)', 'hsl(280, 67%, 60%)']
  
  const allocationData = [
    { name: 'Cash', value: portfolio.cash, color: 'hsl(215, 20%, 65%)' },
    ...portfolio.positions.map((p: any, i: number) => ({
      name: p.ticker,
      value: typeof p.value === 'number' ? p.value : parseFloat(p.value) || 0,
      color: colors[i % colors.length]
    }))
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Portfolio</h2>
        <p className="text-muted-foreground">
          Monitor your current positions and allocation
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-muted shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Value</CardTitle>
            <Wallet className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Portfolio valuation
            </p>
          </CardContent>
        </Card>

        <Card className="border-muted shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cash Balance</CardTitle>
            <DollarSign className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${portfolio.cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Available to invest
            </p>
          </CardContent>
        </Card>

        <Card className="border-muted shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Exposure</CardTitle>
            <Target className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(portfolio.exposure * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Market exposure
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Allocation Chart */}
        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5 text-primary" />
              Asset Allocation
            </CardTitle>
            <CardDescription>
              Portfolio distribution by holdings
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RechartsPie>
                <Pie
                  data={allocationData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {allocationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '0.5rem'
                  }}
                  formatter={(value: any) => [`$${value.toLocaleString()}`, 'Value']}
                />
                <Legend />
              </RechartsPie>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Holdings Summary */}
        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Briefcase className="h-5 w-5 text-primary" />
              Holdings Summary
            </CardTitle>
            <CardDescription>
              Key portfolio metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground mb-1">Positions</p>
                <p className="text-2xl font-bold">{portfolio.positions.length}</p>
              </div>
              
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground mb-1">Invested</p>
                <p className="text-2xl font-bold">
                  ${investedValue.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </p>
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <h4 className="text-sm font-medium">Allocation Breakdown</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 rounded bg-muted/30">
                  <span className="text-sm">Cash</span>
                  <Badge variant="secondary">
                    {((portfolio.cash / portfolio.total_value) * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-muted/30">
                  <span className="text-sm">Equity</span>
                  <Badge variant="default">
                    {((investedValue / portfolio.total_value) * 100).toFixed(1)}%
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Positions Table */}
      <Card className="border-muted shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Current Positions
          </CardTitle>
          <CardDescription>
            Detailed view of your holdings
          </CardDescription>
        </CardHeader>
        <CardContent>
          {portfolio.positions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No positions currently held
            </div>
          ) : (
            <div className="space-y-3">
              {portfolio.positions.map((position: any, i: number) => {
                const value = typeof position.value === 'number' ? position.value : parseFloat(position.value) || 0
                const allocation = (value / portfolio.total_value) * 100
                
                return (
                  <div 
                    key={i}
                    className="flex items-center justify-between p-4 rounded-lg border border-muted hover:border-primary/50 hover:bg-accent transition-all"
                  >
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <span className="text-sm font-bold text-primary">
                          {position.ticker.substring(0, 2)}
                        </span>
                      </div>
                      <div>
                        <p className="font-semibold">{position.ticker}</p>
                        <p className="text-sm text-muted-foreground">
                          {position.quantity} shares
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="font-semibold">
                          ${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          ${(value / position.quantity).toFixed(2)}/share
                        </p>
                      </div>
                      
                      <Badge variant="outline" className="min-w-[60px] justify-center">
                        {allocation.toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
