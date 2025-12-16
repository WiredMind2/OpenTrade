import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHealth } from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { HealthResponse } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import { 
  Activity, 
  TrendingUp, 
  DollarSign, 
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu
} from 'lucide-react'
import { Separator } from '../components/ui/separator'

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHealth()
      setHealth(data)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch health')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} onRetry={fetchHealth} />

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const stats = [
    {
      title: 'System Status',
      value: health?.status || 'Unknown',
      icon: Activity,
      description: 'All systems operational',
      color: 'text-success',
      badge: 'success' as const,
    },
    {
      title: 'Models Loaded',
      value: health?.models_loaded || 0,
      icon: Cpu,
      description: 'ML models ready',
      color: 'text-primary',
      badge: 'default' as const,
    },
    {
      title: 'Uptime',
      value: health?.uptime_seconds ? formatUptime(health.uptime_seconds) : '0h 0m',
      icon: Clock,
      description: 'System uptime',
      color: 'text-blue-500',
      badge: 'secondary' as const,
    },
  ]

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="space-y-1">
        <h2 className="text-xl font-semibold text-tv-text-primary">
          Dashboard
        </h2>
        <p className="text-sm text-tv-text-secondary">
          Welcome back! Here's your trading system overview.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {stats.map((stat, i) => {
          const Icon = stat.icon
          return (
            <Card 
              key={i}
              className="cursor-pointer"
            >
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-tv-text-secondary uppercase">
                  {stat.title}
                </CardTitle>
                <Icon className={`h-4 w-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xl font-semibold text-tv-text-primary">{stat.value}</div>
                    <p className="text-xs text-tv-text-tertiary mt-0.5">
                      {stat.description}
                    </p>
                  </div>
                  <Badge variant={stat.badge} className="ml-2">
                    Active
                  </Badge>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* System Health Card */}
      {health && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                  System Health
                </CardTitle>
                <CardDescription className="mt-1">
                  Detailed system status and performance metrics
                </CardDescription>
              </div>
              <Badge variant="success">
                Operational
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <Separator className="bg-tv-border-secondary" />
            
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 rounded bg-tv-bg-tertiary">
                  <span className="text-xs font-medium text-tv-text-secondary">API Status</span>
                  <Badge variant="success">{health.status}</Badge>
                </div>
                
                <div className="flex items-center justify-between p-2 rounded bg-tv-bg-tertiary">
                  <span className="text-xs font-medium text-tv-text-secondary">Models Active</span>
                  <Badge variant="default">{health.models_loaded}</Badge>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 rounded bg-tv-bg-tertiary">
                  <span className="text-xs font-medium text-tv-text-secondary">Uptime</span>
                  <span className="text-xs font-mono text-tv-text-primary">{health.uptime_seconds.toFixed(0)}s</span>
                </div>
                
                <div className="flex items-center justify-between p-2 rounded bg-tv-bg-tertiary">
                  <span className="text-xs font-medium text-tv-text-secondary">Last Check</span>
                  <span className="text-xs text-tv-text-tertiary">Just now</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Common tasks and shortcuts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-4">
            <Link to="/predictions" className="flex items-center gap-2 p-3 rounded border border-tv-border-secondary hover:border-primary hover:bg-tv-bg-hover transition-tv group">
              <TrendingUp className="h-6 w-6 text-primary group-hover:scale-110 transition-transform" />
              <div>
                <p className="font-medium text-xs text-tv-text-primary">New Prediction</p>
                <p className="text-xs text-tv-text-tertiary">Generate forecast</p>
              </div>
              <span className="ml-auto text-xs text-tv-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity">→</span>
            </Link>
            
            <Link to="/backtests" className="flex items-center gap-2 p-3 rounded border border-tv-border-secondary hover:border-primary hover:bg-tv-bg-hover transition-tv group">
              <BarChart3 className="h-6 w-6 text-tv-blue group-hover:scale-110 transition-transform" />
              <div>
                <p className="font-medium text-xs text-tv-text-primary">Run Backtest</p>
                <p className="text-xs text-tv-text-tertiary">Test strategy</p>
              </div>
              <span className="ml-auto text-xs text-tv-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity">→</span>
            </Link>
            
            <Link to="/portfolio" className="flex items-center gap-2 p-3 rounded border border-tv-border-secondary hover:border-primary hover:bg-tv-bg-hover transition-tv group">
              <DollarSign className="h-6 w-6 text-success group-hover:scale-110 transition-transform" />
              <div>
                <p className="font-medium text-xs text-tv-text-primary">View Portfolio</p>
                <p className="text-xs text-tv-text-tertiary">Check positions</p>
              </div>
              <span className="ml-auto text-xs text-tv-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity">→</span>
            </Link>
            
            <Link to="/models" className="flex items-center gap-2 p-3 rounded border border-tv-border-secondary hover:border-primary hover:bg-tv-bg-hover transition-tv group">
              <Activity className="h-6 w-6 text-tv-orange group-hover:scale-110 transition-transform" />
              <div>
                <p className="font-medium text-xs text-tv-text-primary">Market Data</p>
                <p className="text-xs text-tv-text-tertiary">Live updates</p>
              </div>
              <span className="ml-auto text-xs text-tv-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity">→</span>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
