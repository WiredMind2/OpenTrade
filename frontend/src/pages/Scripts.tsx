import { useEffect, useState, useRef } from 'react'
import {
  listScriptExecutions,
  runPipeline,
  generateMAPredictions,
} from '../services/api'
import websocketService from '../services/websocket'
import { ScriptStatusMessage, PipelineStatusMessage } from '../types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Switch } from '../components/ui/switch'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import {
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  Zap
} from 'lucide-react'
interface ScriptExecution {
  execution_id: string
  script_name: string
  status: string
  start_time: string
  end_time?: string
  duration_seconds?: number
}

interface PipelineExecution extends ScriptExecution {
  current_step?: string
  completed_steps: string[]
  failed_steps: string[]
}

export default function Scripts() {
  const [executions, setExecutions] = useState<ScriptExecution[]>([])
  const [pipelineExecution, setPipelineExecution] = useState<PipelineExecution | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const scriptCleanupsRef = useRef<(() => void)[]>([])
  const pipelineCleanupRef = useRef<(() => void) | null>(null)

  // Form states for different scripts
  const [pipelineSteps, setPipelineSteps] = useState('')

  // MA Prediction form states
  const [maStartDate, setMaStartDate] = useState('2025-01-01')
  const [maEndDate, setMaEndDate] = useState('2025-12-31')
  const [skipOptimization, setSkipOptimization] = useState(false)
  const [shortMaRange, setShortMaRange] = useState('3,5,7')
  const [mediumMaRange, setMediumMaRange] = useState('15,20,25')
  const [longMaRange, setLongMaRange] = useState('40,50,60')
  const [fixedShort, setFixedShort] = useState('5')
  const [fixedMedium, setFixedMedium] = useState('20')
  const [fixedLong, setFixedLong] = useState('50')
  const [maExecution, setMaExecution] = useState<any>(null)

  const fetchExecutions = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await listScriptExecutions()
      setExecutions(res.executions)
    } catch (e: any) {
      setError(e.message || 'Failed to fetch executions')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExecutions()
  }, [])

  // Cleanup WebSocket listeners on unmount
  useEffect(() => {
    return () => {
      scriptCleanupsRef.current.forEach(cleanup => cleanup())
      scriptCleanupsRef.current = []
      if (pipelineCleanupRef.current) {
        pipelineCleanupRef.current()
        pipelineCleanupRef.current = null
      }
    }
  }, [])

  const handleRunPipeline = async () => {
    try {
      setError(null)
      const steps = pipelineSteps.trim() ? pipelineSteps.split(',').map(s => s.trim()) : undefined
      const execution = await runPipeline(steps)
      setPipelineExecution(execution)

      // Register WebSocket listener for pipeline status
      const cleanup = websocketService.registerListener('pipeline_status', (message: PipelineStatusMessage) => {
        if (message.data.execution_id === execution.execution_id) {
          setPipelineExecution(prev => prev ? { ...prev, ...message.data } : null)
          if (message.data.status !== 'running') {
            cleanup()
            pipelineCleanupRef.current = null
          }
        }
      })

      pipelineCleanupRef.current = cleanup

      await fetchExecutions()
    } catch (e: any) {
      setError(e.message || 'Failed to start pipeline')
    }
  }

  const handleGenerateMAPredictions = async () => {
    try {
      setError(null)
      const requestData: any = {
        start_date: maStartDate,
        end_date: maEndDate,
        skip_optimization: skipOptimization
      }

      if (skipOptimization) {
        requestData.fixed_short = parseInt(fixedShort)
        requestData.fixed_medium = parseInt(fixedMedium)
        requestData.fixed_long = parseInt(fixedLong)
      } else {
        requestData.short_ma_range = shortMaRange.split(',').map((s: string) => parseInt(s.trim()))
        requestData.medium_ma_range = mediumMaRange.split(',').map((s: string) => parseInt(s.trim()))
        requestData.long_ma_range = longMaRange.split(',').map((s: string) => parseInt(s.trim()))
      }

      const execution = await generateMAPredictions(requestData)
      setMaExecution(execution)

      // Register WebSocket listener for MA prediction status
      const cleanup = websocketService.registerListener('script_status', (message: ScriptStatusMessage) => {
        if (message.data.execution_id === execution.execution_id) {
          if (message.data.status !== 'running') {
            setMaExecution(prev => prev ? { ...prev, ...message.data } : null)
            cleanup()
            // Remove from cleanups array
            scriptCleanupsRef.current = scriptCleanupsRef.current.filter(c => c !== cleanup)
          }
        }
      })

      scriptCleanupsRef.current.push(cleanup)

      await fetchExecutions()
    } catch (e: any) {
      setError(e.message || 'Failed to start MA prediction generation')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      running: "default",
      completed: "secondary",
      failed: "destructive",
      pending: "outline"
    }
    return <Badge variant={variants[status] || "outline"}>{status}</Badge>
  }

  if (loading && executions.length === 0) {
    return <Loading />
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Scripts</h1>
          <p className="text-muted-foreground">
            Execute batch operations and manage automated pipelines for data processing and model training
          </p>
        </div>
        <Button onClick={fetchExecutions} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* Pipeline Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Full Data Pipeline
          </CardTitle>
          <CardDescription>
            Run the complete data processing pipeline from raw data to trading predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Pipeline Steps (optional)</label>
              <Input
                placeholder="apply_schema,ingest_prices,ingest_news,... (leave empty for all)"
                value={pipelineSteps}
                onChange={(e) => setPipelineSteps(e.target.value)}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Comma-separated list of steps to run
              </p>
            </div>
          </div>
          <Button
            onClick={handleRunPipeline}
            disabled={pipelineExecution?.status === 'running'}
            className="w-full"
          >
            <Play className="h-4 w-4 mr-2" />
            {pipelineExecution?.status === 'running' ? 'Running Pipeline...' : 'Run Full Pipeline'}
          </Button>
        </CardContent>
      </Card>

      {pipelineExecution && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(pipelineExecution.status)}
              Pipeline Execution: {pipelineExecution.execution_id}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm font-medium">Status</p>
                {getStatusBadge(pipelineExecution.status)}
              </div>
              <div>
                <p className="text-sm font-medium">Current Step</p>
                <p className="text-sm text-muted-foreground">
                  {pipelineExecution.current_step || 'None'}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium">Completed</p>
                <p className="text-sm text-muted-foreground">
                  {pipelineExecution.completed_steps.length} steps
                </p>
              </div>
              <div>
                <p className="text-sm font-medium">Failed</p>
                <p className="text-sm text-muted-foreground">
                  {pipelineExecution.failed_steps.length} steps
                </p>
              </div>
            </div>

            {pipelineExecution.completed_steps.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Completed Steps:</p>
                <div className="flex flex-wrap gap-1">
                  {pipelineExecution.completed_steps.map(step => (
                    <Badge key={step} variant="secondary">{step}</Badge>
                  ))}
                </div>
              </div>
            )}

            {pipelineExecution.failed_steps.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Failed Steps:</p>
                <div className="flex flex-wrap gap-1">
                  {pipelineExecution.failed_steps.map(step => (
                    <Badge key={step} variant="destructive">{step}</Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* MA Predictions Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Moving Average Predictions
          </CardTitle>
          <CardDescription>
            Generate trading predictions using moving average crossover strategies with optional optimization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-sm font-medium">Start Date</label>
              <Input
                type="date"
                value={maStartDate}
                onChange={(e) => setMaStartDate(e.target.value)}
              />
            </div>
            <div>
              <label className="text-sm font-medium">End Date</label>
              <Input
                type="date"
                value={maEndDate}
                onChange={(e) => setMaEndDate(e.target.value)}
              />
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="skip-optimization"
              checked={skipOptimization}
              onCheckedChange={setSkipOptimization}
            />
            <label htmlFor="skip-optimization" className="text-sm font-medium">
              Skip Optimization (use fixed MA periods)
            </label>
          </div>

          {!skipOptimization ? (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Short MA Range (comma-separated)</label>
                <Input
                  value={shortMaRange}
                  onChange={(e) => setShortMaRange(e.target.value)}
                  placeholder="3,5,7"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Medium MA Range (comma-separated)</label>
                <Input
                  value={mediumMaRange}
                  onChange={(e) => setMediumMaRange(e.target.value)}
                  placeholder="15,20,25"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Long MA Range (comma-separated)</label>
                <Input
                  value={longMaRange}
                  onChange={(e) => setLongMaRange(e.target.value)}
                  placeholder="40,50,60"
                />
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="text-sm font-medium">Short MA Period</label>
                <Input
                  type="number"
                  value={fixedShort}
                  onChange={(e) => setFixedShort(e.target.value)}
                  placeholder="5"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Medium MA Period</label>
                <Input
                  type="number"
                  value={fixedMedium}
                  onChange={(e) => setFixedMedium(e.target.value)}
                  placeholder="20"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Long MA Period</label>
                <Input
                  type="number"
                  value={fixedLong}
                  onChange={(e) => setFixedLong(e.target.value)}
                  placeholder="50"
                />
              </div>
            </div>
          )}

          <Button
            onClick={handleGenerateMAPredictions}
            disabled={maExecution?.status === 'running'}
            className="w-full"
          >
            <TrendingUp className="h-4 w-4 mr-2" />
            {maExecution?.status === 'running' ? 'Running Optimization...' : 'Run MA Optimization'}
          </Button>
        </CardContent>
      </Card>

      {maExecution && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(maExecution.status)}
              MA Prediction Execution: {maExecution.execution_id}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm font-medium">Status</p>
                {getStatusBadge(maExecution.status)}
              </div>
              <div>
                <p className="text-sm font-medium">Start Time</p>
                <p className="text-sm text-muted-foreground">
                  {new Date(maExecution.start_time).toLocaleString()}
                </p>
              </div>
              {maExecution.end_time && (
                <div>
                  <p className="text-sm font-medium">End Time</p>
                  <p className="text-sm text-muted-foreground">
                    {new Date(maExecution.end_time).toLocaleString()}
                  </p>
                </div>
              )}
              {maExecution.duration_seconds && (
                <div>
                  <p className="text-sm font-medium">Duration</p>
                  <p className="text-sm text-muted-foreground">
                    {maExecution.duration_seconds.toFixed(1)}s
                  </p>
                </div>
              )}
            </div>

            {maExecution.output && (
              <div>
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-xs bg-muted p-2 rounded overflow-x-auto">
                  {maExecution.output}
                </pre>
              </div>
            )}

            {maExecution.error && (
              <div>
                <p className="text-sm font-medium mb-2 text-red-600">Error:</p>
                <pre className="text-xs bg-red-50 p-2 rounded overflow-x-auto">
                  {maExecution.error}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Executions History */}
      <Card>
        <CardHeader>
          <CardTitle>Script Executions</CardTitle>
          <CardDescription>
            History of all script executions and their status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {executions.map((execution) => (
              <div key={execution.execution_id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(execution.status)}
                    <span className="font-medium">{execution.script_name}</span>
                    <Badge variant="outline">{execution.execution_id.slice(-8)}</Badge>
                  </div>
                  {getStatusBadge(execution.status)}
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-muted-foreground">
                  <div>Started: {new Date(execution.start_time).toLocaleString()}</div>
                  {execution.end_time && (
                    <div>Ended: {new Date(execution.end_time).toLocaleString()}</div>
                  )}
                  {execution.duration_seconds && (
                    <div>Duration: {execution.duration_seconds.toFixed(1)}s</div>
                  )}
                </div>
              </div>
            ))}
            {executions.length === 0 && (
              <p className="text-center text-muted-foreground py-8">
                No script executions found
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}