import React, { useEffect, useState, useRef } from 'react'
import {
  executeScript,
  listScriptExecutions,
  runPipeline
} from '../services/api'
import websocketService from '../services/websocket'
import { ScriptStatusMessage, PipelineStatusMessage } from '../types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import {
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Settings,
  Database,
  Brain,
  TrendingUp,
  BarChart3,
  FileText,
  Download,
  Zap
} from 'lucide-react'
import { Separator } from '../components/ui/separator'

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
  const [runningScripts, setRunningScripts] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const scriptCleanupsRef = useRef<(() => void)[]>([])
  const pipelineCleanupRef = useRef<(() => void) | null>(null)

  // Form states for different scripts
  const [pipelineSteps, setPipelineSteps] = useState('')
  const [trainingCsv, setTrainingCsv] = useState('data/training_labels_1d_top10.csv')
  const [modelOutdir, setModelOutdir] = useState('models')
  const [sentimentHorizon, setSentimentHorizon] = useState('1')
  const [tradingStart, setTradingStart] = useState('2020-01-01')
  const [tradingEnd, setTradingEnd] = useState('2025-01-01')
  const [backtestStart, setBacktestStart] = useState('2023-01-01')
  const [backtestEnd, setBacktestEnd] = useState('2023-12-31')

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

  const handleExecuteScript = async (scriptName: string, parameters: Record<string, any> = {}) => {
    try {
      setError(null)
      const execution = await executeScript(scriptName, parameters)
      setRunningScripts(prev => new Set(prev).add(execution.execution_id))

      // Register WebSocket listener for status updates
      const cleanup = websocketService.registerListener('script_status', (message: ScriptStatusMessage) => {
        if (message.data.execution_id === execution.execution_id) {
          if (message.data.status !== 'running') {
            setRunningScripts(prev => {
              const newSet = new Set(prev)
              newSet.delete(execution.execution_id)
              return newSet
            })
            // Update executions list
            setExecutions(prev => prev.map(ex =>
              ex.execution_id === execution.execution_id
                ? { ...ex, status: message.data.status, end_time: message.data.end_time, duration_seconds: message.data.duration_seconds }
                : ex
            ))
            cleanup()
            // Remove from cleanups array
            scriptCleanupsRef.current = scriptCleanupsRef.current.filter(c => c !== cleanup)
          }
        }
      })

      scriptCleanupsRef.current.push(cleanup)

      await fetchExecutions()
    } catch (e: any) {
      setError(e.message || `Failed to execute ${scriptName}`)
    }
  }

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
            Run data processing and machine learning pipeline scripts
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

      <Separator />

      {/* Individual Scripts Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Train Sentiment Model
            </CardTitle>
            <CardDescription>
              Train a LightGBM model to predict stock returns from news sentiment
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Training Data CSV</label>
              <Input
                value={trainingCsv}
                onChange={(e) => setTrainingCsv(e.target.value)}
                placeholder="data/training_labels_1d_top10.csv"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Model Output Directory</label>
              <Input
                value={modelOutdir}
                onChange={(e) => setModelOutdir(e.target.value)}
                placeholder="models"
              />
            </div>
            <Button
              onClick={() => handleExecuteScript('train_sentiment_model', {
                csv: trainingCsv,
                outdir: modelOutdir
              })}
              className="w-full"
            >
              <Brain className="h-4 w-4 mr-2" />
              Train Model
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Generate Sentiment Predictions
            </CardTitle>
            <CardDescription>
              Use trained models to predict sentiment for news articles
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Prediction Horizon (days)</label>
              <Input
                type="number"
                value={sentimentHorizon}
                onChange={(e) => setSentimentHorizon(e.target.value)}
                placeholder="1"
              />
            </div>
            <Button
              onClick={() => handleExecuteScript('generate_sentiment_predictions', {
                horizon: parseInt(sentimentHorizon)
              })}
              className="w-full"
            >
              <FileText className="h-4 w-4 mr-2" />
              Generate Sentiment
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Generate Trading Predictions
            </CardTitle>
            <CardDescription>
              Convert sentiment scores into trading position recommendations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-sm font-medium">Start Date</label>
                <Input
                  type="date"
                  value={tradingStart}
                  onChange={(e) => setTradingStart(e.target.value)}
                />
              </div>
              <div>
                <label className="text-sm font-medium">End Date</label>
                <Input
                  type="date"
                  value={tradingEnd}
                  onChange={(e) => setTradingEnd(e.target.value)}
                />
              </div>
            </div>
            <Button
              onClick={() => handleExecuteScript('generate_trading_predictions', {
                start: tradingStart,
                end: tradingEnd
              })}
              className="w-full"
            >
              <TrendingUp className="h-4 w-4 mr-2" />
              Generate Trading Signals
            </Button>
          </CardContent>
        </Card>

        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Run Backtest
            </CardTitle>
            <CardDescription>
              Execute a backtest using the generated trading predictions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-2 max-w-md">
              <div>
                <label className="text-sm font-medium">Start Date</label>
                <Input
                  type="date"
                  value={backtestStart}
                  onChange={(e) => setBacktestStart(e.target.value)}
                />
              </div>
              <div>
                <label className="text-sm font-medium">End Date</label>
                <Input
                  type="date"
                  value={backtestEnd}
                  onChange={(e) => setBacktestEnd(e.target.value)}
                />
              </div>
            </div>
            <Button
              onClick={() => handleExecuteScript('backtest_runner', {
                start: backtestStart,
                end: backtestEnd
              })}
              className="w-full max-w-md"
            >
              <BarChart3 className="h-4 w-4 mr-2" />
              Run Backtest
            </Button>
          </CardContent>
        </Card>
      </div>

      <Separator />

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