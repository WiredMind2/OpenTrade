import { useEffect, useState, useRef, useMemo } from 'react'
import { listScriptExecutions, runPipeline, runBatchStrategyTraining } from '../services/api'
import websocketService from '../services/websocket'
import { PipelineStatusMessage, ScriptStatusMessage, ScriptExecutionResponse } from '../types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Switch } from '../components/ui/switch'
import { Label } from '../components/ui/label'
import { Progress } from '../components/ui/progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import {
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  BarChart3
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

type BatchTrainRowStatus = 'pending' | 'running' | 'ok' | 'error'

interface BatchTrainRow {
  strategy: string
  status: BatchTrainRowStatus
  detail?: string
  best_metrics?: Record<string, unknown>
}

function parseBatchTrainOutput(
  output: string | null | undefined,
  execRunning: boolean
): { planStrategies: string[]; total: number; rows: BatchTrainRow[]; completedCount: number } {
  const planStrategies: string[] = []
  let total = 0
  const results = new Map<string, { status: 'ok' | 'error'; detail?: string; best_metrics?: Record<string, unknown> }>()

  if (output) {
    for (const line of output.split(/\r?\n/)) {
      const t = line.trim()
      if (!t.startsWith('{')) continue
      try {
        const o = JSON.parse(t) as Record<string, unknown>
        if (o.batch_plan === true) {
          const strategies = o.strategies
          if (Array.isArray(strategies)) {
            planStrategies.length = 0
            for (const s of strategies) {
              if (typeof s === 'string') planStrategies.push(s)
            }
          }
          total = typeof o.total === 'number' ? o.total : planStrategies.length
        } else if (typeof o.strategy === 'string' && (o.status === 'ok' || o.status === 'error')) {
          results.set(o.strategy, {
            status: o.status,
            detail: typeof o.detail === 'string' ? o.detail : undefined,
            best_metrics: o.best_metrics && typeof o.best_metrics === 'object' && o.best_metrics !== null
              ? (o.best_metrics as Record<string, unknown>)
              : undefined,
          })
        }
      } catch {
        /* ignore non-JSON lines */
      }
    }
  }

  if (!total && planStrategies.length) total = planStrategies.length

  const completedCount = [...results.values()].filter(r => r.status === 'ok' || r.status === 'error').length

  const rows: BatchTrainRow[] = []
  if (planStrategies.length) {
    for (let i = 0; i < planStrategies.length; i++) {
      const s = planStrategies[i]
      const r = results.get(s)
      if (r) {
        rows.push({ strategy: s, status: r.status, detail: r.detail, best_metrics: r.best_metrics })
        continue
      }
      if (execRunning) {
        const prevDone = planStrategies.slice(0, i).every(x => results.has(x))
        rows.push({ strategy: s, status: prevDone ? 'running' : 'pending' })
      } else {
        rows.push({ strategy: s, status: 'pending' })
      }
    }
  } else {
    for (const [strategy, r] of results) {
      rows.push({ strategy, status: r.status, detail: r.detail, best_metrics: r.best_metrics })
    }
    rows.sort((a, b) => a.strategy.localeCompare(b.strategy))
  }

  return { planStrategies, total: total || planStrategies.length || rows.length, rows, completedCount }
}

function formatStrategyLabel(id: string) {
  return id.replace(/_/g, ' ')
}

export default function Scripts() {
  const [executions, setExecutions] = useState<ScriptExecution[]>([])
  const [pipelineExecution, setPipelineExecution] = useState<PipelineExecution | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const pipelineCleanupRef = useRef<(() => void) | null>(null)
  const scriptCleanupsRef = useRef<(() => void)[]>([])

  // Form states for different scripts
  const [pipelineSteps, setPipelineSteps] = useState('')

  const [batchTicker, setBatchTicker] = useState('AAPL')
  const [batchStartDate, setBatchStartDate] = useState('2025-01-01')
  const [batchEndDate, setBatchEndDate] = useState('2025-12-31')
  const [batchInitialCapital, setBatchInitialCapital] = useState(100000)
  const [batchObjective, setBatchObjective] = useState('balanced')
  const [batchMaxEvals, setBatchMaxEvals] = useState(8)
  const [batchOptimizerMode, setBatchOptimizerMode] = useState('grid')
  const [batchRandomSeed, setBatchRandomSeed] = useState('')
  const [batchPairTicker, setBatchPairTicker] = useState('')
  const [batchUniverseLimit, setBatchUniverseLimit] = useState(8)
  const [batchStopOnError, setBatchStopOnError] = useState(false)
  const [batchTrainExecution, setBatchTrainExecution] = useState<ScriptExecutionResponse | null>(null)

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

  const handleBatchStrategyTraining = async () => {
    try {
      setError(null)
      const sym = batchTicker.trim().toUpperCase()
      if (!sym) {
        setError('Ticker is required')
        return
      }
      const seedRaw = batchRandomSeed.trim()
      if (seedRaw !== '' && !Number.isFinite(Number(seedRaw))) {
        setError('Random seed must be a number')
        return
      }
      const pair = batchPairTicker.trim().toUpperCase()
      const execution = await runBatchStrategyTraining({
        ticker: sym,
        start_date: batchStartDate,
        end_date: batchEndDate,
        initial_capital: batchInitialCapital,
        objective: batchObjective,
        max_evals: batchMaxEvals,
        optimizer_mode: batchOptimizerMode,
        ...(seedRaw !== '' ? { random_seed: Number(seedRaw) } : {}),
        pair_ticker: pair || null,
        universe_limit: batchUniverseLimit,
        stop_on_error: batchStopOnError,
      })
      setBatchTrainExecution(execution)

      const cleanup = websocketService.registerListener(
        'script_status',
        (message: ScriptStatusMessage) => {
          if (message.data.execution_id === execution.execution_id) {
            setBatchTrainExecution(prev => (prev ? { ...prev, ...message.data } : null))
            if (message.data.status !== 'running') {
              cleanup()
              scriptCleanupsRef.current = scriptCleanupsRef.current.filter(c => c !== cleanup)
              void fetchExecutions()
            }
          }
        }
      )
      scriptCleanupsRef.current.push(cleanup)

      await fetchExecutions()
    } catch (e: any) {
      setError(e.message || 'Failed to start batch strategy training')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <RefreshCw className="h-4 w-4 animate-spin text-primary" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-success" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-destructive" />
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />
    }
  }

  const batchParsed = useMemo(
    () =>
      parseBatchTrainOutput(
        batchTrainExecution?.output,
        batchTrainExecution?.status === 'running'
      ),
    [batchTrainExecution?.output, batchTrainExecution?.status]
  )

  const batchProgressPct =
    batchParsed.total > 0
      ? Math.min(100, Math.round((batchParsed.completedCount / batchParsed.total) * 100))
      : 0

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
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="pipeline-steps">Pipeline steps (optional)</Label>
              <Input
                id="pipeline-steps"
                placeholder="apply_schema,ingest_prices,ingest_news,... (leave empty for all)"
                value={pipelineSteps}
                onChange={(e) => setPipelineSteps(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">Comma-separated list of steps to run</p>
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

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Batch strategy training
          </CardTitle>
          <CardDescription>
            Runs signal-parameter optimization (the same search used by each strategy{"'"}s train endpoint) for
            every supported strategy on one symbol and window. Pairs trading is skipped unless you set a second-leg
            ticker.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="batch-ticker">Ticker</Label>
              <Input
                id="batch-ticker"
                placeholder="AAPL"
                value={batchTicker}
                onChange={(e) => setBatchTicker(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-start">Start date</Label>
              <Input
                id="batch-start"
                type="date"
                value={batchStartDate}
                onChange={(e) => setBatchStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-end">End date</Label>
              <Input
                id="batch-end"
                type="date"
                value={batchEndDate}
                onChange={(e) => setBatchEndDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-capital">Initial capital</Label>
              <Input
                id="batch-capital"
                type="number"
                min={1000}
                step={1000}
                value={batchInitialCapital}
                onChange={(e) => setBatchInitialCapital(Number(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Objective</Label>
              <Select value={batchObjective} onValueChange={setBatchObjective}>
                <SelectTrigger id="batch-objective">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="balanced">balanced</SelectItem>
                  <SelectItem value="sharpe">sharpe</SelectItem>
                  <SelectItem value="return">return</SelectItem>
                  <SelectItem value="drawdown">drawdown</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-max-evals">Max evals per strategy</Label>
              <Input
                id="batch-max-evals"
                type="number"
                min={1}
                max={50}
                value={batchMaxEvals}
                onChange={(e) => setBatchMaxEvals(Number(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label>Optimizer mode</Label>
              <Select value={batchOptimizerMode} onValueChange={setBatchOptimizerMode}>
                <SelectTrigger id="batch-opt-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="grid">grid</SelectItem>
                  <SelectItem value="random">random</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-seed">Random seed (optional)</Label>
              <Input
                id="batch-seed"
                placeholder="e.g. 42"
                value={batchRandomSeed}
                onChange={(e) => setBatchRandomSeed(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-pair">Pair ticker (pairs_trading only)</Label>
              <Input
                id="batch-pair"
                placeholder="Second symbol, optional"
                value={batchPairTicker}
                onChange={(e) => setBatchPairTicker(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-uni">Universe limit</Label>
              <Input
                id="batch-uni"
                type="number"
                min={2}
                max={15}
                value={batchUniverseLimit}
                onChange={(e) => setBatchUniverseLimit(Number(e.target.value))}
              />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <Switch id="batch-stop" checked={batchStopOnError} onCheckedChange={setBatchStopOnError} />
              <Label htmlFor="batch-stop" className="cursor-pointer font-normal">
                Stop on first failure
              </Label>
            </div>
          </div>
          <Button
            type="button"
            onClick={() => void handleBatchStrategyTraining()}
            disabled={batchTrainExecution?.status === 'running'}
            className="w-full sm:w-auto"
          >
            <Play className="h-4 w-4 mr-2" />
            {batchTrainExecution?.status === 'running' ? 'Training batch…' : 'Run batch training'}
          </Button>
          {batchTrainExecution && (
            <div className="space-y-4 rounded-lg border border-border bg-muted/30 p-4">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                {getStatusIcon(batchTrainExecution.status)}
                <span className="font-medium">{batchTrainExecution.execution_id}</span>
                {getStatusBadge(batchTrainExecution.status)}
                {typeof batchTrainExecution.duration_seconds === 'number' ? (
                  <span className="text-muted-foreground">
                    {batchTrainExecution.duration_seconds.toFixed(1)}s
                  </span>
                ) : null}
              </div>

              {batchTrainExecution.status === 'running' ? (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Strategies in this batch</span>
                    <span>
                      {batchParsed.total > 0
                        ? `${batchParsed.completedCount} / ${batchParsed.total} finished`
                        : 'Starting…'}
                    </span>
                  </div>
                  {batchParsed.total > 0 ? (
                    <Progress value={batchProgressPct} />
                  ) : (
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div className="h-full w-1/3 animate-pulse rounded-full bg-primary/70" />
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Each strategy runs up to {batchMaxEvals} optimizer evaluations; progress advances when a strategy
                    completes.
                  </p>
                </div>
              ) : batchParsed.total > 0 ? (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Batch complete</span>
                    <span>
                      {batchParsed.completedCount} / {batchParsed.total} strategies
                    </span>
                  </div>
                  <Progress value={batchProgressPct} />
                </div>
              ) : null}

              {batchParsed.rows.length > 0 ? (
                <div>
                  <p className="text-sm font-medium mb-2">Strategy status</p>
                  <ul className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {batchParsed.rows.map(row => (
                      <li
                        key={row.strategy}
                        className="flex flex-col gap-0.5 rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium capitalize">{formatStrategyLabel(row.strategy)}</span>
                          <span className="shrink-0">
                            {row.status === 'ok' ? (
                              <Badge variant="secondary" className="gap-1">
                                <CheckCircle className="h-3 w-3" /> Done
                              </Badge>
                            ) : row.status === 'error' ? (
                              <Badge variant="destructive" className="gap-1">
                                <XCircle className="h-3 w-3" /> Failed
                              </Badge>
                            ) : row.status === 'running' ? (
                              <Badge variant="default" className="gap-1">
                                <RefreshCw className="h-3 w-3 animate-spin" /> Running
                              </Badge>
                            ) : (
                              <Badge variant="outline">Waiting</Badge>
                            )}
                          </span>
                        </div>
                        {row.status === 'error' && row.detail ? (
                          <p className="text-xs text-destructive whitespace-pre-wrap">{row.detail}</p>
                        ) : null}
                        {row.status === 'ok' && row.best_metrics ? (
                          <p className="text-xs text-muted-foreground">
                            {typeof row.best_metrics.sharpe_ratio === 'number' ? (
                              <>Sharpe {row.best_metrics.sharpe_ratio.toFixed(2)}</>
                            ) : null}
                            {typeof row.best_metrics.total_return === 'number' ? (
                              <>
                                {typeof row.best_metrics.sharpe_ratio === 'number' ? ' · ' : null}
                                Return {(row.best_metrics.total_return * 100).toFixed(1)}%
                              </>
                            ) : null}
                          </p>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {batchTrainExecution.error ? (
                <pre className="text-xs text-destructive whitespace-pre-wrap max-h-40 overflow-auto">
                  {batchTrainExecution.error}
                </pre>
              ) : null}
              {batchTrainExecution.output ? (
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                    Raw log
                  </summary>
                  <pre className="text-xs whitespace-pre-wrap max-h-48 overflow-auto bg-background/80 rounded p-2 border mt-2">
                    {batchTrainExecution.output}
                  </pre>
                </details>
              ) : null}
            </div>
          )}
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

      {/* Executions History */}
      <Card>
        <CardHeader>
          <CardTitle>Script Executions</CardTitle>
          <CardDescription>
            History of all script executions and their status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 space-y-3 overflow-y-auto">
            {executions.map((execution) => (
              <Card key={execution.execution_id} className="shadow-none">
                <CardContent className="space-y-3 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      {getStatusIcon(execution.status)}
                      <span className="font-medium">{execution.script_name}</span>
                      <Badge variant="outline">{execution.execution_id.slice(-8)}</Badge>
                    </div>
                    {getStatusBadge(execution.status)}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground md:grid-cols-4">
                    <div>Started: {new Date(execution.start_time).toLocaleString()}</div>
                    {execution.end_time ? (
                      <div>Ended: {new Date(execution.end_time).toLocaleString()}</div>
                    ) : null}
                    {execution.duration_seconds ? (
                      <div>Duration: {execution.duration_seconds.toFixed(1)}s</div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
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