import { useEffect, useState, useRef, useMemo, useCallback } from 'react'
import {
  listScriptExecutions,
  runPipeline,
  runBatchStrategyTraining,
  getScriptStatus,
  getPipelineStatus,
} from '../services/api'
import websocketService from '../services/websocket'
import {
  PipelineStatusMessage,
  ScriptStatusMessage,
  ScriptExecutionResponse,
  PipelineStatus,
} from '../types'
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
  BarChart3,
  ChevronRight,
  X,
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

  const [executionDetailOpen, setExecutionDetailOpen] = useState(false)
  const [executionDetailListRow, setExecutionDetailListRow] = useState<ScriptExecution | null>(null)
  const [executionDetailScript, setExecutionDetailScript] = useState<ScriptExecutionResponse | null>(null)
  const [executionDetailPipeline, setExecutionDetailPipeline] = useState<PipelineStatus | null>(null)
  const [executionDetailLoading, setExecutionDetailLoading] = useState(false)
  const [executionDetailError, setExecutionDetailError] = useState<string | null>(null)

  const openExecutionDetail = (row: ScriptExecution) => {
    setExecutionDetailListRow(row)
    setExecutionDetailOpen(true)
    setExecutionDetailError(null)
    setExecutionDetailScript(null)
    setExecutionDetailPipeline(null)
  }

  const closeExecutionDetail = useCallback(() => {
    setExecutionDetailOpen(false)
    setExecutionDetailListRow(null)
    setExecutionDetailScript(null)
    setExecutionDetailPipeline(null)
    setExecutionDetailError(null)
  }, [])

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

  useEffect(() => {
    if (!executionDetailOpen || !executionDetailListRow) return

    const id = executionDetailListRow.execution_id
    let cancelled = false

    const load = async (isInitial: boolean) => {
      if (isInitial) setExecutionDetailLoading(true)
      try {
        const script = await getScriptStatus(id)
        if (cancelled) return
        setExecutionDetailScript(script)
        if (script.script_name === 'run_pipeline') {
          const pipe = await getPipelineStatus(id)
          if (!cancelled) setExecutionDetailPipeline(pipe)
        } else {
          setExecutionDetailPipeline(null)
        }
        setExecutionDetailError(null)
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : 'Failed to load execution'
          setExecutionDetailError(msg)
        }
      } finally {
        if (!cancelled && isInitial) setExecutionDetailLoading(false)
      }
    }

    void load(true)
    const interval = setInterval(() => {
      void load(false)
    }, 2000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [executionDetailOpen, executionDetailListRow])

  useEffect(() => {
    if (!executionDetailOpen) return
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') closeExecutionDetail()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [executionDetailOpen, closeExecutionDetail])

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

  const detailBatchParsed = useMemo(
    () =>
      executionDetailScript?.script_name === 'train_all_strategies'
        ? parseBatchTrainOutput(
            executionDetailScript.output,
            executionDetailScript.status === 'running'
          )
        : { planStrategies: [], total: 0, rows: [], completedCount: 0 },
    [executionDetailScript?.output, executionDetailScript?.script_name, executionDetailScript?.status]
  )

  const detailBatchProgressPct =
    detailBatchParsed.total > 0
      ? Math.min(100, Math.round((detailBatchParsed.completedCount / detailBatchParsed.total) * 100))
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
            Execute batch operations and manage automated pipelines for data processing and strategy training
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
            History of all script executions. Click a row to open full logs, stderr, and live progress while a job runs.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 space-y-3 overflow-y-auto">
            {executions.map((execution) => (
              <Card
                key={execution.execution_id}
                className="shadow-none cursor-pointer transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                role="button"
                tabIndex={0}
                onClick={() => openExecutionDetail(execution)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    openExecutionDetail(execution)
                  }
                }}
              >
                <CardContent className="space-y-3 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      {getStatusIcon(execution.status)}
                      <span className="font-medium">{execution.script_name}</span>
                      <Badge variant="outline">{execution.execution_id.slice(-8)}</Badge>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {getStatusBadge(execution.status)}
                      <ChevronRight className="h-4 w-4 text-muted-foreground" aria-hidden />
                    </div>
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

      {executionDetailOpen && executionDetailListRow ? (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center p-4 pb-8 sm:items-center sm:pb-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="execution-detail-title"
        >
          <button
            type="button"
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            aria-label="Close execution details"
            onClick={closeExecutionDetail}
          />
          <div
            className="relative flex max-h-[min(85vh,720px)] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex shrink-0 items-start justify-between gap-3 border-b border-border p-4">
              <div className="min-w-0 space-y-1">
                <h2 id="execution-detail-title" className="truncate text-lg font-semibold">
                  {executionDetailScript?.script_name ?? executionDetailListRow.script_name}
                </h2>
                <p className="font-mono text-xs text-muted-foreground break-all">
                  {executionDetailListRow.execution_id}
                </p>
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  {getStatusIcon(executionDetailScript?.status ?? executionDetailListRow.status)}
                  {getStatusBadge(executionDetailScript?.status ?? executionDetailListRow.status)}
                  {executionDetailScript?.status === 'running' ? (
                    <span className="text-xs text-muted-foreground">Refreshing every 2s</span>
                  ) : null}
                </div>
              </div>
              <Button type="button" variant="ghost" size="icon" onClick={closeExecutionDetail} aria-label="Close">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
              {executionDetailLoading && !executionDetailScript ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Loading details…
                </div>
              ) : null}
              {executionDetailError ? (
                <p className="text-sm text-destructive">{executionDetailError}</p>
              ) : null}

              {executionDetailScript ? (
                <>
                  <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                    <div>
                      <p className="font-medium text-foreground">Started</p>
                      <p className="text-muted-foreground">
                        {new Date(executionDetailScript.start_time).toLocaleString()}
                      </p>
                    </div>
                    {executionDetailScript.end_time ? (
                      <div>
                        <p className="font-medium text-foreground">Ended</p>
                        <p className="text-muted-foreground">
                          {new Date(executionDetailScript.end_time).toLocaleString()}
                        </p>
                      </div>
                    ) : null}
                    {typeof executionDetailScript.duration_seconds === 'number' ? (
                      <div>
                        <p className="font-medium text-foreground">Duration</p>
                        <p className="text-muted-foreground">
                          {executionDetailScript.duration_seconds.toFixed(1)}s
                        </p>
                      </div>
                    ) : null}
                  </div>

                  {executionDetailPipeline ? (
                    <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-sm font-medium">Pipeline progress</p>
                      <div className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                        <div>
                          <span className="text-muted-foreground">Current step</span>
                          <p className="font-medium">{executionDetailPipeline.current_step ?? '—'}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Steps</span>
                          <p className="font-medium">
                            {executionDetailPipeline.completed_steps.length} done
                            {executionDetailPipeline.failed_steps.length
                              ? ` · ${executionDetailPipeline.failed_steps.length} failed`
                              : ''}
                          </p>
                        </div>
                      </div>
                      {executionDetailPipeline.completed_steps.length > 0 ? (
                        <div>
                          <p className="mb-1 text-xs text-muted-foreground">Completed</p>
                          <div className="flex flex-wrap gap-1">
                            {executionDetailPipeline.completed_steps.map((step) => (
                              <Badge key={step} variant="secondary">
                                {step}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      {executionDetailPipeline.failed_steps.length > 0 ? (
                        <div>
                          <p className="mb-1 text-xs text-muted-foreground">Failed</p>
                          <div className="flex flex-wrap gap-1">
                            {executionDetailPipeline.failed_steps.map((step) => (
                              <Badge key={step} variant="destructive">
                                {step}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {executionDetailScript.script_name === 'train_all_strategies' ? (
                    <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-sm font-medium">Batch training progress</p>
                      {executionDetailScript.status === 'running' && detailBatchParsed.total > 0 ? (
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>Strategies</span>
                            <span>
                              {detailBatchParsed.completedCount} / {detailBatchParsed.total} finished
                            </span>
                          </div>
                          <Progress value={detailBatchProgressPct} />
                        </div>
                      ) : detailBatchParsed.total > 0 ? (
                        <div className="space-y-2">
                          <Progress value={detailBatchProgressPct} />
                          <p className="text-xs text-muted-foreground">
                            {detailBatchParsed.completedCount} / {detailBatchParsed.total} strategies
                          </p>
                        </div>
                      ) : executionDetailScript.status === 'running' ? (
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div className="h-full w-1/3 animate-pulse rounded-full bg-primary/70" />
                        </div>
                      ) : null}
                      {detailBatchParsed.rows.length > 0 ? (
                        <ul className="max-h-52 space-y-2 overflow-y-auto pr-1">
                          {detailBatchParsed.rows.map((row) => (
                            <li
                              key={row.strategy}
                              className="rounded-lg border border-border bg-card px-3 py-2 text-sm"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-medium capitalize">
                                  {formatStrategyLabel(row.strategy)}
                                </span>
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
                                <p className="mt-1 text-xs text-destructive whitespace-pre-wrap">{row.detail}</p>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  ) : null}

                  {executionDetailScript.error ? (
                    <div>
                      <p className="mb-2 text-sm font-medium text-destructive">Stderr</p>
                      <pre className="max-h-48 overflow-auto rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs whitespace-pre-wrap">
                        {executionDetailScript.error}
                      </pre>
                    </div>
                  ) : null}
                  {executionDetailScript.output ? (
                    <div>
                      <p className="mb-2 text-sm font-medium">Stdout / logs</p>
                      <pre className="max-h-64 overflow-auto rounded-md border border-border bg-muted/50 p-3 text-xs whitespace-pre-wrap">
                        {executionDetailScript.output}
                      </pre>
                    </div>
                  ) : executionDetailScript.status !== 'running' ? (
                    <p className="text-sm text-muted-foreground">No output captured.</p>
                  ) : (
                    <p className="text-sm text-muted-foreground">Waiting for output…</p>
                  )}
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}