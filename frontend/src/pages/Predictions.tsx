import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  listSavedModels,
  signalsSavedModelsBatch,
  type SavedModel,
  type SavedModelSignal,
} from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Target, Activity, Calendar, BarChart3 } from 'lucide-react'
import OHLCChart from '../components/OHLCChart'
import { NewsSidebar } from '../components/NewsSidebar'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'

const OBJECTIVES = ['balanced', 'sharpe', 'return', 'drawdown'] as const

function degradeVariant(status: string | undefined): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'degraded') return 'destructive'
  if (status === 'watch') return 'secondary'
  return 'outline'
}

function signalActionBadgeVariant(action: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (action === 'buy') return 'default'
  if (action === 'sell') return 'destructive'
  return 'secondary'
}

function signalActionLabel(action: string): string {
  if (action === 'buy') return 'Buy'
  if (action === 'sell') return 'Sell'
  return 'Hold'
}

export default function Predictions() {
  const [selectedTicker, setSelectedTicker] = useState(() => getStoredTicker())

  const [signalAsOfDate, setSignalAsOfDate] = useState('')
  const [objective, setObjective] = useState<string>('balanced')
  const [topN, setTopN] = useState(5)

  const [savedModels, setSavedModels] = useState<SavedModel[]>([])
  const [topSignals, setTopSignals] = useState<SavedModelSignal[]>([])
  const [excludedIds, setExcludedIds] = useState<Set<number>>(() => new Set())
  const [pinnedIds, setPinnedIds] = useState<Set<number>>(() => new Set())

  const [modelsLoading, setModelsLoading] = useState(false)
  const [signalLoading, setSignalLoading] = useState(false)
  const [signalError, setSignalError] = useState<string | null>(null)

  const excludeKey = useMemo(() => [...excludedIds].sort((a, b) => a - b).join(','), [excludedIds])
  const pinKey = useMemo(() => [...pinnedIds].sort((a, b) => a - b).join(','), [pinnedIds])

  const sym = selectedTicker.trim().toUpperCase()

  useEffect(() => {
    setExcludedIds(new Set())
    setPinnedIds(new Set())
  }, [sym])

  useEffect(() => {
    if (!sym) {
      setSavedModels([])
      return
    }
    let cancelled = false
    setModelsLoading(true)
    setSavedModels([])
    void listSavedModels({ ticker: sym, active: true })
      .then((rows) => {
        if (!cancelled) setSavedModels(rows)
      })
      .catch(() => {
        if (!cancelled) setSavedModels([])
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [sym])

  const runTopSignals = useCallback(async () => {
    if (!sym) {
      setTopSignals([])
      return
    }
    setSignalLoading(true)
    setSignalError(null)
    try {
      const trimmedAsOf = signalAsOfDate.trim()
      const rows = await signalsSavedModelsBatch({
        ticker: sym,
        objective,
        top_n: topN,
        include_model_ids: pinKey ? pinKey.split(',').map((s) => parseInt(s, 10)) : [],
        exclude_model_ids: excludeKey ? excludeKey.split(',').map((s) => parseInt(s, 10)) : [],
        ...(trimmedAsOf ? { as_of_date: trimmedAsOf } : {}),
      })
      setTopSignals(rows)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Signal request failed'
      setSignalError(msg)
      setTopSignals([])
    } finally {
      setSignalLoading(false)
    }
  }, [sym, signalAsOfDate, objective, topN, excludeKey, pinKey])

  useEffect(() => {
    if (!sym || savedModels.length === 0) {
      setTopSignals([])
      return
    }
    const t = window.setTimeout(() => {
      void runTopSignals()
    }, 500)
    return () => window.clearTimeout(t)
  }, [sym, savedModels.length, signalAsOfDate, objective, topN, excludeKey, pinKey, runTopSignals])

  const toggleExclude = (id: number) => {
    setExcludedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const togglePin = (id: number) => {
    setPinnedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Predictions</h2>
        <p className="text-muted-foreground">
          Top saved models (ranked using your objective and last stored backtest scores) are evaluated{' '}
          <strong className="text-foreground font-medium">only on the latest daily bar</strong> to suggest buy, sell, or
          hold. Strategies use historical context up through that bar (e.g. moving averages); no multi-month simulation is
          run here. Saved models come from Strategy Performance / Backtests.
        </p>
      </div>

      <div className="space-y-4">
        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Saved models for <span className="font-mono">{sym || '—'}</span>
            </CardTitle>
            <CardDescription>
              Exclude models from the candidate pool or pin favourites. &quot;Top N&quot; picks which ranked models receive
              a live signal at the anchor date below.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Signal as-of (optional)
                </label>
                <Input
                  type="date"
                  value={signalAsOfDate}
                  onChange={(e) => setSignalAsOfDate(e.target.value)}
                  placeholder="Latest bar"
                />
                <p className="text-xs text-muted-foreground">Uses latest close on or before this date.</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Ranking objective</label>
                <select
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={objective}
                  onChange={(e) => setObjective(e.target.value)}
                >
                  {OBJECTIVES.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-muted-foreground">Orders candidates using stored run metrics.</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Top N signals</label>
                <Input
                  type="number"
                  min={1}
                  max={25}
                  value={topN}
                  onChange={(e) => setTopN(Math.min(25, Math.max(1, parseInt(e.target.value, 10) || 1)))}
                />
              </div>
            </div>

            {modelsLoading && (
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Activity className="h-3 w-3 animate-spin" />
                Loading saved models…
              </p>
            )}
            {!modelsLoading && sym && savedModels.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No active saved models for this ticker yet. Save a strategy configuration from Strategy Performance /
                Backtests or the API.
              </p>
            )}

            {savedModels.length > 0 && (
              <div className="overflow-x-auto rounded-md border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/40 text-left">
                      <th className="p-2 font-medium">Model</th>
                      <th className="p-2 font-medium">Strategy</th>
                      <th className="p-2 font-medium">Exclude</th>
                      <th className="p-2 font-medium">Pin</th>
                    </tr>
                  </thead>
                  <tbody>
                    {savedModels.map((m) => (
                      <tr key={m.id} className="border-b last:border-0">
                        <td className="p-2">
                          <span className="font-medium">{m.name}</span>
                          <span className="text-muted-foreground ml-1">#{m.id}</span>
                        </td>
                        <td className="p-2 font-mono text-xs">{m.strategy_name}</td>
                        <td className="p-2">
                          <input
                            type="checkbox"
                            checked={excludedIds.has(m.id)}
                            onChange={() => toggleExclude(m.id)}
                            aria-label={`Exclude ${m.name}`}
                          />
                        </td>
                        <td className="p-2">
                          <input
                            type="checkbox"
                            checked={pinnedIds.has(m.id)}
                            onChange={() => togglePin(m.id)}
                            aria-label={`Pin ${m.name}`}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {signalLoading && (
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Activity className="h-3 w-3 animate-spin" />
                Computing signals…
              </p>
            )}
            {signalError && <p className="text-sm text-destructive">{signalError}</p>}

            {topSignals.length > 0 && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-baseline gap-2">
                  <p className="text-sm font-medium">Today&apos;s stance (anchor bar)</p>
                  <span className="text-xs text-muted-foreground">
                    As of {topSignals[0]?.as_of} · last close {(topSignals[0]?.last_price ?? 0).toFixed(4)}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {topSignals.map((s) => (
                    <Badge
                      key={s.model_id}
                      variant={signalActionBadgeVariant(s.action)}
                      className="gap-1 font-medium"
                      title={s.reason}
                    >
                      <Target className="h-3 w-3" />
                      {s.name || `Model ${s.model_id}`}: {signalActionLabel(s.action)}
                    </Badge>
                  ))}
                </div>

                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/40 text-left">
                        <th className="p-2 font-medium">Model</th>
                        <th className="p-2 font-medium">Signal</th>
                        <th className="text-right p-2 font-medium">Target %</th>
                        <th className="text-right p-2 font-medium">Conf.</th>
                        <th className="p-2 font-medium">Reason</th>
                        <th className="p-2 font-medium">Drift</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topSignals.map((s) => (
                        <tr key={s.model_id} className="border-b last:border-0">
                          <td className="p-2">
                            <span className="font-medium">{s.name || `#${s.model_id}`}</span>
                            <div className="text-xs font-mono text-muted-foreground">{s.strategy_name}</div>
                          </td>
                          <td className="p-2">
                            <Badge variant={signalActionBadgeVariant(s.action)}>{signalActionLabel(s.action)}</Badge>
                          </td>
                          <td className="p-2 text-right font-mono">{(s.target_pct * 100).toFixed(2)}%</td>
                          <td className="p-2 text-right font-mono">
                            {(s.confidence <= 1 ? s.confidence * 100 : s.confidence).toFixed(0)}
                            {(s.confidence <= 1 ? '%' : '')}
                          </td>
                          <td className="p-2 max-w-xs">
                            <span className="text-xs break-words">{s.reason}</span>
                            {s.error && <span className="block text-xs text-destructive mt-1">{s.error}</span>}
                          </td>
                          <td className="p-2">
                            <Badge variant={degradeVariant(s.degrade_status)} className="text-xs">
                              {s.degrade_status}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1 min-w-0">
            <OHLCChart
              symbol={selectedTicker}
              height="600px"
              strategyName=""
              params={{}}
              horizon={30}
              onSymbolChange={(symbol) => {
                const normalized = rememberTicker(symbol)
                if (normalized && normalized !== selectedTicker) {
                  setSelectedTicker(normalized)
                }
              }}
            />
          </div>
          <div className="w-full lg:w-80 xl:w-96 flex-shrink-0">
            <NewsSidebar ticker={selectedTicker} />
          </div>
        </div>
      </div>
    </div>
  )
}
