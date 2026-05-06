import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Separator } from '../components/ui/separator'
import { Skeleton } from '../components/ui/skeleton'
import { ShieldAlert, PieChart, Brain, FlaskConical, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react'
import { getBacktests } from '../services/api'
import type { BacktestResult } from '../types'

// ── Types ────────────────────────────────────────────────────────────────────

interface Rule {
  text: string
  detail?: string
  level: 'critical' | 'important' | 'note'
}

interface Category {
  id: string
  title: string
  description: string
  icon: React.ElementType
  rules: Rule[]
}

interface UserProfile {
  avgSharpe: number
  avgDrawdown: number   // negative, e.g. -0.18
  avgWinRate: number    // 0–1
  avgReturn: number     // e.g. 0.12
  count: number
}

// ── Static content ───────────────────────────────────────────────────────────

const categories: Category[] = [
  {
    id: 'risk',
    title: 'Risk Management',
    description: 'Capital preservation rules that must be respected on every trade.',
    icon: ShieldAlert,
    rules: [
      { text: 'Never risk more than 1–2% of total capital on a single position.', level: 'critical' },
      { text: 'Define your stop-loss before entering any trade.', level: 'critical' },
      { text: 'Keep total open exposure below 20–30% of the portfolio at all times.', level: 'important' },
      { text: 'If drawdown exceeds 10%, reduce position sizes by 50% until recovery.', level: 'important' },
      { text: 'Do not average down on a losing position without a confirmed reversal signal.', level: 'note' },
    ],
  },
  {
    id: 'diversification',
    title: 'Diversification',
    description: 'Spreading exposure to reduce variance without sacrificing returns.',
    icon: PieChart,
    rules: [
      { text: 'No single sector or asset should represent more than 20% of the portfolio.', level: 'critical' },
      { text: 'Combine strategies with low correlation — momentum, mean-reversion, MA crossover.', level: 'important' },
      { text: 'Positions with a correlation above 0.8 compound drawdowns without adding diversification.', level: 'important' },
      { text: 'Maintain 10–20% in cash to deploy during market corrections.', level: 'note' },
    ],
  },
  {
    id: 'backtesting',
    title: 'Backtesting',
    description: 'Validating a strategy before committing real capital.',
    icon: FlaskConical,
    rules: [
      { text: 'Always use walk-forward testing — never optimize over the full historical dataset.', level: 'critical' },
      { text: 'Excessive parameters lead to overfitting. A strategy that only works in-sample is not a strategy.', level: 'critical' },
      { text: 'Account for transaction costs and slippage. Omitting them systematically overstates performance.', level: 'important' },
      { text: 'Validate across at least one full bull and one full bear market cycle.', level: 'important' },
      { text: 'A Sharpe Ratio above 1.5 out-of-sample is a reasonable threshold before live deployment.', level: 'note' },
    ],
  },
  {
    id: 'entries',
    title: 'Entries & Signals',
    description: 'Improving entry quality and timing to maximise risk-adjusted returns.',
    icon: TrendingUp,
    rules: [
      { text: 'Require confirmation from at least two independent indicators before entering.', level: 'important' },
      { text: 'Avoid trading in the 30-minute window surrounding major macro releases (CPI, NFP, FOMC).', level: 'important' },
      { text: 'Require a minimum risk/reward ratio of 1:2 before every entry.', level: 'important' },
      { text: 'The London and New York session opens offer the highest liquidity and tightest spreads.', level: 'note' },
    ],
  },
  {
    id: 'discipline',
    title: 'Discipline',
    description: 'Maintaining operational consistency regardless of market conditions.',
    icon: Brain,
    rules: [
      { text: 'Execute the trading plan without deviation. Never adjust a stop-loss once a position is open.', level: 'critical' },
      { text: 'After three consecutive losses, halt trading and conduct a review session before resuming.', level: 'important' },
      { text: 'Maintain a trading journal documenting entry rationale, exit rationale, and emotional state.', level: 'important' },
      { text: 'Do not increase position sizes following a winning streak. Overconfidence bias is a leading cause of account drawdown.', level: 'note' },
    ],
  },
]

// ── Adaptive logic ────────────────────────────────────────────────────────────

interface FocusArea {
  categoryId: string
  reason: string
  severity: 'high' | 'medium'
}

function deriveFocusAreas(profile: UserProfile): FocusArea[] {
  const areas: FocusArea[] = []

  if (profile.avgDrawdown < -0.10) {
    areas.push({
      categoryId: 'risk',
      reason: `Max drawdown at ${fmtPct(profile.avgDrawdown)} — reduce your position sizes now and stop increasing exposure until you recover.`,
      severity: profile.avgDrawdown < -0.20 ? 'high' : 'medium',
    })
  }

  if (profile.avgSharpe < 1.0) {
    areas.push({
      categoryId: 'diversification',
      reason: `Sharpe ratio of ${profile.avgSharpe.toFixed(2)} — you're taking too much risk for what you're making. Add uncorrelated strategies.`,
      severity: profile.avgSharpe < 0.5 ? 'high' : 'medium',
    })
  }

  if (profile.avgWinRate < 0.45) {
    areas.push({
      categoryId: 'entries',
      reason: `Only ${fmtPct(profile.avgWinRate)} of your trades are winners — be more selective before entering a position.`,
      severity: profile.avgWinRate < 0.35 ? 'high' : 'medium',
    })
  }

  if (profile.avgReturn < 0) {
    areas.push({
      categoryId: 'backtesting',
      reason: `Your backtests are losing money on average (${fmtPct(profile.avgReturn)}). Do not trade this live until you understand why.`,
      severity: 'high',
    })
  }

  if (profile.avgDrawdown < -0.20 && profile.avgReturn < 0) {
    areas.push({
      categoryId: 'discipline',
      reason: `Drawdown at ${fmtPct(profile.avgDrawdown)} and returns in the red — step back, stop trading, and review what went wrong before continuing.`,
      severity: 'high',
    })
  }

  return areas
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function safeNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function computeProfile(backtests: BacktestResult[]): UserProfile | null {
  const completed = backtests.filter((b) => safeNum(b.total_return) !== null)
  if (completed.length === 0) return null

  const avg = (fn: (b: BacktestResult) => number) =>
    completed.reduce((s, b) => s + fn(b), 0) / completed.length

  return {
    avgSharpe:   avg((b) => safeNum(b.sharpe_ratio)  ?? 0),
    avgDrawdown: avg((b) => safeNum(b.max_drawdown)   ?? 0),
    avgWinRate:  avg((b) => safeNum(b.win_rate)       ?? 0),
    avgReturn:   avg((b) => safeNum(b.total_return)   ?? 0),
    count: completed.length,
  }
}

// ── Sub-components ────────────────────────────────────────────────────────────

const levelConfig: Record<string, { label: string; variant: 'destructive' | 'default' | 'secondary' }> = {
  critical:  { label: 'Critical',  variant: 'destructive' },
  important: { label: 'Important', variant: 'default' },
  note:      { label: 'Note',      variant: 'secondary' },
}

function ProfileSnapshot({ profile }: { profile: UserProfile }) {
  const ddColor =
    profile.avgDrawdown < -0.15 ? 'text-destructive' : profile.avgDrawdown < -0.10 ? 'text-warning' : 'text-success'
  const srColor = profile.avgSharpe < 1.0 ? 'text-warning' : 'text-success'
  const wrColor = profile.avgWinRate < 0.45 ? 'text-warning' : 'text-success'
  const retColor = profile.avgReturn < 0 ? 'text-destructive' : 'text-success'

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Your Performance Profile</CardTitle>
        <CardDescription>Averaged across {profile.count} completed backtest{profile.count > 1 ? 's' : ''}.</CardDescription>
      </CardHeader>
      <Separator />
      <CardContent className="pt-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">Avg. Return</span>
            <span className={`text-base font-semibold tabular-nums ${retColor}`}>{fmtPct(profile.avgReturn)}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">Max Drawdown</span>
            <span className={`text-base font-semibold tabular-nums ${ddColor}`}>{fmtPct(profile.avgDrawdown)}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">Sharpe Ratio</span>
            <span className={`text-base font-semibold tabular-nums ${srColor}`}>{profile.avgSharpe.toFixed(2)}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">Win Rate</span>
            <span className={`text-base font-semibold tabular-nums ${wrColor}`}>{fmtPct(profile.avgWinRate)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function FocusCard({ area, category }: { area: FocusArea; category: Category }) {
  const Icon = category.icon
  const focusRules = category.rules.filter((r) => r.level === 'critical' || r.level === 'important')

  return (
    <Card className={area.severity === 'high' ? 'border-destructive/40' : 'border-warning/40'}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
          <CardTitle className="text-sm font-semibold">{category.title}</CardTitle>
          <Badge variant={area.severity === 'high' ? 'destructive' : 'warning'} className="ml-auto shrink-0">
            Focus area
          </Badge>
        </div>
        <CardDescription className="flex items-start gap-1.5 pt-1">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          {area.reason}
        </CardDescription>
      </CardHeader>
      <Separator />
      <CardContent className="pt-0 divide-y divide-border">
        {focusRules.map((rule, i) => {
          const cfg = levelConfig[rule.level]
          return (
            <div key={i} className="flex items-start gap-4 py-3">
              <Badge variant={cfg.variant} className="mt-0.5 shrink-0 w-20 justify-center">
                {cfg.label}
              </Badge>
              <p className="text-sm leading-relaxed text-foreground">{rule.text}</p>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function RulesTab({ category }: { category: Category }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{category.title}</CardTitle>
        <CardDescription>{category.description}</CardDescription>
      </CardHeader>
      <Separator />
      <CardContent className="pt-0 divide-y divide-border">
        {category.rules.map((rule, i) => {
          const cfg = levelConfig[rule.level]
          return (
            <div key={i} className="flex items-start gap-4 py-3">
              <Badge variant={cfg.variant} className="mt-0.5 shrink-0 w-20 justify-center">
                {cfg.label}
              </Badge>
              <div>
                <p className="text-sm leading-relaxed text-foreground">{rule.text}</p>
                {rule.detail && (
                  <p className="mt-0.5 text-xs text-muted-foreground">{rule.detail}</p>
                )}
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function Recommendations() {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getBacktests()
      .then((data) => {
        const backtests: BacktestResult[] = Array.isArray(data)
          ? data
          : Array.isArray(data?.results)
          ? data.results
          : []
        setProfile(computeProfile(backtests))
      })
      .catch(() => setProfile(null))
      .finally(() => setLoading(false))
  }, [])

  const focusAreas = profile ? deriveFocusAreas(profile) : []

  return (
    <div className="space-y-6">

      {/* Performance snapshot */}
      {loading ? (
        <Skeleton className="h-24 w-full rounded-lg" />
      ) : profile ? (
        <ProfileSnapshot profile={profile} />
      ) : null}

      {/* Personalised focus areas */}
      {!loading && profile && focusAreas.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-foreground">Recommended focus areas</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {focusAreas.map((area) => {
              const cat = categories.find((c) => c.id === area.categoryId)!
              return <FocusCard key={area.categoryId} area={area} category={cat} />
            })}
          </div>
        </div>
      )}

      {/* Healthy profile feedback */}
      {!loading && profile && focusAreas.length === 0 && (
        <div className="flex items-start gap-3 rounded-lg border border-success/40 bg-success/5 p-4">
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-success" />
          <div>
            <p className="text-sm font-semibold text-foreground">Your profile looks healthy</p>
            <p className="mt-0.5 text-sm text-muted-foreground">
              No critical areas detected across your {profile.count} backtest{profile.count > 1 ? 's' : ''}. Keep following the guidelines below to maintain consistency.
            </p>
          </div>
        </div>
      )}

      {/* Full reference */}
      <div className="space-y-3">
        {(profile || !loading) && (
          <h2 className="text-sm font-semibold text-foreground">
            {focusAreas.length > 0 ? 'Full reference' : 'Guidelines'}
          </h2>
        )}
        <Tabs defaultValue="risk">
          <TabsList>
            {categories.map((cat) => {
              const Icon = cat.icon
              const isFocus = focusAreas.some((a) => a.categoryId === cat.id)
              return (
                <TabsTrigger key={cat.id} value={cat.id} className="flex items-center gap-1.5">
                  <Icon className="h-3.5 w-3.5" />
                  {cat.title}
                  {isFocus && <span className="h-1.5 w-1.5 rounded-full bg-destructive" />}
                </TabsTrigger>
              )
            })}
          </TabsList>
          {categories.map((cat) => (
            <TabsContent key={cat.id} value={cat.id} className="mt-4">
              <RulesTab category={cat} />
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  )
}
