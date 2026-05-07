import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Separator } from '../components/ui/separator'
import { Skeleton } from '../components/ui/skeleton'
import { ShieldAlert, PieChart, Brain, FlaskConical, TrendingUp, CheckCircle2 } from 'lucide-react'
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
  icon: React.ElementType
  rules: Rule[]
}

interface UserProfile {
  avgSharpe: number
  avgDrawdown: number   // negative, e.g. -0.18
  avgWinRate: number    // 0–1
  avgReturn: number     // e.g. 0.12
}

// ── Static content ───────────────────────────────────────────────────────────

const categories: Category[] = [
  {
    id: 'risk',
    title: 'Risk Management',
    icon: ShieldAlert,
    rules: [
      { text: 'Never put more than 1–2% of your total capital on a single trade.', level: 'critical' },
      { text: 'Always set a stop-loss before entering a trade, not after.', level: 'critical' },
      { text: 'If you have lost more than 10% of your capital, halve your position sizes until you recover.', level: 'important' },
    ],
  },
  {
    id: 'diversification',
    title: 'Diversification',
    icon: PieChart,
    rules: [
      { text: 'Don\'t put more than 20% of your money into one asset or one sector.', level: 'critical' },
      { text: 'Use several different strategies. They won\'t all lose at the same time.', level: 'important' },
      { text: 'Keep 10–20% of your capital in cash so you can act when opportunities appear.', level: 'note' },
    ],
  },
  {
    id: 'backtesting',
    title: 'Backtesting',
    icon: FlaskConical,
    rules: [
      { text: 'Always test your strategy on data it has never seen, not the data you used to build it.', level: 'critical' },
      { text: 'Include transaction fees in your backtest. They eat into profits more than you think.', level: 'critical' },
      { text: 'A profitable backtest does not guarantee real profits. The market changes.', level: 'important' },
    ],
  },
  {
    id: 'entries',
    title: 'Entries and Signals',
    icon: TrendingUp,
    rules: [
      { text: 'Wait for at least two signals to agree before entering a trade.', level: 'important' },
      { text: 'Only enter if the potential gain is at least twice the potential loss (1:2 ratio).', level: 'important' },
    ],
  },
  {
    id: 'discipline',
    title: 'Discipline',
    icon: Brain,
    rules: [
      { text: 'Don\'t change your stop-loss once a trade is open.', level: 'critical' },
      { text: 'If you lose 3 trades in a row, stop and review what went wrong before continuing.', level: 'important' },
      { text: 'Write down every trade: why you entered and why you exited.', level: 'important' },
      { text: 'A winning streak doesn\'t mean you should bet bigger. It usually ends badly.', level: 'note' },
    ],
  },
]

// ── Adaptive logic ────────────────────────────────────────────────────────────

interface FocusArea {
  categoryId: string
  severity: 'high' | 'medium'
}

function deriveFocusAreas(profile: UserProfile): FocusArea[] {
  const areas: FocusArea[] = []

  if (profile.avgDrawdown < -0.10)
    areas.push({ categoryId: 'risk', severity: profile.avgDrawdown < -0.20 ? 'high' : 'medium' })

  if (profile.avgSharpe < 1.0)
    areas.push({ categoryId: 'diversification', severity: profile.avgSharpe < 0.5 ? 'high' : 'medium' })

  if (profile.avgWinRate < 0.45)
    areas.push({ categoryId: 'entries', severity: profile.avgWinRate < 0.35 ? 'high' : 'medium' })

  if (profile.avgReturn < 0)
    areas.push({ categoryId: 'backtesting', severity: 'high' })

  if (profile.avgDrawdown < -0.20 && profile.avgReturn < 0)
    areas.push({ categoryId: 'discipline', severity: 'high' })

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
  const srColor = profile.avgSharpe < 0.5 ? 'text-destructive' : profile.avgSharpe < 1.0 ? 'text-warning' : 'text-success'
  const wrColor = profile.avgWinRate < 0.35 ? 'text-destructive' : profile.avgWinRate < 0.50 ? 'text-warning' : 'text-success'
  const retColor = profile.avgReturn < 0 ? 'text-destructive' : 'text-success'

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold">Your Performance Profile</CardTitle>
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
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
          <CardTitle className="text-sm font-semibold">{category.title}</CardTitle>
        </div>
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
          <h2 className="text-sm font-semibold text-foreground">Recommended for you</h2>
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
            <p className="text-sm font-semibold text-foreground">Everything looks good</p>
            <p className="mt-0.5 text-sm text-muted-foreground">
              No warning on your backtests. Keep it up and stick to the guidelines below.
            </p>
          </div>
        </div>
      )}

      {/* Full reference */}
      <div className="space-y-3">
        {(profile || !loading) && (
          <h2 className="text-sm font-semibold text-foreground">
            {focusAreas.length > 0 ? 'All guidelines' : 'All guidelines'}
          </h2>
        )}
        <Tabs defaultValue="risk">
          <TabsList>
            {categories.map((cat) => {
              const Icon = cat.icon
              return (
                <TabsTrigger key={cat.id} value={cat.id} className="flex items-center gap-1.5">
                  <Icon className="h-3.5 w-3.5" />
                  {cat.title}
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
