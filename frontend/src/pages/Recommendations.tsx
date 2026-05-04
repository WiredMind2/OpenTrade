import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Separator } from '../components/ui/separator'
import { ShieldAlert, PieChart, Brain, FlaskConical, TrendingUp } from 'lucide-react'

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
      {
        text: 'Excessive parameters lead to overfitting. A strategy that only works in-sample is not a strategy.',
        level: 'critical',
      },
      {
        text: 'Account for transaction costs and slippage. Omitting them systematically overstates performance.',
        level: 'important',
      },
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
      {
        text: 'Avoid trading in the 30-minute window surrounding major macro releases (CPI, NFP, FOMC).',
        level: 'important',
      },
      { text: 'Require a minimum risk/reward ratio of 1:2 before every entry.', level: 'important' },
      {
        text: 'The London and New York session opens offer the highest liquidity and tightest spreads.',
        level: 'note',
      },
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
      {
        text: 'Maintain a trading journal documenting entry rationale, exit rationale, and emotional state.',
        level: 'important',
      },
      {
        text: 'Do not increase position sizes following a winning streak. Overconfidence bias is a leading cause of account drawdown.',
        level: 'note',
      },
    ],
  },
]

const levelConfig: Record<string, { label: string; variant: 'destructive' | 'default' | 'secondary' }> = {
  critical:  { label: 'Critical',  variant: 'destructive' },
  important: { label: 'Important', variant: 'default' },
  note:      { label: 'Note',      variant: 'secondary' },
}

export default function Recommendations() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-tv-text-primary">Best Practices</h1>
        <p className="text-sm text-tv-text-secondary mt-1">
          Core guidelines for disciplined, risk-aware trading.
        </p>
      </div>

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
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">{cat.title}</CardTitle>
                <CardDescription>{cat.description}</CardDescription>
              </CardHeader>
              <Separator />
              <CardContent className="pt-0 divide-y divide-border">
                {cat.rules.map((rule, i) => {
                  const cfg = levelConfig[rule.level]
                  return (
                    <div key={i} className="flex items-start gap-4 py-3">
                      <Badge variant={cfg.variant} className="mt-0.5 shrink-0 w-20 justify-center">
                        {cfg.label}
                      </Badge>
                      <div>
                        <p className="text-sm text-tv-text-primary leading-relaxed">{rule.text}</p>
                        {rule.detail && (
                          <p className="text-xs text-tv-text-secondary mt-0.5">{rule.detail}</p>
                        )}
                      </div>
                    </div>
                  )
                })}
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}
