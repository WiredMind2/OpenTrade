import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import {
  ShieldAlert,
  Scale,
  PieChart,
  Brain,
  FlaskConical,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react'

interface Recommendation {
  text: string
  level: 'critical' | 'important' | 'tip'
}

interface Category {
  title: string
  description: string
  icon: React.ElementType
  recommendations: Recommendation[]
}

const categories: Category[] = [
  {
    title: 'Risk Management',
    description: 'Fundamental rules to preserve capital',
    icon: ShieldAlert,
    recommendations: [
      { text: 'Never risk more than 1–2% of total capital on a single position.', level: 'critical' },
      { text: 'Always define a stop-loss before entering a position.', level: 'critical' },
      { text: 'Keep total portfolio exposure below 20–30% across all open positions simultaneously.', level: 'important' },
      { text: 'If drawdown exceeds 10%, cut position sizes in half until recovery.', level: 'important' },
      { text: 'Never average down on a losing position without a confirmed reversal signal.', level: 'tip' },
    ],
  },
  {
    title: 'Position Sizing',
    description: 'Calculate the optimal size for each trade',
    icon: Scale,
    recommendations: [
      { text: 'Use the Kelly Criterion or a fractional Kelly (0.25–0.5) to calibrate position sizes.', level: 'important' },
      { text: 'Reduce position sizes during high-volatility periods (elevated VIX or abnormally high ATR).', level: 'important' },
      { text: 'Adjust size based on liquidity: avoid representing more than 1% of average daily volume.', level: 'tip' },
    ],
  },
  {
    title: 'Diversification',
    description: 'Spread risk effectively across assets',
    icon: PieChart,
    recommendations: [
      { text: 'Do not concentrate more than 20% of the portfolio in a single sector or asset.', level: 'critical' },
      { text: 'Combine uncorrelated strategies (momentum, mean-reversion, MA crossover) to reduce overall variance.', level: 'important' },
      { text: 'Avoid positions with correlation above 0.8: they amplify drawdowns without adding real diversification.', level: 'important' },
      { text: 'Keep 10–20% in cash to seize opportunities during market corrections.', level: 'tip' },
    ],
  },
  {
    title: 'Backtesting & Validation',
    description: 'Ensure strategies are robust before deployment',
    icon: FlaskConical,
    recommendations: [
      { text: 'Always perform walk-forward testing: do not optimize over the full historical period.', level: 'critical' },
      { text: 'Beware of overfitting: a strategy with too many parameters rarely generalises out-of-sample.', level: 'critical' },
      { text: 'Ensure backtest results account for transaction fees and slippage.', level: 'important' },
      { text: 'Test the strategy across at least 2 full market cycles (bull + bear).', level: 'important' },
      { text: 'A Sharpe Ratio > 1.5 out-of-sample is a reasonable minimum confidence threshold before deployment.', level: 'tip' },
    ],
  },
  {
    title: 'Signals & Entries',
    description: 'Maximise the quality of entry points',
    icon: TrendingUp,
    recommendations: [
      { text: 'Confirm a signal on at least two independent indicators before entering.', level: 'important' },
      { text: 'Avoid trading in the 30 minutes before and after major macroeconomic releases.', level: 'important' },
      { text: 'A minimum risk/reward ratio of 1:2 must be visible before every entry.', level: 'important' },
      { text: 'Prefer entering at the London/New York session open: liquidity is higher and slippage is reduced.', level: 'tip' },
    ],
  },
  {
    title: 'Discipline & Psychology',
    description: 'Maintain operational rigour at all times',
    icon: Brain,
    recommendations: [
      { text: 'Strictly follow the defined trading plan — never move a stop-loss once in a position.', level: 'critical' },
      { text: 'After 3 consecutive losses, stop and review before resuming.', level: 'important' },
      { text: 'Keep a trading journal: record entry/exit reasons and emotional state.', level: 'important' },
      { text: 'Do not oversize after a winning streak: overconfidence bias is one of the most common causes of ruin.', level: 'tip' },
    ],
  },
]

const levelConfig = {
  critical: { label: 'Critical', variant: 'destructive' as const, icon: AlertTriangle },
  important: { label: 'Important', variant: 'default' as const, icon: CheckCircle2 },
  tip: { label: 'Tip', variant: 'secondary' as const, icon: CheckCircle2 },
}

export default function Recommendations() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-tv-text-primary">Recommendations</h1>
        <p className="text-sm text-tv-text-secondary mt-1">
          Best practices for disciplined and sustainable trading
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {categories.map((category) => {
          const Icon = category.icon
          return (
            <Card key={category.title}>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Icon className="h-5 w-5 text-primary shrink-0" />
                  {category.title}
                </CardTitle>
                <CardDescription>{category.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {category.recommendations.map((rec, i) => {
                  const config = levelConfig[rec.level]
                  return (
                    <div
                      key={i}
                      className="flex items-start gap-3 p-2 rounded bg-tv-bg-tertiary"
                    >
                      <Badge variant={config.variant} className="mt-0.5 shrink-0 text-xs">
                        {config.label}
                      </Badge>
                      <p className="text-sm text-tv-text-primary leading-snug">{rec.text}</p>
                    </div>
                  )
                })}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
