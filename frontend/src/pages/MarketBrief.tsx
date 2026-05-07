import { type FormEvent, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Gauge,
  Newspaper,
  RefreshCw,
  Search,
  Sparkles,
} from 'lucide-react'
import { getBacktests, getPriceHistory, type PriceHistoryRow } from '../services/api'
import { getNews, type NewsArticle } from '../api/news'
import type { BacktestResult } from '../types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import ErrorMessage from '../components/ErrorMessage'
import { Skeleton } from '../components/ui/skeleton'
import { getRememberedTicker, getStoredTicker, rememberTicker } from '../utils/tickerMemory'

type BriefBacktest = BacktestResult & {
  id?: string | number
  status?: string
  error?: string
  ticker?: string | null
  params?: Record<string, any>
}

type MarketMove = {
  ticker: string
  date: string
  close: number
  previousClose: number
  changePct: number
  intradayRangePct: number
  volume: number
  risk: 'lower' | 'moderate' | 'elevated'
}

const MARKET_RISK_BASKET = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'AMZN', 'META']

function pct(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function moversLeaders(moves: MarketMove[], limit: number): MarketMove[] {
  return [...moves].filter((m) => m.changePct > 0).sort((a, b) => b.changePct - a.changePct).slice(0, limit)
}

function moversLaggards(moves: MarketMove[], limit: number): MarketMove[] {
  return [...moves].filter((m) => m.changePct < 0).sort((a, b) => a.changePct - b.changePct).slice(0, limit)
}

function leaderTone(move: MarketMove): 'Constructive' | 'Watch' {
  if (move.changePct > 0.01 && move.risk !== 'elevated') return 'Constructive'
  return 'Watch'
}

function laggardTone(move: MarketMove): 'Cautious' | 'Watch' {
  if (move.changePct < -0.01 && move.risk !== 'elevated') return 'Cautious'
  return 'Watch'
}

function summarizeNews(articles: NewsArticle[]) {
  const positives = articles.filter((a) => a.sentiment === 'positive').length
  const negatives = articles.filter((a) => a.sentiment === 'negative').length
  const highImpact = articles.filter((a) => a.impact === 'high').length
  if (articles.length === 0) return 'No recent headlines are loaded yet.'
  if (highImpact > 0) return `${highImpact} high-impact headline${highImpact > 1 ? 's' : ''} deserve attention.`
  if (positives > negatives) return 'Headline tone is leaning positive.'
  if (negatives > positives) return 'Headline tone is leaning cautious.'
  return 'Headline tone is mixed.'
}

function buildMarketMove(ticker: string, rows: PriceHistoryRow[]): MarketMove | null {
  if (rows.length < 2) return null
  const latest = rows[0]
  const previous = rows[1]
  const close = Number(latest.close)
  const previousClose = Number(previous.close)
  const high = Number(latest.high)
  const low = Number(latest.low)
  if (!Number.isFinite(close) || !Number.isFinite(previousClose) || previousClose <= 0) return null
  const changePct = (close - previousClose) / previousClose
  const intradayRangePct =
    Number.isFinite(high) && Number.isFinite(low) && close > 0
      ? Math.abs(high - low) / close
      : Math.abs(changePct)
  const absMove = Math.abs(changePct)
  const risk: MarketMove['risk'] =
    absMove >= 0.025 || intradayRangePct >= 0.04
      ? 'elevated'
      : absMove >= 0.01 || intradayRangePct >= 0.02
      ? 'moderate'
      : 'lower'
  return {
    ticker,
    date: latest.date,
    close,
    previousClose,
    changePct,
    intradayRangePct,
    volume: Number(latest.volume || 0),
    risk,
  }
}

function marketTake(move: MarketMove) {
  if (move.changePct > 0.01 && move.risk !== 'elevated') {
    return 'Price is moving up with acceptable risk. A smaller position can be considered if it fits the plan.'
  }
  if (move.changePct > 0) {
    return 'Price is up, but risk needs sizing discipline. Avoid chasing an extended move.'
  }
  if (move.changePct < -0.01) {
    return 'Price is moving down. Wait for stabilization unless the strategy is explicitly bearish.'
  }
  return 'Price is mostly flat. The market is not giving a strong directional signal yet.'
}

function riskBadgeVariant(risk: MarketMove['risk']) {
  if (risk === 'elevated') return 'destructive' as const
  if (risk === 'moderate') return 'warning' as const
  return 'success' as const
}

function riskDecision(move: MarketMove) {
  if (move.changePct > 0.01 && move.risk === 'lower') return 'Risk can be considered'
  if (move.changePct > 0 && move.risk !== 'elevated') return 'Use reduced size'
  if (move.risk === 'elevated') return 'Avoid chasing'
  if (move.changePct < -0.01) return 'Wait for support'
  return 'No clear edge'
}

function backtestTicker(run: BriefBacktest | undefined | null): string {
  return String(run?.ticker || run?.params?.ticker || '').trim().toUpperCase()
}

export default function MarketBrief() {
  const [backtests, setBacktests] = useState<BriefBacktest[]>([])
  const [news, setNews] = useState<NewsArticle[]>([])
  const [marketMoves, setMarketMoves] = useState<MarketMove[]>([])
  const [searchedMove, setSearchedMove] = useState<MarketMove | null>(null)
  const [preferredTicker, setPreferredTicker] = useState(() => getStoredTicker())
  const [searchTicker, setSearchTicker] = useState(() => getStoredTicker())
  const [searchingTicker, setSearchingTicker] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadBrief = async () => {
    setLoading(true)
    setError(null)
    try {
      const backtestData = await getBacktests()
      const safeBacktests = Array.isArray(backtestData) ? backtestData : []
      setBacktests(safeBacktests)

      const rememberedTicker = getRememberedTicker()
      const lastBacktestTicker = backtestTicker(safeBacktests.find((item: BriefBacktest) => item.ticker || item.params?.ticker))
      const newsTicker = rememberedTicker || lastBacktestTicker || MARKET_RISK_BASKET[0]
      const symbols = Array.from(
        new Set([
          ...(rememberedTicker ? [rememberedTicker] : []),
          ...(lastBacktestTicker ? [lastBacktestTicker] : []),
          ...MARKET_RISK_BASKET,
        ])
      ).slice(0, 8) as string[]
      const moveResults = await Promise.allSettled(
        symbols.map(async (symbol) => buildMarketMove(symbol, await getPriceHistory(symbol, 2)))
      )
      setMarketMoves(
        moveResults
          .map((result) => (result.status === 'fulfilled' ? result.value : null))
          .filter((move): move is MarketMove => move != null)
      )
      const newsData = await getNews(newsTicker)
      setNews(Array.isArray(newsData) ? newsData.slice(0, 8) : [])
      setPreferredTicker(rememberedTicker)
      setSearchTicker(newsTicker)
      setSearchedMove(null)
      setSearchError(null)
    } catch (e: any) {
      setError(e.message || 'Failed to load market brief')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadBrief()
  }, [])

  const leaders = useMemo(() => moversLeaders(marketMoves, 4), [marketMoves])
  const laggards = useMemo(() => moversLaggards(marketMoves, 4), [marketMoves])
  const completedBacktests = useMemo(
    () => backtests.filter((item) => (item.status ?? item.metrics?.status ?? 'completed') === 'completed'),
    [backtests]
  )
  const lastRelevantBacktest = useMemo(
    () =>
      backtests
        .filter((item) => item.ticker || item.params?.ticker)
        .slice()
        .sort((a, b) => new Date(b.timestamp || b.completed_at || b.end_date).getTime() - new Date(a.timestamp || a.completed_at || a.end_date).getTime())[0],
    [backtests]
  )
  const lastRelevantTicker = backtestTicker(lastRelevantBacktest)
  const targetTicker = searchedMove?.ticker || preferredTicker || lastRelevantTicker
  const bestBacktest = useMemo(
    () => completedBacktests.slice().sort((a, b) => Number(b.total_return || 0) - Number(a.total_return || 0))[0],
    [completedBacktests]
  )
  const failedBacktests = backtests.filter((item) => (item.status ?? item.metrics?.status) === 'failed').length
  const newsSummary = summarizeNews(news)
  const leadMove = useMemo(
    () =>
      searchedMove ||
      marketMoves.find((move) => preferredTicker && move.ticker === preferredTicker) ||
      marketMoves.find((move) => lastRelevantTicker && move.ticker === lastRelevantTicker) ||
      marketMoves.slice().sort((a, b) => Math.abs(b.changePct) - Math.abs(a.changePct))[0],
    [lastRelevantTicker, marketMoves, preferredTicker, searchedMove]
  )
  const sortedMarketMoves = useMemo(
    () => marketMoves.slice().sort((a, b) => Math.abs(b.changePct) - Math.abs(a.changePct)),
    [marketMoves]
  )
  const displayMarketMoves = useMemo(() => {
    if (!leadMove) return sortedMarketMoves.slice(0, 5)
    return [
      leadMove,
      ...sortedMarketMoves.filter((move) => move.ticker !== leadMove.ticker),
    ].slice(0, 5)
  }, [leadMove, sortedMarketMoves])
  const leadMoveSource = searchedMove
    ? 'searched ticker'
    : preferredTicker && leadMove?.ticker === preferredTicker
    ? 'selected ticker'
    : lastRelevantTicker && leadMove?.ticker === lastRelevantTicker
    ? 'last backtest'
    : 'tracked market basket'
  const upMoves = marketMoves.filter((move) => move.changePct > 0).length
  const downMoves = marketMoves.filter((move) => move.changePct < 0).length

  const searchMarketTicker = async (event?: FormEvent) => {
    event?.preventDefault()
    const symbol = searchTicker.trim().toUpperCase()
    if (!symbol) return
    rememberTicker(symbol)
    setPreferredTicker(symbol)
    setSearchingTicker(true)
    setSearchError(null)
    try {
      const move = buildMarketMove(symbol, await getPriceHistory(symbol, 2))
      if (!move) {
        setSearchedMove(null)
        setSearchError(`No recent price history found for ${symbol}.`)
        return
      }
      setSearchedMove(move)
      setMarketMoves((current) => [move, ...current.filter((item) => item.ticker !== move.ticker)])
      try {
        const newsData = await getNews(symbol)
        setNews(Array.isArray(newsData) ? newsData.slice(0, 8) : [])
      } catch {
        setNews([])
        setSearchError(`Loaded ${symbol} direction, but no related headlines could be fetched.`)
      }
    } catch (e: any) {
      setSearchedMove(null)
      setSearchError(e.message || `Failed to load ${symbol}.`)
    } finally {
      setSearchingTicker(false)
    }
  }

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-72" />
        <div className="grid gap-3 md:grid-cols-3">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-28 w-full" />)}
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      </div>
    )
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={loadBrief} />
  }

  return (
    <div className="space-y-5">

      <div className="grid gap-3 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Sparkles className="h-4 w-4 text-primary" />
              Basket breadth
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">
              {marketMoves.length === 0 ? 'No breadth data' : upMoves >= downMoves ? 'Risk-On Watch' : 'Defensive Watch'}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {marketMoves.length === 0
                ? 'Load price history for the basket to see how many names are up vs down on the latest bar.'
                : `${upMoves} of ${marketMoves.length} tracked names up on the latest daily bar vs ${downMoves} down (price history only).`}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Newspaper className="h-4 w-4 text-primary" />
              {targetTicker ? `${targetTicker} News Tone` : 'News Tone'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">{news.length} Headlines</p>
            <p className="mt-2 text-sm text-muted-foreground">{newsSummary}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <BarChart3 className="h-4 w-4 text-primary" />
              Market Direction
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold">
              {leadMove
                ? leadMove.changePct > 0
                  ? `${leadMove.ticker} Moving Up`
                  : leadMove.changePct < 0
                  ? `${leadMove.ticker} Moving Down`
                  : `${leadMove.ticker} Flat`
                : 'No Price Data'}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              {leadMove
                ? `${pct(leadMove.changePct)} on ${leadMove.date}, based on the ${leadMoveSource}. Basket: ${upMoves} up, ${downMoves} down.`
                : "Load price data to read today's market direction."}
            </p>
          </CardContent>
        </Card>
      </div>

      {failedBacktests > 0 && (
        <div className="rounded border border-destructive/30 bg-destructive/5 p-3 text-sm">
          <div className="flex items-center gap-2 font-medium text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {failedBacktests} recent backtest{failedBacktests > 1 ? 's' : ''} failed.
          </div>
          <p className="mt-1 text-muted-foreground">Treat performance conclusions cautiously until the failed runs are resolved.</p>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ArrowUpRight className="h-5 w-5 text-success" />
              Constructive Watchlist
            </CardTitle>
            <CardDescription>Names with the largest positive daily change among the tracked set.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {leaders.length === 0 ? (
              <p className="text-sm text-muted-foreground">No up days in the tracked basket yet.</p>
            ) : leaders.map((item) => (
              <div key={item.ticker} className="rounded bg-tv-bg-tertiary p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-mono text-lg font-semibold">{item.ticker}</p>
                    <p className="text-xs text-muted-foreground">{item.date}</p>
                  </div>
                  <div className="text-right">
                    <Badge variant={leaderTone(item) === 'Constructive' ? 'success' : 'secondary'}>
                      {leaderTone(item)}
                    </Badge>
                    <p className="mt-1 text-sm font-medium text-success">{pct(item.changePct)}</p>
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Intraday range {pct(item.intradayRangePct)} · {item.risk} risk
                </p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ArrowDownRight className="h-5 w-5 text-destructive" />
              Caution Watchlist
            </CardTitle>
            <CardDescription>Names with the largest negative daily change among the tracked set.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {laggards.length === 0 ? (
              <p className="text-sm text-muted-foreground">No down days in the tracked basket yet.</p>
            ) : laggards.map((item) => (
              <div key={item.ticker} className="rounded bg-tv-bg-tertiary p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-mono text-lg font-semibold">{item.ticker}</p>
                    <p className="text-xs text-muted-foreground">{item.date}</p>
                  </div>
                  <div className="text-right">
                    <Badge variant={laggardTone(item) === 'Cautious' ? 'destructive' : 'secondary'}>
                      {laggardTone(item)}
                    </Badge>
                    <p className="mt-1 text-sm font-medium text-destructive">{pct(item.changePct)}</p>
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Intraday range {pct(item.intradayRangePct)} · {item.risk} risk
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Newspaper className="h-5 w-5 text-primary" />
              {targetTicker ? `${targetTicker} Headlines` : 'Headlines To Read'}
            </CardTitle>
            <CardDescription>
              News is filtered to the market direction ticker or the searched ticker.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {news.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {targetTicker ? `No related headlines loaded for ${targetTicker} yet.` : 'No headlines loaded yet.'}
              </p>
            ) : news.slice(0, 5).map((article) => (
              <a
                key={article.url || article.title}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded bg-tv-bg-tertiary p-3 transition-colors hover:bg-tv-bg-hover"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={article.sentiment === 'positive' ? 'success' : article.sentiment === 'negative' ? 'destructive' : 'secondary'}>
                    {article.sentiment ?? 'neutral'}
                  </Badge>
                  <Badge variant={article.impact === 'high' ? 'warning' : 'outline'}>
                    {article.impact ?? 'low'} impact
                  </Badge>
                </div>
                <p className="mt-2 text-sm font-medium leading-snug">{article.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{article.source}</p>
              </a>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Gauge className="h-5 w-5 text-primary" />
                  Today's Market Risk
                </CardTitle>
                <CardDescription>
                  {targetTicker
                    ? `Focused on ${targetTicker}; search any ticker to replace it.`
                    : 'Search any ticker or use the tracked market basket.'}
                </CardDescription>
              </div>
              <form onSubmit={searchMarketTicker} className="flex w-full gap-2 sm:w-64">
                <Input
                  value={searchTicker}
                  onChange={(event) => setSearchTicker(event.target.value.toUpperCase())}
                  placeholder="Search ticker"
                  className="font-mono uppercase"
                  aria-label="Search ticker market direction"
                />
                <Button type="submit" size="sm" disabled={searchingTicker || !searchTicker.trim()}>
                  <Search className="mr-2 h-4 w-4" />
                  {searchingTicker ? 'Loading' : 'Search'}
                </Button>
              </form>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {searchError && (
              <div className="rounded border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
                {searchError}
              </div>
            )}
            {displayMarketMoves.length === 0 ? (
              <p className="text-sm text-muted-foreground">No recent price movement is available yet.</p>
            ) : displayMarketMoves.map((move) => (
              <div key={`${move.ticker}-${move.date}`} className="rounded bg-tv-bg-tertiary p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="font-mono text-base font-semibold">{move.ticker}</p>
                      {leadMove?.ticker === move.ticker && (
                        <Badge variant="outline">{leadMoveSource}</Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {move.date} close ${move.close.toFixed(2)} vs ${move.previousClose.toFixed(2)}
                    </p>
                  </div>
                  <Badge variant={move.changePct >= 0 ? 'success' : 'destructive'}>
                    {move.changePct >= 0 ? '+' : ''}{pct(move.changePct)}
                  </Badge>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  {marketTake(move)}
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Badge variant={riskBadgeVariant(move.risk)}>{move.risk} risk</Badge>
                  <Badge variant={move.changePct > 0 && move.risk !== 'elevated' ? 'success' : move.risk === 'elevated' ? 'destructive' : 'secondary'}>
                    {riskDecision(move)}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Range {pct(move.intradayRangePct)} | Volume {new Intl.NumberFormat().format(move.volume)}
                  </span>
                </div>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">
              Source: latest two daily candles from /data/prices. This is a risk read, not financial advice.
            </p>
            {bestBacktest && (
              <div className="rounded border border-border/70 p-3 text-xs text-muted-foreground">
                Strategy context: best recent completed run was {bestBacktest.strategy_name} at {pct(Number(bestBacktest.total_return || 0))}.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
