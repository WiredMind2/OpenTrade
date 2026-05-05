import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  getTickerPriceOnDate,
  getUdfQuotes,
  getUdfSymbolInfo,
  type PriceDailyRow,
  type UdfQuote,
  type UdfSymbolInfo,
} from '../services/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Calendar, Route } from 'lucide-react'
import OHLCChart from '../components/OHLCChart'
import { NewsSidebar } from '../components/NewsSidebar'
import { getStoredTicker, rememberTicker } from '../utils/tickerMemory'

const popularSymbols = [
  { symbol: 'AAPL', label: 'Apple' },
  { symbol: 'MSFT', label: 'Microsoft' },
  { symbol: 'NVDA', label: 'Nvidia' },
  { symbol: 'TSLA', label: 'Tesla' },
  { symbol: 'SPY', label: 'S&P 500 ETF' },
  { symbol: 'QQQ', label: 'Nasdaq ETF' },
  { symbol: 'BTC-USD', label: 'Bitcoin' },
  { symbol: 'ETH-USD', label: 'Ethereum' },
  { symbol: 'XAUUSD=X', label: 'Gold spot' },
  { symbol: 'GC=F', label: 'Gold futures' },
  { symbol: 'CL=F', label: 'Crude oil' },
]

function formatQuotePrice(value: number | null | undefined, currency = 'USD') {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  const decimals = Math.abs(value) >= 100 ? 2 : 4
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value)
  } catch {
    return `${value.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })} ${currency}`
  }
}

function formatSigned(value: number | undefined, digits = 2) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-'
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`
}

export default function Predictions() {
  const [selectedTicker, setSelectedTicker] = useState(() => getStoredTicker())
  const [symbolInfo, setSymbolInfo] = useState<UdfSymbolInfo | null>(null)
  const [quote, setQuote] = useState<UdfQuote | null>(null)
  const [priceOnDate, setPriceOnDate] = useState<PriceDailyRow | null>(null)
  const [signalAsOfDate, setSignalAsOfDate] = useState('')

  const chartStrategyParams = useMemo(() => ({}), [])
  const sym = selectedTicker.trim().toUpperCase()
  const tradePlanHref = `/trade-plan${sym ? `?ticker=${encodeURIComponent(sym)}${signalAsOfDate ? `&date=${signalAsOfDate}` : ''}` : ''}`
  const strategyPerformanceHref = `/strategy-performance${sym ? `?ticker=${encodeURIComponent(sym)}` : ''}`

  useEffect(() => {
    if (!sym) return
    let cancelled = false
    const loadHeader = async () => {
      try {
        const [info, quotes, asOfPrice] = await Promise.all([
          getUdfSymbolInfo(sym),
          getUdfQuotes([sym]),
          getTickerPriceOnDate(sym, signalAsOfDate.trim() || undefined),
        ])
        if (cancelled) return
        setSymbolInfo(info)
        setQuote(quotes[0] ?? null)
        setPriceOnDate(asOfPrice)
      } catch {
        if (!cancelled) {
          setSymbolInfo(null)
          setQuote(null)
          setPriceOnDate(null)
        }
      }
    }
    void loadHeader()
    const interval = window.setInterval(() => void loadHeader(), signalAsOfDate.trim() ? 60000 : 15000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [sym, signalAsOfDate])

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Predictions</h2>
        <p className="text-muted-foreground">
          Watch the current ticker, check the chart, then build the actual entry, stop, targets, and position size in
          Trade Plan using your chosen strategy.
        </p>
      </div>

      <div className="space-y-4">
        <Card className="border-muted shadow-md">
          <CardContent className="p-0">
            <div className="flex flex-col gap-3 border-b px-3 py-3 md:flex-row md:items-center md:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-lg font-semibold leading-tight">{sym || 'Select ticker'}</h3>
                  <Badge variant="outline">{symbolInfo?.exchange || 'MARKET'}</Badge>
                  <Badge variant="secondary">{symbolInfo?.currency_code || 'USD'}</Badge>
                </div>
                <p className="truncate text-sm text-muted-foreground">
                  {symbolInfo?.description || 'Loading symbol details'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-semibold tabular-nums">
                  {formatQuotePrice(priceOnDate?.close ?? quote?.v?.lp, symbolInfo?.currency_code || 'USD')}
                </p>
                <p className="text-xs text-muted-foreground">
                  Price date: {priceOnDate?.date || 'latest available'}
                </p>
                {!signalAsOfDate.trim() && (
                  <p
                    className={`text-sm tabular-nums ${
                      Number(quote?.v?.ch ?? 0) >= 0 ? 'text-success' : 'text-destructive'
                    }`}
                  >
                    {formatSigned(quote?.v?.ch)} ({formatSigned(quote?.v?.chp)}%)
                  </p>
                )}
              </div>
            </div>
            <div className="flex gap-2 overflow-x-auto px-3 py-2">
              {popularSymbols.map((item) => (
                <Button
                  key={item.symbol}
                  type="button"
                  size="sm"
                  variant={sym === item.symbol ? 'default' : 'outline'}
                  className="shrink-0"
                  title={item.label}
                  onClick={() => {
                    const normalized = rememberTicker(item.symbol)
                    setSelectedTicker(normalized)
                  }}
                >
                  {item.symbol}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-muted shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Route className="h-5 w-5 text-primary" />
              Build the trading decision
            </CardTitle>
            <CardDescription>
              Pick a price date to replay history, or leave it empty so Trade Plan uses the latest available market data.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-[minmax(220px,1fr)_auto] md:items-end">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Price date
                </label>
                <Input
                  type="date"
                  value={signalAsOfDate}
                  onChange={(e) => setSignalAsOfDate(e.target.value)}
                  placeholder="Latest bar"
                />
                <p className="text-xs text-muted-foreground">Uses latest close on or before this date.</p>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button asChild type="button" variant="outline" className="w-full sm:w-auto">
                  <Link to={tradePlanHref}>Build trade plan</Link>
                </Button>
                <Button asChild type="button" variant="ghost" className="w-full sm:w-auto">
                  <Link to={strategyPerformanceHref}>Compare strategies</Link>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1 min-w-0">
            <OHLCChart
              symbol={selectedTicker}
              height="600px"
              strategyName=""
              params={chartStrategyParams}
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
