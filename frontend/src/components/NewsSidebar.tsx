import { useEffect, useState, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { getNews, type NewsArticle } from '@/api/news'

interface NewsSidebarProps {
  ticker: string
}

type SentimentFilter = 'all' | 'positive' | 'negative'

function cleanSummary(summary: string | null | undefined): string {
  if (!summary) return ''
  // Remove HTML artifacts
  let cleaned = summary.replace(/<[^>]*>/g, '')
  // Remove encoded characters like â€¦
  cleaned = cleaned.replace(/â€¦/g, '...')
  cleaned = cleaned.replace(/â€/g, '')
  cleaned = cleaned.replace(/\[\+\d+ chars\]/g, '')
  // Decode common HTML entities
  cleaned = cleaned.replace(/&nbsp;/g, ' ')
  cleaned = cleaned.replace(/&amp;/g, '&')
  cleaned = cleaned.replace(/&lt;/g, '<')
  cleaned = cleaned.replace(/&gt;/g, '>')
  cleaned = cleaned.replace(/&quot;/g, '"')
  return cleaned.trim()
}

function formatPublishedDate(dateString: string | null | undefined): string {
  if (!dateString) return ''
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
    
    if (diffHours < 1) {
      const diffMins = Math.floor(diffMs / (1000 * 60))
      return `${diffMins}m ago`
    } else if (diffHours < 24) {
      return `${diffHours}h ago`
    } else if (diffDays < 7) {
      return `${diffDays}d ago`
    } else {
      return date.toLocaleDateString()
    }
  } catch {
    return ''
  }
}

function truncateSummary(summary: string, maxLength: number = 100): string {
  if (summary.length <= maxLength) return summary
  return summary.substring(0, maxLength).trim() + '...'
}

function SentimentBadge({ sentiment }: { sentiment?: string }) {
  const colors = {
    positive: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    negative: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
  }
  const labels = {
    positive: 'Positive',
    negative: 'Negative',
    neutral: 'Neutral'
  }
  const sent = sentiment || 'neutral'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colors[sent as keyof typeof colors]}`}>
      {labels[sent as keyof typeof labels]}
    </span>
  )
}

function ImpactBadge({ impact }: { impact?: string }) {
  const colors = {
    high: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    low: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
  }
  const labels = {
    high: 'High',
    medium: 'Medium',
    low: 'Low'
  }
  const imp = impact || 'low'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colors[imp as keyof typeof colors]}`}>
      {labels[imp as keyof typeof labels]}
    </span>
  )
}

export function NewsSidebar({ ticker }: NewsSidebarProps) {
  const [articles, setArticles] = useState<NewsArticle[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sentimentFilter, setSentimentFilter] = useState<SentimentFilter>('all')

  const fetchNews = async () => {
    try {
      setError(null)
      console.log('Fetching news for ticker:', ticker || 'global')
      const data = await getNews(ticker || '')
      console.log('Fetched news articles:', data.length)
      setArticles(data.slice(0, 10))
    } catch (err) {
      console.error('Failed to fetch news:', err)
      setError(err instanceof Error ? err.message : 'Failed to load news')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    setLoading(true)
    fetchNews()
    
    // Auto refresh every 60 seconds
    const interval = setInterval(fetchNews, 60000)
    return () => clearInterval(interval)
  }, [ticker])

  const filteredArticles = useMemo(() => {
    if (sentimentFilter === 'all') return articles
    return articles.filter(a => a.sentiment === sentimentFilter)
  }, [articles, sentimentFilter])

  if (loading) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{ticker ? `${ticker} News` : 'Latest News'}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{ticker ? `${ticker} News` : 'Latest News'}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500 text-sm">{error}</p>
        </CardContent>
      </Card>
    )
  }

  if (articles.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{ticker ? `${ticker} News` : 'Latest News'}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            {ticker ? `No news available for ${ticker}` : 'No news available'}
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-shrink-0">
        <CardTitle>{ticker ? `${ticker} News` : 'Latest News'}</CardTitle>
        <div className="flex gap-1 mt-2">
          <Button
            variant={sentimentFilter === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSentimentFilter('all')}
            className="text-xs px-2 py-1 h-6"
          >
            All
          </Button>
          <Button
            variant={sentimentFilter === 'positive' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSentimentFilter('positive')}
            className="text-xs px-2 py-1 h-6"
          >
            Positive
          </Button>
          <Button
            variant={sentimentFilter === 'negative' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSentimentFilter('negative')}
            className="text-xs px-2 py-1 h-6"
          >
            Negative
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto">
        <div className="space-y-3">
          {filteredArticles.map((article, index) => (
            <div key={index} className="border-b border-border pb-3 last:border-0">
              <div className="flex items-center gap-2 mb-1">
                <SentimentBadge sentiment={article.sentiment} />
                <ImpactBadge impact={article.impact} />
              </div>
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block hover:text-primary transition-colors"
              >
                <h3 className="font-medium text-sm line-clamp-2">
                  {article.title}
                </h3>
              </a>
              <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                <span>{article.source}</span>
                <span>•</span>
                <span>{formatPublishedDate(article.published_at)}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                {truncateSummary(cleanSummary(article.summary))}
              </p>
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-xs text-primary hover:underline mt-2"
              >
                Open article →
              </a>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}