import instance from '../services/api'

export interface NewsArticle {
  ticker: string
  title: string
  summary: string
  source: string
  url: string
  published_at: string
  relevance_score: number
  sentiment?: 'positive' | 'negative' | 'neutral'
  impact?: 'low' | 'medium' | 'high'
}

export const getNews = async (ticker: string): Promise<NewsArticle[]> => {
  const q = ticker.trim() ? `?ticker=${encodeURIComponent(ticker.trim())}` : ''
  const response = await instance.get(`/api/news${q}`)
  return response.data
}