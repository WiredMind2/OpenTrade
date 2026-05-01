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
  const response = await instance.get(`/api/news?ticker=${ticker}`)
  return response.data
}