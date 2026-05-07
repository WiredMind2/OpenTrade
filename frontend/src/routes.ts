import React from 'react'

// Route-level code splitting + explicit prefetch hooks.
// Each `loadX()` function is used both by React.lazy() and by hover/idle prefetch.

export const loadMarketBrief = () => import('./pages/MarketBrief')
export const loadBacktests = () => import('./pages/Backtests')
export const loadScripts = () => import('./pages/Scripts')
export const loadStrategyPerformance = () => import('./pages/StrategyPerformance')
export const loadRecommendations = () => import('./pages/Recommendations')
export const loadTradePlan = () => import('./pages/TradePlan')
export const loadPredictions = () => import('./pages/Predictions')

export const MarketBriefPage = React.lazy(loadMarketBrief)
export const BacktestsPage = React.lazy(loadBacktests)
export const PredictionsPage = React.lazy(loadPredictions)
export const ScriptsPage = React.lazy(loadScripts)
export const StrategyPerformancePage = React.lazy(loadStrategyPerformance)
export const TradePlanPage = React.lazy(loadTradePlan)
export const RecommendationsPage = React.lazy(loadRecommendations)

const routePrefetchers: Record<string, () => void> = {
  '/brief': () => void loadMarketBrief(),
  '/backtests': () => void loadBacktests(),
  '/predictions': () => void loadPredictions(),
  '/scripts': () => void loadScripts(),
  '/strategy-performance': () => void loadStrategyPerformance(),
  '/trade-plan': () => void loadTradePlan(),
  '/recommendations': () => void loadRecommendations(),
}

export function prefetchRoute(path: string) {
  routePrefetchers[path]?.()
}

