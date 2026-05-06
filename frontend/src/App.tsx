import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './components/ThemeProvider'
import { Sidebar, MobileSidebar } from './components/Sidebar'
import {
  MarketBriefPage,
  BacktestsPage,
  PredictionsPage,
  ScriptsPage,
  StrategyPerformancePage,
  TradePlanPage,
  RecommendationsPage,
} from './routes'

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false)

  return (
    <ThemeProvider defaultTheme="dark">
      <div
        className="flex min-h-screen bg-background"
        style={
          {
            // Keep desktop content aligned with a fixed-position sidebar
            '--sidebar-width': sidebarCollapsed ? '3.5rem' : '14rem',
          } as React.CSSProperties
        }
      >
        {/* Desktop Sidebar */}
        <div className="hidden md:block">
          <Sidebar collapsed={sidebarCollapsed} onCollapsedChange={setSidebarCollapsed} />
        </div>

        {/* Main Content */}
        <div className="flex-1 md:pl-[var(--sidebar-width)]">
          {/* Mobile Header */}
          <header className="sticky top-0 z-30 flex h-12 items-center gap-3 border-b border-tv-border-primary bg-tv-bg-secondary px-3 md:hidden">
            <MobileSidebar />
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-tv-text-primary">
                OpenTrade
              </span>
            </div>
          </header>

          {/* Page Content */}
          <main className="p-3 md:p-4 animate-fade-in">
            <React.Suspense
              fallback={
                <div className="py-10 text-center text-sm text-tv-text-tertiary">
                  Loading…
                </div>
              }
            >
              <Routes>
                <Route path="/" element={<Navigate to="/predictions" replace />} />
                <Route path="/brief" element={<MarketBriefPage />} />
                <Route path="/backtests" element={<BacktestsPage />} />
                <Route path="/predictions" element={<PredictionsPage />} />
                <Route path="/scripts" element={<ScriptsPage />} />
                <Route path="/strategy-performance" element={<StrategyPerformancePage />} />
                <Route path="/trade-plan" element={<TradePlanPage />} />
                <Route path="/recommendations" element={<RecommendationsPage />} />
              </Routes>
            </React.Suspense>
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}
