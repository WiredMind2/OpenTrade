import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import MarketBrief from './pages/MarketBrief'
import Predictions from './pages/Predictions'
import Backtests from './pages/Backtests'
import Scripts from './pages/Scripts'
import StrategyPerformance from './pages/StrategyPerformance'
import Recommendations from './pages/Recommendations'
import { ThemeProvider } from './components/ThemeProvider'
import { Sidebar, MobileSidebar } from './components/Sidebar'

export default function App() {
  return (
    <ThemeProvider defaultTheme="dark">
      <div className="flex min-h-screen bg-background">
        {/* Desktop Sidebar */}
        <div className="hidden md:block">
          <Sidebar />
        </div>

        {/* Main Content */}
        <div className="flex-1 md:pl-56">
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
            <Routes>
              <Route path="/" element={<Navigate to="/brief" replace />} />
              <Route path="/brief" element={<MarketBrief />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/backtests" element={<Backtests />} />
              <Route path="/scripts" element={<Scripts />} />
              <Route path="/strategy-performance" element={<StrategyPerformance />} />
              <Route path="/recommendations" element={<Recommendations />} />
            </Routes>
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}
