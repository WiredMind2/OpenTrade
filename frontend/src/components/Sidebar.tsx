import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  TrendingUp,
  BarChart3,
  Wallet,
  Menu,
  X,
  Activity,
  Settings,
  Brain,
  LineChart,
  BookOpen
} from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { ThemeToggle } from './ThemeToggle'

interface SidebarProps {
  className?: string
}

const navigation = [
  { name: 'Predictions', href: '/predictions', icon: TrendingUp },
  { name: 'Backtests', href: '/backtests', icon: BarChart3 },
  { name: 'Portfolio', href: '/portfolio', icon: Wallet },
  { name: 'Strategies', href: '/strategies', icon: Brain },
  { name: 'Performance', href: '/strategy-performance', icon: LineChart },
  { name: 'Scripts', href: '/scripts', icon: Settings },
  { name: 'Recommendations', href: '/recommendations', icon: BookOpen },
]

export function Sidebar({ className }: SidebarProps) {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen transition-all duration-200',
        collapsed ? 'w-14' : 'w-56',
        'border-r border-tv-border-primary bg-tv-bg-secondary',
        className
      )}
    >
      <div className="flex h-full flex-col">
        {/* Logo/Header */}
        <div className="flex h-12 items-center justify-between px-3 border-b border-tv-border-primary">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              <span className="font-semibold text-sm text-tv-text-primary">
                TradeBot
              </span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            className="ml-auto h-7 w-7"
          >
            {collapsed ? (
              <Menu className="h-4 w-4" />
            ) : (
              <X className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-0.5 p-2 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            const Icon = item.icon
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  'flex items-center gap-2 rounded px-2 py-1.5 text-sm font-medium transition-tv',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-tv-text-secondary hover:text-tv-text-primary hover:bg-tv-bg-hover',
                  collapsed && 'justify-center'
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && (
                  <span>{item.name}</span>
                )}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-tv-border-primary p-2">
          <div className={cn('flex items-center', collapsed ? 'justify-center' : 'justify-between')}>
            <ThemeToggle />
            {!collapsed && (
              <div className="text-xs text-tv-text-tertiary">
                v1.0.0
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}

export function MobileSidebar() {
  const [open, setOpen] = useState(false)
  const location = useLocation()

  // Close sidebar when route changes
  React.useEffect(() => {
    setOpen(false)
  }, [location.pathname])

  return (
    <>
      {/* Mobile Menu Button */}
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        onClick={() => setOpen(!open)}
      >
        <Menu className="h-5 w-5" />
      </Button>

      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-0 z-50 h-screen w-56 border-r border-tv-border-primary bg-tv-bg-secondary transition-transform duration-200 md:hidden',
          open ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex h-full flex-col">
          <div className="flex h-12 items-center justify-between px-3 border-b border-tv-border-primary">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              <span className="font-semibold text-sm text-tv-text-primary">
                TradeBot
              </span>
            </div>
            <Button variant="ghost" size="icon" onClick={() => setOpen(false)} className="h-7 w-7">
              <X className="h-4 w-4" />
            </Button>
          </div>

          <nav className="flex-1 space-y-0.5 p-2 overflow-y-auto">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              const Icon = item.icon
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    'flex items-center gap-2 rounded px-2 py-1.5 text-sm font-medium transition-tv',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-tv-text-secondary hover:text-tv-text-primary hover:bg-tv-bg-hover'
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <span>{item.name}</span>
                </Link>
              )
            })}
          </nav>

          <div className="border-t border-tv-border-primary p-2">
            <div className="flex items-center justify-between">
              <ThemeToggle />
              <div className="text-xs text-tv-text-tertiary">v1.0.0</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
