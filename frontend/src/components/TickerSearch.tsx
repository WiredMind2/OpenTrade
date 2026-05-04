import React, { useCallback, useEffect, useRef, useState } from 'react'
import { searchUdfSymbols, type UdfSearchSymbol } from '../services/api'
import { cn } from '@/lib/utils'

interface TickerSearchProps {
  value: string
  onChange: (ticker: string) => void
  placeholder?: string
  className?: string
}

export default function TickerSearch({ value, onChange, placeholder = 'Search a ticker', className }: TickerSearchProps) {
  const [inputValue, setInputValue] = useState(value)
  const [results, setResults] = useState<UdfSearchSymbol[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setInputValue(value)
  }, [value])

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [])

  const search = useCallback(async (query: string) => {
    if (query.length < 1) {
      setResults([])
      setOpen(false)
      setActiveIndex(-1)
      return
    }
    setLoading(true)
    try {
      const data = await searchUdfSymbols(query, '', 8)
      setResults(data)
      setOpen(data.length > 0)
      setActiveIndex(-1)
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value.toUpperCase()
    setInputValue(raw)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => search(raw), 250)
  }

  const selectTicker = (symbol: string) => {
    const upper = symbol.toUpperCase()
    setInputValue(upper)
    onChange(upper)
    setOpen(false)
    setResults([])
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (activeIndex >= 0 && results[activeIndex]) {
        selectTicker(results[activeIndex].symbol)
      } else {
        onChange(inputValue)
        setOpen(false)
      }
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const handleBlur = (e: React.FocusEvent) => {
    if (!containerRef.current?.contains(e.relatedTarget as Node)) {
      setOpen(false)
      onChange(inputValue)
    }
  }

  return (
    <div ref={containerRef} className={cn('relative', className)} onBlur={handleBlur}>
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onFocus={() => inputValue.length > 0 && results.length > 0 && setOpen(true)}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        className="flex h-8 w-full rounded border border-input bg-background px-3 py-1.5 text-sm text-foreground transition-tv placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:border-primary hover:border-tv-border-hover"
      />
      {loading && (
        <div className="absolute right-2 top-1/2 -translate-y-1/2">
          <div className="h-3 w-3 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        </div>
      )}
      {open && results.length > 0 && (
        <ul
          role="listbox"
          className="absolute z-50 mt-1 w-full max-h-60 overflow-auto rounded-md border border-input bg-background shadow-lg text-sm"
        >
          {results.map((r, i) => (
            <li
              key={r.symbol}
              role="option"
              aria-selected={i === activeIndex}
              onMouseDown={(e) => { e.preventDefault(); selectTicker(r.symbol) }}
              onMouseEnter={() => setActiveIndex(i)}
              className={cn(
                'flex items-center gap-2 px-3 py-2 cursor-pointer',
                i === activeIndex ? 'bg-primary/10 text-primary' : 'hover:bg-muted/60'
              )}
            >
              <span className="font-semibold w-20 shrink-0 truncate">{r.symbol}</span>
              <span className="text-muted-foreground truncate flex-1">{r.description}</span>
              <span className="text-xs text-muted-foreground/70 shrink-0">{r.exchange}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
