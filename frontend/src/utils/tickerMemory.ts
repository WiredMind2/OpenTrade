const ACTIVE_TICKER_KEY = 'opentrade.activeTicker'
export const DEFAULT_TICKER = 'AAPL'

export function normalizeTicker(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase()
}

export function getStoredTicker(): string {
  return getRememberedTicker() || DEFAULT_TICKER
}

export function getRememberedTicker(): string {
  if (typeof window === 'undefined') return ''
  const stored = normalizeTicker(window.localStorage.getItem(ACTIVE_TICKER_KEY))
  return stored
}

export function rememberTicker(value: string | null | undefined): string {
  const normalized = normalizeTicker(value)
  if (!normalized) return ''
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(ACTIVE_TICKER_KEY, normalized)
  }
  return normalized
}
