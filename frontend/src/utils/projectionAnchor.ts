export interface ProjectionAnchor {
  latestPrice: number
  latestTime: number
}

interface ProjectionAnchorOptions {
  maxAttempts?: number
  retryDelayMs?: number
  fallbackAnchor?: () => Promise<ProjectionAnchor | null>
}

const DEFAULT_MAX_ATTEMPTS = 8
const DEFAULT_RETRY_DELAY_MS = 300

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function resolveProjectionAnchor(
  getLatestPrice: () => number | null | undefined,
  getLatestTime: () => number | null | undefined,
  options: ProjectionAnchorOptions = {}
): Promise<ProjectionAnchor | null> {
  const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS
  const retryDelayMs = options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const latestPrice = getLatestPrice()
    const latestTime = getLatestTime()

    if (typeof latestPrice === 'number' && Number.isFinite(latestPrice)
      && typeof latestTime === 'number' && Number.isFinite(latestTime)) {
      return { latestPrice, latestTime }
    }

    if (attempt < maxAttempts - 1) {
      await sleep(retryDelayMs)
    }
  }

  if (options.fallbackAnchor) {
    try {
      return await options.fallbackAnchor()
    } catch {
      return null
    }
  }

  return null
}
