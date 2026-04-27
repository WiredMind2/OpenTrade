import { getMockPredictionsForTicker } from '../utils/mockPredictions'

describe('mockPredictions', () => {
  it('starts the first projection point at base time (no +1 day offset)', () => {
    const baseTimeMs = Date.UTC(2026, 3, 24, 0, 0, 0) // 24 Apr 2026 UTC
    const projections = getMockPredictionsForTicker('AAPL', 273, baseTimeMs)

    expect(projections.length).toBeGreaterThan(0)
    expect(projections[0].points.length).toBeGreaterThan(0)
    expect(projections[0].points[0].time).toBe(Math.floor(baseTimeMs / 1000))
  })
})
