import { resolveProjectionAnchor } from '../utils/projectionAnchor'

describe('resolveProjectionAnchor', () => {
  it('returns anchor immediately when latest data exists', async () => {
    const anchor = await resolveProjectionAnchor(
      () => 201.25,
      () => 1714176000,
      { maxAttempts: 2, retryDelayMs: 1 }
    )

    expect(anchor).toEqual({
      latestPrice: 201.25,
      latestTime: 1714176000,
    })
  })

  it('retries until chart data is available', async () => {
    let calls = 0
    const getLatestPrice = jest.fn(() => {
      calls += 1
      return calls < 3 ? null : 199.75
    })
    const getLatestTime = jest.fn(() => (calls < 3 ? null : 1714176000))

    const anchor = await resolveProjectionAnchor(
      getLatestPrice,
      getLatestTime,
      { maxAttempts: 5, retryDelayMs: 1 }
    )

    expect(anchor).toEqual({
      latestPrice: 199.75,
      latestTime: 1714176000,
    })
    expect(getLatestPrice).toHaveBeenCalledTimes(3)
  })

  it('returns null after max retries when data is missing', async () => {
    const anchor = await resolveProjectionAnchor(
      () => null,
      () => null,
      { maxAttempts: 3, retryDelayMs: 1 }
    )

    expect(anchor).toBeNull()
  })

  it('uses fallback anchor when chart anchor never becomes available', async () => {
    const fallbackAnchor = jest.fn(async () => ({
      latestPrice: 273.43,
      latestTime: 1776902400,
    }))

    const anchor = await resolveProjectionAnchor(
      () => null,
      () => null,
      { maxAttempts: 2, retryDelayMs: 1, fallbackAnchor }
    )

    expect(fallbackAnchor).toHaveBeenCalledTimes(1)
    expect(anchor).toEqual({
      latestPrice: 273.43,
      latestTime: 1776902400,
    })
  })
})
