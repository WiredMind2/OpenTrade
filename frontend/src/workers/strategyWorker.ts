interface ProjectionPoint {
  time: number;
  price: number;
}

interface WorkerMessage {
  strategy: string;
  startPoint: { time: number; price: number };
  params: Record<string, any>;
  horizon: number;
}

self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  const { strategy, startPoint, params, horizon } = e.data;
  let points: ProjectionPoint[] = [];

  if (strategy === 'moving_average') {
    // Simulate trend continuation with deterministic volatility
    let currentPrice = startPoint.price;
    let currentTime = startPoint.time;
    const volatility = params.volatility || 0.01; // default 1% volatility factor
    const trend = params.trend || 0; // default no trend (price change per step)
    const interval = params.interval || 60000; // default 1 minute in ms

    for (let i = 1; i <= horizon; i++) {
      currentTime += interval;
      // Deterministic simulation: linear trend + sinusoidal volatility
      const change = trend + Math.sin(i * 0.1) * volatility * currentPrice;
      currentPrice += change;
      // Round to 2 decimal places for price precision
      currentPrice = Math.round(currentPrice * 100) / 100;
      points.push({ time: currentTime, price: currentPrice });
    }
  }

  // For other strategies, could add more cases here

  self.postMessage(points);
};