import { PredictionProjection, PredictionProjectionPoint } from '../types';

/**
 * Mock prediction data generator for demonstration purposes
 * Generates template prediction projection data based on design specifications
 */

const PREDICTION_COLORS = [
  '#3B82F6', // Blue
  '#8B5CF6', // Purple
  '#10B981', // Green
  '#F59E0B', // Amber
  '#EF4444', // Red
  '#06B6D4', // Cyan
];

const MODEL_NAMES = [
  'Linear Regression',
  'LSTM Neural Network',
  'Random Forest',
  'Gradient Boosting',
  'ARIMA Time Series',
  'Prophet Forecast'
];

/**
 * Generate a single prediction projection point
 */
function generatePredictionPoint(
  baseTime: number,
  basePrice: number,
  index: number,
  totalPoints: number,
  volatility: number = 0.02
): PredictionProjectionPoint {
  const timeOffset = index * 24 * 60 * 60 * 1000; // 1 day in milliseconds
  const time = baseTime + timeOffset;

  // Generate price with some trend and volatility
  const trend = Math.sin(index / totalPoints * Math.PI) * 0.1; // Slight sinusoidal trend
  const randomChange = (Math.random() - 0.5) * volatility * basePrice;
  const price = basePrice * (1 + trend) + randomChange;

  // Confidence decreases over time
  const confidence = Math.max(0.3, 0.9 - (index / totalPoints) * 0.4);

  // Generate confidence bounds
  const boundWidth = (1 - confidence) * price * 0.2;
  const upperBound = price + boundWidth;
  const lowerBound = price - boundWidth;

  return {
    time: Math.floor(time / 1000), // Convert to Unix timestamp
    price: Math.max(0.01, price),
    confidence,
    upperBound: Math.max(0.01, upperBound),
    lowerBound: Math.max(0.01, lowerBound)
  };
}

/**
 * Generate a complete prediction projection
 */
function generatePredictionProjection(
  ticker: string,
  baseTime: number,
  basePrice: number,
  horizon: number,
  modelIndex: number
): PredictionProjection {
  const points: PredictionProjectionPoint[] = [];
  const numPoints = Math.max(5, Math.min(30, horizon)); // 5-30 points based on horizon

  // Adjust volatility based on model type (some models are more volatile)
  const baseVolatility = 0.02 + (modelIndex * 0.005);

  for (let i = 0; i < numPoints; i++) {
    const point = generatePredictionPoint(baseTime, basePrice, i, numPoints, baseVolatility);
    points.push(point);
  }

  const avgConfidence = points.reduce((sum, p) => sum + p.confidence, 0) / points.length;

  return {
    id: `${ticker}_${MODEL_NAMES[modelIndex].toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`,
    ticker,
    modelName: MODEL_NAMES[modelIndex],
    horizon,
    points,
    confidence: avgConfidence,
    color: PREDICTION_COLORS[modelIndex % PREDICTION_COLORS.length],
    createdAt: new Date().toISOString(),
    metadata: {
      volatility: baseVolatility,
      modelVersion: `v${modelIndex + 1}`,
      trainingDataPoints: Math.floor(Math.random() * 10000) + 5000
    }
  };
}

/**
 * Generate mock prediction projections for a given ticker
 */
export function generateMockPredictions(
  ticker: string,
  basePrice: number = 100,
  baseTime: number = Date.now(),
  numModels: number = 3,
  horizon: number = 14
): PredictionProjection[] {
  const projections: PredictionProjection[] = [];

  // Generate projections from different models
  for (let i = 0; i < numModels; i++) {
    const projection = generatePredictionProjection(ticker, baseTime, basePrice, horizon, i);
    projections.push(projection);
  }

  return projections;
}

/**
 * Generate mock prediction data with multiple tickers
 */
export function generateMockPredictionData() {
  const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
  const basePrices = [180, 380, 140, 155, 250];
  const projections: PredictionProjection[] = [];

  tickers.forEach((ticker, index) => {
    const tickerProjections = generateMockPredictions(
      ticker,
      basePrices[index],
      Date.now(),
      2, // 2 models per ticker
      10 // 10 day horizon
    );
    projections.push(...tickerProjections);
  });

  return {
    projections,
    lastUpdated: new Date().toISOString()
  };
}

/**
 * Get prediction projections for a specific ticker
 */
export function getMockPredictionsForTicker(
  ticker: string,
  basePrice?: number,
  baseTime?: number
): PredictionProjection[] {
  const defaultPrices: Record<string, number> = {
    'AAPL': 180,
    'MSFT': 380,
    'GOOGL': 140,
    'AMZN': 155,
    'TSLA': 250,
    'NVDA': 450,
    'META': 320
  };

  const price = basePrice || defaultPrices[ticker] || 100;
  const time = baseTime || Date.now();
  return generateMockPredictions(ticker, price, time, 3, 14);
}

/**
 * Format prediction point for tooltip display
 */
export function formatPredictionTooltip(point: PredictionProjectionPoint): string {
  const price = `$${point.price.toFixed(2)}`;
  const confidence = `${(point.confidence * 100).toFixed(0)}%`;
  const bounds = point.upperBound && point.lowerBound
    ? ` ($${point.lowerBound.toFixed(2)} - $${point.upperBound.toFixed(2)})`
    : '';

  return `${price}${bounds}\nConfidence: ${confidence}`;
}