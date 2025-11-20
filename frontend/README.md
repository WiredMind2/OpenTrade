# Frontend for Trading Backtester

_Updated: 2025-11-20_

A modern React TypeScript frontend for the Trading Backtester API, featuring interactive OHLC candlestick charts.

## Features

- **Interactive Charts**: OHLC candlestick charts with volume histograms
- **Real-time Data**: Live price data visualization
- **Advanced Features**: Customizable colors and responsive design
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- **TypeScript**: Full type safety with comprehensive interfaces
- **Performance**: Optimized with React.memo, useMemo, and data caching
## Advanced Chart Features

The OHLCChart component supports advanced features for enhanced trading analysis:

- **Prediction Overlays**: Display AI-generated trading predictions directly on the chart
- **Confidence Bands**: Visualize prediction confidence intervals with shaded bands
- **Prediction Aggregation**: Aggregate multiple predictions for comprehensive market insights

## Components


### OHLCChart

Low-level chart component using TradingView's lightweight-charts library.

```tsx
<OHLCChart
  data={chartData}
  showVolume={true}
  showConfidence={true}
  height="500px"
/>
```

**Props:**
- `data`: Array of OHLC data points
- `showVolume`: Display volume histogram
- `bullishColor`: Color for up candles
- `bearishColor`: Color for down candles
- `height`: Chart container height

## Setup

### Prerequisites
- Node.js 16+ (Node 18 recommended)
- npm or yarn

### Installation

```bash
cd frontend
# PowerShell
$env:VITE_API_BASE = 'http://localhost:8000'
npm install
# Or on Unix-like shells
export VITE_API_BASE='http://localhost:8000'
```

### Configuration

Set the API base URL (optional, defaults to `http://localhost:8000`):
# Start the frontend dev server
npm run dev

```bash
To run the frontend and backend together during development, run the backend (`uvicorn backend.main:app --reload`) in one terminal and then start the frontend in another.
# Linux/Mac
export VITE_API_BASE='http://localhost:8000'

# Windows PowerShell
# Build the frontend for production
npm run build
$env:VITE_API_BASE = 'http://localhost:8000'

# Windows CMD
set VITE_API_BASE=http://localhost:8000
```
npm run type-check

### Development

```bash
npm run dev
```

### Build
npm run build
```

### Type Checking

```bash
npm run type-check
```




## API Integration

The frontend communicates with the backend API for:

- **Predictions**: `/predictions/recent`
- **Health Checks**: `/health`

All API calls include proper error handling and loading states.

## Styling

Built with Tailwind CSS for consistent, responsive design. Custom chart colors can be configured via component props.

## Performance

- Memoized computations with React.useMemo
- Optimized re-renders with React.memo
- Lazy loading of chart components
