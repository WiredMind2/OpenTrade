import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OHLCChart from '../components/OHLCChart';
import { projectStrategy } from '../services/strategyApi';

jest.mock('../services/strategyApi', () => ({
  projectStrategy: jest.fn(),
}));

// Mock the TradingView library
jest.mock('../../public/charting_library/charting_library.esm.js', () => ({
  widget: jest.fn(),
}));

// Mock the datafeed
jest.mock('../services/tradingViewUDF', () => ({
  default: jest.fn().mockImplementation(function() {
    return {
      // Mock datafeed methods as needed
    };
  }),
}));

// Mock ChartProjectionManager
jest.mock('../lib/ChartProjectionManager', () => ({
  attachProjectionManager: jest.fn(),
  detachProjectionManager: jest.fn(),
}));

describe('OHLCChart Projection Integration', () => {
  const mockWidget = {
    onChartReady: jest.fn(),
    chart: jest.fn(),
    remove: jest.fn(),
  };

  const mockChart = {
    createMultipointShape: jest.fn(),
    createShape: jest.fn(),
    removeEntity: jest.fn(),
    lastBar: jest.fn(),
    symbolExt: jest.fn(),
    timeScale: jest.fn(),
    priceScale: jest.fn(),
    subscribeClick: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (projectStrategy as jest.Mock).mockResolvedValue([
      { time: '2021-01-01T00:00:00Z', price: 150.0 },
      { time: '2021-01-02T00:00:00Z', price: 152.0 },
      { time: '2021-01-03T00:00:00Z', price: 151.5 },
    ]);

    // Mock TradingView widget constructor
    const mockTradingView = require('../../public/charting_library/charting_library.esm.js');
    mockTradingView.widget.mockImplementation((options: any) => {
      // Simulate chart ready callback
      setTimeout(() => {
        options.onChartReady && options.onChartReady();
      }, 0);
      return mockWidget;
    });

    // Setup chart mock
    mockWidget.chart.mockReturnValue(mockChart);
    mockChart.createMultipointShape.mockReturnValue('line-entity-id');
    mockChart.createShape.mockReturnValue('marker-entity-id');
    mockChart.lastBar.mockReturnValue({
      time: 1609459200000,
      close: 150.5,
    });
    mockChart.symbolExt.mockReturnValue({ tick_size: 0.01 });
    mockChart.timeScale.mockReturnValue({
      coordinateToTime: jest.fn(() => 1609459200),
    });
    mockChart.priceScale.mockReturnValue({
      coordinateToPrice: jest.fn(() => 150.5),
    });
  });

  afterEach(() => {
    jest.clearAllTimers();
  });

  it('integrates projection functionality with chart clicks', async () => {
    const { attachProjectionManager } = require('../lib/ChartProjectionManager');

    render(
      <OHLCChart
        symbol="AAPL"
        strategyName="moving_average"
        params={{ window: 20 }}
        horizon={3}
        mode="interactive"
      />
    );

    // Wait for chart initialization
    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    // Verify projection manager was attached
    expect(attachProjectionManager).toHaveBeenCalledWith(
      mockWidget,
      expect.objectContaining({
        onProjectionRequest: expect.any(Function),
        onProjectionRendered: expect.any(Function),
      })
    );

    // Get the projection options passed to attachProjectionManager
    const projectionOptions = attachProjectionManager.mock.calls[0][1];

    // Test the projection request flow
    const startPoint = { time: 1609459200, price: 150.0 };
    const result = await projectionOptions.onProjectionRequest(startPoint);

    // Verify API call was made
    expect(projectStrategy).toHaveBeenCalledWith(
      'moving_average',
      'AAPL',
      '2021-01-01T00:00:00.000Z',
      150.0,
      { window: 20 },
      3
    );

    // Verify response transformation
    expect(result).toEqual([
      { time: 1609459200, open: 150.0, high: 150.0, low: 150.0, close: 150.0, predicted: true },
      { time: 1609545600, open: 152.0, high: 152.0, low: 152.0, close: 152.0, predicted: true },
      { time: 1609632000, open: 151.5, high: 151.5, low: 151.5, close: 151.5, predicted: true },
    ]);
  });

  it('renders projection entities on chart when projection succeeds', async () => {
    (projectStrategy as jest.Mock).mockResolvedValueOnce([
      { time: '2021-01-01T00:00:00Z', price: 150.0 },
      { time: '2021-01-02T00:00:00Z', price: 152.0 },
    ]);

    const { attachProjectionManager } = require('../lib/ChartProjectionManager');

    render(
      <OHLCChart
        symbol="AAPL"
        strategyName="moving_average"
        params={{}}
        horizon={2}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    const projectionOptions = attachProjectionManager.mock.calls[0][1];

    // Simulate projection rendering
    const transformedPoints = [
      { time: 1609459200, open: 150.0, high: 150.0, low: 150.0, close: 150.0, predicted: true },
      { time: 1609545600, open: 152.0, high: 152.0, low: 152.0, close: 152.0, predicted: true },
    ];

    projectionOptions.onProjectionRendered(transformedPoints);

    // Verify chart entities were created
    expect(mockChart.createMultipointShape).toHaveBeenCalledWith(
      [
        { time: 1609459200, price: 150.0 },
        { time: 1609545600, price: 152.0 },
      ],
      {
        shape: 'trend_line',
        lock: true,
        disableSelection: true,
        disableSave: true,
        overrides: {
          linestyle: 2,
          linewidth: 2,
          linecolor: '#FF6B35',
          transparency: 0,
        },
      }
    );

    // Verify start marker was created
    expect(mockChart.createShape).toHaveBeenCalledWith(
      { time: 1609459200, price: 150.0 },
      {
        shape: 'arrow_up',
        lock: true,
        disableSelection: true,
        disableSave: true,
        overrides: {
          color: '#10B981',
          transparency: 0,
        },
      }
    );
  });

  it('handles projection API errors gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    (projectStrategy as jest.Mock).mockRejectedValueOnce(new Error('Projection failed'));

    const { attachProjectionManager } = require('../lib/ChartProjectionManager');

    render(
      <OHLCChart
        symbol="AAPL"
        strategyName="invalid_strategy"
        params={{}}
        horizon={1}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    const projectionOptions = attachProjectionManager.mock.calls[0][1];

    // Attempt projection request
    const startPoint = { time: 1609459200, price: 150.0 };

    await expect(projectionOptions.onProjectionRequest(startPoint)).rejects.toThrow();

    consoleSpy.mockRestore();
  });

  it('renders multiple live prediction overlays for current symbol', async () => {
    render(
      <OHLCChart
        symbol="AAPL"
        showPredictionProjections={true}
        predictionProjections={[
          {
            id: 'AAPL_moving_average_1',
            ticker: 'AAPL',
            modelName: 'moving_average',
            horizon: 3,
            points: [
              { time: 1609459200, price: 150, confidence: 0.8, upperBound: 152, lowerBound: 148 },
              { time: 1609545600, price: 151, confidence: 0.79, upperBound: 153, lowerBound: 149 },
            ],
            confidence: 0.8,
            color: '#3B82F6',
            createdAt: '2026-04-27T00:00:00Z',
          },
          {
            id: 'AAPL_sentiment_ml_1',
            ticker: 'AAPL',
            modelName: 'sentiment_ml',
            horizon: 3,
            points: [
              { time: 1609459200, price: 149, confidence: 0.74, upperBound: 151, lowerBound: 147 },
              { time: 1609545600, price: 152, confidence: 0.72, upperBound: 154, lowerBound: 150 },
            ],
            confidence: 0.73,
            color: '#8B5CF6',
            createdAt: '2026-04-27T00:00:00Z',
          },
        ]}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    // Only the 2 primary prediction lines should be rendered.
    expect(mockChart.createMultipointShape).toHaveBeenCalledTimes(2);
    expect(mockChart.createMultipointShape).toHaveBeenCalledWith(
      [
        { time: 1609459200, price: 150 },
        { time: 1609545600, price: 151 },
      ],
      expect.objectContaining({
        shape: 'trend_line',
        overrides: expect.objectContaining({ linecolor: '#3B82F6' }),
      })
    );
    expect(mockChart.createMultipointShape).toHaveBeenCalledWith(
      [
        { time: 1609459200, price: 149 },
        { time: 1609545600, price: 152 },
      ],
      expect.objectContaining({
        shape: 'trend_line',
        overrides: expect.objectContaining({ linecolor: '#8B5CF6' }),
      })
    );
  });

  it('clears existing projections before rendering new ones', async () => {
    (projectStrategy as jest.Mock).mockResolvedValueOnce([
      { time: '2021-01-01T00:00:00Z', price: 150.0 },
    ]);

    const { attachProjectionManager } = require('../lib/ChartProjectionManager');

    render(
      <OHLCChart
        symbol="AAPL"
        strategyName="moving_average"
        params={{}}
        horizon={1}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    const projectionOptions = attachProjectionManager.mock.calls[0][1];

    // First projection
    const firstPoints = [
      { time: 1609459200, open: 150.0, high: 150.0, low: 150.0, close: 150.0, predicted: true },
    ];
    projectionOptions.onProjectionRendered(firstPoints);

    // Second projection (should clear first)
    const secondPoints = [
      { time: 1609545600, open: 152.0, high: 152.0, low: 152.0, close: 152.0, predicted: true },
    ];
    projectionOptions.onProjectionRendered(secondPoints);

    // Verify entities were removed
    expect(mockChart.removeEntity).toHaveBeenCalledWith('line-entity-id');
    expect(mockChart.removeEntity).toHaveBeenCalledWith('marker-entity-id');
  });

  it('updates projection settings via ref methods', async () => {
    const chartRef = React.createRef<any>();

    (projectStrategy as jest.Mock).mockResolvedValue([
      { time: '2021-01-01T00:00:00Z', price: 150.0 },
    ]);

    render(
      <OHLCChart
        ref={chartRef}
        symbol="AAPL"
        strategyName="moving_average"
        params={{ window: 20 }}
        horizon={1}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    // Update projection strategy via ref
    chartRef.current?.setProjectionStrategy(
      'sentiment_ml',
      { threshold: 0.8 },
      5,
      'server-side'
    );

    // Verify the chart component state would be updated
    // (This is more of an integration test to ensure the ref API works)
    expect(chartRef.current).toBeDefined();
  });

  it('exposes latest anchor values through ref methods', async () => {
    const chartRef = React.createRef<any>();
    mockChart.lastBar.mockReturnValue({
      time: 1609459200000, // milliseconds
      close: 150.5,
    });

    render(<OHLCChart ref={chartRef} symbol="AAPL" />);

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    expect(chartRef.current?.getLatestPrice()).toBe(150.5);
    expect(chartRef.current?.getLatestTime()).toBe(1609459200);
  });

  it('clears projections via ref method', async () => {
    const chartRef = React.createRef<any>();

    render(
      <OHLCChart
        ref={chartRef}
        symbol="AAPL"
        strategyName="moving_average"
        params={{}}
        horizon={1}
      />
    );

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    // Clear projections via ref
    chartRef.current?.clearProjections();

    // Verify chart entities would be removed
    expect(mockChart.removeEntity).toHaveBeenCalled();
  });

  it('handles chart initialization errors', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    const mockTradingView = require('../../public/charting_library/charting_library.esm.js');
    mockTradingView.widget.mockImplementation(() => {
      throw new Error('Chart initialization failed');
    });

    render(<OHLCChart symbol="AAPL" />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to initialize TradingView chart:', expect.any(Error));
    });

    consoleSpy.mockRestore();
  });

  it('cleans up resources on unmount', async () => {
    const { unmount } = render(<OHLCChart symbol="AAPL" />);

    await waitFor(() => {
      expect(mockWidget.onChartReady).toHaveBeenCalled();
    });

    const { detachProjectionManager } = require('../lib/ChartProjectionManager');

    unmount();

    expect(detachProjectionManager).toHaveBeenCalled();
    expect(mockWidget.remove).toHaveBeenCalled();
  });
});