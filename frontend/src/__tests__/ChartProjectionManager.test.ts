import { attachProjectionManager, detachProjectionManager, ProjectionManagerOptions } from '../lib/ChartProjectionManager';

// Mock the TradingView widget
const mockChart = {
  subscribeClick: jest.fn(),
  timeScale: jest.fn(),
  priceScale: jest.fn(),
  symbolExt: jest.fn(),
};

const mockWidget = {
  onChartReady: jest.fn(),
  chart: jest.fn(() => mockChart),
};

describe('ChartProjectionManager', () => {
  let mockOptions: ProjectionManagerOptions;
  let mockOnProjectionRequest: jest.Mock;
  let mockOnProjectionRendered: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();

    mockOnProjectionRequest = jest.fn();
    mockOnProjectionRendered = jest.fn();

    mockOptions = {
      onProjectionRequest: mockOnProjectionRequest,
      onProjectionRendered: mockOnProjectionRendered,
    };

    // Reset mock implementations
    mockWidget.onChartReady.mockImplementation((callback) => callback());
    mockChart.subscribeClick.mockReturnValue(jest.fn());
    mockChart.timeScale.mockReturnValue({
      coordinateToTime: jest.fn(),
    });
    mockChart.priceScale.mockReturnValue({
      coordinateToPrice: jest.fn(),
    });
    mockChart.symbolExt.mockReturnValue({
      tick_size: 0.01,
    });
  });

  afterEach(() => {
    // Clean up after each test
    detachProjectionManager();
  });

  describe('attachProjectionManager', () => {
    it('attaches click handler to chart when widget is ready', () => {
      attachProjectionManager(mockWidget as any, mockOptions);

      expect(mockWidget.onChartReady).toHaveBeenCalledWith(expect.any(Function));
      expect(mockChart.subscribeClick).toHaveBeenCalledWith(expect.any(Function));
    });

    it('detaches previous manager when attaching new one', () => {
      const unsubscribeMock = jest.fn();
      mockChart.subscribeClick.mockReturnValue(unsubscribeMock);

      // Attach first manager
      attachProjectionManager(mockWidget as any, mockOptions);
      expect(unsubscribeMock).not.toHaveBeenCalled();

      // Attach second manager (should detach first)
      attachProjectionManager(mockWidget as any, mockOptions);
      expect(unsubscribeMock).toHaveBeenCalled();
    });

    it('handles click events and converts coordinates to time and price', () => {
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(1609459200); // 2021-01-01 00:00:00 UTC
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.5);

      mockOnProjectionRequest.mockResolvedValue([]);

      attachProjectionManager(mockWidget as any, mockOptions);

      // Simulate click
      clickHandler({ x: 100, y: 200 });

      expect(mockChart.timeScale().coordinateToTime).toHaveBeenCalledWith(100);
      expect(mockChart.priceScale().coordinateToPrice).toHaveBeenCalledWith(200);
    });

    it('rounds price to appropriate tick size', () => {
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(1609459200);
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.123456);
      mockChart.symbolExt.mockReturnValue({ tick_size: 0.01 });

      mockOnProjectionRequest.mockResolvedValue([]);

      attachProjectionManager(mockWidget as any, mockOptions);

      clickHandler({ x: 100, y: 200 });

      expect(mockOnProjectionRequest).toHaveBeenCalledWith(
        { time: 1609459200, price: 150.12 },
        'default',
        {},
        10
      );
    });

    it('uses default tick size when symbol info is not available', () => {
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(1609459200);
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.123456);
      mockChart.symbolExt.mockReturnValue(null);

      mockOnProjectionRequest.mockResolvedValue([]);

      attachProjectionManager(mockWidget as any, mockOptions);

      clickHandler({ x: 100, y: 200 });

      expect(mockOnProjectionRequest).toHaveBeenCalledWith(
        { time: 1609459200, price: 150.12 },
        'default',
        {},
        10
      );
    });

    it('ignores clicks when coordinate conversion returns null', () => {
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(null);
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.5);

      attachProjectionManager(mockWidget as any, mockOptions);

      clickHandler({ x: 100, y: 200 });

      expect(mockOnProjectionRequest).not.toHaveBeenCalled();
    });

    it('calls onProjectionRendered when projection request succeeds', async () => {
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(1609459200);
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.5);

      const mockPoints = [
        { time: 1609459200, price: 150.5 },
        { time: 1609545600, price: 152.0 },
      ];
      mockOnProjectionRequest.mockResolvedValue(mockPoints);

      attachProjectionManager(mockWidget as any, mockOptions);

      clickHandler({ x: 100, y: 200 });

      await new Promise(process.nextTick); // Wait for promise resolution

      expect(mockOnProjectionRendered).toHaveBeenCalledWith(mockPoints);
    });

    it('logs error when projection request fails', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      const clickHandler = jest.fn();
      mockChart.subscribeClick.mockImplementation((handler) => {
        clickHandler.mockImplementation(handler);
        return jest.fn();
      });

      mockChart.timeScale().coordinateToTime.mockReturnValue(1609459200);
      mockChart.priceScale().coordinateToPrice.mockReturnValue(150.5);

      const mockError = new Error('Projection failed');
      mockOnProjectionRequest.mockRejectedValue(mockError);

      attachProjectionManager(mockWidget as any, mockOptions);

      clickHandler({ x: 100, y: 200 });

      await new Promise(process.nextTick); // Wait for promise rejection

      expect(consoleSpy).toHaveBeenCalledWith('Projection request failed:', mockError);
      consoleSpy.mockRestore();
    });
  });

  describe('detachProjectionManager', () => {
    it('unsubscribes from chart click events', () => {
      const unsubscribeMock = jest.fn();
      mockChart.subscribeClick.mockReturnValue(unsubscribeMock);

      attachProjectionManager(mockWidget as any, mockOptions);
      detachProjectionManager();

      expect(unsubscribeMock).toHaveBeenCalled();
    });

    it('clears attached widget reference', () => {
      const unsubscribeMock = jest.fn();
      mockChart.subscribeClick.mockReturnValue(unsubscribeMock);

      attachProjectionManager(mockWidget as any, mockOptions);
      detachProjectionManager();

      // Try to attach again - should not call unsubscribe on previous subscription
      attachProjectionManager(mockWidget as any, mockOptions);
      expect(unsubscribeMock).toHaveBeenCalledTimes(1);
    });

    it('handles detach when no manager is attached', () => {
      expect(() => detachProjectionManager()).not.toThrow();
    });
  });
});