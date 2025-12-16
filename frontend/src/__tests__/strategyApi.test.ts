import { getStrategies, projectStrategy } from '../services/strategyApi';

describe('strategyApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  describe('getStrategies', () => {
    it('fetches strategies successfully', async () => {
      const mockStrategies = [
        {
          name: 'moving_average',
          description: 'Moving Average Strategy',
          type: 'technical',
          parameters_schema: { window: { type: 'int', default: 20 } },
          can_train: false
        },
        {
          name: 'sentiment_ml',
          description: 'Sentiment ML Strategy',
          type: 'ml',
          parameters_schema: { threshold: { type: 'float', default: 0.8 } },
          can_train: true
        }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStrategies,
      });

      const result = await getStrategies();

      expect(global.fetch).toHaveBeenCalledWith('/api/strategies');
      expect(result).toEqual(mockStrategies);
    });

    it('throws error when response is not ok', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      await expect(getStrategies()).rejects.toThrow('HTTP error! status: 500');
    });

    it('throws error when fetch fails', async () => {
      const mockError = new Error('Network error');
      (global.fetch as jest.Mock).mockRejectedValueOnce(mockError);

      await expect(getStrategies()).rejects.toThrow('Network error');
    });
  });

  describe('projectStrategy', () => {
    const mockProjectionData = [
      { time: '2021-01-01T00:00:00Z', price: 150.0 },
      { time: '2021-01-02T00:00:00Z', price: 152.5 },
      { time: '2021-01-03T00:00:00Z', price: 151.0 },
    ];

    it('projects strategy successfully with all parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      const result = await projectStrategy(
        'moving_average',
        'AAPL',
        '2021-01-01T00:00:00Z',
        150.0,
        { window: 20, threshold: 0.02 },
        3
      );

      expect(global.fetch).toHaveBeenCalledWith('/api/strategies/moving_average/project', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: 'AAPL',
          startTime: '2021-01-01T00:00:00Z',
          startPrice: 150.0,
          params: { window: 20, threshold: 0.02 },
          horizon: 3,
        }),
      });
      expect(result).toEqual(mockProjectionData);
    });

    it('projects strategy with minimal parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      const result = await projectStrategy(
        'sentiment_ml',
        'TSLA',
        '2021-01-01T12:00:00Z',
        800.5,
        {},
        1
      );

      expect(global.fetch).toHaveBeenCalledWith('/api/strategies/sentiment_ml/project', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: 'TSLA',
          startTime: '2021-01-01T12:00:00Z',
          startPrice: 800.5,
          params: {},
          horizon: 1,
        }),
      });
      expect(result).toEqual(mockProjectionData);
    });

    it('throws error when response is not ok', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
      });

      await expect(projectStrategy('invalid_strategy', 'AAPL', '2021-01-01T00:00:00Z', 150.0, {}, 1))
        .rejects.toThrow('HTTP error! status: 400');
    });

    it('throws error when fetch fails', async () => {
      const mockError = new Error('Connection failed');
      (global.fetch as jest.Mock).mockRejectedValueOnce(mockError);

      await expect(projectStrategy('moving_average', 'AAPL', '2021-01-01T00:00:00Z', 150.0, {}, 1))
        .rejects.toThrow('Connection failed');
    });

    it('handles different strategy names', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      await projectStrategy('custom_strategy', 'GOOGL', '2021-01-01T00:00:00Z', 2500.0, {}, 1);

      expect(global.fetch).toHaveBeenCalledWith('/api/strategies/custom_strategy/project', expect.any(Object));
    });

    it('handles different symbols', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      await projectStrategy('moving_average', 'MSFT', '2021-01-01T00:00:00Z', 300.0, {}, 1);

      const callArgs = (global.fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.symbol).toBe('MSFT');
    });

    it('handles different time formats', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      const customTime = '2021-12-31T23:59:59.999Z';
      await projectStrategy('moving_average', 'AAPL', customTime, 150.0, {}, 1);

      const callArgs = (global.fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.startTime).toBe(customTime);
    });

    it('handles different price values', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      await projectStrategy('moving_average', 'AAPL', '2021-01-01T00:00:00Z', 0.01, {}, 1);

      const callArgs = (global.fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.startPrice).toBe(0.01);
    });

    it('handles complex parameter objects', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      const complexParams = {
        window: 50,
        threshold: 0.05,
        use_short: true,
        nested: {
          value: 42,
          list: [1, 2, 3]
        }
      };

      await projectStrategy('moving_average', 'AAPL', '2021-01-01T00:00:00Z', 150.0, complexParams, 1);

      const callArgs = (global.fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.params).toEqual(complexParams);
    });

    it('handles different horizon values', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProjectionData,
      });

      await projectStrategy('moving_average', 'AAPL', '2021-01-01T00:00:00Z', 150.0, {}, 100);

      const callArgs = (global.fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(callArgs.body);
      expect(body.horizon).toBe(100);
    });
  });
});