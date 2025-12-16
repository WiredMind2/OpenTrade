import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import StrategySelector from '../components/StrategySelector';

const mockStrategies = [
  {
    name: 'moving_average',
    description: 'Moving Average Strategy',
    type: 'technical',
    parameters_schema: {
      window: { type: 'int', default: 20, description: 'Moving average window' },
      threshold: { type: 'float', default: 0.02, description: 'Signal threshold' },
      use_short: { type: 'boolean', default: true, description: 'Use short positions' }
    },
    can_train: false
  },
  {
    name: 'sentiment_ml',
    description: 'Sentiment ML Strategy',
    type: 'ml',
    parameters_schema: {
      confidence_threshold: { type: 'float', default: 0.8, description: 'Confidence threshold' },
      model_path: { type: 'string', default: '/models/sentiment', description: 'Model path' }
    },
    can_train: true
  }
];

describe('StrategySelector', () => {
  const mockOnStrategyChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  it('renders loading state initially', () => {
    (global.fetch as jest.Mock).mockImplementationOnce(() =>
      new Promise(() => {}) // Never resolves
    );

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    expect(screen.getByText('Select Strategy:')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('fetches and displays strategies on mount', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByText('-- Select a Strategy --')).toBeInTheDocument();
      expect(screen.getByText('moving_average - Moving Average Strategy')).toBeInTheDocument();
      expect(screen.getByText('sentiment_ml - Sentiment ML Strategy')).toBeInTheDocument();
    });
  });

  it('handles fetch error gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch strategies:', expect.any(Error));
    });

    consoleSpy.mockRestore();
  });

  it('calls onStrategyChange when strategy is selected', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'moving_average' } });

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenLastCalledWith('moving_average', {
        window: 20,
        threshold: 0.02,
        use_short: true
      });
    });
  });

  it('renders parameters when strategy is selected', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'moving_average' } });

    await waitFor(() => {
      expect(screen.getByText('Parameters')).toBeInTheDocument();
      expect(screen.getByText('window:')).toBeInTheDocument();
      expect(screen.getByText('threshold:')).toBeInTheDocument();
      expect(screen.getByText('use_short:')).toBeInTheDocument();
    });
  });

  it('renders different input types for parameters', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'moving_average' } });

    await waitFor(() => {
      // Check for number input (int type)
      const windowInput = screen.getByDisplayValue('20');
      expect(windowInput).toHaveAttribute('type', 'number');
      expect(windowInput).toHaveAttribute('step', '1');

      // Check for number input (float type)
      const thresholdInput = screen.getByDisplayValue('0.02');
      expect(thresholdInput).toHaveAttribute('type', 'number');
      expect(thresholdInput).toHaveAttribute('step', 'any');

      // Check for checkbox (boolean type)
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });
  });

  it('renders string input for unknown parameter types', async () => {
    const strategiesWithUnknownType = [{
      name: 'custom',
      description: 'Custom Strategy',
      type: 'custom',
      parameters_schema: {
        custom_param: { type: 'unknown', default: 'default_value', description: 'Custom parameter' }
      },
      can_train: false
    }];

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => strategiesWithUnknownType,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'custom' } });

    await waitFor(() => {
      const input = screen.getByDisplayValue('default_value');
      expect(input).toHaveAttribute('type', 'text');
    });
  });

  it('updates parameters and calls onStrategyChange when parameter values change', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'moving_average' } });

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenCalledWith('moving_average', {
        window: 20,
        threshold: 0.02,
        use_short: true
      });
    });

    // Clear previous calls
    mockOnStrategyChange.mockClear();

    // Change a parameter value
    const windowInput = screen.getByDisplayValue('20');
    fireEvent.change(windowInput, { target: { value: '30' } });

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenCalledWith('moving_average', {
        window: 30,
        threshold: 0.02,
        use_short: true
      });
    });
  });

  it('calls onStrategyChange with empty values when no strategy is selected', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: '' } });

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenCalledWith('', {});
    });
  });

  it('shows parameter descriptions', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStrategies,
    });

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'moving_average' } });

    await waitFor(() => {
      expect(screen.getByText('Moving average window')).toBeInTheDocument();
      expect(screen.getByText('Signal threshold')).toBeInTheDocument();
      expect(screen.getByText('Use short positions')).toBeInTheDocument();
    });
  });
});