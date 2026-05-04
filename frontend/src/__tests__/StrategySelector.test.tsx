import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import StrategySelector from '../components/StrategySelector'
import { getStrategies } from '../api/strategies'

jest.mock('../api/strategies', () => ({
  getStrategies: jest.fn(),
}))

const mockStrategies = [
  {
    name: 'moving_average',
    description: 'Moving Average Strategy',
    type: 'technical',
    parameters_schema: {
      window: { type: 'int', default: 20, description: 'Moving average window' },
      threshold: { type: 'float', default: 0.02, description: 'Signal threshold' },
      use_short: { type: 'boolean', default: true, description: 'Use short positions' },
    },
    can_train: false,
  },
  {
    name: 'sentiment_ml',
    description: 'Sentiment ML Strategy',
    type: 'ml',
    parameters_schema: {
      confidence_threshold: { type: 'float', default: 0.8, description: 'Confidence threshold' },
      model_path: { type: 'string', default: '/models/sentiment', description: 'Model path' },
    },
    can_train: true,
  },
]

describe('StrategySelector', () => {
  const mockOnStrategyChange = jest.fn()
  const mockedGetStrategies = getStrategies as jest.MockedFunction<typeof getStrategies>

  beforeEach(() => {
    jest.clearAllMocks()
    mockedGetStrategies.mockResolvedValue(mockStrategies as never)
  })

  it('renders loading state before strategies resolve', async () => {
    mockedGetStrategies.mockImplementationOnce(() => new Promise(() => {}))

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    expect(screen.getByText('Strategy')).toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.getByText('Loading strategies…')).toBeInTheDocument()
  })

  it('fetches strategies and selects the first by default', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => {
      expect(mockedGetStrategies).toHaveBeenCalled()
    })

    expect(mockOnStrategyChange).toHaveBeenCalledWith('moving_average', {
      window: 20,
      threshold: 0.02,
      use_short: true,
    })

    const trigger = screen.getByRole('combobox')
    await user.click(trigger)
    expect(await screen.findByRole('option', { name: /sentiment_ml/i })).toBeInTheDocument()
  })

  it('handles fetch error gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {})
    mockedGetStrategies.mockRejectedValueOnce(new Error('Network error'))

    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch strategies:', expect.any(Error))
    })

    expect(await screen.findByText('Unable to load strategies. Please try again later.')).toBeInTheDocument()
    consoleSpy.mockRestore()
  })

  it('calls onStrategyChange when another strategy is chosen', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => expect(screen.getByRole('combobox')).not.toBeDisabled())

    mockOnStrategyChange.mockClear()

    await user.click(screen.getByRole('combobox'))
    await user.click(screen.getByRole('option', { name: /sentiment_ml/i }))

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenLastCalledWith('sentiment_ml', {
        confidence_threshold: 0.8,
        model_path: '/models/sentiment',
      })
    })
  })

  it('renders parameters for the selected strategy', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /parameters/i })).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: /parameters/i }))

    expect(screen.getByText('window')).toBeInTheDocument()
    expect(screen.getByText('threshold')).toBeInTheDocument()
    expect(screen.getByText('use_short')).toBeInTheDocument()
  })

  it('renders appropriate controls for parameter types', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /parameters/i })).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /parameters/i }))
    await waitFor(() => expect(screen.getByDisplayValue('20')).toBeInTheDocument())

    const windowInput = screen.getByDisplayValue('20')
    expect(windowInput).toHaveAttribute('type', 'number')
    expect(windowInput).toHaveAttribute('step', '1')

    const thresholdInput = screen.getByDisplayValue('0.02')
    expect(thresholdInput).toHaveAttribute('type', 'number')
    expect(thresholdInput).toHaveAttribute('step', 'any')

    expect(screen.getByRole('switch')).toBeChecked()
  })

  it('renders string input for unknown parameter types', async () => {
    const strategiesWithUnknownType = [
      {
        name: 'custom',
        description: 'Custom Strategy',
        type: 'custom',
        parameters_schema: {
          custom_param: { type: 'unknown', default: 'default_value', description: 'Custom parameter' },
        },
        can_train: false,
      },
    ]

    mockedGetStrategies.mockResolvedValueOnce(strategiesWithUnknownType as never)

    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => expect(screen.getByRole('combobox')).not.toBeDisabled())

    await user.click(screen.getByRole('combobox'))
    await user.click(screen.getByRole('option', { name: /custom/i }))

    await user.click(screen.getByRole('button', { name: /parameters/i }))

    await waitFor(() => {
      const input = screen.getByDisplayValue('default_value')
      expect(input).toHaveAttribute('type', 'text')
    })
  })

  it('updates parameters and calls onStrategyChange when values change', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenCalledWith('moving_average', {
        window: 20,
        threshold: 0.02,
        use_short: true,
      })
    })

    await user.click(screen.getByRole('button', { name: /parameters/i }))
    await waitFor(() => expect(screen.getByDisplayValue('20')).toBeInTheDocument())

    mockOnStrategyChange.mockClear()

    const windowInput = screen.getByDisplayValue('20')
    fireEvent.change(windowInput, { target: { value: '30' } })

    await waitFor(() => {
      expect(mockOnStrategyChange).toHaveBeenCalledWith('moving_average', {
        window: 30,
        threshold: 0.02,
        use_short: true,
      })
    })
  })

  it('shows parameter descriptions', async () => {
    const user = userEvent.setup()
    render(<StrategySelector onStrategyChange={mockOnStrategyChange} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /parameters/i })).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /parameters/i }))

    await waitFor(() => {
      expect(screen.getByText('Moving average window')).toBeInTheDocument()
      expect(screen.getByText('Signal threshold')).toBeInTheDocument()
      expect(screen.getByText('Use short positions')).toBeInTheDocument()
    })
  })
})
