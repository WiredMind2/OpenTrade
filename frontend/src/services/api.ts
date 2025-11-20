import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const instance = axios.create({
  baseURL: API_BASE,
  timeout: 30000
})

export default instance

// Script execution API functions
export const executeScript = async (scriptName: string, parameters: Record<string, any> = {}) => {
  const response = await instance.post('/scripts/execute', {
    script_name: scriptName,
    parameters
  })
  return response.data
}

export const getScriptStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/status/${executionId}`)
  return response.data
}

export const listScriptExecutions = async () => {
  const response = await instance.get('/scripts/executions')
  return response.data
}

export const runPipeline = async (steps?: string[]) => {
  const data = steps ? { steps } : undefined
  const response = await instance.post('/scripts/pipeline/run', data)
  return response.data
}

export const getPipelineStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/pipeline/status/${executionId}`)
  return response.data
}

// Predictions API functions
export const getPredictions = async () => {
  const response = await instance.get('/predictions/recent')
  return response.data
}

export const getTickers = async () => {
  const response = await instance.get('/predictions/tickers')
  return response.data
}

export const createPrediction = async (ticker: string, horizon: string) => {
  const response = await instance.post('/predict', { ticker: ticker.toUpperCase(), horizon })
  return response.data
}


// Portfolio API functions
export const getPortfolio = async () => {
  const response = await instance.get('/portfolio/current')
  return response.data
}

// Health API functions
export const getHealth = async () => {
  const response = await instance.get('/health')
  return response.data
}

// Backtests API functions
export const getBacktests = async () => {
  const response = await instance.get('/trading/backtest')
  return response.data
}

export const runBacktest = async (data: {
  strategy_name: string
  start_date: string
  end_date: string
  initial_capital: number
  parameters?: Record<string, any>
}) => {
  const response = await instance.post('/backtest', data)
  return response.data
}

// Mock data generators (for when API is unavailable)
export const generateMockChartData = (totalReturn: number, points: number = 20) => {
  const data = []
  for (let i = 0; i <= points; i++) {
    data.push({
      day: i,
      value: 100000 * (1 + (totalReturn / 100) * (i / points))
    })
  }
  return data
}
