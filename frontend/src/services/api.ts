import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const instance = axios.create({
  baseURL: API_BASE,
  timeout: 30000
})


// Add response interceptor for logging
instance.interceptors.response.use(
  (response) => {
    console.log(`API ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`)
    return response
  },
  (error) => {
    console.error(`API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} - ${error.response?.status || 'Network Error'}`, error)
    return Promise.reject(error)
  }
)

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

// MA Predictions API functions
export const generateMAPredictions = async (data: {
  start_date: string
  end_date: string
  short_ma_range?: number[]
  medium_ma_range?: number[]
  long_ma_range?: number[]
  skip_optimization?: boolean
  fixed_short?: number
  fixed_medium?: number
  fixed_long?: number
}) => {
  const response = await instance.post('/scripts/generate-ma-predictions', data)
  return response.data
}

export const getMAPredictionStatus = async (executionId: string) => {
  const response = await instance.get(`/scripts/generate-ma-predictions/status/${executionId}`)
  return response.data
}

// Model API functions
export const getModels = async () => {
  console.log('Fetching models from /api/models')
  const response = await instance.get('/api/models')
  console.log('Models response:', response.data)
  return response.data
}

export const predictWithModel = async (modelName: string, inputs: Record<string, any>, config: Record<string, any>) => {
  const response = await instance.post(`/api/models/${modelName}/predict`, {
    inputs,
    config
  })
  return response.data
}

export const retrainModel = async (modelName: string, trainingPayload: Record<string, any>, config: Record<string, any>, options: Record<string, any>) => {
  const response = await instance.post(`/api/models/${modelName}/retrain`, {
    training_payload: trainingPayload,
    config,
    options
  })
  return response.data
}

export const getJobStatus = async (jobId: string) => {
  const response = await instance.get(`/jobs/${jobId}`)
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
