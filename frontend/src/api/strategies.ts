import instance from '../services/api'

export interface StrategyMetadata {
  name: string
  description: string
  type: string
  can_train: boolean
  parameters_schema: any
  model_info?: any
}

export const listStrategies = async (): Promise<StrategyMetadata[]> => {
  const response = await instance.get('/api/strategies')
  return response.data
}

export const getStrategy = async (name: string): Promise<StrategyMetadata> => {
  const response = await instance.get(`/api/strategies/${name}`)
  return response.data
}

export const trainStrategy = async (strategyName: string, config: Record<string, any>) => {
  const response = await instance.post(`/api/strategies/${strategyName}/train`, config)
  return response.data
}