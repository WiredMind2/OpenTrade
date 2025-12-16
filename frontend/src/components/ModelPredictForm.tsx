import React, { useState } from 'react'
import { predictWithModel } from '../services/api'
import { ModelSummary, ModelPredictResponse } from '../types'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import Loading from './Loading'
import ErrorMessage from './ErrorMessage'

interface ModelPredictFormProps {
  model: ModelSummary
}

export default function ModelPredictForm({ model }: ModelPredictFormProps) {
  const [formData, setFormData] = useState({
    start_date: '',
    end_date: '',
    tickers: '',
    short_ma: '5',
    medium_ma: '20',
    long_ma: '50',
    skip_optimization: false
  })
  const [predictions, setPredictions] = useState<ModelPredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const inputs = {
        start_date: formData.start_date,
        end_date: formData.end_date,
        tickers: formData.tickers.split(',').map(t => t.trim()),
        ma_periods: {
          short: parseInt(formData.short_ma),
          medium: parseInt(formData.medium_ma),
          long: parseInt(formData.long_ma)
        },
        skip_optimization: formData.skip_optimization
      }

      const config = {} // Additional config if needed

      const result = await predictWithModel(model.name, inputs, config)
      setPredictions(result)
    } catch (err) {
      setError('Failed to generate predictions')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Generate Predictions</CardTitle>
          <CardDescription>
            Configure parameters for {model.name} model predictions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Start Date</label>
                <Input
                  type="date"
                  value={formData.start_date}
                  onChange={(e) => handleInputChange('start_date', e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">End Date</label>
                <Input
                  type="date"
                  value={formData.end_date}
                  onChange={(e) => handleInputChange('end_date', e.target.value)}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Tickers (comma-separated)</label>
              <Input
                value={formData.tickers}
                onChange={(e) => handleInputChange('tickers', e.target.value)}
                placeholder="AAPL, MSFT, GOOGL"
                required
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Short MA</label>
                <Input
                  type="number"
                  value={formData.short_ma}
                  onChange={(e) => handleInputChange('short_ma', e.target.value)}
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Medium MA</label>
                <Input
                  type="number"
                  value={formData.medium_ma}
                  onChange={(e) => handleInputChange('medium_ma', e.target.value)}
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Long MA</label>
                <Input
                  type="number"
                  value={formData.long_ma}
                  onChange={(e) => handleInputChange('long_ma', e.target.value)}
                  min="1"
                />
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="skip_optimization"
                checked={formData.skip_optimization}
                onChange={(e) => handleInputChange('skip_optimization', e.target.checked)}
              />
              <label htmlFor="skip_optimization" className="text-sm">
                Skip optimization (use fixed MA periods)
              </label>
            </div>

            <Button type="submit" disabled={loading}>
              {loading ? 'Generating...' : 'Generate Predictions'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {loading && <Loading />}

      {error && <ErrorMessage message={error} />}

      {predictions && (
        <Card>
          <CardHeader>
            <CardTitle>Predictions Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Date</th>
                    <th className="text-left p-2">Ticker</th>
                    <th className="text-left p-2">Signal</th>
                    <th className="text-left p-2">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.predictions.map((pred: any, index: number) => (
                    <tr key={index} className="border-b">
                      <td className="p-2">{pred.date || 'N/A'}</td>
                      <td className="p-2">{pred.ticker || 'N/A'}</td>
                      <td className="p-2">{pred.signal || 'N/A'}</td>
                      <td className="p-2">{pred.confidence ? pred.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}