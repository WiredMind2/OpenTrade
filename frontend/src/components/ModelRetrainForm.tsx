import React, { useState } from 'react'
import { retrainModel, getJobStatus } from '../services/api'
import { ModelSummary, RetrainResponse, JobStatus } from '../types'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import Loading from './Loading'
import ErrorMessage from './ErrorMessage'

interface ModelRetrainFormProps {
  model: ModelSummary
}

export default function ModelRetrainForm({ model }: ModelRetrainFormProps) {
  const [formData, setFormData] = useState({
    dataset_file: null as File | null,
    config_params: '',
    background: true
  })
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [polling, setPolling] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    setFormData(prev => ({ ...prev, dataset_file: file }))
  }

  const handleInputChange = (field: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.dataset_file) {
      setError('Please select a dataset file')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // For now, we'll simulate the training payload
      // In a real implementation, you'd upload the file and get a reference
      const trainingPayload = {
        dataset_path: formData.dataset_file.name, // This would be the uploaded file path
        file_size: formData.dataset_file.size
      }

      const config = formData.config_params ? JSON.parse(formData.config_params) : {}
      const options = { background: formData.background }

      const result: RetrainResponse = await retrainModel(model.name, trainingPayload, config, options)

      if (result.job_id) {
        setJobStatus({
          id: result.job_id,
          model_name: model.name,
          status: result.status,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          config: JSON.stringify(config),
          result: '',
          error: ''
        })
        startPolling(result.job_id)
      } else {
        setJobStatus({
          id: 'sync',
          model_name: model.name,
          status: result.status,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          config: JSON.stringify(config),
          result: JSON.stringify(result.model_meta || {}),
          error: ''
        })
      }
    } catch (err) {
      setError('Failed to start retraining')
      console.error('Retraining error:', err)
    } finally {
      setLoading(false)
    }
  }

  const startPolling = (jobId: string) => {
    setPolling(true)
    const poll = async () => {
      try {
        const status = await getJobStatus(jobId)
        setJobStatus(status)

        if (status.status === 'completed' || status.status === 'failed') {
          setPolling(false)
        } else {
          setTimeout(poll, 2000) // Poll every 2 seconds
        }
      } catch (err) {
        console.error('Polling error:', err)
        setPolling(false)
      }
    }
    poll()
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Retrain Model</CardTitle>
          <CardDescription>
            Upload new training data to retrain the {model.name} model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Dataset File</label>
              <Input
                type="file"
                accept=".csv,.json,.parquet"
                onChange={handleFileChange}
                required
              />
              {formData.dataset_file && (
                <p className="text-sm text-muted-foreground mt-1">
                  Selected: {formData.dataset_file.name} ({(formData.dataset_file.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Config Parameters (JSON)</label>
              <textarea
                className="flex min-h-[80px] w-full rounded border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                value={formData.config_params}
                onChange={(e) => handleInputChange('config_params', e.target.value)}
                placeholder='{"learning_rate": 0.001, "epochs": 100}'
              />
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="background"
                checked={formData.background}
                onChange={(e) => handleInputChange('background', e.target.checked)}
              />
              <label htmlFor="background" className="text-sm">
                Run in background
              </label>
            </div>

            <Button type="submit" disabled={loading}>
              {loading ? 'Starting...' : 'Start Retraining'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {jobStatus && (
        <Card>
          <CardHeader>
            <CardTitle>Training Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Job ID:</span>
                <span className="font-mono text-sm">{jobStatus.id}</span>
              </div>
              <div className="flex justify-between">
                <span>Status:</span>
                <span className={`font-medium ${
                  jobStatus.status === 'completed' ? 'text-green-600' :
                  jobStatus.status === 'failed' ? 'text-red-600' :
                  jobStatus.status === 'running' ? 'text-blue-600' : 'text-yellow-600'
                }`}>
                  {jobStatus.status}
                  {polling && ' (polling...)'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Created:</span>
                <span className="text-sm">{new Date(jobStatus.created_at).toLocaleString()}</span>
              </div>
              {jobStatus.status === 'completed' && jobStatus.result && (
                <div>
                  <span className="font-medium">Result:</span>
                  <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-x-auto">
                    {jobStatus.result}
                  </pre>
                </div>
              )}
              {jobStatus.status === 'failed' && jobStatus.error && (
                <div>
                  <span className="font-medium text-red-600">Error:</span>
                  <pre className="text-xs bg-red-50 p-2 rounded mt-1 overflow-x-auto">
                    {jobStatus.error}
                  </pre>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {loading && <Loading />}
      {error && <ErrorMessage message={error} />}
    </div>
  )
}