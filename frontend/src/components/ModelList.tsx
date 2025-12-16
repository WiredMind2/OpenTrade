import React, { useState, useEffect } from 'react'
import { getModels } from '../services/api'
import { ModelSummary } from '../types'
import ModelCard from './ModelCard'
import Loading from './Loading'
import ErrorMessage from './ErrorMessage'

interface ModelListProps {
  onModelSelect: (model: ModelSummary) => void
}

export default function ModelList({ onModelSelect }: ModelListProps) {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true)
        const data = await getModels()
        setModels(data)
        setError(null)
      } catch (err) {
        setError('Failed to load models')
        console.error('Error fetching models:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchModels()
  }, [])

  if (loading) {
    return <Loading />
  }

  if (error) {
    return <ErrorMessage message={error} />
  }

  if (models.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted-foreground">No models available</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {models.map((model) => (
        <ModelCard
          key={model.name}
          model={model}
          onSelect={onModelSelect}
        />
      ))}
    </div>
  )
}