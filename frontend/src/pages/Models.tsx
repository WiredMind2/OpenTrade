import React, { useState } from 'react'
import { ModelSummary } from '../types'
import ModelList from '../components/ModelList'
import ModelDetail from '../components/ModelDetail'

export default function Models() {
  const [selectedModel, setSelectedModel] = useState<ModelSummary | null>(null)

  const handleModelSelect = (model: ModelSummary) => {
    setSelectedModel(model)
  }

  const handleBack = () => {
    setSelectedModel(null)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Models</h1>
        <p className="text-muted-foreground">
          Manage machine learning models including training, retraining, and deployment for trading predictions
        </p>
      </div>

      {selectedModel ? (
        <ModelDetail model={selectedModel} onBack={handleBack} />
      ) : (
        <ModelList onModelSelect={handleModelSelect} />
      )}
    </div>
  )
}