import React from 'react'
import { ModelSummary } from '../types'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import ModelPredictForm from './ModelPredictForm'
import ModelRetrainForm from './ModelRetrainForm'

interface ModelDetailProps {
  model: ModelSummary
  onBack: () => void
}

export default function ModelDetail({ model, onBack }: ModelDetailProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          ← Back to Models
        </button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{model.name}</CardTitle>
          <CardDescription className="text-base">{model.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <span className="text-sm text-muted-foreground">Type:</span>
              <Badge variant="secondary" className="ml-2">{model.type}</Badge>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Version:</span>
              <span className="ml-2">{model.version}</span>
            </div>
          </div>
          <div>
            <span className="text-sm text-muted-foreground">Capabilities:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {model.capabilities.map((capability) => (
                <Badge key={capability} variant="outline">
                  {capability}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="predict" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="retrain">Retrain</TabsTrigger>
        </TabsList>

        <TabsContent value="predict" className="mt-6">
          <ModelPredictForm model={model} />
        </TabsContent>

        <TabsContent value="retrain" className="mt-6">
          <ModelRetrainForm model={model} />
        </TabsContent>
      </Tabs>
    </div>
  )
}