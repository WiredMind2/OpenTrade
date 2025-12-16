import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { ModelSummary } from '../types'

interface ModelCardProps {
  model: ModelSummary
  onSelect: (model: ModelSummary) => void
}

export default function ModelCard({ model, onSelect }: ModelCardProps) {
  return (
    <Card
      className="cursor-pointer hover:shadow-lg transition-shadow"
      onClick={() => onSelect(model)}
    >
      <CardHeader>
        <CardTitle className="text-lg">{model.name}</CardTitle>
        <CardDescription>{model.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Type:</span>
            <Badge variant="secondary">{model.type}</Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Version:</span>
            <span className="text-sm">{model.version}</span>
          </div>
          <div className="flex flex-wrap gap-1">
            <span className="text-sm text-muted-foreground mr-2">Capabilities:</span>
            {model.capabilities.map((capability) => (
              <Badge key={capability} variant="outline" className="text-xs">
                {capability}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}