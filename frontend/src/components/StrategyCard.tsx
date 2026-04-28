import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { trainStrategy } from '../api/strategies'
import { StrategyMetadata } from '../types'
import { Play, Loader2 } from 'lucide-react'

interface StrategyCardProps {
  strategy: StrategyMetadata
}

export default function StrategyCard({ strategy }: StrategyCardProps) {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingError, setTrainingError] = useState<string | null>(null)

  const handleTrain = async () => {
    setIsTraining(true)
    setTrainingError(null)

    try {
      await trainStrategy(strategy.name, {}) // Empty config for now
      // You could show a success message or refresh the strategies list
      alert('Training started successfully!')
    } catch (error: any) {
      setTrainingError(error.message || 'Failed to start training')
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <CardTitle className="text-lg">{strategy.name}</CardTitle>
        <CardDescription>{strategy.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Type:</span>
            <Badge variant="secondary">{strategy.type}</Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Trainable:</span>
            <Badge variant={strategy.can_train ? "default" : "outline"}>
              {strategy.can_train ? "Yes" : "No"}
            </Badge>
          </div>
          {strategy.can_train && (
            <div className="pt-2">
              <Button
                onClick={handleTrain}
                disabled={isTraining}
                size="sm"
                className="w-full"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Train Strategy
                  </>
                )}
              </Button>
              {trainingError && (
                <p className="text-sm text-destructive mt-2">{trainingError}</p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}