import React from 'react'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  return (
    <Card className="border-destructive/50 bg-destructive/5 animate-fade-in">
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 shrink-0" />
          <div className="flex-1">
            <h3 className="font-semibold text-destructive mb-2">Error</h3>
            <p className="text-sm text-muted-foreground mb-4">{message}</p>
            {onRetry && (
              <Button
                onClick={onRetry}
                variant="outline"
                size="sm"
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Retry
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}