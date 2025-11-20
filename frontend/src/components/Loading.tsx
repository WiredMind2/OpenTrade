import React from 'react'
import { Loader2 } from 'lucide-react'

export default function Loading() {
  return (
    <div className="flex flex-col items-center justify-center p-8 animate-fade-in">
      <Loader2 className="h-8 w-8 animate-spin text-primary mb-3" />
      <p className="text-xs text-tv-text-secondary">Loading...</p>
    </div>
  )
}