import React, { useState, useEffect, useRef } from 'react'
import { getStrategies } from '../api/strategies'
import { Input } from './ui/input'
import { Label } from './ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import { Switch } from './ui/switch'
import { ChevronDown } from 'lucide-react'

interface ParameterSchema {
  type: string
  default: unknown
  description: string
}

interface Strategy {
  name: string
  description: string
  type: string
  parameters_schema: Record<string, ParameterSchema>
  can_train: boolean
}

interface StrategySelectorProps {
  onStrategyChange: (strategy: string, params: Record<string, unknown>) => void
}

const StrategySelector: React.FC<StrategySelectorProps> = ({ onStrategyChange }) => {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [error, setError] = useState<string | null>(null)
  const [parametersOpen, setParametersOpen] = useState(false)
  const onStrategyChangeRef = useRef(onStrategyChange)
  useEffect(() => {
    onStrategyChangeRef.current = onStrategyChange
  })

  useEffect(() => {
    getStrategies()
      .then((data: Strategy[]) => {
        setStrategies(data)
        if (data.length > 0) {
          const first = data[0]
          const initialParams: Record<string, unknown> = {}
          for (const [key, schema] of Object.entries(first.parameters_schema)) {
            initialParams[key] = schema.default
          }
          setSelectedStrategy(first.name)
          setParams(initialParams)
          onStrategyChangeRef.current(first.name, initialParams)
        }
      })
      .catch((err) => {
        console.error('Failed to fetch strategies:', err)
        setError('Unable to load strategies. Please try again later.')
      })
  }, [])

  const handleStrategyChange = (name: string) => {
    setSelectedStrategy(name)
    const strategy = strategies.find((s) => s.name === name)
    if (!strategy) {
      setParams({})
      onStrategyChange('', {})
      return
    }

    const initialParams: Record<string, unknown> = {}
    for (const [key, schema] of Object.entries(strategy.parameters_schema)) {
      initialParams[key] = schema.default
    }
    setParams(initialParams)
    onStrategyChange(name, initialParams)
  }

  const handleParamChange = (key: string, value: unknown) => {
    const newParams = { ...params, [key]: value }
    setParams(newParams)
    onStrategyChange(selectedStrategy, newParams)
  }

  const renderParamInput = (key: string, schema: ParameterSchema) => {
    const value = params[key]
    const narrowInput = 'min-w-0 max-w-full'
    switch (schema.type) {
      case 'int':
        return (
          <Input
            id={`param-${key}`}
            type="number"
            step={1}
            className={narrowInput}
            value={String(value ?? '')}
            onChange={(e) => handleParamChange(key, parseInt(e.target.value, 10))}
            placeholder={schema.description}
          />
        )
      case 'float':
        return (
          <Input
            id={`param-${key}`}
            type="number"
            step="any"
            className={narrowInput}
            value={String(value ?? '')}
            onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
            placeholder={schema.description}
          />
        )
      case 'string':
        return (
          <Input
            id={`param-${key}`}
            type="text"
            className={narrowInput}
            value={String(value ?? '')}
            onChange={(e) => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
          />
        )
      case 'bool':
      case 'boolean':
        return (
          <div className="flex min-w-0 items-start gap-2 pt-1">
            <Switch
              id={`param-${key}`}
              className="mt-0.5 shrink-0"
              checked={Boolean(value)}
              onCheckedChange={(checked) => handleParamChange(key, checked)}
            />
            <span className="min-w-0 flex-1 break-words text-sm text-muted-foreground">{schema.description}</span>
          </div>
        )
      default:
        return (
          <Input
            id={`param-${key}`}
            type="text"
            className={narrowInput}
            value={String(value ?? '')}
            onChange={(e) => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
          />
        )
    }
  }

  return (
    <div className="min-w-0 max-w-full space-y-3">
      <div className="min-w-0 space-y-2">
        <Label htmlFor="strategy-select">Strategy</Label>
        {strategies.length === 0 ? (
          <div
            id="strategy-select"
            className="flex h-9 min-w-0 items-center rounded-lg border border-input bg-muted px-3 text-sm text-muted-foreground"
          >
            Loading strategies…
          </div>
        ) : (
          <Select value={selectedStrategy} onValueChange={handleStrategyChange}>
            <SelectTrigger
              id="strategy-select"
              className="h-auto min-h-9 w-full min-w-0 items-start py-2 [&>span]:whitespace-normal [&>span]:break-words [&>span]:line-clamp-none [&>span]:overflow-visible"
            >
              <SelectValue placeholder="Choose a strategy" />
            </SelectTrigger>
            <SelectContent>
              {strategies.map((strategy) => (
                <SelectItem key={strategy.name} value={strategy.name} textValue={`${strategy.name} ${strategy.description}`}>
                  <span className="block w-full min-w-0 max-w-full truncate font-medium leading-tight">
                    {strategy.name}
                  </span>
                  <span className="mt-0.5 block w-full min-w-0 max-w-full whitespace-normal break-words text-xs leading-snug text-muted-foreground [overflow-wrap:anywhere] [word-break:break-word]">
                    {strategy.description}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {selectedStrategy ? (
        <div className="min-w-0 border-t border-border pt-3">
          {(() => {
            const entries = Object.entries(
              strategies.find((s) => s.name === selectedStrategy)?.parameters_schema || {}
            )
            if (entries.length === 0) return null
            return (
              <>
                <button
                  type="button"
                  onClick={() => setParametersOpen((o) => !o)}
                  className="flex w-full items-center justify-between gap-2 rounded-md py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground transition-colors hover:text-foreground"
                  aria-expanded={parametersOpen}
                >
                  <span>
                    Parameters
                    <span className="ml-1.5 normal-case font-normal text-muted-foreground/80">
                      ({entries.length})
                    </span>
                  </span>
                  <ChevronDown
                    className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${parametersOpen ? 'rotate-180' : ''}`}
                    aria-hidden
                  />
                </button>
                {parametersOpen ? (
                  <div className="mt-3 grid min-w-0 grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
                    {entries.map(([key, schema]) => (
                      <div key={key} className="min-w-0 space-y-2">
                        <Label htmlFor={`param-${key}`} className="block min-w-0 break-words capitalize">
                          {key}
                        </Label>
                        <div
                          className={
                            schema.type === 'bool' || schema.type === 'boolean'
                              ? 'min-w-0'
                              : 'min-w-0 w-full max-w-full'
                          }
                        >
                          {renderParamInput(key, schema)}
                        </div>
                        {schema.type !== 'bool' && schema.type !== 'boolean' ? (
                          <p className="min-w-0 break-words text-xs text-muted-foreground">{schema.description}</p>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : null}
              </>
            )
          })()}
        </div>
      ) : null}
    </div>
  )
}

export default StrategySelector
