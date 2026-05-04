import React, { useState, useEffect } from 'react';
import { getStrategies } from '../api/strategies';

interface ParameterSchema {
  type: string;
  default: any;
  description: string;
}

interface Strategy {
  name: string;
  description: string;
  type: string;
  parameters_schema: Record<string, ParameterSchema>;
  can_train: boolean;
}

interface StrategySelectorProps {
  onStrategyChange: (strategy: string, params: Record<string, any>) => void;
}

const StrategySelector: React.FC<StrategySelectorProps> = ({ onStrategyChange }) => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [params, setParams] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStrategies()
      .then((data: Strategy[]) => {
        setStrategies(data);
        if (data.length > 0) {
          const first = data[0];
          const initialParams: Record<string, any> = {};
          for (const [key, schema] of Object.entries(first.parameters_schema)) {
            initialParams[key] = schema.default;
          }
          setSelectedStrategy(first.name);
          setParams(initialParams);
          onStrategyChange(first.name, initialParams);
        }
      })
      .catch(err => {
        console.error('Failed to fetch strategies:', err);
        setError('Unable to load strategies. Please try again later.');
      });
  }, []);

  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const name = e.target.value;
    setSelectedStrategy(name);
    const strategy = strategies.find(s => s.name === name);
    if (!strategy) {
      setParams({});
      onStrategyChange('', {});
      return;
    }

    const initialParams: Record<string, any> = {};
    for (const [key, schema] of Object.entries(strategy.parameters_schema)) {
      initialParams[key] = schema.default;
    }
    setParams(initialParams);
    onStrategyChange(name, initialParams);
  };

  const handleParamChange = (key: string, value: any) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
    onStrategyChange(selectedStrategy, newParams);
  };

  const inputClass = "w-full h-9 rounded-md border border-input bg-secondary px-3 py-1 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"

  const renderParamInput = (key: string, schema: ParameterSchema) => {
    const value = params[key];
    switch (schema.type) {
      case 'int':
        return (
          <input
            id={`param-${key}`}
            type="number"
            step="1"
            value={value}
            onChange={e => handleParamChange(key, parseInt(e.target.value, 10))}
            placeholder={schema.description}
            className={inputClass}
          />
        );
      case 'float':
        return (
          <input
            id={`param-${key}`}
            type="number"
            step="any"
            value={value}
            onChange={e => handleParamChange(key, parseFloat(e.target.value))}
            placeholder={schema.description}
            className={inputClass}
          />
        );
      case 'string':
        return (
          <input
            id={`param-${key}`}
            type="text"
            value={value}
            onChange={e => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
            className={inputClass}
          />
        );
      case 'bool':
      case 'boolean':
        return (
          <input
            id={`param-${key}`}
            type="checkbox"
            checked={value}
            onChange={e => handleParamChange(key, e.target.checked)}
            className="h-4 w-4 rounded border-input accent-primary cursor-pointer"
          />
        );
      default:
        return (
          <input
            id={`param-${key}`}
            type="text"
            value={value}
            onChange={e => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
            className={inputClass}
          />
        );
    }
  };

  return (
    <div className="space-y-1.5">
      <label htmlFor="strategy-select" className="text-sm font-medium text-foreground">
        Select Strategy
      </label>
      <select
        id="strategy-select"
        value={selectedStrategy}
        onChange={handleStrategyChange}
        className={inputClass}
      >
        <option value="" className="bg-secondary text-foreground">-- Select a Strategy --</option>
        {strategies.map(strategy => (
          <option key={strategy.name} value={strategy.name} className="bg-secondary text-foreground">
            {strategy.name} - {strategy.description}
          </option>
        ))}
      </select>

      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      {selectedStrategy && (
        <div className="space-y-3 pt-1">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Parameters</p>
          {Object.entries(strategies.find(s => s.name === selectedStrategy)?.parameters_schema || {}).map(([key, schema]) => (
            <div key={key} className="space-y-1">
              <label htmlFor={`param-${key}`} className="text-sm font-medium text-foreground capitalize">
                {key}
              </label>
              <div className={schema.type === 'bool' || schema.type === 'boolean' ? '' : 'w-full'}>
                {renderParamInput(key, schema)}
              </div>
              <p className="text-xs text-muted-foreground">{schema.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default StrategySelector;