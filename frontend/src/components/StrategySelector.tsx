import React, { useState, useEffect } from 'react';
import { getStrategies } from '../services/strategyApi';

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
    if (strategy) {
      const initialParams: Record<string, any> = {};
      for (const [key, schema] of Object.entries(strategy.parameters_schema)) {
        initialParams[key] = schema.default;
      }
      setParams(initialParams);
      onStrategyChange(name, initialParams);
    }
  };

  const handleParamChange = (key: string, value: any) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
    onStrategyChange(selectedStrategy, newParams);
  };

  const renderParamInput = (key: string, schema: ParameterSchema) => {
    const value = params[key];
    switch (schema.type) {
      case 'int':
        return (
          <input
            type="number"
            step="1"
            value={value}
            onChange={e => handleParamChange(key, parseInt(e.target.value, 10))}
            placeholder={schema.description}
          />
        );
      case 'float':
        return (
          <input
            type="number"
            step="any"
            value={value}
            onChange={e => handleParamChange(key, parseFloat(e.target.value))}
            placeholder={schema.description}
          />
        );
      case 'string':
        return (
          <input
            type="text"
            value={value}
            onChange={e => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
          />
        );
      case 'bool':
      case 'boolean':
        return (
          <input
            type="checkbox"
            checked={value}
            onChange={e => handleParamChange(key, e.target.checked)}
          />
        );
      default:
        return (
          <input
            type="text"
            value={value}
            onChange={e => handleParamChange(key, e.target.value)}
            placeholder={schema.description}
          />
        );
    }
  };

  return (
    <div>
      <label htmlFor="strategy-select">Select Strategy:</label>
      <select id="strategy-select" value={selectedStrategy} onChange={handleStrategyChange}>
        {strategies.map(strategy => (
          <option key={strategy.name} value={strategy.name}>
            {strategy.name} - {strategy.description}
          </option>
        ))}
      </select>

      {error && (
        <p style={{ color: 'red', marginTop: '0.5rem' }}>{error}</p>
      )}

      {selectedStrategy && (
        <div>
          <h3>Parameters</h3>
          {Object.entries(strategies.find(s => s.name === selectedStrategy)?.parameters_schema || {}).map(([key, schema]) => (
            <div key={key} style={{ marginBottom: '10px' }}>
              <label htmlFor={`param-${key}`}>{key}: </label>
              {renderParamInput(key, schema)}
              <small style={{ display: 'block', color: '#666' }}>{schema.description}</small>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default StrategySelector;