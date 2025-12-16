import React, { useState, useEffect } from 'react';

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

  useEffect(() => {
    console.log('Fetching strategies from /api/strategies');
    fetch('/api/strategies')
      .then(res => {
        console.log('Strategies fetch response status:', res.status);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then((data: Strategy[]) => {
        console.log('Fetched strategies:', data);
        setStrategies(data);
      })
      .catch(err => {
        console.error('Failed to fetch strategies:', err);
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
    } else {
      setParams({});
      onStrategyChange('', {});
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
        <option value="">-- Select a Strategy --</option>
        {strategies.map(strategy => (
          <option key={strategy.name} value={strategy.name}>
            {strategy.name} - {strategy.description}
          </option>
        ))}
      </select>

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