CREATE TABLE IF NOT EXISTS monte_carlo_simulations (
    id SERIAL PRIMARY KEY,
    simulation_id INTEGER,
    timestamp TIMESTAMP,
    price DECIMAL,
    returns DECIMAL
);

CREATE TABLE IF NOT EXISTS monte_carlo_aggregated (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER,
    num_simulations INTEGER,
    mean_return DECIMAL,
    std_return DECIMAL,
    confidence_lower DECIMAL,
    confidence_upper DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);