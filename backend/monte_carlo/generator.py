import numpy as np

class MonteCarloGenerator:
    def generate_price_path(self, initial_price, drift, volatility, time_horizon, num_steps):
        dt = time_horizon / num_steps
        price_path = [initial_price]
        for _ in range(num_steps):
            shock = np.random.normal(0, 1)
            price = price_path[-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock)
            price_path.append(price)
        return price_path