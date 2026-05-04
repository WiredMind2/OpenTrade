"""
Statistical analysis utilities for Monte Carlo simulations.
"""
import numpy as np
from typing import List, Dict, Any


def calculate_confidence_intervals(data: List[float], confidence_level: float = 0.95) -> tuple:
    """Calculate confidence intervals using percentile method."""
    lower = np.percentile(data, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(data, (1 + confidence_level) / 2 * 100)
    return lower, upper


def calculate_value_at_risk(returns: List[float], confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) for a given confidence level."""
    return -np.percentile(returns, (1 - confidence_level) * 100)


def calculate_expected_shortfall(returns: List[float], confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (CVaR) for a given confidence level."""
    var = calculate_value_at_risk(returns, confidence_level)
    return -np.mean([r for r in returns if r <= -var])


def aggregate_monte_carlo_results(simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple Monte Carlo simulations."""
    if not simulation_results:
        return {}

    # Extract key metrics
    final_values = [r['final_value'] for r in simulation_results]
    total_returns = [r['total_return'] for r in simulation_results]

    # Basic statistics
    mean_final_value = np.mean(final_values)
    std_final_value = np.std(final_values)
    mean_total_return = np.mean(total_returns)
    std_total_return = np.std(total_returns)

    # Confidence intervals (95% default)
    conf_lower_return, conf_upper_return = calculate_confidence_intervals(total_returns)

    # Extreme cases
    worst_case_return = np.min(total_returns)
    best_case_return = np.max(total_returns)

    # Probability of positive return
    probability_positive_return = np.mean([1 if r > 0 else 0 for r in total_returns])

    return {
        'mean_final_value': mean_final_value,
        'std_final_value': std_final_value,
        'mean_total_return': mean_total_return,
        'std_total_return': std_total_return,
        'confidence_lower_return': conf_lower_return,
        'confidence_upper_return': conf_upper_return,
        'worst_case_return': worst_case_return,
        'best_case_return': best_case_return,
        'probability_positive_return': probability_positive_return
    }