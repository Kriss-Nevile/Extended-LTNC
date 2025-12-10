"""
Fitness functions for the Functional Programming GA implementation.
"""
from typing import Tuple, Dict, Callable
import random
from functools import reduce

from .chromosome import Chromosome


# Type alias for fitness function
FitnessFunction = Callable[[Chromosome], float]


def onemax_fitness(chromosome: Chromosome) -> float:
    """
    Calculate OneMax fitness - count the number of 1s.
    
    This is a pure function with no side effects.
    
    Args:
        chromosome: The chromosome to evaluate.
        
    Returns:
        Number of 1s in the chromosome.
    """
    return float(sum(chromosome))


def create_onemax_fitness(length: int) -> Dict:
    """
    Create a OneMax fitness configuration.
    
    Args:
        length: Expected chromosome length.
        
    Returns:
        Dictionary with fitness function and metadata.
    """
    return {
        'name': 'OneMax',
        'evaluate': onemax_fitness,
        'optimal': float(length),
        'length': length
    }


def knapsack_fitness(chromosome: Chromosome, 
                     values: Tuple[int, ...], 
                     weights: Tuple[int, ...], 
                     capacity: int) -> float:
    """
    Calculate Knapsack fitness.
    
    Returns total value if within capacity, 0 if over capacity.
    This is a pure function with no side effects.
    
    Args:
        chromosome: The chromosome to evaluate (selection of items).
        values: Tuple of item values.
        weights: Tuple of item weights.
        capacity: Maximum weight capacity.
        
    Returns:
        Total value if valid, 0 if overweight.
    """
    total_value = sum(v * g for v, g in zip(values, chromosome))
    total_weight = sum(w * g for w, g in zip(weights, chromosome))
    
    if total_weight > capacity:
        return 0.0
    return float(total_value)


def create_knapsack_problem(n_items: int, seed: int = 42) -> Dict:
    """
    Create a Knapsack problem configuration.
    
    Uses immutable data structures (tuples) for values and weights.
    
    Args:
        n_items: Number of items.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with fitness function and problem data.
    """
    rng = random.Random(seed)   
    
    # Generate random values and weights as immutable tuples
    values = tuple(rng.randint(1, 100) for _ in range(n_items))
    weights = tuple(rng.randint(1, 50) for _ in range(n_items))
    
    # Capacity = 40% of total weight
    capacity = int(0.4 * sum(weights))
    
    # Calculate approximate optimal using greedy
    optimal = _greedy_knapsack_approximation(values, weights, capacity)
    
    # Create a closure that captures the problem data
    def evaluate(chromosome: Chromosome) -> float:
        return knapsack_fitness(chromosome, values, weights, capacity)
    
    return {
        'name': 'Knapsack',
        'evaluate': evaluate,
        'optimal': optimal,
        'values': values,
        'weights': weights,
        'capacity': capacity,
        'n_items': n_items
    }


def _greedy_knapsack_approximation(values: Tuple[int, ...], 
                                    weights: Tuple[int, ...], 
                                    capacity: int) -> float:
    """
    Calculate greedy approximation of optimal knapsack value.
    
    Args:
        values: Item values.
        weights: Item weights.
        capacity: Knapsack capacity.
        
    Returns:
        Approximate optimal value.
    """
    # Sort items by value-to-weight ratio (descending)
    ratios = tuple(sorted(
        [(values[i] / weights[i], i) for i in range(len(values))],
        reverse=True
    ))
    
    def add_item(acc: Tuple[int, int], ratio_index: Tuple[float, int]) -> Tuple[int, int]:
        total_value, total_weight = acc
        _, i = ratio_index
        
        if total_weight + weights[i] <= capacity:
            return (total_value + values[i], total_weight + weights[i])
        return acc
    
    final_value, _ = reduce(add_item, ratios, (0, 0))
    
    return float(final_value)


def get_solution_details(chromosome: Chromosome,
                         values: Tuple[int, ...],
                         weights: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Get detailed solution information for knapsack.
    
    Args:
        chromosome: The solution chromosome.
        values: Item values.
        weights: Item weights.
        
    Returns:
        Tuple of (total_value, total_weight).
    """
    total_value = sum(v * g for v, g in zip(values, chromosome))
    total_weight = sum(w * g for w, g in zip(weights, chromosome))
    return total_value, total_weight
