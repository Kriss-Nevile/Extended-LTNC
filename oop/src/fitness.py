"""
Fitness function classes for evaluating chromosomes.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import random

from .chromosome import Chromosome


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    @abstractmethod
    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Evaluate the fitness of a chromosome.
        
        Args:
            chromosome: The chromosome to evaluate.
            
        Returns:
            The fitness value (higher is better).
        """
        pass
    
    @abstractmethod
    def get_optimal_fitness(self) -> float:
        """
        Get the theoretical optimal fitness value.
        
        Returns:
            The maximum possible fitness value.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the fitness function."""
        pass


class OneMaxFitness(FitnessFunction):
    """
    OneMax problem fitness function.
    
    The goal is to maximize the number of 1s in the chromosome.
    Fitness = count of 1s in the bitstring.
    """
    
    def __init__(self, length: int = 100):
        """
        Initialize OneMax fitness function.
        
        Args:
            length: Expected length of chromosomes.
        """
        self._length = length
    
    def evaluate(self, chromosome: Chromosome) -> float:
        """Count the number of 1s in the chromosome."""
        fitness = sum(chromosome.genes)
        chromosome.fitness = fitness
        return fitness
    
    def get_optimal_fitness(self) -> float:
        """Optimal is when all genes are 1."""
        return float(self._length)
    
    @property
    def name(self) -> str:
        return "OneMax"


class KnapsackFitness(FitnessFunction):
    """
    0/1 Knapsack problem fitness function.
    
    Each gene represents whether an item is selected (1) or not (0).
    Fitness = total value of selected items if within capacity, else 0.
    """
    
    def __init__(self, n_items: int = 100, seed: int = 42):
        """
        Initialize Knapsack fitness function with random items.
        
        Args:
            n_items: Number of items.
            seed: Random seed for reproducibility.
        """
        self._n_items = n_items
        self._seed = seed
        
        # Generate random values and weights
        rng = random.Random(seed)
        self._values: List[int] = [rng.randint(1, 100) for _ in range(n_items)]
        self._weights: List[int] = [rng.randint(1, 50) for _ in range(n_items)]
        
        # Capacity = 40% of total weight
        self._capacity = int(0.4 * sum(self._weights))
        
        # Calculate optimal fitness (approximation for reference)
        self._optimal = self._calculate_optimal_approximation()
    
    @property
    def values(self) -> List[int]:
        """Get the item values."""
        return list(self._values)
    
    @property
    def weights(self) -> List[int]:
        """Get the item weights."""
        return list(self._weights)
    
    @property
    def capacity(self) -> int:
        """Get the knapsack capacity."""
        return self._capacity
    
    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Evaluate knapsack fitness.
        
        Returns total value if within capacity, 0 otherwise.
        """
        genes = chromosome.genes
        total_value = sum(v * g for v, g in zip(self._values, genes))
        total_weight = sum(w * g for w, g in zip(self._weights, genes))
        
        if total_weight > self._capacity:
            fitness = 0.0
        else:
            fitness = float(total_value)
        
        chromosome.fitness = fitness
        return fitness
    
    def _calculate_optimal_approximation(self) -> float:
        """Calculate an approximation of optimal fitness using greedy approach."""
        # Value-to-weight ratio greedy approximation
        ratios = [(self._values[i] / self._weights[i], i) 
                  for i in range(self._n_items)]
        ratios.sort(reverse=True)
        
        total_value = 0
        total_weight = 0
        for _, i in ratios:
            if total_weight + self._weights[i] <= self._capacity:
                total_value += self._values[i]
                total_weight += self._weights[i]
        
        return float(total_value)
    
    def get_optimal_fitness(self) -> float:
        """Get the approximated optimal fitness."""
        return self._optimal
    
    @property
    def name(self) -> str:
        return "Knapsack"
    
    def get_solution_details(self, chromosome: Chromosome) -> Tuple[int, int]:
        """
        Get detailed solution information.
        
        Returns:
            Tuple of (total_value, total_weight)
        """
        genes = chromosome.genes
        total_value = sum(v * g for v, g in zip(self._values, genes))
        total_weight = sum(w * g for w, g in zip(self._weights, genes))
        return total_value, total_weight
