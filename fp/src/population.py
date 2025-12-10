"""
Population functions for the Functional Programming GA implementation.

All functions are pure and use immutable data structures (tuples).
"""
from typing import Tuple, Callable, Dict
import random

from .chromosome import Chromosome, create_random_chromosome


# Type for individual with fitness: (chromosome, fitness)
Individual = Tuple[Chromosome, float]

# Type for population: tuple of individuals
Population = Tuple[Individual, ...]


def create_population(size: int, 
                      chromosome_length: int, 
                      seed: int) -> Tuple[Chromosome, ...]:
    """
    Create a random population of chromosomes.
    
    Pure function - creates unique seeds for each chromosome.
    
    Args:
        size: Number of individuals.
        chromosome_length: Length of each chromosome.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of chromosomes.
    """
    return tuple(
        create_random_chromosome(chromosome_length, seed + i)
        for i in range(size)
    )


def evaluate_population(chromosomes: Tuple[Chromosome, ...],
                        fitness_func: Callable[[Chromosome], float]) -> Population:
    """
    Evaluate fitness of all chromosomes in a population.
    
    Uses map to apply fitness function to all chromosomes.
    
    Args:
        chromosomes: Tuple of chromosomes to evaluate.
        fitness_func: Function that evaluates a chromosome's fitness.
        
    Returns:
        Tuple of (chromosome, fitness) pairs.
    """
    # Use map for functional style
    return tuple(
        (chrom, fitness_func(chrom))
        for chrom in chromosomes
    )


def get_best(population: Population) -> Individual:
    """
    Get the best individual from the population.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        
    Returns:
        The individual with highest fitness.
    """
    return max(population, key=lambda ind: ind[1])


def get_worst(population: Population) -> Individual:
    """
    Get the worst individual from the population.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        
    Returns:
        The individual with lowest fitness.
    """
    return min(population, key=lambda ind: ind[1])


def get_elite(population: Population, count: int) -> Tuple[Chromosome, ...]:
    """
    Get the top chromosomes by fitness.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        count: Number of elite chromosomes to return.
        
    Returns:
        Tuple of top chromosomes.
    """
    sorted_pop = sorted(population, key=lambda ind: ind[1], reverse=True)
    return tuple(ind[0] for ind in sorted_pop[:count])


def get_fitness_stats(population: Population) -> Dict:
    """
    Calculate fitness statistics for the population.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        
    Returns:
        Dictionary with min, max, avg fitness.
    """
    fitnesses = tuple(ind[1] for ind in population)
    return {
        'min': min(fitnesses),
        'max': max(fitnesses),
        'avg': sum(fitnesses) / len(fitnesses)
    }


def get_average_fitness(population: Population) -> float:
    """
    Calculate the average fitness of the population.
    
    Uses reduce/sum pattern.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        
    Returns:
        Average fitness value.
    """
    total = sum(ind[1] for ind in population)
    return total / len(population)


def sort_by_fitness(population: Population, 
                    descending: bool = True) -> Population:
    """
    Sort population by fitness.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        descending: If True, best first.
        
    Returns:
        Sorted population.
    """
    return tuple(sorted(population, key=lambda ind: ind[1], reverse=descending))


def filter_valid(population: Population, 
                 min_fitness: float = 0.0) -> Population:
    """
    Filter population to only include valid individuals.
    
    Uses filter higher-order function.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        min_fitness: Minimum fitness threshold.
        
    Returns:
        Filtered population.
    """
    return tuple(filter(lambda ind: ind[1] > min_fitness, population))
