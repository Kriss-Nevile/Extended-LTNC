"""
Selection functions for the Functional Programming GA implementation.

All functions are pure and use higher-order functions.
"""
from typing import Tuple, List, Callable
import random

from .chromosome import Chromosome


# Type for individual with fitness: (chromosome, fitness)
Individual = Tuple[Chromosome, float]

# Type for population: tuple of individuals
Population = Tuple[Individual, ...]


def tournament_select(population: Population, 
                      tournament_size: int, 
                      seed: int) -> Chromosome:
    """
    Select one chromosome using tournament selection.
    
    Selects k random individuals and returns the best one.
    Pure function - creates its own RNG from seed.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        tournament_size: Number of individuals in tournament.
        seed: Random seed for reproducibility.
        
    Returns:
        The winning chromosome.
    """
    rng = random.Random(seed)
    # Randomly sample tournament participants
    tournament_indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    tournament = [population[i] for i in tournament_indices]
    
    # Select the best (highest fitness)
    winner = max(tournament, key=lambda ind: ind[1])
    return winner[0]


def select_parents(population: Population,
                   num_parents: int,
                   tournament_size: int,
                   seed: int) -> Tuple[Chromosome, ...]:
    """
    Select multiple parents using tournament selection.
    
    Uses map to apply selection repeatedly.
    Pure function - creates unique seeds for each selection.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        num_parents: Number of parents to select.
        tournament_size: Tournament size for each selection.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of selected parent chromosomes.
    """
    # Use list comprehension with unique seeds for each selection
    parents = tuple(
        tournament_select(population, tournament_size, seed + i)
        for i in range(num_parents)
    )
    return parents


def select_pair(population: Population,
                tournament_size: int,
                seed: int) -> Tuple[Chromosome, Chromosome]:
    """
    Select a pair of parents for crossover.
    
    Pure function - creates unique seeds for each parent selection.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        tournament_size: Tournament size.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of two parent chromosomes.
    """
    parent1 = tournament_select(population, tournament_size, seed)
    parent2 = tournament_select(population, tournament_size, seed + 1)
    return parent1, parent2
