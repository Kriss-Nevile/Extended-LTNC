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
                      rng: random.Random) -> Chromosome:
    """
    Select one chromosome using tournament selection.
    
    Selects k random individuals and returns the best one.
    This is a pure function (given the same RNG state).
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        tournament_size: Number of individuals in tournament.
        rng: Random number generator.
        
    Returns:
        The winning chromosome.
    """
    # Randomly sample tournament participants
    tournament_indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    tournament = [population[i] for i in tournament_indices]
    
    # Select the best (highest fitness)
    winner = max(tournament, key=lambda ind: ind[1])
    return winner[0]


def select_parents(population: Population,
                   num_parents: int,
                   tournament_size: int,
                   rng: random.Random) -> Tuple[Chromosome, ...]:
    """
    Select multiple parents using tournament selection.
    
    Uses map to apply selection repeatedly.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        num_parents: Number of parents to select.
        tournament_size: Tournament size for each selection.
        rng: Random number generator.
        
    Returns:
        Tuple of selected parent chromosomes.
    """
    # Use list comprehension (more Pythonic than map for this case)
    parents = tuple(
        tournament_select(population, tournament_size, rng)
        for _ in range(num_parents)
    )
    return parents


def select_pair(population: Population,
                tournament_size: int,
                rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Select a pair of parents for crossover.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        tournament_size: Tournament size.
        rng: Random number generator.
        
    Returns:
        Tuple of two parent chromosomes.
    """
    parent1 = tournament_select(population, tournament_size, rng)
    parent2 = tournament_select(population, tournament_size, rng)
    return parent1, parent2


def roulette_select(population: Population, rng: random.Random) -> Chromosome:
    """
    Select one chromosome using roulette wheel selection.
    
    Selection probability is proportional to fitness.
    
    Args:
        population: Tuple of (chromosome, fitness) pairs.
        rng: Random number generator.
        
    Returns:
        The selected chromosome.
    """
    total_fitness = sum(ind[1] for ind in population)
    
    if total_fitness <= 0:
        # If all fitness values are 0, use uniform random selection
        return population[rng.randrange(len(population))][0]
    
    pick = rng.uniform(0, total_fitness)
    current = 0.0
    
    for chromosome, fitness in population:
        current += fitness
        if current >= pick:
            return chromosome
    
    # Fallback: return last chromosome
    return population[-1][0]
