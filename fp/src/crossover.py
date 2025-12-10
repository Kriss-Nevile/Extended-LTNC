"""
Crossover functions for the Functional Programming GA implementation.

All functions are pure and return new chromosomes without modifying inputs.
"""
from typing import Tuple
import random

from .chromosome import Chromosome


def one_point_crossover(parent1: Chromosome, 
                        parent2: Chromosome, 
                        seed: int) -> Tuple[Chromosome, Chromosome]:
    """
    Perform one-point crossover on two parent chromosomes.
    
    Creates two offspring by swapping genes after a random crossover point.
    Pure function - creates its own RNG from seed.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of two offspring chromosomes.
    """
    rng = random.Random(seed)
    length = len(parent1)
    crossover_point = rng.randint(1, length - 1)
    
    # Create offspring using tuple slicing (immutable operations)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2


def crossover_pair(parent1: Chromosome,
                   parent2: Chromosome,
                   crossover_prob: float,
                   seed: int) -> Tuple[Chromosome, Chromosome]:
    """
    Conditionally perform crossover based on probability.
    
    Pure function - creates its own RNG from seed.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        crossover_prob: Probability of performing crossover.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of two offspring (either crossed over or copies of parents).
    """
    rng = random.Random(seed)
    if rng.random() < crossover_prob:
        return one_point_crossover(parent1, parent2, seed + 1)
    else:
        # Return copies (tuples are immutable, so same reference is fine)
        return parent1, parent2
