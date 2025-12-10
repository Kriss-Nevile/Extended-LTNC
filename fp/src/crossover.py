"""
Crossover functions for the Functional Programming GA implementation.

All functions are pure and return new chromosomes without modifying inputs.
"""
from typing import Tuple
import random

from .chromosome import Chromosome


def one_point_crossover(parent1: Chromosome, 
                        parent2: Chromosome, 
                        rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Perform one-point crossover on two parent chromosomes.
    
    Creates two offspring by swapping genes after a random crossover point.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        rng: Random number generator.
        
    Returns:
        Tuple of two offspring chromosomes.
    """
    length = len(parent1)
    crossover_point = rng.randint(1, length - 1)
    
    # Create offspring using tuple slicing (immutable operations)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2


def crossover_pair(parent1: Chromosome,
                   parent2: Chromosome,
                   crossover_prob: float,
                   rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Conditionally perform crossover based on probability.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        crossover_prob: Probability of performing crossover.
        rng: Random number generator.
        
    Returns:
        Tuple of two offspring (either crossed over or copies of parents).
    """
    if rng.random() < crossover_prob:
        return one_point_crossover(parent1, parent2, rng)
    else:
        # Return copies (tuples are immutable, so same reference is fine)
        return parent1, parent2

# was this used?
def two_point_crossover(parent1: Chromosome,
                        parent2: Chromosome,
                        rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Perform two-point crossover on two parent chromosomes.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        rng: Random number generator.
        
    Returns:
        Tuple of two offspring chromosomes.
    """
    length = len(parent1)
    point1 = rng.randint(1, length - 2)
    point2 = rng.randint(point1 + 1, length - 1)
    
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return offspring1, offspring2


# was this used?
def uniform_crossover(parent1: Chromosome,
                      parent2: Chromosome,
                      rng: random.Random) -> Tuple[Chromosome, Chromosome]:
    """
    Perform uniform crossover on two parent chromosomes.
    
    Each gene is independently chosen from one of the parents.
    
    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.
        rng: Random number generator.
        
    Returns:
        Tuple of two offspring chromosomes.
    """
    offspring1_genes = []
    offspring2_genes = []
    
    for g1, g2 in zip(parent1, parent2):
        if rng.random() < 0.5:
            offspring1_genes.append(g1)
            offspring2_genes.append(g2)
        else:
            offspring1_genes.append(g2)
            offspring2_genes.append(g1)
    
    return tuple(offspring1_genes), tuple(offspring2_genes)
