"""
Mutation functions for the Functional Programming GA implementation.

All functions are pure and return new chromosomes without modifying inputs.
"""
from typing import Tuple
import random

from .chromosome import Chromosome


def bit_flip_mutate(chromosome: Chromosome, 
                    mutation_prob: float, 
                    rng: random.Random) -> Chromosome:
    """
    Perform bit-flip mutation on a chromosome.
    
    Each gene has a probability of being flipped.
    This is a pure function - returns a new chromosome.
    
    Args:
        chromosome: The chromosome to mutate.
        mutation_prob: Probability of flipping each gene.
        rng: Random number generator.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    # Use tuple comprehension for immutable result
    mutated_genes = tuple(
        1 - gene if rng.random() < mutation_prob else gene
        for gene in chromosome
    )
    return mutated_genes


def mutate_chromosome(chromosome: Chromosome,
                      chromosome_length: int,
                      rng: random.Random) -> Chromosome:
    """
    Mutate a chromosome using the standard 1/L mutation rate.
    
    Args:
        chromosome: The chromosome to mutate.
        chromosome_length: Length of chromosomes (for calculating rate).
        rng: Random number generator.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    mutation_prob = 1.0 / chromosome_length
    return bit_flip_mutate(chromosome, mutation_prob, rng)


def swap_mutate(chromosome: Chromosome, 
                mutation_prob: float,
                rng: random.Random) -> Chromosome:
    """
    Perform swap mutation on a chromosome.
    
    Swaps two random positions with given probability.
    
    Args:
        chromosome: The chromosome to mutate.
        mutation_prob: Probability of performing swap.
        rng: Random number generator.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    if rng.random() >= mutation_prob:
        return chromosome
    
    length = len(chromosome)
    if length < 2:
        return chromosome
    
    pos1 = rng.randrange(length)
    pos2 = rng.randrange(length)
    while pos2 == pos1:
        pos2 = rng.randrange(length)
    
    # Create new chromosome with swapped genes
    genes = list(chromosome)
    genes[pos1], genes[pos2] = genes[pos2], genes[pos1]
    return tuple(genes)


def inversion_mutate(chromosome: Chromosome,
                     mutation_prob: float,
                     rng: random.Random) -> Chromosome:
    """
    Perform inversion mutation on a chromosome.
    
    Reverses a random segment with given probability.
    
    Args:
        chromosome: The chromosome to mutate.
        mutation_prob: Probability of performing inversion.
        rng: Random number generator.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    if rng.random() >= mutation_prob:
        return chromosome
    
    length = len(chromosome)
    if length < 2:
        return chromosome
    
    pos1 = rng.randint(0, length - 2)
    pos2 = rng.randint(pos1 + 1, length - 1)
    
    # Create new chromosome with reversed segment
    new_genes = chromosome[:pos1] + chromosome[pos1:pos2+1][::-1] + chromosome[pos2+1:]
    return new_genes
