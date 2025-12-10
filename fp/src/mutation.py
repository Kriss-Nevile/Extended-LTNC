"""
Mutation functions for the Functional Programming GA implementation.

All functions are pure and return new chromosomes without modifying inputs.
"""
from typing import Tuple
import random

from .chromosome import Chromosome


def bit_flip_mutate(chromosome: Chromosome, 
                    mutation_prob: float, 
                    seed: int) -> Chromosome:
    """
    Perform bit-flip mutation on a chromosome.
    
    Each gene has a probability of being flipped.
    Pure function - creates its own RNG from seed.
    
    Args:
        chromosome: The chromosome to mutate.
        mutation_prob: Probability of flipping each gene.
        seed: Random seed for reproducibility.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    rng = random.Random(seed)
    # Use tuple comprehension for immutable result
    mutated_genes = tuple(
        1 - gene if rng.random() < mutation_prob else gene
        for gene in chromosome
    )
    return mutated_genes


def mutate_chromosome(chromosome: Chromosome,
                      chromosome_length: int,
                      seed: int) -> Chromosome:
    """
    Mutate a chromosome using the standard 1/L mutation rate.
    
    Pure function - creates its own RNG from seed.
    
    Args:
        chromosome: The chromosome to mutate.
        chromosome_length: Length of chromosomes (for calculating rate).
        seed: Random seed for reproducibility.
        
    Returns:
        A new chromosome (potentially mutated).
    """
    mutation_prob = 1.0 / chromosome_length
    return bit_flip_mutate(chromosome, mutation_prob, seed)
