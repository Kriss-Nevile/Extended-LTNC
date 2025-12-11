"""
Chromosome functions for the Functional Programming GA implementation.

All functions are pure they don't modify their inputs and have no side effects. (Using seeds for randomness.)
Chromosomes are represented as immutable tuples.
"""
from typing import Tuple, Optional
import random


# Type alias for chromosome representation
Chromosome = Tuple[int, ...]


def create_chromosome(genes: Tuple[int, ...]) -> Chromosome:
    """
    Create a chromosome from a tuple of genes.
    
    Args:
        genes: Tuple of binary values (0 or 1).
        
    Returns:
        A chromosome (immutable tuple of genes).
    """
    return tuple(genes)


def create_random_chromosome(length: int, seed: int) -> Chromosome:
    """
    Create a random chromosome of given length.
    
    Pure function - creates its own RNG from seed.
    
    Args:
        length: Number of genes in the chromosome.
        seed: Random seed for reproducibility.
        
    Returns:
        A random chromosome.
    """
    rng = random.Random(seed)
    return tuple(rng.randint(0, 1) for _ in range(length))


def copy_chromosome(chromosome: Chromosome) -> Chromosome:
    """
    Create a copy of a chromosome.
    
    Since chromosomes are immutable tuples, this just returns the same tuple.
    
    Args:
        chromosome: The chromosome to copy.
        
    Returns:
        A copy of the chromosome (same tuple since immutable).
    """
    return chromosome  # Tuples are immutable, no need for deep copy


def get_gene(chromosome: Chromosome, index: int) -> int:
    """
    Get a gene at a specific index.
    
    Args:
        chromosome: The chromosome.
        index: Index of the gene.
        
    Returns:
        The gene value (0 or 1).
    """
    return chromosome[index]


def set_gene(chromosome: Chromosome, index: int, value: int) -> Chromosome:
    """
    Create a new chromosome with a gene set to a new value.
    
    Args:
        chromosome: The original chromosome.
        index: Index of the gene to change.
        value: New value for the gene.
        
    Returns:
        A new chromosome with the modified gene.
    """
    genes = list(chromosome)
    genes[index] = value
    return tuple(genes)


def flip_gene(chromosome: Chromosome, index: int) -> Chromosome:
    """
    Create a new chromosome with a flipped gene.
    
    Args:
        chromosome: The original chromosome.
        index: Index of the gene to flip.
        
    Returns:
        A new chromosome with the flipped gene.
    """
    genes = list(chromosome)
    genes[index] = 1 - genes[index]
    return tuple(genes)


def chromosome_length(chromosome: Chromosome) -> int:
    """
    Get the length of a chromosome.
    
    Args:
        chromosome: The chromosome.
        
    Returns:
        Number of genes in the chromosome.
    """
    return len(chromosome)


def chromosomes_equal(chrom1: Chromosome, chrom2: Chromosome) -> bool:
    """
    Check if two chromosomes are equal.
    
    Args:
        chrom1: First chromosome.
        chrom2: Second chromosome.
        
    Returns:
        True if chromosomes have identical genes.
    """
    return chrom1 == chrom2
