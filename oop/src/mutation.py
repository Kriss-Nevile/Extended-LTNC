"""
Mutation strategy classes for genetic algorithm.
"""
from abc import ABC, abstractmethod
import random

from .chromosome import Chromosome


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    
    def __init__(self, probability: float):
        """
        Initialize mutation strategy.
        
        Args:
            probability: Probability of mutation per gene.
        """
        self._probability = probability
    
    @property
    def probability(self) -> float:
        """Get the mutation probability."""
        return self._probability
    
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome (may be modified in place or a copy).
        """
        pass


class BitFlipMutation(MutationStrategy):
    """
    Bit-flip mutation strategy.
    
    Each gene has a probability of being flipped (0 -> 1 or 1 -> 0).
    Default probability is 1/L where L is chromosome length.
    """
    
    def __init__(self, probability: float = None, chromosome_length: int = 100):
        """
        Initialize bit-flip mutation.
        
        Args:
            probability: Probability per bit. If None, uses 1/chromosome_length.
            chromosome_length: Length of chromosomes (used for default probability).
        """
        if probability is None:
            probability = 1.0 / chromosome_length
        super().__init__(probability)
        self._chromosome_length = chromosome_length
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform bit-flip mutation.
        
        Each bit has a probability of being flipped.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        mutated = chromosome.copy()
        
        for i in range(mutated.length):
            if random.random() < self._probability:
                mutated.flip_gene(i)
        
        return mutated


class SwapMutation(MutationStrategy):
    """
    Swap mutation strategy.
    
    Randomly selects two positions and swaps their genes.
    """
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform swap mutation.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        if random.random() >= self._probability:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        length = mutated.length
        
        if length < 2:
            return mutated
        
        # Select two different positions
        pos1 = random.randint(0, length - 1)
        pos2 = random.randint(0, length - 1)
        while pos2 == pos1:
            pos2 = random.randint(0, length - 1)
        
        # Swap genes
        gene1 = mutated.get_gene(pos1)
        gene2 = mutated.get_gene(pos2)
        mutated.set_gene(pos1, gene2)
        mutated.set_gene(pos2, gene1)
        
        return mutated


class InversionMutation(MutationStrategy):
    """
    Inversion mutation strategy.
    
    Randomly selects a segment and reverses it.
    """
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform inversion mutation.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        if random.random() >= self._probability:
            return chromosome.copy()
        
        genes = chromosome.genes
        length = len(genes)
        
        if length < 2:
            return Chromosome(genes)
        
        # Select two positions
        pos1 = random.randint(0, length - 2)
        pos2 = random.randint(pos1 + 1, length - 1)
        
        # Reverse the segment
        new_genes = genes[:pos1] + genes[pos1:pos2+1][::-1] + genes[pos2+1:]
        
        return Chromosome(new_genes)
