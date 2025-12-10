"""
Crossover strategy classes for genetic algorithm.
Implementation checked
"""
from abc import ABC, abstractmethod
from typing import Tuple
import random

from .chromosome import Chromosome


class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""
    
    def __init__(self, probability: float = 0.9):
        """
        Initialize crossover strategy.
        
        Args:
            probability: Probability of performing crossover.
        """
        self._probability = probability
    
    @property
    def probability(self) -> float:
        """Get the crossover probability."""
        return self._probability
    
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover on two parent chromosomes.
        
        Args:
            parent1: First parent chromosome.
            parent2: Second parent chromosome.
            
        Returns:
            Tuple of two offspring chromosomes.
        """
        pass
    
    def should_crossover(self) -> bool:
        """Determine if crossover should occur based on probability."""
        return random.random() < self._probability


class OnePointCrossover(CrossoverStrategy):
    """
    One-point crossover strategy.
    
    A random crossover point is selected, and genes are swapped
    between parents after that point.
    """
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform one-point crossover.
        
        Args:
            parent1: First parent chromosome.
            parent2: Second parent chromosome.
            
        Returns:
            Tuple of two offspring chromosomes.
        """
        if not self.should_crossover():
            # Return copies of parents without crossover
            return parent1.copy(), parent2.copy()
        
        length = parent1.length
        
        # Select random crossover point (1 to length-1)
        crossover_point = random.randint(1, length - 1)
        
        # Create offspring genes
        genes1 = parent1.genes
        genes2 = parent2.genes
        
        offspring1_genes = genes1[:crossover_point] + genes2[crossover_point:]
        offspring2_genes = genes2[:crossover_point] + genes1[crossover_point:]
        
        return Chromosome(offspring1_genes), Chromosome(offspring2_genes)