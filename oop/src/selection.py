"""
Selection strategy classes for genetic algorithm.
"""
from abc import ABC, abstractmethod
from typing import List
import random

from .chromosome import Chromosome


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select chromosomes from the population for reproduction.
        
        Args:
            population: List of chromosomes to select from.
            num_parents: Number of parents to select.
            
        Returns:
            List of selected chromosomes.
        """
        pass


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.
    
    Randomly selects k individuals and returns the best one.
    This process is repeated to get the required number of parents.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament (k).
        """
        self._tournament_size = tournament_size
    
    @property
    def tournament_size(self) -> int:
        """Get the tournament size."""
        return self._tournament_size
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select parents using tournament selection.
        
        Args:
            population: List of chromosomes to select from.
            num_parents: Number of parents to select.
            
        Returns:
            List of selected parent chromosomes.
        """
        selected = []
        
        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament = random.sample(population, min(self._tournament_size, len(population)))
            
            # Select the best from the tournament
            winner = max(tournament, key=lambda c: c.fitness if c.fitness is not None else float('-inf'))
            selected.append(winner.copy())
        
        return selected
    
    def select_pair(self, population: List[Chromosome]) -> tuple:
        """
        Select a pair of parents for crossover.
        
        Args:
            population: List of chromosomes to select from.
            
        Returns:
            Tuple of two parent chromosomes.
        """
        parents = self.select(population, 2)
        return parents[0], parents[1]
