"""
Population class for managing a collection of chromosomes.
"""
from typing import List, Optional, Callable
import random

from .chromosome import Chromosome
from .fitness import FitnessFunction


class Population:
    """
    Represents a population of chromosomes in the genetic algorithm.
    
    Manages creation, evaluation, and tracking of chromosomes.
    """
    
    def __init__(self, 
                 size: int = 100, 
                 chromosome_length: int = 100,
                 fitness_function: Optional[FitnessFunction] = None,
                 chromosomes: Optional[List[Chromosome]] = None):
        """
        Initialize the population.
        
        Args:
            size: Number of individuals in the population.
            chromosome_length: Length of each chromosome.
            fitness_function: Function to evaluate chromosomes.
            chromosomes: Optional list of pre-existing chromosomes.
        """
        self._size = size
        self._chromosome_length = chromosome_length
        self._fitness_function = fitness_function
        
        if chromosomes is not None:
            self._chromosomes = list(chromosomes)
        else:
            self._chromosomes = [Chromosome(length=chromosome_length) for _ in range(size)]
        
        self._generation = 0
        self._best_chromosome: Optional[Chromosome] = None
        self._best_fitness: float = float('-inf')
    
    @property
    def size(self) -> int:
        """Get the population size."""
        return self._size
    
    @property
    def chromosomes(self) -> List[Chromosome]:
        """Get the list of chromosomes."""
        return self._chromosomes
    
    @property
    def generation(self) -> int:
        """Get the current generation number."""
        return self._generation
    
    @generation.setter
    def generation(self, value: int):
        """Set the generation number."""
        self._generation = value
    
    @property
    def best_chromosome(self) -> Optional[Chromosome]:
        """Get the best chromosome found so far."""
        return self._best_chromosome
    
    @property
    def best_fitness(self) -> float:
        """Get the best fitness found so far."""
        return self._best_fitness
    
    def set_chromosomes(self, chromosomes: List[Chromosome]):
        """Replace the current chromosomes with new ones."""
        self._chromosomes = list(chromosomes)
    
    def evaluate(self):
        """Evaluate fitness of all chromosomes in the population."""
        if self._fitness_function is None:
            raise ValueError("No fitness function set")
        
        for chromosome in self._chromosomes:
            fitness = self._fitness_function.evaluate(chromosome)
            
            # Track best chromosome
            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._best_chromosome = chromosome.copy()
    
    def get_best(self) -> Chromosome:
        """Get the best chromosome in the current population."""
        return max(self._chromosomes, 
                   key=lambda c: c.fitness if c.fitness is not None else float('-inf'))
    
    def get_worst(self) -> Chromosome:
        """Get the worst chromosome in the current population."""
        return min(self._chromosomes,
                   key=lambda c: c.fitness if c.fitness is not None else float('inf'))
    
    def get_average_fitness(self) -> float:
        """Calculate the average fitness of the population."""
        fitnesses = [c.fitness for c in self._chromosomes if c.fitness is not None]
        if not fitnesses:
            return 0.0
        return sum(fitnesses) / len(fitnesses)
    
    def get_fitness_stats(self) -> dict:
        """Get statistics about the population's fitness."""
        fitnesses = [c.fitness for c in self._chromosomes if c.fitness is not None]
        if not fitnesses:
            return {'min': 0, 'max': 0, 'avg': 0, 'best_ever': self._best_fitness}
        
        return {
            'min': min(fitnesses),
            'max': max(fitnesses),
            'avg': sum(fitnesses) / len(fitnesses),
            'best_ever': self._best_fitness
        }
    
    def get_sorted_by_fitness(self, reverse: bool = True) -> List[Chromosome]:
        """
        Get chromosomes sorted by fitness.
        
        Args:
            reverse: If True, sort descending (best first).
            
        Returns:
            Sorted list of chromosomes.
        """
        return sorted(self._chromosomes,
                      key=lambda c: c.fitness if c.fitness is not None else float('-inf'),
                      reverse=reverse)
    
    def get_elite(self, count: int) -> List[Chromosome]:
        """
        Get the top chromosomes by fitness.
        
        Args:
            count: Number of elite chromosomes to return.
            
        Returns:
            List of top chromosomes.
        """
        sorted_pop = self.get_sorted_by_fitness(reverse=True)
        return [c.copy() for c in sorted_pop[:count]]
    
    def __len__(self) -> int:
        """Return the population size."""
        return len(self._chromosomes)
    
    def __iter__(self):
        """Iterate over chromosomes."""
        return iter(self._chromosomes)
    
    def __getitem__(self, index: int) -> Chromosome:
        """Get chromosome by index."""
        return self._chromosomes[index]
