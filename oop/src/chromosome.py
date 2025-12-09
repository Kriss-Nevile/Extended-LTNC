"""
Chromosome class representing a candidate solution in the genetic algorithm.
"""
import random
from typing import List, Optional


class Chromosome:
    """
    Represents a candidate solution as a binary string (bitstring).
    
    Attributes:
        genes: List of binary values (0 or 1)
        fitness: Cached fitness value
    """
    
    def __init__(self, genes: Optional[List[int]] = None, length: int = 100):
        """
        Initialize a chromosome.
        
        Args:
            genes: Optional list of binary values. If None, random genes are generated.
            length: Length of the chromosome if genes are not provided.
        """
        if genes is not None:
            self._genes = list(genes)  # Create a copy for encapsulation
        else:
            self._genes = [random.randint(0, 1) for _ in range(length)]
        self._fitness: Optional[float] = None
    
    @property
    def genes(self) -> List[int]:
        """Get a copy of the genes (encapsulation)."""
        return list(self._genes)
    
    @property
    def length(self) -> int:
        """Get the length of the chromosome."""
        return len(self._genes)
    
    @property
    def fitness(self) -> Optional[float]:
        """Get the fitness value."""
        return self._fitness
    
    @fitness.setter
    def fitness(self, value: float):
        """Set the fitness value."""
        self._fitness = value
    
    def get_gene(self, index: int) -> int:
        """Get a specific gene value."""
        return self._genes[index]
    
    def set_gene(self, index: int, value: int):
        """Set a specific gene value and invalidate fitness cache."""
        self._genes[index] = value
        self._fitness = None  # Invalidate cached fitness
    
    def flip_gene(self, index: int):
        """Flip a specific gene (0 -> 1 or 1 -> 0)."""
        self._genes[index] = 1 - self._genes[index]
        self._fitness = None  # Invalidate cached fitness
    
    def copy(self) -> 'Chromosome':
        """Create a deep copy of this chromosome."""
        new_chromosome = Chromosome(genes=self._genes)
        new_chromosome._fitness = self._fitness
        return new_chromosome
    
    def __repr__(self) -> str:
        """String representation of the chromosome."""
        gene_str = ''.join(map(str, self._genes[:20]))
        if len(self._genes) > 20:
            gene_str += '...'
        return f"Chromosome(genes={gene_str}, fitness={self._fitness})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on genes."""
        if not isinstance(other, Chromosome):
            return False
        return self._genes == other._genes
    
    def __hash__(self) -> int:
        """Hash based on genes."""
        return hash(tuple(self._genes))
