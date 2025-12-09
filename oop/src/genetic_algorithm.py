"""
Main Genetic Algorithm class that coordinates the evolutionary process.
"""
from typing import List, Dict, Optional, Callable
import random
import time

from .chromosome import Chromosome
from .population import Population
from .selection import SelectionStrategy, TournamentSelection
from .crossover import CrossoverStrategy, OnePointCrossover
from .mutation import MutationStrategy, BitFlipMutation
from .fitness import FitnessFunction


class GeneticAlgorithm:
    """
    Genetic Algorithm class that coordinates the evolutionary process.
    
    This class manages the main loop of the GA, including selection,
    crossover, mutation, and replacement operations.
    """
    
    def __init__(self,
                 fitness_function: FitnessFunction,
                 population_size: int = 100,
                 chromosome_length: int = 100,
                 selection_strategy: Optional[SelectionStrategy] = None,
                 crossover_strategy: Optional[CrossoverStrategy] = None,
                 mutation_strategy: Optional[MutationStrategy] = None,
                 elitism_count: int = 2,
                 max_generations: int = 300,
                 seed: int = 42):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            fitness_function: Function to evaluate chromosome fitness.
            population_size: Number of individuals in the population.
            chromosome_length: Length of each chromosome (bitstring).
            selection_strategy: Strategy for selecting parents.
            crossover_strategy: Strategy for crossover operation.
            mutation_strategy: Strategy for mutation operation.
            elitism_count: Number of elite individuals to preserve.
            max_generations: Maximum number of generations.
            seed: Random seed for reproducibility.
        """
        # Set random seed for reproducibility
        self._seed = seed
        random.seed(seed)
        
        self._fitness_function = fitness_function
        self._population_size = population_size
        self._chromosome_length = chromosome_length
        self._elitism_count = elitism_count
        self._max_generations = max_generations
        
        # Initialize strategies with defaults if not provided
        self._selection_strategy = selection_strategy or TournamentSelection(tournament_size=3)
        self._crossover_strategy = crossover_strategy or OnePointCrossover(probability=0.9)
        self._mutation_strategy = mutation_strategy or BitFlipMutation(
            probability=1.0/chromosome_length,
            chromosome_length=chromosome_length
        )
        
        # Initialize population
        self._population = Population(
            size=population_size,
            chromosome_length=chromosome_length,
            fitness_function=fitness_function
        )
        
        # Tracking variables
        self._history: List[Dict] = []
        self._best_solution: Optional[Chromosome] = None
        self._best_fitness: float = float('-inf')
        self._runtime: float = 0.0
    
    @property
    def population(self) -> Population:
        """Get the current population."""
        return self._population
    
    @property
    def history(self) -> List[Dict]:
        """Get the fitness history."""
        return self._history
    
    @property
    def best_solution(self) -> Optional[Chromosome]:
        """Get the best solution found."""
        return self._best_solution
    
    @property
    def best_fitness(self) -> float:
        """Get the best fitness found."""
        return self._best_fitness
    
    @property
    def runtime(self) -> float:
        """Get the total runtime in seconds."""
        return self._runtime
    
    def _create_next_generation(self) -> List[Chromosome]:
        """
        Create the next generation of chromosomes.
        
        Returns:
            List of chromosomes for the next generation.
        """
        next_generation = []
        
        # Elitism: preserve the best individuals
        elite = self._population.get_elite(self._elitism_count)
        next_generation.extend(elite)
        
        # Generate offspring to fill the rest of the population
        while len(next_generation) < self._population_size:
            # Select parents
            parents = self._selection_strategy.select(self._population.chromosomes, 2)
            parent1, parent2 = parents[0], parents[1]
            
            # Crossover
            offspring1, offspring2 = self._crossover_strategy.crossover(parent1, parent2)
            
            # Mutation
            offspring1 = self._mutation_strategy.mutate(offspring1)
            offspring2 = self._mutation_strategy.mutate(offspring2)
            
            next_generation.append(offspring1)
            if len(next_generation) < self._population_size:
                next_generation.append(offspring2)
        
        return next_generation
    
    def _update_tracking(self, generation: int):
        """
        Update tracking variables and history.
        
        Args:
            generation: Current generation number.
        """
        stats = self._population.get_fitness_stats()
        current_best = self._population.get_best()
        
        # Update best solution if improved
        if current_best.fitness is not None and current_best.fitness > self._best_fitness:
            self._best_fitness = current_best.fitness
            self._best_solution = current_best.copy()
        
        # Record history
        self._history.append({
            'generation': generation,
            'best_fitness': stats['max'],
            'average_fitness': stats['avg'],
            'worst_fitness': stats['min'],
            'best_ever': self._best_fitness
        })
    
    def run(self, verbose: bool = True, callback: Optional[Callable[[int, Dict], None]] = None) -> Dict:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: If True, print progress information.
            callback: Optional callback function called each generation.
            
        Returns:
            Dictionary containing results.
        """
        # Reset random seed
        random.seed(self._seed)
        
        # Reset tracking
        self._history = []
        self._best_solution = None
        self._best_fitness = float('-inf')
        
        # Reinitialize population with seed
        self._population = Population(
            size=self._population_size,
            chromosome_length=self._chromosome_length,
            fitness_function=self._fitness_function
        )
        
        start_time = time.time()
        
        # Evaluate initial population
        self._population.evaluate()
        self._update_tracking(0)
        
        if verbose:
            print(f"Generation 0: Best = {self._history[-1]['best_fitness']:.2f}, "
                  f"Avg = {self._history[-1]['average_fitness']:.2f}")
        
        # Main evolutionary loop
        for generation in range(1, self._max_generations + 1):
            # Create next generation
            next_gen_chromosomes = self._create_next_generation()
            
            # Update population
            self._population.set_chromosomes(next_gen_chromosomes)
            self._population.generation = generation
            
            # Evaluate new population
            self._population.evaluate()
            self._update_tracking(generation)
            
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: Best = {self._history[-1]['best_fitness']:.2f}, "
                      f"Avg = {self._history[-1]['average_fitness']:.2f}")
            
            # Call callback if provided
            if callback:
                callback(generation, self._history[-1])
            
            # Check if optimal solution found
            optimal = self._fitness_function.get_optimal_fitness()
            if self._best_fitness >= optimal:
                if verbose:
                    print(f"Optimal solution found at generation {generation}!")
                break
        
        self._runtime = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Final Results for {self._fitness_function.name}:")
            print(f"Best Fitness: {self._best_fitness:.2f}")
            print(f"Optimal Fitness: {self._fitness_function.get_optimal_fitness():.2f}")
            print(f"Runtime: {self._runtime:.2f} seconds")
            print(f"{'='*50}\n")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Get the results of the algorithm run.
        
        Returns:
            Dictionary containing results and history.
        """
        return {
            'problem': self._fitness_function.name,
            'best_fitness': self._best_fitness,
            'optimal_fitness': self._fitness_function.get_optimal_fitness(),
            'generations_run': len(self._history),
            'runtime_seconds': self._runtime,
            'history': self._history,
            'parameters': {
                'population_size': self._population_size,
                'chromosome_length': self._chromosome_length,
                'elitism_count': self._elitism_count,
                'max_generations': self._max_generations,
                'crossover_probability': self._crossover_strategy.probability,
                'mutation_probability': self._mutation_strategy.probability,
                'selection_tournament_size': getattr(self._selection_strategy, 'tournament_size', None),
                'seed': self._seed
            }
        }
