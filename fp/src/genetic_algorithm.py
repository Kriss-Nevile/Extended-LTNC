"""
Main Genetic Algorithm functions for the Functional Programming implementation.

Implements GA using pure functions, immutability, and function composition.
No classes are used - only functions and data structures.
"""
from typing import Tuple, Dict, List, Callable, Optional
import random
import time

from .chromosome import Chromosome, create_random_chromosome
from .population import (
    Population, Individual, 
    create_population, evaluate_population, 
    get_best, get_elite, get_fitness_stats
)
from .selection import select_pair
from .crossover import crossover_pair
from .mutation import bit_flip_mutate


def create_ga_config(chromosome_length: int = 100,
                     population_size: int = 100,
                     max_generations: int = 300,
                     crossover_prob: float = 0.9,
                     elitism_count: int = 2,
                     tournament_size: int = 3,
                     seed: int = 42) -> Dict:
    """
    Create a configuration dictionary for the GA.
    
    This uses a dictionary instead of a class to maintain FP style.
    
    Args:
        chromosome_length: Length of chromosomes.
        population_size: Number of individuals.
        max_generations: Maximum generations to run.
        crossover_prob: Probability of crossover.
        elitism_count: Number of elite individuals to preserve.
        tournament_size: Size of tournament selection.
        seed: Random seed for reproducibility.
        
    Returns:
        Configuration dictionary.
    """
    return {
        'chromosome_length': chromosome_length,
        'population_size': population_size,
        'max_generations': max_generations,
        'crossover_prob': crossover_prob,
        'mutation_prob': 1.0 / chromosome_length,
        'elitism_count': elitism_count,
        'tournament_size': tournament_size,
        'seed': seed
    }


def create_offspring(population: Population,
                     config: Dict,
                     seed: int) -> Tuple[Chromosome, Chromosome]:
    """
    Create two offspring from the population.
    
    Applies selection, crossover, and mutation.
    Pure function - creates unique seeds for each operation.
    
    Args:
        population: Current population.
        config: GA configuration.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of two offspring chromosomes.
    """
    # Selection (uses seed and seed+1 internally)
    parent1, parent2 = select_pair(
        population, 
        config['tournament_size'], 
        seed
    )
    
    # Crossover
    offspring1, offspring2 = crossover_pair(
        parent1, parent2,
        config['crossover_prob'],
        seed + 2
    )
    
    # Mutation (use different seeds for each offspring)
    offspring1 = bit_flip_mutate(offspring1, config['mutation_prob'], seed + 3)
    offspring2 = bit_flip_mutate(offspring2, config['mutation_prob'], seed + 4)
    
    return offspring1, offspring2


def create_next_generation(population: Population,
                           config: Dict,
                           seed: int) -> Tuple[Chromosome, ...]:
    """
    Create the next generation of chromosomes.
    
    Uses elitism and creates offspring to fill the population.
    Pure function - creates unique seeds for each offspring pair.
    
    Args:
        population: Current population with fitness.
        config: GA configuration.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of chromosomes for next generation.
    """
    pop_size = config['population_size']
    elitism = config['elitism_count']
    
    # Preserve elite individuals
    elite = get_elite(population, elitism)
    
    # Generate offspring with unique seeds for each pair
    offspring = list(elite)
    pair_index = 0
    
    while len(offspring) < pop_size:
        # Each offspring pair gets a unique seed offset (5 operations per pair)
        child1, child2 = create_offspring(population, config, seed + pair_index * 5)
        offspring.append(child1)
        if len(offspring) < pop_size:
            offspring.append(child2)
        pair_index += 1
    
    return tuple(offspring)


def run_generation(population: Population,
                   fitness_func: Callable[[Chromosome], float],
                   config: Dict,
                   seed: int) -> Population:
    """
    Run one generation of the GA.
    
    Creates next generation and evaluates fitness.
    Pure function - uses seed for all random operations.
    
    Args:
        population: Current population with fitness.
        fitness_func: Function to evaluate chromosomes.
        config: GA configuration.
        seed: Random seed for reproducibility.
        
    Returns:
        New population with fitness values.
    """
    # Create next generation chromosomes
    next_gen_chromosomes = create_next_generation(population, config, seed)
    
    # Evaluate the new population
    return evaluate_population(next_gen_chromosomes, fitness_func)


def run_ga(fitness_config: Dict,
           ga_config: Optional[Dict] = None,
           verbose: bool = True) -> Dict:
    """
    Run the genetic algorithm.
    
    This is the main entry point for running the GA.
    Uses recursion-like iteration while maintaining functional style.
    
    Args:
        fitness_config: Dictionary with fitness function and metadata.
        ga_config: GA configuration (uses defaults if not provided).
        verbose: Whether to print progress.
        
    Returns:
        Results dictionary with history and best solution.
    """
    # Use default config if not provided
    if ga_config is None:
        ga_config = create_ga_config()
    
    # Base seed for all random operations
    base_seed = ga_config['seed']
    
    # Extract fitness function
    fitness_func = fitness_config['evaluate']
    optimal = fitness_config['optimal']
    problem_name = fitness_config['name']
    
    start_time = time.time()
    
    # Create initial population (each chromosome gets unique seed)
    chromosomes = create_population(
        ga_config['population_size'],
        ga_config['chromosome_length'],
        base_seed
    )
    
    # Evaluate initial population
    population = evaluate_population(chromosomes, fitness_func)
    
    # Track history (using list for accumulation, converted to tuple at end)
    history = []
    
    # Track best solution
    best_individual = get_best(population)
    best_fitness = best_individual[1]
    best_chromosome = best_individual[0]
    
    # Record initial stats
    stats = get_fitness_stats(population)
    history.append({
        'generation': 0,
        'best_fitness': stats['max'],
        'average_fitness': stats['avg'],
        'worst_fitness': stats['min'],
        'best_ever': best_fitness
    })
    
    if verbose:
        print(f"Generation 0: Best = {stats['max']:.2f}, Avg = {stats['avg']:.2f}")
    
    # Main evolution loop
    for generation in range(1, ga_config['max_generations'] + 1):
        # Create next generation with unique seed per generation
        generation_seed = base_seed + generation * 10000
        population = run_generation(population, fitness_func, ga_config, generation_seed)
        
        # Update best
        current_best = get_best(population)
        if current_best[1] > best_fitness:
            best_fitness = current_best[1]
            best_chromosome = current_best[0]
        
        # Record stats
        stats = get_fitness_stats(population)
        history.append({
            'generation': generation,
            'best_fitness': stats['max'],
            'average_fitness': stats['avg'],
            'worst_fitness': stats['min'],
            'best_ever': best_fitness
        })
        
        if verbose and generation % 50 == 0:
            print(f"Generation {generation}: Best = {stats['max']:.2f}, Avg = {stats['avg']:.2f}")
        
        # Check if optimal found
        if best_fitness >= optimal:
            if verbose:
                print(f"Optimal solution found at generation {generation}!")
            break
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Final Results for {problem_name}:")
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Optimal Fitness: {optimal:.2f}")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"{'='*50}\n")
    
    # Return immutable results
    return {
        'problem': problem_name,
        'best_fitness': best_fitness,
        'best_chromosome': best_chromosome,
        'optimal_fitness': optimal,
        'generations_run': len(history),
        'runtime_seconds': runtime,
        'history': tuple(history),
        'parameters': {
            'population_size': ga_config['population_size'],
            'chromosome_length': ga_config['chromosome_length'],
            'elitism_count': ga_config['elitism_count'],
            'max_generations': ga_config['max_generations'],
            'crossover_probability': ga_config['crossover_prob'],
            'mutation_probability': ga_config['mutation_prob'],
            'selection_tournament_size': ga_config['tournament_size'],
            'seed': ga_config['seed']
        }
    }


# Functional composition helpers

def compose(*functions):
    """
    Compose multiple functions into a single function.
    
    compose(f, g, h)(x) = f(g(h(x)))
    
    Args:
        *functions: Functions to compose.
        
    Returns:
        Composed function.
    """
    def composed(x):
        result = x
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


def pipe(*functions):
    """
    Pipe data through multiple functions (left to right).
    
    pipe(f, g, h)(x) = h(g(f(x)))
    
    Args:
        *functions: Functions to pipe through.
        
    Returns:
        Piped function.
    """
    def piped(x):
        result = x
        for func in functions:
            result = func(result)
        return result
    return piped
