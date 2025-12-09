"""
OOP Genetic Algorithm - Main Runner Script

This script runs the Genetic Algorithm for both OneMax and Knapsack problems
using Object-Oriented Programming principles.
"""
import sys
import os
import json
import random

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genetic_algorithm import GeneticAlgorithm
from src.fitness import OneMaxFitness, KnapsackFitness
from src.selection import TournamentSelection
from src.crossover import OnePointCrossover
from src.mutation import BitFlipMutation


def save_results(results: dict, filename: str):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def plot_fitness_curve(history: list, title: str, filename: str):
    """Plot and save the fitness evolution curve."""
    try:
        import matplotlib.pyplot as plt
        
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['average_fitness'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'g--', label='Average Fitness', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'{title} - Fitness Evolution (OOP)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {filename}")
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")


def run_onemax():
    """Run the Genetic Algorithm on the OneMax problem."""
    print("\n" + "="*60)
    print("ONEMAX PROBLEM (OOP Implementation)")
    print("="*60 + "\n")
    
    # Configuration
    chromosome_length = 100
    population_size = 100
    max_generations = 300
    seed = 42
    
    # Create fitness function
    fitness_function = OneMaxFitness(length=chromosome_length)
    
    # Create strategies
    selection = TournamentSelection(tournament_size=3)
    crossover = OnePointCrossover(probability=0.9)
    mutation = BitFlipMutation(probability=1.0/chromosome_length, chromosome_length=chromosome_length)
    
    # Create and run GA
    ga = GeneticAlgorithm(
        fitness_function=fitness_function,
        population_size=population_size,
        chromosome_length=chromosome_length,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        elitism_count=2,
        max_generations=max_generations,
        seed=seed
    )
    
    results = ga.run(verbose=True)
    
    return results


def run_knapsack():
    """Run the Genetic Algorithm on the Knapsack problem."""
    print("\n" + "="*60)
    print("0/1 KNAPSACK PROBLEM (OOP Implementation)")
    print("="*60 + "\n")
    
    # Configuration
    n_items = 100
    population_size = 100
    max_generations = 300
    seed = 42
    
    # Create fitness function
    fitness_function = KnapsackFitness(n_items=n_items, seed=seed)
    
    print(f"Knapsack Configuration:")
    print(f"  - Number of items: {n_items}")
    print(f"  - Capacity: {fitness_function.capacity}")
    print(f"  - Total weight of all items: {sum(fitness_function.weights)}")
    print(f"  - Total value of all items: {sum(fitness_function.values)}")
    print()
    
    # Create strategies
    selection = TournamentSelection(tournament_size=3)
    crossover = OnePointCrossover(probability=0.9)
    mutation = BitFlipMutation(probability=1.0/n_items, chromosome_length=n_items)
    
    # Create and run GA
    ga = GeneticAlgorithm(
        fitness_function=fitness_function,
        population_size=population_size,
        chromosome_length=n_items,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        elitism_count=2,
        max_generations=max_generations,
        seed=seed
    )
    
    results = ga.run(verbose=True)
    
    return results


def main():
    """Main function to run both problems and save results."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    reports_dir = os.path.join(project_root, 'reports')
    
    # Run OneMax problem
    onemax_results = run_onemax()
    
    # Run Knapsack problem
    knapsack_results = run_knapsack()
    
    # Combine results
    all_results = {
        'implementation': 'OOP',
        'onemax': onemax_results,
        'knapsack': knapsack_results
    }
    
    # Save results to JSON
    results_file = os.path.join(reports_dir, 'results_oop.json')
    save_results(all_results, results_file)
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    onemax_plot = os.path.join(reports_dir, 'onemax_curve.png')
    knapsack_plot = os.path.join(reports_dir, 'knapsack_curve.png')
    
    plot_fitness_curve(onemax_results['history'], 'OneMax Problem', onemax_plot)
    plot_fitness_curve(knapsack_results['history'], 'Knapsack Problem', knapsack_plot)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (OOP Implementation)")
    print("="*60)
    print(f"\nOneMax Problem:")
    print(f"  Best Fitness: {onemax_results['best_fitness']}/{onemax_results['optimal_fitness']}")
    print(f"  Runtime: {onemax_results['runtime_seconds']:.2f}s")
    
    print(f"\nKnapsack Problem:")
    print(f"  Best Fitness: {knapsack_results['best_fitness']:.0f}")
    print(f"  Approximate Optimal: {knapsack_results['optimal_fitness']:.0f}")
    print(f"  Runtime: {knapsack_results['runtime_seconds']:.2f}s")
    
    return all_results


if __name__ == '__main__':
    main()
