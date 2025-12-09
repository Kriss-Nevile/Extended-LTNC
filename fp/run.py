"""
FP Genetic Algorithm - Main Runner Script

This script runs the Genetic Algorithm for both OneMax and Knapsack problems
using Functional Programming principles.
"""
import sys
import os
import json

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genetic_algorithm import run_ga, create_ga_config
from src.fitness import create_onemax_fitness, create_knapsack_problem


def save_results(results: dict, filename: str):
    """
    Save results to a JSON file.
    
    Converts tuples to lists for JSON serialization.
    """
    # Convert tuples to lists for JSON serialization
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples(item) for item in obj]
        return obj
    
    serializable = convert_tuples(results)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)


def plot_fitness_curve(history: tuple, title: str, filename: str):
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
        plt.title(f'{title} - Fitness Evolution (FP)')
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
    print("ONEMAX PROBLEM (FP Implementation)")
    print("="*60 + "\n")
    
    # Configuration
    chromosome_length = 100
    seed = 42
    
    # Create fitness configuration
    fitness_config = create_onemax_fitness(length=chromosome_length)
    
    # Create GA configuration
    ga_config = create_ga_config(
        chromosome_length=chromosome_length,
        population_size=100,
        max_generations=300,
        crossover_prob=0.9,
        elitism_count=2,
        tournament_size=3,
        seed=seed
    )
    
    # Run GA
    results = run_ga(fitness_config, ga_config, verbose=True)
    
    # Convert history tuple to list for JSON
    results_dict = dict(results)
    results_dict['history'] = list(results['history'])
    
    return results_dict


def run_knapsack():
    """Run the Genetic Algorithm on the Knapsack problem."""
    print("\n" + "="*60)
    print("0/1 KNAPSACK PROBLEM (FP Implementation)")
    print("="*60 + "\n")
    
    # Configuration
    n_items = 100
    seed = 42
    
    # Create fitness configuration (includes problem data)
    fitness_config = create_knapsack_problem(n_items=n_items, seed=seed)
    
    print(f"Knapsack Configuration:")
    print(f"  - Number of items: {n_items}")
    print(f"  - Capacity: {fitness_config['capacity']}")
    print(f"  - Total weight of all items: {sum(fitness_config['weights'])}")
    print(f"  - Total value of all items: {sum(fitness_config['values'])}")
    print()
    
    # Create GA configuration
    ga_config = create_ga_config(
        chromosome_length=n_items,
        population_size=100,
        max_generations=300,
        crossover_prob=0.9,
        elitism_count=2,
        tournament_size=3,
        seed=seed
    )
    
    # Run GA
    results = run_ga(fitness_config, ga_config, verbose=True)
    
    # Convert for JSON serialization
    results_dict = dict(results)
    results_dict['history'] = list(results['history'])
    if 'best_chromosome' in results_dict:
        results_dict['best_chromosome'] = list(results_dict['best_chromosome'])
    
    return results_dict


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
        'implementation': 'FP',
        'onemax': onemax_results,
        'knapsack': knapsack_results
    }
    
    # Save results to JSON
    results_file = os.path.join(reports_dir, 'results_fp.json')
    save_results(all_results, results_file)
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots (with FP suffix to not overwrite OOP plots)
    onemax_plot = os.path.join(reports_dir, 'onemax_curve_fp.png')
    knapsack_plot = os.path.join(reports_dir, 'knapsack_curve_fp.png')
    
    plot_fitness_curve(tuple(onemax_results['history']), 'OneMax Problem', onemax_plot)
    plot_fitness_curve(tuple(knapsack_results['history']), 'Knapsack Problem', knapsack_plot)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (FP Implementation)")
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
