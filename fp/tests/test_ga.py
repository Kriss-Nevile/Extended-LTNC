"""
Unit tests for the FP Genetic Algorithm implementation.
"""
import sys
import os
import unittest
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chromosome import (
    create_chromosome, create_random_chromosome, copy_chromosome,
    flip_gene, chromosome_length, set_gene
)
from src.fitness import (
    onemax_fitness, create_onemax_fitness,
    knapsack_fitness, create_knapsack_problem
)
from src.selection import tournament_select, select_parents, select_pair
from src.crossover import one_point_crossover, crossover_pair
from src.mutation import bit_flip_mutate, mutate_chromosome
from src.population import (
    create_population, evaluate_population,
    get_best, get_elite, get_fitness_stats
)
from src.genetic_algorithm import run_ga, create_ga_config


class TestChromosome(unittest.TestCase):
    """Test cases for chromosome functions."""
    
    def test_create_chromosome(self):
        """Test creating a chromosome from genes."""
        genes = (1, 0, 1, 1, 0)
        chrom = create_chromosome(genes)
        self.assertEqual(chrom, genes)
        self.assertIsInstance(chrom, tuple)
    
    def test_create_random_chromosome(self):
        """Test creating a random chromosome."""
        seed = 42
        chrom = create_random_chromosome(50, seed)
        self.assertEqual(len(chrom), 50)
        self.assertTrue(all(g in [0, 1] for g in chrom))
    
    def test_chromosome_immutability(self):
        """Test that chromosomes are immutable tuples."""
        genes = (1, 0, 1, 1, 0)
        chrom = create_chromosome(genes)
        
        # Tuples are immutable
        with self.assertRaises(TypeError):
            chrom[0] = 0
    
    def test_flip_gene(self):
        """Test gene flipping creates new chromosome."""
        original = (1, 0, 1, 1, 0)
        flipped = flip_gene(original, 0)
        
        # Original unchanged
        self.assertEqual(original[0], 1)
        # New chromosome has flipped gene
        self.assertEqual(flipped[0], 0)
        self.assertNotEqual(original, flipped)
    
    def test_set_gene(self):
        """Test setting a gene creates new chromosome."""
        original = (1, 0, 1, 1, 0)
        modified = set_gene(original, 1, 1)
        
        self.assertEqual(original[1], 0)
        self.assertEqual(modified[1], 1)


class TestFitness(unittest.TestCase):
    """Test cases for fitness functions."""
    
    def test_onemax_fitness(self):
        """Test OneMax fitness calculation."""
        # All ones
        all_ones = tuple([1] * 10)
        self.assertEqual(onemax_fitness(all_ones), 10)
        
        # All zeros
        all_zeros = tuple([0] * 10)
        self.assertEqual(onemax_fitness(all_zeros), 0)
        
        # Mixed
        mixed = (1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
        self.assertEqual(onemax_fitness(mixed), 6)
    
    def test_onemax_fitness_config(self):
        """Test OneMax fitness configuration."""
        config = create_onemax_fitness(length=100)
        
        self.assertEqual(config['name'], 'OneMax')
        self.assertEqual(config['optimal'], 100)
        self.assertTrue(callable(config['evaluate']))
    
    def test_knapsack_fitness(self):
        """Test Knapsack fitness calculation."""
        values = (10, 20, 30)
        weights = (5, 10, 15)
        capacity = 20
        
        # Select items 1 and 2 (weight = 15, value = 30)
        selection = (1, 1, 0)
        fitness = knapsack_fitness(selection, values, weights, capacity)
        self.assertEqual(fitness, 30)
        
        # Over capacity - select all (weight = 30 > 20)
        all_items = (1, 1, 1)
        fitness = knapsack_fitness(all_items, values, weights, capacity)
        self.assertEqual(fitness, 0)
    
    def test_knapsack_problem_creation(self):
        """Test knapsack problem configuration."""
        config = create_knapsack_problem(n_items=10, seed=42)
        
        self.assertEqual(config['name'], 'Knapsack')
        self.assertEqual(len(config['values']), 10)
        self.assertEqual(len(config['weights']), 10)
        self.assertTrue(callable(config['evaluate']))


class TestSelection(unittest.TestCase):
    """Test cases for selection functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        # Create population with known fitness values
        self.population = tuple(
            (tuple([1] * i + [0] * (10 - i)), float(i))
            for i in range(10)
        )
    
    def test_tournament_select(self):
        """Test tournament selection."""
        selected = tournament_select(self.population, 3, self.seed)
        
        self.assertIsInstance(selected, tuple)
        self.assertEqual(len(selected), 10)
    
    def test_select_parents(self):
        """Test selecting multiple parents."""
        parents = select_parents(self.population, 4, 3, self.seed)
        
        self.assertEqual(len(parents), 4)
        for parent in parents:
            self.assertIsInstance(parent, tuple)
    
    def test_tournament_favors_better(self):
        """Test that tournament selection favors better individuals."""
        seed = 42

        selected_fitness = []
        for i in range(100):
            # use a unique derived seed per draw so selections vary
            winner = tournament_select(self.population, 3, seed + i)
            # Find fitness of selected (sum of bits equals fitness in this setup)
            fitness = sum(winner)
            selected_fitness.append(fitness)
        
        avg_selected = sum(selected_fitness) / len(selected_fitness)
        avg_population = sum(ind[1] for ind in self.population) / len(self.population)
        
        self.assertGreater(avg_selected, avg_population)


class TestCrossover(unittest.TestCase):
    """Test cases for crossover functions."""
    
    def test_one_point_crossover(self):
        """Test one-point crossover."""
        seed = 42
        parent1 = tuple([1] * 10)
        parent2 = tuple([0] * 10)
        
        offspring1, offspring2 = one_point_crossover(parent1, parent2, seed)
        
        self.assertEqual(len(offspring1), 10)
        self.assertEqual(len(offspring2), 10)
        
        # Offspring should be complements
        for g1, g2 in zip(offspring1, offspring2):
            self.assertEqual(g1 + g2, 1)
    
    def test_crossover_with_probability(self):
        """Test crossover respects probability."""
        seed = 42
        parent1 = tuple([1] * 10)
        parent2 = tuple([0] * 10)
        
        # With 0 probability, should return copies
        off1, off2 = crossover_pair(parent1, parent2, 0.0, seed)
        self.assertEqual(off1, parent1)
        self.assertEqual(off2, parent2)
    
    def test_crossover_immutability(self):
        """Test that crossover doesn't modify parents."""
        seed = 42
        parent1 = tuple([1] * 10)
        parent2 = tuple([0] * 10)
        
        original_p1 = parent1
        original_p2 = parent2
        
        one_point_crossover(parent1, parent2, seed)
        
        # Parents unchanged
        self.assertEqual(parent1, original_p1)
        self.assertEqual(parent2, original_p2)


class TestMutation(unittest.TestCase):
    """Test cases for mutation functions."""
    
    def test_bit_flip_mutate(self):
        """Test bit-flip mutation."""
        seed = 42
        original = tuple([0] * 100)
        
        mutated = bit_flip_mutate(original, 0.1, seed)
        
        # Some genes should have flipped
        ones_count = sum(mutated)
        self.assertGreater(ones_count, 0)
        self.assertLess(ones_count, 100)
    
    def test_mutation_immutability(self):
        """Test that mutation doesn't modify original."""
        seed = 42
        original = tuple([0] * 10)
        
        mutated = bit_flip_mutate(original, 0.5, seed)
        
        # Original unchanged
        self.assertEqual(original, tuple([0] * 10))
    
    def test_mutate_chromosome_function(self):
        """Test the standard mutation function."""
        seed = 42
        original = tuple([0] * 100)
        
        mutated = mutate_chromosome(original, 100, seed)
        
        self.assertEqual(len(mutated), 100)
        self.assertIsInstance(mutated, tuple)


class TestPopulation(unittest.TestCase):
    """Test cases for population functions."""
    
    def test_create_population(self):
        """Test population creation."""
        seed = 42
        pop = create_population(20, 50, seed)
        
        self.assertEqual(len(pop), 20)
        for chrom in pop:
            self.assertEqual(len(chrom), 50)
            self.assertIsInstance(chrom, tuple)
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        chromosomes = (
            tuple([1] * 10),
            tuple([0] * 10),
            tuple([1, 0] * 5)
        )
        
        evaluated = evaluate_population(chromosomes, onemax_fitness)
        
        self.assertEqual(len(evaluated), 3)
        self.assertEqual(evaluated[0][1], 10)  # All ones
        self.assertEqual(evaluated[1][1], 0)   # All zeros
        self.assertEqual(evaluated[2][1], 5)   # Half ones
    
    def test_get_best(self):
        """Test getting best individual."""
        population = (
            (tuple([0] * 10), 0.0),
            (tuple([1] * 10), 10.0),
            (tuple([1, 0] * 5), 5.0)
        )
        
        best = get_best(population)
        
        self.assertEqual(best[1], 10.0)
    
    def test_get_elite(self):
        """Test getting elite individuals."""
        population = tuple(
            (tuple([1] * i + [0] * (10 - i)), float(i))
            for i in range(10)
        )
        
        elite = get_elite(population, 2)
        
        self.assertEqual(len(elite), 2)
        # Should be the two best (highest fitness)
        self.assertEqual(sum(elite[0]), 9)
        self.assertEqual(sum(elite[1]), 8)
    
    def test_get_fitness_stats(self):
        """Test fitness statistics calculation."""
        population = (
            (tuple([0] * 10), 0.0),
            (tuple([1] * 10), 10.0),
            (tuple([1, 0] * 5), 5.0)
        )
        
        stats = get_fitness_stats(population)
        
        self.assertEqual(stats['min'], 0.0)
        self.assertEqual(stats['max'], 10.0)
        self.assertEqual(stats['avg'], 5.0)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for the genetic algorithm."""
    
    def test_create_ga_config(self):
        """Test GA configuration creation."""
        config = create_ga_config(
            chromosome_length=50,
            population_size=20,
            seed=42
        )
        
        self.assertEqual(config['chromosome_length'], 50)
        self.assertEqual(config['population_size'], 20)
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['mutation_prob'], 1.0/50)
    
    def test_ga_improvement(self):
        """Test that GA improves fitness over generations."""
        fitness_config = create_onemax_fitness(length=50)
        ga_config = create_ga_config(
            chromosome_length=50,
            population_size=50,
            max_generations=50,
            seed=42
        )
        
        results = run_ga(fitness_config, ga_config, verbose=False)
        
        initial_best = results['history'][0]['best_fitness']
        final_best = results['best_fitness']
        
        self.assertGreaterEqual(final_best, initial_best)
    
    def test_ga_reproducibility(self):
        """Test that GA produces reproducible results."""
        fitness_config = create_onemax_fitness(length=50)
        ga_config = create_ga_config(
            chromosome_length=50,
            population_size=20,
            max_generations=20,
            seed=42
        )
        
        results1 = run_ga(fitness_config, ga_config, verbose=False)
        results2 = run_ga(fitness_config, ga_config, verbose=False)
        
        self.assertEqual(results1['best_fitness'], results2['best_fitness'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete FP GA system."""
    
    def test_onemax_convergence(self):
        """Test that GA converges on OneMax problem."""
        fitness_config = create_onemax_fitness(length=20)
        ga_config = create_ga_config(
            chromosome_length=20,
            population_size=50,
            max_generations=100,
            seed=42
        )
        
        results = run_ga(fitness_config, ga_config, verbose=False)
        
        # Should achieve at least 90% of optimal
        optimal = fitness_config['optimal']
        self.assertGreaterEqual(results['best_fitness'], 0.9 * optimal)
    
    def test_knapsack_validity(self):
        """Test that Knapsack solutions are valid."""
        fitness_config = create_knapsack_problem(n_items=20, seed=42)
        ga_config = create_ga_config(
            chromosome_length=20,
            population_size=50,
            max_generations=50,
            seed=42
        )
        
        results = run_ga(fitness_config, ga_config, verbose=False)
        
        # Best solution should have positive fitness (valid solution)
        self.assertGreater(results['best_fitness'], 0)
    
    def test_pure_functions_no_side_effects(self):
        """Test that functions don't have side effects."""
        seed = 42
        
        # Create same chromosome twice
        chrom1 = create_random_chromosome(10, seed)
        chrom2 = create_random_chromosome(10, seed)
        
        self.assertEqual(chrom1, chrom2)
        
        # Evaluate should return same result
        fitness1 = onemax_fitness(chrom1)
        fitness2 = onemax_fitness(chrom1)
        
        self.assertEqual(fitness1, fitness2)


class TestFunctionalPatterns(unittest.TestCase):
    """Test functional programming patterns used in the implementation."""
    
    def test_immutable_data_structures(self):
        """Test that tuples are used for immutability."""
        seed = 42
        pop = create_population(5, 10, seed)
        
        # Population is a tuple of tuples
        self.assertIsInstance(pop, tuple)
        for chrom in pop:
            self.assertIsInstance(chrom, tuple)
    
    def test_higher_order_functions(self):
        """Test use of higher-order functions."""
        # evaluate_population uses map internally
        chromosomes = (tuple([1] * 5), tuple([0] * 5))
        evaluated = evaluate_population(chromosomes, onemax_fitness)
        
        self.assertEqual(len(evaluated), 2)
    
    def test_function_as_parameter(self):
        """Test passing functions as parameters."""
        def custom_fitness(chrom):
            return float(len(chrom) - sum(chrom))  # Count zeros
        
        chromosomes = (tuple([1] * 5), tuple([0] * 5))
        evaluated = evaluate_population(chromosomes, custom_fitness)
        
        # All ones should have 0 zeros
        self.assertEqual(evaluated[0][1], 0)
        # All zeros should have 5 zeros
        self.assertEqual(evaluated[1][1], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
