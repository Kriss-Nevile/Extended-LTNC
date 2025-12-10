"""
Unit tests for the OOP Genetic Algorithm implementation.
"""
import sys
import os
import unittest
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chromosome import Chromosome
from src.population import Population
from src.fitness import OneMaxFitness, KnapsackFitness
from src.selection import TournamentSelection
from src.crossover import OnePointCrossover
from src.mutation import BitFlipMutation
from src.genetic_algorithm import GeneticAlgorithm


class TestChromosome(unittest.TestCase):
    """Test cases for the Chromosome class."""
    
    def test_chromosome_creation_with_length(self):
        """Test creating a chromosome with specified length."""
        random.seed(42)
        chrom = Chromosome(length=50)
        self.assertEqual(chrom.length, 50)
        self.assertTrue(all(g in [0, 1] for g in chrom.genes))
    
    def test_chromosome_creation_with_genes(self):
        """Test creating a chromosome with specified genes."""
        genes = [1, 0, 1, 1, 0]
        chrom = Chromosome(genes=genes)
        self.assertEqual(chrom.genes, genes)
        self.assertEqual(chrom.length, 5)
    
    def test_chromosome_copy(self):
        """Test chromosome copy functionality."""
        genes = [1, 0, 1, 1, 0]
        chrom1 = Chromosome(genes=genes)
        chrom1.fitness = 3.0
        chrom2 = chrom1.copy()
        
        self.assertEqual(chrom1.genes, chrom2.genes)
        self.assertEqual(chrom1.fitness, chrom2.fitness)
        
        # Ensure it's a deep copy
        chrom2.flip_gene(0)
        self.assertNotEqual(chrom1.genes, chrom2.genes)
    
    def test_flip_gene(self):
        """Test gene flipping."""
        genes = [1, 0, 1, 1, 0]
        chrom = Chromosome(genes=genes)
        chrom.flip_gene(0)
        self.assertEqual(chrom.get_gene(0), 0)
        chrom.flip_gene(1)
        self.assertEqual(chrom.get_gene(1), 1)


class TestFitness(unittest.TestCase):
    """Test cases for fitness functions."""
    
    def test_onemax_fitness(self):
        """Test OneMax fitness calculation."""
        fitness_func = OneMaxFitness(length=10)
        
        # All ones should have fitness = length
        all_ones = Chromosome(genes=[1] * 10)
        self.assertEqual(fitness_func.evaluate(all_ones), 10)
        
        # All zeros should have fitness = 0
        all_zeros = Chromosome(genes=[0] * 10)
        self.assertEqual(fitness_func.evaluate(all_zeros), 0)
        
        # Mixed should count ones
        mixed = Chromosome(genes=[1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        self.assertEqual(fitness_func.evaluate(mixed), 6)
    
    def test_onemax_optimal(self):
        """Test OneMax optimal fitness."""
        fitness_func = OneMaxFitness(length=100)
        self.assertEqual(fitness_func.get_optimal_fitness(), 100)
    
    def test_knapsack_fitness(self):
        """Test Knapsack fitness calculation."""
        fitness_func = KnapsackFitness(n_items=10, seed=42)
        
        # All zeros should have fitness = 0 (nothing selected)
        empty = Chromosome(genes=[0] * 10)
        self.assertEqual(fitness_func.evaluate(empty), 0)
        
        # All ones might exceed capacity
        full = Chromosome(genes=[1] * 10)
        fitness = fitness_func.evaluate(full)
        
        total_weight = sum(fitness_func.weights)
        if total_weight > fitness_func.capacity:
            # If overweight, fitness should be 0
            self.assertEqual(fitness, 0)
    
    def test_knapsack_capacity_constraint(self):
        """Test that knapsack respects capacity constraint."""
        fitness_func = KnapsackFitness(n_items=5, seed=42)
        
        # Create a chromosome and check if over capacity results in 0 fitness
        genes = [1] * 5  # Select all items
        chrom = Chromosome(genes=genes)
        
        total_weight = sum(fitness_func.weights)
        fitness = fitness_func.evaluate(chrom)
        
        if total_weight > fitness_func.capacity:
            self.assertEqual(fitness, 0)
        else:
            self.assertGreater(fitness, 0)


class TestSelection(unittest.TestCase):
    """Test cases for selection strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        self.fitness_func = OneMaxFitness(length=10)
        self.population = Population(
            size=10,
            chromosome_length=10,
            fitness_function=self.fitness_func
        )
        self.population.evaluate()
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        selection = TournamentSelection(tournament_size=3)
        parents = selection.select(self.population.chromosomes, 4)
        
        self.assertEqual(len(parents), 4)
        for parent in parents:
            self.assertIsInstance(parent, Chromosome)
    
    def test_tournament_selection_selects_better(self):
        """Test that tournament selection tends to select better individuals."""
        random.seed(42)
        
        # Create population with known fitness values
        chromosomes = []
        for i in range(10):
            # Create chromosomes with increasing fitness
            genes = [1] * i + [0] * (10 - i)
            chrom = Chromosome(genes=genes)
            chrom.fitness = i
            chromosomes.append(chrom)
        
        selection = TournamentSelection(tournament_size=3)
        
        # Run many selections
        selected_fitness = []
        for _ in range(100):
            parent = selection.select(chromosomes, 1)[0]
            selected_fitness.append(parent.fitness)
        
        # Average selected fitness should be above average population fitness
        avg_selected = sum(selected_fitness) / len(selected_fitness)
        avg_population = sum(c.fitness for c in chromosomes) / len(chromosomes)
        
        self.assertGreater(avg_selected, avg_population)


class TestCrossover(unittest.TestCase):
    """Test cases for crossover strategies."""
    
    def test_one_point_crossover(self):
        """Test one-point crossover."""
        random.seed(42)
        
        parent1 = Chromosome(genes=[1] * 10)
        parent2 = Chromosome(genes=[0] * 10)
        
        crossover = OnePointCrossover(probability=1.0)  # Always crossover
        offspring1, offspring2 = crossover.crossover(parent1, parent2)
        
        # Offspring should be different from parents (with high probability)
        # and should contain genes from both parents
        genes1 = offspring1.genes
        genes2 = offspring2.genes
        
        # Check that offspring have mixed genes
        self.assertEqual(len(genes1), 10)
        self.assertEqual(len(genes2), 10)
        
        # Offspring should be complements of each other
        for g1, g2 in zip(genes1, genes2):
            self.assertEqual(g1 + g2, 1)
    
    def test_crossover_probability(self):
        """Test that crossover respects probability."""
        parent1 = Chromosome(genes=[1] * 10)
        parent2 = Chromosome(genes=[0] * 10)
        
        # With 0 probability, offspring should be copies of parents
        crossover = OnePointCrossover(probability=0.0)
        
        for _ in range(10):
            offspring1, offspring2 = crossover.crossover(parent1, parent2)
            self.assertEqual(offspring1.genes, parent1.genes)
            self.assertEqual(offspring2.genes, parent2.genes)


class TestMutation(unittest.TestCase):
    """Test cases for mutation strategies."""
    
    def test_bit_flip_mutation(self):
        """Test bit-flip mutation."""
        random.seed(42)
        
        original = Chromosome(genes=[0] * 100)
        mutation = BitFlipMutation(probability=0.1, chromosome_length=100)
        
        mutated = mutation.mutate(original)
        
        # Some genes should have flipped
        ones_count = sum(mutated.genes)
        self.assertGreater(ones_count, 0)
        self.assertLess(ones_count, 100)
    
    def test_mutation_creates_copy(self):
        """Test that mutation creates a new chromosome."""
        original = Chromosome(genes=[0] * 10)
        mutation = BitFlipMutation(probability=0.5, chromosome_length=10)
        
        mutated = mutation.mutate(original)
        
        # Original should be unchanged
        self.assertEqual(original.genes, [0] * 10)


class TestPopulation(unittest.TestCase):
    """Test cases for the Population class."""
    
    def test_population_creation(self):
        """Test population creation."""
        random.seed(42)
        pop = Population(size=20, chromosome_length=50)
        
        self.assertEqual(pop.size, 20)
        self.assertEqual(len(pop.chromosomes), 20)
        for chrom in pop:
            self.assertEqual(chrom.length, 50)
    
    def test_population_evaluation(self):
        """Test population evaluation."""
        random.seed(42)
        fitness_func = OneMaxFitness(length=10)
        pop = Population(size=10, chromosome_length=10, fitness_function=fitness_func)
        
        pop.evaluate()
        
        for chrom in pop:
            self.assertIsNotNone(chrom.fitness)
    
    def test_get_best(self):
        """Test getting best chromosome."""
        fitness_func = OneMaxFitness(length=10)
        
        # Create population with known best
        chromosomes = [Chromosome(genes=[0] * 10) for _ in range(9)]
        best_chrom = Chromosome(genes=[1] * 10)
        chromosomes.append(best_chrom)
        
        pop = Population(
            size=10,
            chromosome_length=10,
            fitness_function=fitness_func,
            chromosomes=chromosomes
        )
        pop.evaluate()
        
        best = pop.get_best()
        self.assertEqual(best.fitness, 10)
    
    def test_elite_selection(self):
        """Test elite chromosome selection."""
        fitness_func = OneMaxFitness(length=10)
        
        chromosomes = []
        for i in range(10):
            genes = [1] * i + [0] * (10 - i)
            chromosomes.append(Chromosome(genes=genes))
        
        pop = Population(
            size=10,
            chromosome_length=10,
            fitness_function=fitness_func,
            chromosomes=chromosomes
        )
        pop.evaluate()
        
        elite = pop.get_elite(2)
        
        self.assertEqual(len(elite), 2)
        self.assertEqual(elite[0].fitness, 9)
        self.assertEqual(elite[1].fitness, 8)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for the GeneticAlgorithm class."""
    
    def test_ga_initialization(self):
        """Test GA initialization."""
        fitness_func = OneMaxFitness(length=50)
        ga = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=20,
            chromosome_length=50,
            max_generations=10,
            seed=42
        )
        
        self.assertEqual(ga.population.size, 20)
    
    def test_ga_improvement(self):
        """Test that GA improves fitness over generations."""
        random.seed(42)
        
        fitness_func = OneMaxFitness(length=50)
        ga = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=50,
            chromosome_length=50,
            max_generations=50,
            seed=42
        )
        
        results = ga.run(verbose=False)
        
        # Best fitness should be better than average initial fitness
        initial_best = results['history'][0]['best_fitness']
        final_best = results['best_fitness']
        
        self.assertGreaterEqual(final_best, initial_best)
    
    def test_ga_reproducibility(self):
        """Test that GA produces reproducible results with same seed."""
        fitness_func = OneMaxFitness(length=50)
        
        ga1 = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=20,
            chromosome_length=50,
            max_generations=20,
            seed=42
        )
        results1 = ga1.run(verbose=False)
        
        ga2 = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=20,
            chromosome_length=50,
            max_generations=20,
            seed=42
        )
        results2 = ga2.run(verbose=False)
        
        self.assertEqual(results1['best_fitness'], results2['best_fitness'])
        self.assertEqual(len(results1['history']), len(results2['history']))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete GA system."""
    
    def test_onemax_convergence(self):
        """Test that GA converges on OneMax problem."""
        fitness_func = OneMaxFitness(length=20)
        ga = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=50,
            chromosome_length=20,
            max_generations=100,
            seed=42
        )
        
        results = ga.run(verbose=False)
        
        # Should achieve at least 90% of optimal
        optimal = fitness_func.get_optimal_fitness()
        self.assertGreaterEqual(results['best_fitness'], 0.9 * optimal)
    
    def test_knapsack_validity(self):
        """Test that Knapsack solutions are valid."""
        fitness_func = KnapsackFitness(n_items=20, seed=42)
        ga = GeneticAlgorithm(
            fitness_function=fitness_func,
            population_size=50,
            chromosome_length=20,
            max_generations=50,
            seed=42
        )
        
        results = ga.run(verbose=False)
        
        # Best solution should have non-zero fitness (valid solution)
        self.assertGreater(results['best_fitness'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
