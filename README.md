# Genetic Algorithm Implementation: OOP vs Functional Programming

## Student Information
- **Name**: Ngô Kỳ Nam
- **Student ID**: 2212135

## Project Overview
This project implements a Genetic Algorithm (GA) to solve two optimization problems:
1. **OneMax Problem**: Maximize the number of 1s in a binary string of length 100
2. **0/1 Knapsack Problem**: Maximize value of selected items within capacity constraint

The implementation is done twice using two programming paradigms:
- **Object-Oriented Programming (OOP)**
- **Functional Programming (FP)**

## Project Structure
```
Extended-LTNC/
├── README.md
├── LICENSE
├── oop/
│   ├── run.py                 # Main runner for OOP implementation
│   ├── src/
│   │   ├── chromosome.py      # Chromosome class
│   │   ├── population.py      # Population class
│   │   ├── fitness.py         # Fitness function classes
│   │   ├── selection.py       # Selection strategies
│   │   ├── crossover.py       # Crossover strategies
│   │   ├── mutation.py        # Mutation strategies
│   │   └── genetic_algorithm.py # Main GA class
│   └── tests/
│       └── test_ga.py         # Unit tests
├── fp/
│   ├── run.py                 # Main runner for FP implementation
│   ├── src/
│   │   ├── chromosome.py      # Chromosome functions
│   │   ├── population.py      # Population functions
│   │   ├── fitness.py         # Fitness functions
│   │   ├── selection.py       # Selection functions
│   │   ├── crossover.py       # Crossover functions
│   │   ├── mutation.py        # Mutation functions
│   │   └── genetic_algorithm.py # Main GA functions
│   └── tests/
│       └── test_ga.py         # Unit tests
└── reports/
    ├── onemax_curve.png       # OneMax fitness evolution plot for OOP
    ├── knapsack_curve.png     # Knapsack fitness evolution plot for OOP
    ├── onemax_curve_fp.png       # OneMax fitness evolution plot for FP
    ├── knapsack_curve_fp.png     # Knapsack fitness evolution plot for FP
    ├── results_oop.json       # OOP results
    └── results_fp.json        # FP results
```

## How to Run

### Prerequisites
- Python 3.8 or higher
- matplotlib 

### Installation
```bash
# Clone the repository
git clone https://github.com/Kriss-Nevile/Extended-LTNC
cd Extended-LTNC

# Optional: Install matplotlib for plots
pip install matplotlib
```

### Running the OOP Implementation
```bash
python oop/run.py
```

### Running the FP Implementation
```bash
python fp/run.py
```

### Running Tests
```bash
# Or using unittest
python oop/tests/test_ga.py
python fp/tests/test_ga.py
```

## GA Configuration

| Parameter        | Value                    |
|------------------|--------------------------|
| Representation   | Bitstring (binary)       |
| Population Size  | 100                      |
| Chromosome Length| 100                      |
| Selection        | Tournament (k=3)         |
| Crossover        | One-point (prob=0.9)     |
| Mutation         | Bit-flip (prob=1/L)      |
| Elitism          | 2 individuals            |
| Generations      | 300                      |
| Random Seed      | 42                       |

## Design Explanations

### Important Note: Prioritizing Purity in FP

I realized while coding that trying to use the same random object in both versions was causing issues. In OOP, letting the object update its own state works perfectly. But in the FP version, passing that mutable object around broke the rules of purity. I decided to prioritize clean code over identical results, the FP version uses manually passed seeds to ensure purity. This means the two implementations produce slightly different evolutionary paths, but it ensures the FP approach remains true to its paradigm.

### In particular:

- **OOP Approach:**  
The OOP implementation uses a `random.Random` objects initialized with `seed=42` that is frequently passed around and mutated throughout execution. Each random operation (selection, crossover, mutation) advances the internal state of this shared RNG. This stateful approach is natural in OOP, where objects maintain and modify their internal state.

- **FP Approach:**  
The FP implementation takes a different path to maintain functional purity. Instead of passing a mutable `Random` object, each function accepts an integer `seed` parameter and creates its own local RNG instance. This ensures that:
    - Functions remain pure: the same seed always produces the same output
    - No shared state is mutated across function calls
    - Each operation is independently reproducible

To maintain diversity across multiple random operations (e.g., selecting many parents, creating a population), the FP code derives unique seeds by adding offsets (i.e. `seed + i`). This deterministic seed derivation preserves reproducibility in a different way compared to OOP

**Trade-off:**  
This design choice means the OOP and FP versions, while starting from the same initial seed and using identical GA parameters, will produce slightly different evolutionary trajectories. The OOP version's RNG state evolves as a single continuous stream, while the FP version creates independent RNG streams for each operation. Both approaches are valid and produce comparable results, but the difference illustrates a fundamental paradigm distinction: stateful evolution vs. pure transformation.

### OOP Design

The OOP implementation follows classic object-oriented design patterns:

**Core Classes:**
1. **Chromosome**: Encapsulates gene data with methods for manipulation
2. **Population**: Manages a collection of chromosomes with evaluation methods
3. **GeneticAlgorithm**: Coordinates the evolution process

**Strategy Pattern:**
- `SelectionStrategy` (abstract) → `TournamentSelection`, 
- `CrossoverStrategy` (abstract) → `OnePointCrossover`, 
- `MutationStrategy` (abstract) → `BitFlipMutation`, `SwapMutation`, 
- `FitnessFunction` (abstract) → `OneMaxFitness`, `KnapsackFitness`

**Key Principles Applied:**
- **Encapsulation**: Internal state is protected (using `_` prefix), accessed via properties (though this is more like a suggestion, since python don't actually enforce this)
- **Abstraction**: Abstract base classes define interfaces for strategies
- **Polymorphism**: Different strategies can be swapped without changing the GA class
- **Single Responsibility**: Each class has one clear purpose

### FP Design

The FP implementation follows functional programming principles:

**Key Characteristics:**
1. **Immutable Data**: Chromosomes are tuples (immutable), all operations return new data
2. **Pure Functions**: Functions have no side effects; same input → same output, guaranteed
3. **Seed-Based Randomness**: Each function accepts an integer `seed` and creates its own local RNG, avoiding shared mutable state
4. **Higher-Order Functions**: Functions take/return other functions (e.g., `evaluate_population`)
5. **Function Composition**: Complex operations built from simple, composable functions

**Data Representation:**
- Chromosomes: `Tuple[int, ...]` - immutable tuples of 0s and 1s
- Population: `Tuple[Tuple[Chromosome, float], ...]` - tuples of (chromosome, fitness) pairs


**Purity Through Seed Management:**
Every function that requires randomness accepts a `seed: int` parameter instead of a mutable `Random` object. Inside each function, a fresh `random.Random(seed)` instance is created, used, and discarded. This approach ensures:
- No hidden state mutations
- With this each function can be tested in isolation without RNG state concerns

**Key Functions:**
- `create_random_chromosome(length, seed)` - creates a chromosome from a seed
- `flip_gene()`, `set_gene()` - pure chromosome transformations
- `onemax_fitness()`, `knapsack_fitness()` - pure fitness calculations
- `tournament_select(population, tournament_size, seed)` - selection with seed
- `one_point_crossover(parent1, parent2, seed)` - crossover with seed
- `bit_flip_mutate(chromosome, mutation_prob, seed)` - mutation with seed
- `run_ga()` - main algorithm coordinator that derives unique seeds per generation

## Comparison: OOP vs FP

Building this Genetic Algorithm in both paradigms really highlighted their differences. Particularly:

- OOP (The Intuitive Choice):
This approach felt the most natural for a simulation. A chromosome behaves like an object, so grouping its genes and fitness together just made sense. The structure was easy to organize, and design patterns let me swap out mutation or crossover strategies without touching the rest of the system.

- FP (The Safe Choice):
The FP version was all about reliability. With every function being pure, I never had to worry about hidden side effects or unexpected state changes. If a function worked once, it would always work. The immutability enforced a kind of rigor that made the whole flow of logic feel clean and predictable.

The Randomness Trade-off:
Randomness was the biggest contrast between the two. In OOP, it was effortless—just let a shared random generator evolve naturally. In FP, staying pure meant explicitly threading seeds through functions. It required more boilerplate, but the payoff was perfect reproducibility and isolation.

Final Verdict:
For a production-ready library, I’d pick OOP for its flexibility and maintainability. But if the goal is mathematical clarity and minimizing bugs, the discipline of FP is hard to beat. Both paradigms worked well—they just operate under different rules.

## Output Files

After running both implementations, the following files are generated in `reports/`:
- `results_oop.json`: Complete results from OOP implementation
- `results_fp.json`: Complete results from FP implementation  
- `onemax_curve.png`: Fitness evolution plot for OneMax (OOP)
- `knapsack_curve.png`: Fitness evolution plot for Knapsack (OOP)
- `onemax_curve_fp.png`: Fitness evolution plot for OneMax (FP)
- `knapsack_curve_fp.png`: Fitness evolution plot for Knapsack (FP)
