#!/usr/bin/env python3
"""
Find the optimal ordering of items to maximize pairwise compression gains.
Uses multiple algorithms: greedy construction, simulated annealing, and genetic algorithm.
"""

import random
import math
import time
from typing import List, Dict, Tuple, Set

def parse_individual_sizes(filename):
    """Parse individual compressed sizes from file."""
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item_id, size = line.split(' - ')
            sizes[item_id] = int(size)
    return sizes

def parse_pair_sizes(filename):
    """Parse pair compressed sizes from file."""
    pairs = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items, size = line.split(' - ')
            item_x, item_y = items.split('_')
            pairs[(item_x, item_y)] = int(size)
            pairs[(item_y, item_x)] = int(size)
    return pairs

def compute_gain_matrix(items: List[str], individual_sizes: Dict, pair_sizes: Dict) -> Dict[Tuple[str, str], float]:
    """Precompute gain matrix for all pairs."""
    gain_matrix = {}
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i != j:
                size1 = individual_sizes[item1]
                size2 = individual_sizes[item2]
                pair_size = pair_sizes.get((item1, item2), size1 + size2)
                gain = pair_size - (size1 + size2)
                gain_matrix[(item1, item2)] = gain
    return gain_matrix

def evaluate_order(order: List[str], gain_matrix: Dict) -> float:
    """Calculate total gain for a given ordering."""
    total_gain = 0
    for i in range(len(order) - 1):
        total_gain += gain_matrix[(order[i], order[i+1])]
    return total_gain

def greedy_construction(items: List[str], gain_matrix: Dict, num_starts: int = 20) -> List[str]:
    """Greedy construction from multiple starting points."""
    best_order = None
    best_gain = float('inf')
    
    for start_idx in range(min(num_starts, len(items))):
        start_item = items[start_idx]
        order = [start_item]
        remaining = set(items) - {start_item}
        
        while remaining:
            last_item = order[-1]
            best_item = None
            best_gain_local = float('inf')
            
            for item in remaining:
                gain = gain_matrix[(last_item, item)]
                if gain < best_gain_local:
                    best_gain_local = gain
                    best_item = item
            
            order.append(best_item)
            remaining.remove(best_item)
        
        total_gain = evaluate_order(order, gain_matrix)
        if total_gain < best_gain:
            best_gain = total_gain
            best_order = order
    
    return best_order

def two_opt(order: List[str], gain_matrix: Dict, max_iterations: int = 1000) -> List[str]:
    """2-opt local optimization."""
    current_order = order[:]
    current_gain = evaluate_order(current_order, gain_matrix)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(current_order) - 1):
            for j in range(i + 2, len(current_order)):
                # Create new order by reversing segment i+1 to j
                new_order = current_order[:i+1] + current_order[i+1:j+1][::-1] + current_order[j+1:]
                new_gain = evaluate_order(new_order, gain_matrix)
                
                if new_gain < current_gain:
                    current_order = new_order
                    current_gain = new_gain
                    improved = True
                    break
            if improved:
                break
    
    return current_order

def simulated_annealing(items: List[str], gain_matrix: Dict, 
                       initial_temp: float = 1000, cooling_rate: float = 0.995,
                       max_iterations: int = 5000) -> List[str]:
    """Simulated annealing for optimization."""
    current_order = items[:]
    random.shuffle(current_order)
    current_gain = evaluate_order(current_order, gain_matrix)
    
    best_order = current_order[:]
    best_gain = current_gain
    
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # Generate neighbor by swapping two random elements
        i, j = random.sample(range(len(items)), 2)
        if i == j:
            continue
            
        # Create new order
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_gain = evaluate_order(new_order, gain_matrix)
        
        gain_change = new_gain - current_gain
        
        # Accept or reject
        if gain_change < 0 or math.exp(-gain_change / temp) > random.random():
            current_order = new_order
            current_gain = new_gain
            
            if current_gain < best_gain:
                best_order = current_order[:]
                best_gain = current_gain
        
        temp *= cooling_rate
        
        if temp < 1e-6:
            break
    
    return best_order

def genetic_algorithm(items: List[str], gain_matrix: Dict,
                     population_size: int = 50, generations: int = 100,
                     mutation_rate: float = 0.1) -> List[str]:
    """Genetic algorithm for optimization."""
    
    def create_individual():
        ind = items[:]
        random.shuffle(ind)
        return ind
    
    def crossover(parent1, parent2):
        # Ordered crossover
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        
        # Copy segment from parent1
        child[start:end] = parent1[start:end]
        
        # Fill remaining from parent2
        ptr = 0
        for i in range(size):
            if child[i] is None:
                while parent2[ptr] in child:
                    ptr += 1
                child[i] = parent2[ptr]
                ptr += 1
        
        return child
    
    def mutate(individual):
        # Swap mutation
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    best_individual = None
    best_gain = float('inf')
    
    for generation in range(generations):
        # Evaluate fitness
        gains = [evaluate_order(ind, gain_matrix) for ind in population]
        
        # Update best
        for i, gain in enumerate(gains):
            if gain < best_gain:
                best_gain = gain
                best_individual = population[i][:]
        
        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            tournament_indices = random.sample(range(len(population)), 3)
            tournament = [(population[idx], gains[idx]) for idx in tournament_indices]
            winner = min(tournament, key=lambda x: x[1])[0]  # Min gain is better
            new_population.append(winner[:])
        
        # Crossover and mutation
        population = []
        for i in range(0, population_size, 2):
            if i + 1 < len(new_population):
                parent1 = new_population[i]
                parent2 = new_population[i + 1]
                child1 = mutate(crossover(parent1, parent2))
                child2 = mutate(crossover(parent2, parent1))
                population.extend([child1, child2])
        
        # Keep population size constant
        if len(population) > population_size:
            population = population[:population_size]
        elif len(population) < population_size:
            # Add random individuals if needed
            while len(population) < population_size:
                population.append(create_individual())
    
    return best_individual

def run_algorithm_with_2opt(algorithm_func, items, gain_matrix, algorithm_name):
    """Run an algorithm followed by 2-opt optimization."""
    print(f"Running {algorithm_name}...")
    try:
        order = algorithm_func(items, gain_matrix)
        gain_before = evaluate_order(order, gain_matrix)
        order = two_opt(order, gain_matrix)
        gain_after = evaluate_order(order, gain_matrix)
        improvement = gain_before - gain_after
        print(f"  {algorithm_name}: {gain_after:.0f} bytes (2-opt improved by {improvement:.0f} bytes)")
        return order, gain_after
    except Exception as e:
        print(f"  {algorithm_name} failed: {e}")
        return None, float('inf')

def hybrid_optimization(items: List[str], gain_matrix: Dict, 
                       time_limit: int = 30) -> Tuple[List[str], float]:
    """Hybrid approach using multiple algorithms with time limit."""
    start_time = time.time()
    best_order = None
    best_gain = float('inf')
    
    # Define algorithms to try
    algorithms = [
        ("greedy construction", lambda items, gm: greedy_construction(items, gm, num_starts=20)),
        ("simulated annealing", simulated_annealing),
        ("genetic algorithm", genetic_algorithm),
    ]
    
    # Run each algorithm
    for name, algorithm in algorithms:
        if time.time() - start_time < time_limit:
            order, gain = run_algorithm_with_2opt(algorithm, items, gain_matrix, name)
            if order is not None and gain < best_gain:
                best_order = order
                best_gain = gain
    
    # Multi-start random with remaining time
    print("Running multi-start random search...")
    iterations = 0
    while time.time() - start_time < time_limit and iterations < 100:
        random_order = items[:]
        random.shuffle(random_order)
        order = two_opt(random_order, gain_matrix)
        gain = evaluate_order(order, gain_matrix)
        
        if gain < best_gain:
            best_order = order
            best_gain = gain
            print(f"  Random start {iterations}: Improved to {gain:.0f}")
        
        iterations += 1
    
    return best_order, best_gain

def analyze_results(best_order: List[str], individual_sizes: Dict, pair_sizes: Dict, gain_matrix: Dict):
    """Analyze and display results."""
    best_gain = evaluate_order(best_order, gain_matrix)
    
    print("\n" + "="*80)
    print("BEST ORDERING FOUND:")
    print("="*80)
    print(f"Total compression gain: {best_gain:.0f} bytes")
    print(f"Average gain per adjacent pair: {best_gain/(len(best_order)-1):.2f} bytes")
    
    # Calculate total independent size and solid size
    total_independent = sum(individual_sizes[item] for item in best_order)
    total_solid = total_independent + best_gain
    compression_ratio = total_solid / total_independent if total_independent > 0 else 0
    
    print(f"Total independent size: {total_independent} bytes")
    print(f"Total solid size: {total_solid:.0f} bytes")
    print(f"Compression ratio: {compression_ratio:.4f}")
    
    print(f"\nOrdering ({len(best_order)} items):")
    for i, item in enumerate(best_order):
        print(f"{i:3d}: {item} ({individual_sizes[item]} bytes)")
    
    # Show top pairs by gain
    print("\n" + "="*80)
    print("TOP 10 ADJACENT PAIRS BY GAIN:")
    print("="*80)
    print(f"{'Pair':<20} {'Size X':>10} {'Size Y':>10} {'Solid':>10} {'Gain':>10} {'Savings%':>10}")
    print("-"*80)
    
    pair_gains = []
    for i in range(len(best_order) - 1):
        item1, item2 = best_order[i], best_order[i+1]
        size1 = individual_sizes[item1]
        size2 = individual_sizes[item2]
        pair_size = pair_sizes[(item1, item2)]
        gain = pair_size - (size1 + size2)
        savings_pct = -100.0 * gain / (size1 + size2) if (size1 + size2) > 0 else 0
        pair_gains.append((item1, item2, size1, size2, pair_size, gain, savings_pct))
    
    # Sort by gain (most negative first)
    pair_gains.sort(key=lambda x: x[5])
    
    for i, (item1, item2, size1, size2, pair_size, gain, savings_pct) in enumerate(pair_gains[:10]):
        print(f"{item1}_{item2:<13} {size1:>10} {size2:>10} {pair_size:>10} {gain:>10} {savings_pct:>9.2f}%")
    
    return best_gain, total_independent, total_solid, compression_ratio

def main():
    print("Loading data...")
    individual_sizes = parse_individual_sizes('compressed_sizes.txt')
    pair_sizes = parse_pair_sizes('pair_compressed_sizes.txt')
    
    items = sorted(individual_sizes.keys())
    print(f"Found {len(items)} items")
    print(f"Found {len(pair_sizes)//2} unique pairs")
    
    print("\nPrecomputing gain matrix...")
    gain_matrix = compute_gain_matrix(items, individual_sizes, pair_sizes)
    
    print("\nFinding optimal ordering using hybrid approach...")
    start_time = time.time()
    best_order, best_gain = hybrid_optimization(items, gain_matrix, time_limit=60)
    end_time = time.time()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    best_gain, total_independent, total_solid, compression_ratio = analyze_results(
        best_order, individual_sizes, pair_sizes, gain_matrix
    )
    
    # Save ordering to file
    output_file = 'optimal_order-ds.txt'
    with open(output_file, 'w') as f:
        f.write("# Optimal ordering of items to maximize pairwise compression gains\n")
        f.write(f"# Total gain: {best_gain:.0f} bytes\n")
        f.write(f"# Average gain per pair: {best_gain/(len(best_order)-1):.2f} bytes\n")
        f.write(f"# Total independent size: {total_independent} bytes\n")
        f.write(f"# Total solid size: {total_solid:.0f} bytes\n")
        f.write(f"# Compression ratio: {compression_ratio:.4f}\n\n")
        for item in best_order:
            f.write(item + "\n")
    
    print(f"\nOrdering saved to: {output_file}")

if __name__ == '__main__':
    main()
