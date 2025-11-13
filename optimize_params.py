#!/usr/bin/env python3
"""
Parameter optimization script for LSTM compressor
Uses both Genetic Algorithm (GA) and Gradient-Free optimization
"""

import subprocess
import time
import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

# Try to import optimization libraries
try:
    from scipy.optimize import differential_evolution, minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from deap import base, creator, tools, algorithms
    import random
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("Warning: DEAP not available for GA. Install with: pip install deap")


# Parameter bounds (min, max)
# lstm_input_size is not optimized - it depends on input file (fixed at 128)
PARAM_BOUNDS = {
    'ppmd_order': (2, 256),
    'lstm_num_cells': (10, 200),
    'lstm_num_layers': (1, 10),
    'lstm_horizon': (1, 99),
    'lstm_learning_rate': (0.001, 0.5),
    'lstm_gradient_clip': (0.5, 10.0),
    'update_limit': (500, 10000),
}

# Default values
PARAM_DEFAULTS = {
    'ppmd_order': 12,
    'lstm_input_size': 128,
    'lstm_num_cells': 90,
    'lstm_num_layers': 3,
    'lstm_horizon': 10,
    'lstm_learning_rate': 0.05,
    'lstm_gradient_clip': 2.0,
    'update_limit': 3000,
}

# Parameter types (int or float)
PARAM_TYPES = {
    'ppmd_order': int,
    'lstm_num_cells': int,
    'lstm_num_layers': int,
    'lstm_horizon': int,
    'lstm_learning_rate': float,
    'lstm_gradient_clip': float,
    'update_limit': int,
}


@dataclass
class TestResult:
    """Results from testing a parameter set"""
    params: Dict
    csize: Optional[int] = None
    ctime: Optional[float] = None
    dtime: Optional[float] = None
    valid: bool = False
    metric: Optional[float] = None
    error: Optional[str] = None


class ParamOptimizer:
    """Optimizer for LSTM compressor parameters"""

    def __init__(self, input_file: str, coder_path: str = "./coder",
                 num_threads: int = 7, skip_decompression: bool = False,
                 timeout_multiplier: float = 2.0):
        self.input_file = input_file
        self.coder_path = coder_path
        self.num_threads = num_threads
        self.skip_decompression = skip_decompression
        self.timeout_multiplier = timeout_multiplier

        # Cache for tested parameter sets
        self.cache: Dict[str, TestResult] = {}
        self.cache_lock = threading.Lock()

        # Track best result
        self.best_result: Optional[TestResult] = None
        self.best_lock = threading.Lock()

        # Track worst time for timeout calculation
        self.worst_ctime: float = 60.0  # Initial estimate
        self.worst_dtime: float = 60.0

        # Metric parameters
        self.uspeed = 240 * 1000 / 8  # 30000 bytes/sec
        self.dspeed = 4 * 1000 * 1000 / 8  # 500000 bytes/sec
        self.nusers = 3

        # Temporary directory for test files
        self.temp_dir = Path("/tmp/lstm_optimize")
        self.temp_dir.mkdir(exist_ok=True)

    def params_to_key(self, params: Dict) -> str:
        """Convert parameter dict to a hashable key"""
        # Round floats for consistent hashing
        rounded = {}
        for k, v in params.items():
            if isinstance(v, float):
                rounded[k] = round(v, 6)
            else:
                rounded[k] = v
        return json.dumps(rounded, sort_keys=True)

    def calculate_metric(self, csize: int, ctime: float, dtime: float) -> float:
        """Calculate optimization metric (lower is better)"""
        metric = (ctime + csize / self.uspeed +
                 self.nusers * (csize / self.dspeed + dtime))
        return metric

    def run_coder(self, mode: str, input_file: str, output_file: str,
                  params: Dict, timeout: float) -> Tuple[bool, float, Optional[str]]:
        """Run coder with given parameters and timeout"""
        # Build command
        cmd = [self.coder_path, mode, input_file, output_file]

        # Add parameters (lstm_input_size is always 128, not optimized)
        params_with_fixed = params.copy()
        params_with_fixed['lstm_input_size'] = 128

        for key in ['ppmd_order', 'lstm_input_size', 'lstm_num_cells',
                    'lstm_num_layers', 'lstm_horizon', 'lstm_learning_rate',
                    'lstm_gradient_clip', 'update_limit']:
            if key in params_with_fixed:
                cmd.append(str(params_with_fixed[key]))

        # Run with timeout, redirect output
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            elapsed = time.time() - start_time
            return True, elapsed, None
        except subprocess.TimeoutExpired:
            return False, timeout, "Timeout"
        except subprocess.CalledProcessError as e:
            return False, 0.0, f"Exit code {e.returncode}"
        except Exception as e:
            return False, 0.0, str(e)

    def test_params(self, params: Dict) -> TestResult:
        """Test a parameter set and return results"""
        # Check cache first
        key = self.params_to_key(params)
        with self.cache_lock:
            if key in self.cache:
                return self.cache[key]

        result = TestResult(params=params.copy())

        # Create unique temporary files
        thread_id = threading.get_ident()
        compressed_file = self.temp_dir / f"compressed_{thread_id}.tmp"
        decompressed_file = self.temp_dir / f"decompressed_{thread_id}.tmp"

        try:
            # Test compression
            timeout = self.worst_ctime * self.timeout_multiplier
            success, ctime, error = self.run_coder(
                'e', self.input_file, str(compressed_file), params, timeout
            )

            if not success:
                result.error = f"Compression failed: {error}"
                result.valid = False
            else:
                result.ctime = ctime
                result.csize = os.path.getsize(compressed_file)

                # Update worst time
                if ctime > self.worst_ctime:
                    self.worst_ctime = ctime

                # Test decompression
                if self.skip_decompression:
                    result.dtime = ctime  # Assume same time
                    result.valid = True
                else:
                    timeout = self.worst_dtime * self.timeout_multiplier
                    success, dtime, error = self.run_coder(
                        'd', str(compressed_file), str(decompressed_file),
                        params, timeout
                    )

                    if not success:
                        result.error = f"Decompression failed: {error}"
                        result.valid = False
                    else:
                        result.dtime = dtime
                        result.valid = True

                        # Update worst time
                        if dtime > self.worst_dtime:
                            self.worst_dtime = dtime

                # Calculate metric if valid
                if result.valid:
                    result.metric = self.calculate_metric(
                        result.csize, result.ctime, result.dtime
                    )

        finally:
            # Clean up temporary files
            for f in [compressed_file, decompressed_file]:
                if f.exists():
                    f.unlink()

        # Cache result
        with self.cache_lock:
            self.cache[key] = result

        # Update best result
        if result.valid:
            with self.best_lock:
                if self.best_result is None or result.metric < self.best_result.metric:
                    self.best_result = result

        # Print progress
        self.print_result(result)

        return result

    def print_result(self, result: TestResult):
        """Print test result"""
        print("\n" + "="*80)
        print("Current params:")
        self.print_params(result.params)

        if result.valid:
            print(f"Stats: size={result.csize} ctime={result.ctime:.2f}s "
                  f"dtime={result.dtime:.2f}s metric={result.metric:.2f}")
        else:
            print(f"INVALID: {result.error}")

        if self.best_result and self.best_result.valid:
            print("\nBest params:")
            self.print_params(self.best_result.params)
            print(f"Best stats: size={self.best_result.csize} "
                  f"ctime={self.best_result.ctime:.2f}s "
                  f"dtime={self.best_result.dtime:.2f}s "
                  f"metric={self.best_result.metric:.2f}")
        print("="*80)
        sys.stdout.flush()

    def print_params(self, params: Dict):
        """Print parameter set"""
        print(" ".join(f"{v}" for k, v in sorted(params.items())))

    def params_array_to_dict(self, arr: np.ndarray) -> Dict:
        """Convert parameter array to dict with proper types"""
        param_names = sorted(PARAM_BOUNDS.keys())
        params = {}
        for i, name in enumerate(param_names):
            value = arr[i]
            # Clip to bounds
            min_val, max_val = PARAM_BOUNDS[name]
            value = np.clip(value, min_val, max_val)
            # Convert to proper type
            if PARAM_TYPES[name] == int:
                params[name] = int(round(value))
            else:
                params[name] = float(value)
        return params

    def params_dict_to_array(self, params: Dict) -> np.ndarray:
        """Convert parameter dict to array"""
        param_names = sorted(PARAM_BOUNDS.keys())
        return np.array([params[name] for name in param_names])

    def objective_function(self, arr: np.ndarray) -> float:
        """Objective function for optimization (lower is better)"""
        params = self.params_array_to_dict(arr)
        result = self.test_params(params)

        if result.valid:
            return result.metric
        else:
            # Return large penalty for invalid results
            return 1e9

    def optimize_differential_evolution(self, max_iter: int = 50):
        """Optimize using differential evolution (gradient-free)"""
        if not HAS_SCIPY:
            print("scipy not available. Install with: pip install scipy")
            return

        print(f"\nStarting Differential Evolution optimization (max_iter={max_iter})...")

        # Get bounds as list of tuples
        param_names = sorted(PARAM_BOUNDS.keys())
        bounds = [PARAM_BOUNDS[name] for name in param_names]

        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            workers=self.num_threads,
            maxiter=max_iter,
            disp=True,
            updating='deferred',  # Parallel evaluation
            seed=42
        )

        print(f"\nOptimization complete!")
        print(f"Best metric: {result.fun:.2f}")
        print(f"Best parameters:")
        best_params = self.params_array_to_dict(result.x)
        self.print_params(best_params)

    def optimize_genetic_algorithm(self, population_size: int = 50,
                                   generations: int = 50):
        """Optimize using genetic algorithm (DEAP)"""
        if not HAS_DEAP:
            print("DEAP not available. Install with: pip install deap")
            return

        print(f"\nStarting Genetic Algorithm optimization "
              f"(pop={population_size}, gen={generations})...")

        # Setup DEAP
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Parameter generation
        param_names = sorted(PARAM_BOUNDS.keys())
        for i, name in enumerate(param_names):
            min_val, max_val = PARAM_BOUNDS[name]
            if PARAM_TYPES[name] == int:
                toolbox.register(f"attr_{i}", random.randint, min_val, max_val)
            else:
                toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)

        # Individual and population
        attrs = [getattr(toolbox, f"attr_{i}") for i in range(len(param_names))]
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        attrs, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def eval_individual(individual):
            return (self.objective_function(np.array(individual)),)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        low=[PARAM_BOUNDS[n][0] for n in param_names],
                        up=[PARAM_BOUNDS[n][1] for n in param_names],
                        eta=20.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Thread pool for parallel evaluation
        toolbox.register("map", ThreadPoolExecutor(max_workers=self.num_threads).map)

        # Create initial population
        pop = toolbox.population(n=population_size)

        # Run GA
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                           ngen=generations, verbose=True)

        print("\nGenetic Algorithm optimization complete!")

    def optimize_hybrid(self, ga_generations: int = 20, de_maxiter: int = 20):
        """Hybrid optimization: GA first, then DE refinement"""
        print("Starting hybrid optimization (GA + DE)...")

        if HAS_DEAP:
            print("\nPhase 1: Genetic Algorithm (exploration)")
            self.optimize_genetic_algorithm(population_size=30,
                                           generations=ga_generations)

        if HAS_SCIPY and self.best_result:
            print("\nPhase 2: Differential Evolution (refinement)")
            self.optimize_differential_evolution(max_iter=de_maxiter)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize LSTM compressor parameters"
    )
    parser.add_argument("input_file", help="Input file for testing")
    parser.add_argument("--coder", default="./coder",
                       help="Path to coder binary (default: ./coder)")
    parser.add_argument("--threads", type=int, default=7,
                       help="Number of threads (default: 7)")
    parser.add_argument("--skip-decompress", action="store_true",
                       help="Skip decompression test (assume same time as compression)")
    parser.add_argument("--method", choices=['de', 'ga', 'hybrid'], default='hybrid',
                       help="Optimization method (default: hybrid)")
    parser.add_argument("--max-iter", type=int, default=50,
                       help="Max iterations for DE (default: 50)")
    parser.add_argument("--generations", type=int, default=50,
                       help="Generations for GA (default: 50)")
    parser.add_argument("--population", type=int, default=50,
                       help="Population size for GA (default: 50)")

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Check coder binary exists
    if not os.path.exists(args.coder):
        print(f"Error: Coder binary not found: {args.coder}")
        return 1

    # Create optimizer
    optimizer = ParamOptimizer(
        input_file=args.input_file,
        coder_path=args.coder,
        num_threads=args.threads,
        skip_decompression=args.skip_decompress
    )

    # Run optimization
    if args.method == 'de':
        if not HAS_SCIPY:
            print("Error: scipy required for DE. Install with: pip install scipy")
            return 1
        optimizer.optimize_differential_evolution(max_iter=args.max_iter)
    elif args.method == 'ga':
        if not HAS_DEAP:
            print("Error: DEAP required for GA. Install with: pip install deap")
            return 1
        optimizer.optimize_genetic_algorithm(
            population_size=args.population,
            generations=args.generations
        )
    elif args.method == 'hybrid':
        if not HAS_SCIPY and not HAS_DEAP:
            print("Error: scipy and/or DEAP required. Install with: "
                  "pip install scipy deap")
            return 1
        optimizer.optimize_hybrid(
            ga_generations=args.generations // 2,
            de_maxiter=args.max_iter // 2
        )

    # Print final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    if optimizer.best_result:
        print("\nBest parameters found:")
        optimizer.print_params(optimizer.best_result.params)
        print(f"\nBest metric: {optimizer.best_result.metric:.2f}")
        print(f"Size: {optimizer.best_result.csize} bytes")
        print(f"Compression time: {optimizer.best_result.ctime:.2f}s")
        print(f"Decompression time: {optimizer.best_result.dtime:.2f}s")
        print(f"\nTested {len(optimizer.cache)} unique parameter sets")
    else:
        print("\nNo valid results found!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
