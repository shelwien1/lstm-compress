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
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import platform

# Windows-specific imports for crash dialog suppression
if platform.system() == 'Windows':
    import ctypes
    from ctypes import wintypes

    # Disable Windows Error Reporting dialogs
    SEM_NOGPFAULTERRORBOX = 0x0002
    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOOPENFILEERRORBOX = 0x8000

    def disable_crash_dialogs():
        """Disable Windows crash dialogs for the current process"""
        # Set error mode to prevent crash dialog boxes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX)

    # Subprocess creation flags for Windows
    CREATE_NO_WINDOW = 0x08000000
    SUBPROCESS_FLAGS = CREATE_NO_WINDOW
else:
    def disable_crash_dialogs():
        """No-op on non-Windows systems"""
        pass

    SUBPROCESS_FLAGS = 0

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
    cmd_compress: Optional[str] = None
    cmd_decompress: Optional[str] = None


class ParamOptimizer:
    """Optimizer for LSTM compressor parameters"""

    def __init__(self, input_file: str, log_file: str = "optimization.log",
                 num_threads: int = 7, skip_decompression: bool = False,
                 timeout_multiplier: float = 2.0):
        self.input_file = input_file
        self.num_threads = num_threads
        self.skip_decompression = skip_decompression
        self.timeout_multiplier = timeout_multiplier
        self.log_file = log_file

        # Check for coder binary (both coder and coder.exe)
        self.coder_path = None
        for name in ["./coder", "./coder.exe"]:
            if os.path.exists(name):
                self.coder_path = name
                break

        if self.coder_path is None:
            print("Error: Coder binary not found (tried ./coder and ./coder.exe)")
            print("Please run ./build.sh first to compile the coder.")
            sys.exit(1)

        # Disable crash dialogs (Windows only)
        disable_crash_dialogs()

        # Cache for tested parameter sets
        self.cache: Dict[str, TestResult] = {}
        self.cache_lock = threading.Lock()

        # Track best result
        self.best_result: Optional[TestResult] = None
        self.best_lock = threading.Lock()

        # Track test count
        self.test_count = 0
        self.count_lock = threading.Lock()

        # Track worst time for timeout calculation
        self.worst_ctime: float = 60.0  # Initial estimate
        self.worst_dtime: float = 60.0

        # Metric parameters
        self.uspeed = 240 * 1000 / 8  # 30000 bytes/sec
        self.dspeed = 4 * 1000 * 1000 / 8  # 500000 bytes/sec
        self.nusers = 3

        # Temporary directory for test files (in current directory for cross-platform)
        self.temp_dir = Path("./lstm_optimize_temp")
        self.temp_dir.mkdir(exist_ok=True)

        # Compute input file hash once for verification
        self.input_file_hash = self.compute_file_hash(self.input_file)

        # Initialize log file
        self.log_lock = threading.Lock()
        with open(self.log_file, 'w') as f:
            f.write("LSTM Compressor Parameter Optimization Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Input file hash: {self.input_file_hash}\n")
            f.write(f"Threads: {self.num_threads}\n")
            f.write(f"Skip decompression: {self.skip_decompression}\n")
            f.write(f"Metric: ctime + csize/{self.uspeed} + {self.nusers}*(csize/{self.dspeed} + dtime)\n")
            f.write("=" * 80 + "\n\n")

        # Run baseline test with default parameters
        self.run_baseline()

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def run_baseline(self):
        """Run baseline test with default parameters to establish known good result"""
        print("\n=== Running baseline test with default parameters ===\n")

        # Use default parameters
        baseline_params = PARAM_DEFAULTS.copy()
        # Remove lstm_input_size as it's not optimized
        if 'lstm_input_size' in baseline_params:
            del baseline_params['lstm_input_size']

        # Run the test
        result = self.test_params(baseline_params)

        if result.valid:
            print(f"\nBaseline established: metric={result.metric:.2f}")
        else:
            print(f"\nWARNING: Baseline test failed: {result.error}")
            print("Continuing with optimization anyway...")

        print()

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
                  params: Dict, timeout: float) -> Tuple[bool, float, Optional[str], str]:
        """Run coder with given parameters and timeout

        Returns: (success, elapsed_time, error_message, command_line)
        """
        # Build command
        cmd = [self.coder_path, mode, input_file, output_file]

        # Add parameters by name (lstm_input_size is always 128, not optimized)
        params_with_fixed = params.copy()
        params_with_fixed['lstm_input_size'] = 128

        for key in ['ppmd_order', 'lstm_input_size', 'lstm_num_cells',
                    'lstm_num_layers', 'lstm_horizon', 'lstm_learning_rate',
                    'lstm_gradient_clip', 'update_limit']:
            if key in params_with_fixed:
                cmd.append(f"{key}={params_with_fixed[key]}")

        # Create command line string for logging
        cmd_str = ' '.join(cmd)

        # Run with timeout, redirect output, and suppress crash dialogs
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=SUBPROCESS_FLAGS  # Suppress crash dialogs on Windows
            )
            elapsed = time.time() - start_time

            # Check return code - treat crashes as failures
            if result.returncode != 0:
                # Negative return codes typically indicate crashes/signals
                if result.returncode < 0:
                    return False, 0.0, f"Crashed (signal {-result.returncode})", cmd_str
                else:
                    # Print exit code in hex and decimal
                    return False, 0.0, f"Exit code 0x{result.returncode:X} ({result.returncode})", cmd_str

            return True, elapsed, None, cmd_str
        except subprocess.TimeoutExpired:
            return False, timeout, "Timeout", cmd_str
        except subprocess.CalledProcessError as e:
            return False, 0.0, f"Exit code 0x{e.returncode:X} ({e.returncode})", cmd_str
        except Exception as e:
            return False, 0.0, str(e), cmd_str

    def test_params(self, params: Dict) -> TestResult:
        """Test a parameter set and return results"""
        # Check cache first
        key = self.params_to_key(params)
        with self.cache_lock:
            if key in self.cache:
                return self.cache[key]

        # Increment test count for new test
        with self.count_lock:
            self.test_count += 1
            current_test = self.test_count

        result = TestResult(params=params.copy())

        # Create unique temporary files
        thread_id = threading.get_ident()
        compressed_file = self.temp_dir / f"compressed_{thread_id}.tmp"
        decompressed_file = self.temp_dir / f"decompressed_{thread_id}.tmp"

        try:
            # Test compression
            timeout = self.worst_ctime * self.timeout_multiplier
            success, ctime, error, cmd = self.run_coder(
                'e', self.input_file, str(compressed_file), params, timeout
            )
            result.cmd_compress = cmd

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
                    success, dtime, error, dcmd = self.run_coder(
                        'd', str(compressed_file), str(decompressed_file),
                        params, timeout
                    )
                    result.cmd_decompress = dcmd

                    if not success:
                        result.error = f"Decompression failed: {error}"
                        result.valid = False
                    else:
                        result.dtime = dtime

                        # Verify decompressed file matches original by comparing hashes
                        decompressed_hash = self.compute_file_hash(str(decompressed_file))
                        if decompressed_hash != self.input_file_hash:
                            result.error = "Decompressed file does not match original (data corruption)"
                            result.valid = False
                        else:
                            result.valid = True

                            # Update worst time only if valid
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

        # Log to file
        self.log_result(result)

        # Print progress
        self.print_result(result, current_test)

        return result

    def log_result(self, result: TestResult):
        """Log test result to file"""
        with self.log_lock:
            with open(self.log_file, 'a') as f:
                # Write timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] ")

                # Write parameters
                param_names = sorted(result.params.keys())
                param_str = " ".join(f"{result.params[k]}" for k in param_names)
                f.write(f"Params: {param_str}")

                # Write results
                if result.valid:
                    f.write(f" | size={result.csize} ctime={result.ctime:.3f}s "
                           f"dtime={result.dtime:.3f}s metric={result.metric:.3f}")
                else:
                    f.write(f" | INVALID: {result.error}")

                # Write command lines
                if result.cmd_compress:
                    f.write(f"\n  CMD_COMPRESS: {result.cmd_compress}")
                if result.cmd_decompress:
                    f.write(f"\n  CMD_DECOMPRESS: {result.cmd_decompress}")

                f.write("\n")

    def print_result(self, result: TestResult, test_num: int):
        """Print test result"""
        print()  # Empty line

        # Current params on one line
        param_str = " ".join(f"{v}" for k, v in sorted(result.params.items()))
        if result.valid:
            print(f"Test {test_num}: {param_str} | size={result.csize} "
                  f"ctime={result.ctime:.2f}s dtime={result.dtime:.2f}s "
                  f"metric={result.metric:.2f}")
        else:
            print(f"Test {test_num}: {param_str} | INVALID: {result.error}")

        # Best params on one line
        if self.best_result and self.best_result.valid:
            best_param_str = " ".join(f"{v}" for k, v in sorted(self.best_result.params.items()))
            print(f"Best:      {best_param_str} | size={self.best_result.csize} "
                  f"ctime={self.best_result.ctime:.2f}s dtime={self.best_result.dtime:.2f}s "
                  f"metric={self.best_result.metric:.2f}")

        sys.stdout.flush()

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

        # Create integrality constraint (True for integer parameters)
        integrality = [PARAM_TYPES[name] == int for name in param_names]

        # Callback to print iteration progress
        iteration = [0]
        def callback(xk, convergence):
            iteration[0] += 1
            print(f"\n--- DE Iteration {iteration[0]}/{max_iter} ---")
            return False

        # Run optimization with integrality constraints
        result = differential_evolution(
            self.objective_function,
            bounds,
            integrality=integrality,  # Enforce integer constraints
            workers=self.num_threads,
            maxiter=max_iter,
            disp=False,
            updating='deferred',  # Parallel evaluation
            seed=42,
            callback=callback
        )

        print(f"\n\nDifferential Evolution complete!")
        print(f"Best metric: {result.fun:.2f}")
        param_str = " ".join(f"{v}" for k, v in sorted(self.params_array_to_dict(result.x).items()))
        print(f"Best parameters: {param_str}")

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

        # Helper function to enforce integer constraints
        def enforce_integer_constraints(individual):
            """Round integer parameters to ensure validity"""
            for i, name in enumerate(param_names):
                if PARAM_TYPES[name] == int:
                    individual[i] = round(individual[i])
                    # Clip to bounds
                    min_val, max_val = PARAM_BOUNDS[name]
                    individual[i] = max(min_val, min(max_val, individual[i]))
            return individual,

        # Decorator for genetic operators to enforce integer constraints
        def check_integers(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                # Apply integer constraints to all modified individuals
                if hasattr(args[0], '__iter__'):
                    for ind in args[:2] if len(args) >= 2 else [args[0]]:
                        enforce_integer_constraints(ind)
                return result
            return wrapper

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", check_integers(tools.cxTwoPoint))
        toolbox.register("mutate", check_integers(tools.mutPolynomialBounded),
                        low=[PARAM_BOUNDS[n][0] for n in param_names],
                        up=[PARAM_BOUNDS[n][1] for n in param_names],
                        eta=20.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Thread pool for parallel evaluation
        toolbox.register("map", ThreadPoolExecutor(max_workers=self.num_threads).map)

        # Create initial population
        pop = toolbox.population(n=population_size)

        # Run GA with generation tracking
        print(f"\n--- GA Generation 0/{generations} (Initial population) ---")

        # Evaluate initial population
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution loop
        for gen in range(1, generations + 1):
            print(f"\n--- GA Generation {gen}/{generations} ---")

            # Select offspring
            offspring = toolbox.select(pop, len(pop))
            offspring = list(toolbox.map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            pop[:] = offspring

        print("\n\nGenetic Algorithm optimization complete!")

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
    parser.add_argument("--log", default="optimization.log",
                       help="Log file for results (default: optimization.log)")
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

    # Create optimizer (will check for ./coder binary)
    optimizer = ParamOptimizer(
        input_file=args.input_file,
        log_file=args.log,
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
    print("\n\n=== OPTIMIZATION COMPLETE ===")
    if optimizer.best_result:
        param_str = " ".join(f"{v}" for k, v in sorted(optimizer.best_result.params.items()))
        print(f"\nBest: {param_str} | size={optimizer.best_result.csize} "
              f"ctime={optimizer.best_result.ctime:.2f}s dtime={optimizer.best_result.dtime:.2f}s "
              f"metric={optimizer.best_result.metric:.2f}")
        print(f"Tested {len(optimizer.cache)} unique parameter sets")
        print(f"Results logged to: {args.log}")

        # Log final summary
        with open(args.log, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("OPTIMIZATION COMPLETE\n")
            f.write("="*80 + "\n")
            param_names = sorted(optimizer.best_result.params.keys())
            param_str = " ".join(f"{optimizer.best_result.params[k]}" for k in param_names)
            f.write(f"Best params: {param_str}\n")
            f.write(f"Best metric: {optimizer.best_result.metric:.3f}\n")
            f.write(f"Size: {optimizer.best_result.csize} bytes\n")
            f.write(f"Compression time: {optimizer.best_result.ctime:.3f}s\n")
            f.write(f"Decompression time: {optimizer.best_result.dtime:.3f}s\n")
            f.write(f"Tested {len(optimizer.cache)} unique parameter sets\n")
    else:
        print("\nNo valid results found!")
        print(f"Results logged to: {args.log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
