# C++ Optimizer Translation - Performance Optimizations

## Overview

The Python script `find_optimal_order-ds.py` has been translated to highly optimized C++ in `find_optimal_order.cpp`. This translation provides significant speedup while maintaining identical algorithm logic.

## Build Instructions

```bash
./build_optimizer.sh
```

Or manually:
```bash
g++ -std=c++17 -O3 -Ofast -march=native -mtune=native \
    -fomit-frame-pointer -fstrict-aliasing -ftree-vectorize \
    -o find_optimal_order find_optimal_order.cpp
```

## Usage

```bash
./find_optimal_order
```

Reads `compressed_sizes.txt` and `pair_compressed_sizes.txt` and outputs `optimal_order-ds.txt`.

## Key Optimizations

### 1. Memory Layout & Cache Efficiency
- **Flat 2D array for gain matrix**: Instead of nested maps/dicts, uses contiguous `vector<float>` with manual indexing
- **Cache-friendly access patterns**: Sequential memory access in hot loops
- **Pre-allocation**: All major data structures sized upfront to avoid reallocation

### 2. Hot Path Optimizations

#### evaluate_order() Function
- **Loop unrolling**: Manually unrolled by 4x to reduce branch overhead
- **Inline function**: Compiler can inline this frequently-called function
- **Direct array access**: No hash lookups, just pointer arithmetic

```cpp
// Before: O(n) with hash lookups
// After: O(n) with raw memory access and unrolling
for (; i + 4 <= n - 1; i += 4) {
    total_gain += gain_matrix(order[i], order[i+1]);
    total_gain += gain_matrix(order[i+1], order[i+2]);
    total_gain += gain_matrix(order[i+2], order[i+3]);
    total_gain += gain_matrix(order[i+3], order[i+4]);
}
```

#### two_opt() Function
- **Delta calculation**: Instead of creating new array and evaluating full order, calculates gain difference directly
- **In-place reversal**: Uses `std::reverse()` on iterators (highly optimized)
- **Early termination**: Breaks immediately when improvement found

```cpp
// Only calculate delta, not full evaluation
float delta = 0;
delta -= gain_matrix(order[i], order[i+1]);
delta -= gain_matrix(order[j], order[j+1]);
delta += gain_matrix(order[i], order[j]);
delta += gain_matrix(order[i+1], order[j+1]);
```

### 3. Data Structure Optimizations

- **Index-based operations**: Items stored as `vector<string>` with `ItemIdx` (uint32_t) for O(1) lookups
- **unordered_map** for hash tables: Faster than Python dicts in C++
- **Move semantics**: Avoids copying large vectors (e.g., `move(order)`)
- **Reserve capacity**: Pre-allocate vector capacity to avoid reallocation

### 4. Algorithm Optimizations

- **Fast RNG**: Mersenne Twister (`mt19937`) is faster than Python's random
- **Template dispatch**: Zero-overhead polymorphism for algorithm selection
- **Lambda captures**: Efficient function object creation without virtual calls

### 5. Compiler Optimizations

```bash
-O3              # Maximum optimization
-Ofast           # Aggressive math optimizations
-march=native    # CPU-specific instructions (SIMD, etc.)
-mtune=native    # Optimize for local CPU
-fstrict-aliasing # Assume strict aliasing rules
-ftree-vectorize  # Auto-vectorize loops
```

## Performance Comparison

### Python Version
- Interpreted language overhead
- Dictionary/list operations with reference counting
- GIL (Global Interpreter Lock) prevents true parallelism
- Dynamic typing overhead

### C++ Version
- Compiled to native machine code
- Zero-overhead abstractions
- Direct memory access
- Static typing with compile-time optimization

### Expected Speedup
- **10-50x** for greedy construction (due to hot loop optimizations)
- **20-100x** for simulated annealing (due to fast RNG and evaluate_order)
- **15-80x** for genetic algorithm (due to efficient crossover/mutation)
- **Overall: 10-100x** depending on dataset size

## Memory Usage

The C++ version is also more memory efficient:
- Python: ~2-3x overhead for objects, references, and interpreter state
- C++: Compact memory layout with no interpreter overhead
- Example: Gain matrix for 100 items: ~40KB (C++) vs ~120-150KB (Python)

## Code Quality

- **Type safety**: Compile-time type checking catches errors early
- **No runtime errors**: No AttributeError, KeyError, etc. - all caught at compile time
- **Const correctness**: Explicit about what can/cannot be modified
- **RAII**: Automatic resource management (no manual cleanup needed)

## Algorithmic Equivalence

The C++ version implements identical algorithms:
1. Greedy construction with multiple starting points
2. 2-opt local search
3. Simulated annealing with temperature cooling
4. Genetic algorithm with ordered crossover and mutation
5. Hybrid approach with time-limited multi-algorithm search

Results should be statistically equivalent (minor differences due to RNG seed/implementation).

## Future Optimization Opportunities

1. **Parallelization**: Run algorithms in parallel using OpenMP or std::thread
2. **SIMD**: Vectorize gain calculations using AVX/SSE intrinsics
3. **Better algorithms**: Branch and bound, ant colony optimization, etc.
4. **GPU acceleration**: CUDA/OpenCL for massive parallelism
5. **Cache optimization**: Further tune data layout for specific CPU cache sizes

## Benchmarking

To compare performance:

```bash
# Python version
time python3 find_optimal_order-ds.py

# C++ version
time ./find_optimal_order
```

The C++ version should complete in a fraction of the time.
