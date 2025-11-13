# LSTM Compressor Parameter Optimizer

This script optimizes the parameters of the LSTM compressor using genetic algorithms (GA) and differential evolution (DE).

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements_optimizer.txt
```

Or install individually:
```bash
pip install numpy scipy deap
```

2. Build the coder binary:
```bash
./build.sh
```

## Usage

Basic usage:
```bash
./optimize_params.py <input_file> [options]
```

The script automatically looks for `./coder` (Linux/Mac) or `./coder.exe` (Windows) and will exit if not found.

**Note:** On Windows, crash dialogs are automatically suppressed to prevent interruption during optimization.

### Options

- `--log FILE`: Log file for all test results (default: optimization.log)
- `--threads N`: Number of parallel threads (default: 7)
- `--skip-decompress`: Skip decompression test, assume same time as compression
- `--method METHOD`: Optimization method: 'de', 'ga', or 'hybrid' (default: hybrid)
- `--max-iter N`: Max iterations for differential evolution (default: 50)
- `--generations N`: Generations for genetic algorithm (default: 50)
- `--population N`: Population size for GA (default: 50)

### Examples

1. **Quick test with hybrid optimization (recommended):**
```bash
./optimize_params.py input_file.txt --threads 7 --method hybrid
```

2. **Genetic algorithm only:**
```bash
./optimize_params.py input_file.txt --method ga --generations 100 --population 50
```

3. **Differential evolution only:**
```bash
./optimize_params.py input_file.txt --method de --max-iter 100
```

4. **Skip decompression for faster testing:**
```bash
./optimize_params.py input_file.txt --skip-decompress --threads 14
```

5. **Custom log file:**
```bash
./optimize_params.py input_file.txt --log my_results.log --threads 7
```

Replace `input_file.txt` with the actual file you want to optimize compression for.

## How it Works

### Parameters Optimized

The script optimizes these parameters (ignoring `ppmd_memory` and `lstm_input_size`):

| Parameter | Range | Type | Default |
|-----------|-------|------|---------|
| ppmd_order | 2-256 | int | 12 |
| lstm_num_cells | 10-200 | int | 90 |
| lstm_num_layers | 1-10 | int | 3 |
| lstm_horizon | 1-99 | int | 10 |
| lstm_learning_rate | 0.001-0.5 | float | 0.05 |
| lstm_gradient_clip | 0.5-10.0 | float | 2.0 |
| update_limit | 500-10000 | int | 3000 |

**Note:** `lstm_input_size` is fixed at 128 (depends on input file, not optimized)

### Optimization Metric

The optimization minimizes this metric (lower is better):

```
metric = ctime + csize/uspeed + Nusers*(csize/dspeed + dtime)
```

Where:
- `ctime`: Compression time (seconds)
- `csize`: Compressed file size (bytes)
- `dtime`: Decompression time (seconds)
- `uspeed`: Upload speed = 30,000 bytes/sec (240 kbps)
- `dspeed`: Download speed = 500,000 bytes/sec (4 Mbps)
- `Nusers`: Number of users = 3

This metric balances:
- Compression time (one-time cost)
- Storage/bandwidth cost (compressed size)
- User experience (decompression time × number of users)

### Optimization Methods

1. **Differential Evolution (DE)**:
   - Gradient-free evolutionary algorithm
   - Good for global optimization
   - Robust to local minima
   - Parallel evaluation

2. **Genetic Algorithm (GA)**:
   - Population-based evolutionary approach
   - Exploration-focused
   - Good for initial search
   - Uses DEAP library

3. **Hybrid (Recommended)**:
   - Starts with GA for exploration
   - Refines with DE for exploitation
   - Best balance of exploration and optimization

### Features

- **Caching**: All parameter sets are cached to avoid retesting
- **Timeout protection**: Kills runs that take >200% of worst known time
- **Crash handling**: Crashes are detected and logged without showing error dialogs (Windows)
- **Progress tracking**: Prints current and best results after each test
- **Parallel execution**: Uses thread pool for parallel testing
- **Robust error handling**: Invalid results are marked and penalized
- **Output redirection**: Coder stdout/stderr redirected to avoid clutter
- **Result logging**: All test results are logged to a file with timestamps
- **Cross-platform**: Works on Linux, Mac, and Windows

## Output

### Console Output

After each test, the script prints:
- Current parameter values (space-separated)
- Current stats: size, compression time, decompression time, metric
- Best parameter values found so far
- Best stats

Example console output:
```
--- GA Generation 1/50 ---

Test 1: 12 90 3 10 0.05 2.0 3000 | size=1234 ctime=5.23s dtime=2.15s metric=15.67
Best:   12 90 3 10 0.05 2.0 3000 | size=1234 ctime=5.23s dtime=2.15s metric=15.67

Test 2: 10 75 2 8 0.03 1.5 2500 | size=1198 ctime=4.85s dtime=1.98s metric=14.32
Best:   10 75 2 8 0.03 1.5 2500 | size=1198 ctime=4.85s dtime=1.98s metric=14.32

Test 3: 15 120 4 20 0.1 3.0 5000 | INVALID: Timeout
Best:   10 75 2 8 0.03 1.5 2500 | size=1198 ctime=4.85s dtime=1.98s metric=14.32
```

Each test shows:
- Test number
- Parameter values (space-separated)
- Results (size, compression time, decompression time, metric) or INVALID
- Current best result

Generation/iteration markers show optimization progress.

### Log File

All test results are written to the log file (default: `optimization.log`) with timestamps. Each line contains:
- Timestamp
- Parameter values (space-separated)
- Test results: compressed size, compression time, decompression time, metric
- Or "INVALID" with error message if the test failed

Example log file content:
```
LSTM Compressor Parameter Optimization Log
================================================================================
Input file: input_file.txt
Threads: 7
Skip decompression: False
Metric: ctime + csize/30000.0 + 3*(csize/500000.0 + dtime)
================================================================================

[2025-01-15 10:23:45] Params: 12 90 3 10 0.05 2.0 3000 | size=1234 ctime=5.234s dtime=2.156s metric=15.678
[2025-01-15 10:24:12] Params: 10 75 2 8 0.03 1.5 2500 | size=1198 ctime=4.856s dtime=1.987s metric=14.321
[2025-01-15 10:24:38] Params: 15 120 4 20 0.1 3.0 5000 | INVALID: Timeout

================================================================================
OPTIMIZATION COMPLETE
================================================================================
Best params: 10 75 2 8 0.03 1.5 2500
Best metric: 14.321
Size: 1198 bytes
Compression time: 4.856s
Decompression time: 1.987s
Tested 48 unique parameter sets
```

This log file can be analyzed later to:
- Review all tested parameter combinations
- Track optimization progress over time
- Resume or refine optimization with different methods

## Tips

1. **Start small**: Test with a small file first to verify setup
2. **Use more threads**: If you have CPU cores available, increase `--threads`
3. **Skip decompression**: Use `--skip-decompress` for faster exploration
4. **Review the log**: Check `optimization.log` (or your custom log file) to analyze all tested combinations
5. **Multiple runs**: The script uses random seeds, so run multiple times and compare results

## Troubleshooting

**Error: "Coder binary not found"**
- Run `./build.sh` to compile the coder first

**Error: "scipy not available"** or **"DEAP not available"**
- Install dependencies: `pip install scipy deap`

**All tests timing out**
- Try a smaller input file
- Increase timeout multiplier in the code (default 2.0)

**Memory errors**
- Reduce `--threads`
- Use smaller parameter bounds

## Advanced Usage

To modify parameter bounds, edit `PARAM_BOUNDS` in the script:

```python
PARAM_BOUNDS = {
    'ppmd_order': (2, 256),
    'lstm_num_cells': (10, 200),
    'lstm_horizon': (1, 99),
    # ... etc
}
```

Note: `lstm_input_size` is fixed at 128 and not included in optimization.

To change the metric weights, modify in `ParamOptimizer.__init__`:

```python
self.uspeed = 240 * 1000 / 8  # Upload speed
self.dspeed = 4 * 1000 * 1000 / 8  # Download speed
self.nusers = 3  # Number of users
```
