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

### Options

- `--coder PATH`: Path to coder binary (default: ./coder)
- `--threads N`: Number of parallel threads (default: 7)
- `--skip-decompress`: Skip decompression test, assume same time as compression
- `--method METHOD`: Optimization method: 'de', 'ga', or 'hybrid' (default: hybrid)
- `--max-iter N`: Max iterations for differential evolution (default: 50)
- `--generations N`: Generations for genetic algorithm (default: 50)
- `--population N`: Population size for GA (default: 50)

### Examples

1. **Quick test with hybrid optimization (recommended):**
```bash
./optimize_params.py build.sh --threads 7 --method hybrid
```

2. **Genetic algorithm only:**
```bash
./optimize_params.py build.sh --method ga --generations 100 --population 50
```

3. **Differential evolution only:**
```bash
./optimize_params.py build.sh --method de --max-iter 100
```

4. **Skip decompression for faster testing:**
```bash
./optimize_params.py build.sh --skip-decompress --threads 14
```

5. **Custom input file:**
```bash
./optimize_params.py /path/to/large/file.txt --threads 7
```

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
- **Progress tracking**: Prints current and best results after each test
- **Parallel execution**: Uses thread pool for parallel testing
- **Robust error handling**: Invalid results are marked and penalized
- **Output redirection**: Coder stdout/stderr redirected to avoid clutter

## Output

After each test, the script prints:
- Current parameter values (space-separated)
- Current stats: size, compression time, decompression time, metric
- Best parameter values found so far
- Best stats

Example output:
```
================================================================================
Current params:
12 90 3 10 0.05 2.0 3000
Stats: size=1234 ctime=5.23s dtime=2.15s metric=15.67

Best params:
10 75 2 8 0.03 1.5 2500
Best stats: size=1198 ctime=4.85s dtime=1.98s metric=14.32
================================================================================
```

## Tips

1. **Start small**: Test with a small file first to verify setup
2. **Use more threads**: If you have CPU cores available, increase `--threads`
3. **Skip decompression**: Use `--skip-decompress` for faster exploration
4. **Save results**: Redirect output to a file to save all results:
   ```bash
   ./optimize_params.py input.txt 2>&1 | tee optimization.log
   ```
5. **Multiple runs**: The script uses random seeds, so run multiple times and compare

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
