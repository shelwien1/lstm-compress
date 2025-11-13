
# Improvements and fixes for `optimize_params.py`

**Goal:** help the optimizer find better results faster and fix the `result.txt` import / "suspicious-good" handling you mentioned. This document contains 
- diagnosis / likely causes, 
- concrete fixes and code patches (copy-paste ready), 
- additional robustness and UX improvements (cache persistence, warm starts, smarter timeouts, logging).

---

## 1) High-level diagnosis (why `result.txt` import may not become active)
- The script currently runs `run_baseline()` **before** `test_previous_best()`. That means the baseline becomes `self.best_result` first; when the previous-best from `result.txt` is tested later it will replace `self.best_result` **only if** it is strictly better than the baseline by `improvement_epsilon`. If you expected the `result.txt` parameters to *be used as the starting best regardless of metric*, that's the reason they didn't "become best".
- `params_to_key` currently hashes only the keys provided in the tested dict. If some params are missing (e.g. parsing omitted a parameter), the key is different from the canonical representation — you want canonical keys including every optimized parameter to avoid duplicates / mis-caching.
- Suspicious-good logic currently retests once and then accepts the retest (the code already mostly does that), but it will *always* compare suspiciousness against `self.best_result` — which may be the baseline, so the first suspicious result can be spuriously flagged. Also, a single retest may still be noisy; better to allow configurable multiple retests or a short ensemble (median of N runs).
- Warm-start for DE currently supplies a single vector in `init` — `differential_evolution` expects an initial population (or a 2D array). Providing a single vector may not do what you expect (it may error or effectively be ignored). Better: create a small initial population around the previous-best (perturbations).

---

## 2) Concrete fixes (copy these patches into your script)

### A — Prefer previous `result.txt` as starting best (recommended change in `__init__`)
Replace the two lines at the end of `__init__`:

```py
        # Run baseline test with default parameters
        self.run_baseline()

        # Test previous best parameters from result.txt if it exists
        self.test_previous_best()
```

with this (move previous-best testing earlier and only run baseline if no valid previous-best was accepted, or add a CLI flag to force acceptance):

```py
        # Test previous best parameters from result.txt if it exists and accept them as starting best
        # (if you prefer baseline as the guaranteed known-good, swap order or use --prefer-baseline flag)
        self.test_previous_best()

        # If no valid previous best was found, run baseline to establish a starting point
        if self.best_result is None:
            self.run_baseline()
        else:
            print("\nUsing previous best from result.txt as initial best.\n")
```

This makes `result.txt` parameters the *initial* `best_result` (if valid) and avoids the baseline overriding them.

---

### B — Canonicalize parameters for caching (replace `params_to_key` implementation)
Replace your `params_to_key` with this version that fills missing entries from defaults (or bounds) and normalizes floats/ints consistently:

```py
    def params_to_key(self, params: Dict) -> str:
        '''Canonicalize params (include all optimized keys) and return an MD5 key.'''
        normalized = {}
        # Ensure every parameter in PARAM_BOUNDS is present in the canonical key
        for name in sorted(PARAM_BOUNDS.keys()):
            if name in params:
                v = params[name]
            else:
                # prefer explicit default if available, otherwise use middle of bounds
                if name in PARAM_DEFAULTS:
                    v = PARAM_DEFAULTS[name]
                else:
                    lo, hi = PARAM_BOUNDS[name]
                    # For log-scale params the bounds are in log-space, but defaults are expected in real space.
                    v = (10 ** ((lo + hi) / 2.0)) if name in LOG_SCALE_PARAMS else (lo + hi) / 2.0

            if name in PARAM_TYPES and PARAM_TYPES[name] == int:
                normalized[name] = int(round(v))
            elif isinstance(v, float):
                precision = FLOAT_PRECISION.get(name, 4)
                normalized[name] = round(float(v), precision)
            else:
                normalized[name] = v

        json_str = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
```

Why: this avoids different keys for equivalent parameter sets with missing keys and makes caching deterministic and stable between runs.

---

### C — Improve suspicious-good handling (retest up to `RETEST_MAX` times and accept if consistent)

Find the block near the end of `test_params` that starts with `# Check for suspiciously good results and retest if needed` and replace it with this improved logic:

```py
        # Check for suspiciously good results and retest if needed
        if result.valid and not is_retest and self.best_result and self.best_result.valid:
            suspiciously_good = False
            reasons = []

            if result.csize * 2 <= self.best_result.csize:
                suspiciously_good = True
                reasons.append(f"csize {result.csize} vs best {self.best_result.csize}")

            if result.ctime * 2 <= self.best_result.ctime:
                suspiciously_good = True
                reasons.append(f"ctime {result.ctime:.2f}s vs best {self.best_result.ctime:.2f}s")

            if result.dtime * 2 <= self.best_result.dtime:
                suspiciously_good = True
                reasons.append(f"dtime {result.dtime:.2f}s vs best {self.best_result.dtime:.2f}s")

            if suspiciously_good:
                print(f"\nSuspiciously good result detected ({', '.join(reasons)}). Retesting up to 3 times...")
                # Perform multiple retests and accept if results are consistent (median within 10%)
                RETEST_MAX = 2
                collected = [result.metric]
                for i in range(RETEST_MAX - 1):
                    r = self.test_params(params, is_retest=True)
                    if not r.valid:
                        # a retest failed: tiebreak in favor of previous best (keep current result invalid)
                        print(f"Retest {i+1} FAILED: {r.error}")
                        collected.append(None)
                        break
                    collected.append(r.metric)

                # Filter out failed retests
                good_runs = [m for m in collected if m is not None]
                if len(good_runs) == 0:
                    # All retests failed — keep original (invalid) path
                    return result
                median_metric = sorted(good_runs)[len(good_runs)//2]
                # If the median metric is within 10% of the suspicious run, accept it
                if abs(median_metric - result.metric) / max(1.0, result.metric) <= 0.10:
                    print("Suspicious result reproduced; accepting it.")
                    # continue to caching and best-result update below (we want to accept it)
                else:
                    print("Suspicious result not reproduced consistently; discarding.")
                    # Keep the previous best; treat the suspicious run as an outlier (mark invalid)
                    result.valid = False
                    result.error = "Suspicious result not reproduced on retest"
                    # Cache the rejected run too for reproducibility and continue
                    with self.cache_lock:
                        self.cache[self.params_to_key(params)] = result
                    # Log and return
                    self.log_result(result)
                    self.print_result(result, self.test_count)
                    return result
```

Notes:
- We retest up to 3 times (configurable) and accept if the median metric is within 10% of the suspicious value. This avoids throwing away true improvements while protecting against transient flukes.
- `is_retest=True` prevents recursive retest loops.

---

### D — Make `differential_evolution` warm-start build a full initial population (replace the warm-start code block)

Replace the `if init_population:` handling in `optimize_differential_evolution` with this helper that builds a population by perturbing `best_result`:

```py
        # Add init parameter if warm starting
        if warm_start and self.best_result and self.best_result.valid:
            print("Using warm start from best result to initialize DE population...")
            base_arr = self.params_dict_to_array(self.best_result.params)
            popsize = 15  # or pick based on default DE population size heuristic
            init_population = []
            rng = np.random.default_rng(42)
            for i in range(popsize):
                # small gaussian perturbation in the scaled coordinates
                noise = rng.normal(scale=0.02, size=base_arr.shape) * (np.array([b[1]-b[0] for b in bounds]))
                cand = base_arr + noise
                # Clip to bounds
                for j, (lo, hi) in enumerate(bounds):
                    cand[j] = np.clip(cand[j], lo, hi)
                init_population.append(cand)
            de_kwargs['init'] = np.array(init_population)
```

This builds a reasonable initial set of candidates around the previous best instead of a single vector that DE may ignore.

---

### E — Persist cache to disk across runs (quick implementation)

Add two small helper methods and call them from `__init__` / `cleanup` so the cache survives process restarts. Add near top of class (after `self.cache` declaration):

```py
        # Persistent cache file
        self.cache_file = Path("optimize_cache.json")
        self._load_cache()
```

And add these methods to the class:

```py
    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    raw = json.load(f)
                # raw should be mapping key -> serialized TestResult-ish dict
                for k, v in raw.items():
                    tr = TestResult(params=v.get('params', {}),
                                    csize=v.get('csize'),
                                    ctime=v.get('ctime'),
                                    dtime=v.get('dtime'),
                                    valid=v.get('valid', False),
                                    metric=v.get('metric'),
                                    error=v.get('error'),
                                    cmd_compress=v.get('cmd_compress'),
                                    cmd_decompress=v.get('cmd_decompress'))
                    self.cache[k] = tr
                print(f"Loaded {len(self.cache)} cached results from {self.cache_file}")
            except Exception as e:
                print(f"Warning: failed to load cache: {e}")

    def _save_cache(self):
        try:
            out = {}
            with self.cache_lock:
                for k, v in self.cache.items():
                    out[k] = {
                        'params': v.params,
                        'csize': v.csize,
                        'ctime': v.ctime,
                        'dtime': v.dtime,
                        'valid': v.valid,
                        'metric': v.metric,
                        'error': v.error,
                        'cmd_compress': v.cmd_compress,
                        'cmd_decompress': v.cmd_decompress,
                    }
            with open(self.cache_file, 'w') as f:
                json.dump(out, f)
        except Exception as e:
            print(f"Warning: failed to save cache: {e}")
```

And call `_save_cache()` from `cleanup()` (and optionally at program end).

---

### F — Smarter timeout management
Current `worst_ctime` / `worst_dtime` can grow indefinitely when a single slow job appears; use rolling median + cap. Replace usage where you compute timeout with a helper `get_timeout(kind)` (add small history arrays and helper methods). Full snippet in main doc above.

Why: this prevents single outlier runs from inflating the timeout and keeps retries responsive so the optimizer explores faster.

---

## 3) Smaller improvements and UX niceties

- **DE population size**: pick DE population proportional to dimension (default rule `popsize = 15 * dim` or let `scipy` default) — warm-start population should match that shape.
- **DEAP improvements**: use `ProcessPoolExecutor` for GA evaluations (each evaluation spawns a subprocess, so process-based parallelism is safe and may scale better). But watch out for object pickling with bound methods (you may need a top-level wrapper function).
- **Floating parameter bounds check**: apply `np.clip` and explicit rounding right before `run_coder` so you never pass invalid numeric strings to the binary (and log that corrected value).
- **Deterministic random seed options**: add `--seed` to make experiments reproducible.

---

## 4) Example: small end-to-end patch to accept previous-best and add stronger suspicious handling
Below is a concise patch you can drop into the class (near where `test_previous_best` and `run_baseline` are used in `__init__`), already shown above, plus a minimal `RETEST_MAX` constant at top of file:

Add near top-level constants:

```py
# Number of retests to attempt for suspicious results (1 initial run + RETEST_MAX-1 retests)
RETEST_MAX = 2
RETEST_ACCEPT_REL_DIFF = 0.10  # accept if median result within 10%
```

Then use code from sections A and C above. These two changes alone will make `result.txt` parameters become the initial `best_result` **and** will retest suspicious runs multiple times before discarding them.

---

## 5) Suggested workflow after applying patches
1. Add `--verbose` and `--accept-previous-best` CLI switches.
2. Run the optimizer once on a medium-sized input. Let it create `optimize_cache.json` and `result.txt`.
3. If a suspicious improvement appears, check the per-run log output (with verbose) and examine the cached entries in `optimize_cache.json` (you now persist all runs and can compare them offline).
4. Resume the optimization: the optimizer will warm-start from the `result.txt` parameters and explore near that region first.

---

## 6) Quick checklist to implement or test if you prefer smaller changes first
- [ ] Move `test_previous_best()` ahead of `run_baseline()` (or use `--prefer-previous` flag). **High impact**.
- [ ] Canonicalize params used for cache keys. **High impact**.
- [ ] Increase retest count and accept reproducible suspicious results (median rule). **High impact**.
- [ ] Save/load cache to JSON to persist knowledge across runs. **Medium impact**.
- [ ] Warm-start DE with a full initial population. **Medium impact**.
- [ ] Switch GA map() to ProcessPoolExecutor for better CPU utilization. **Low/medium impact**.

---

## 7) Appendix — small utility snippets you may find handy

**Bounded append helper (keep history limited):**

```py
    def _append_history(self, hist, value, limit=200):
        hist.append(value)
        if len(hist) > limit:
            del hist[0:(len(hist)-limit)]
```

**Serialize TestResult (if you want richer cache records):**

```py
def tr_to_dict(tr: TestResult):
    return {
        'params': tr.params,
        'csize': tr.csize,
        'ctime': tr.ctime,
        'dtime': tr.dtime,
        'valid': tr.valid,
        'metric': tr.metric,
        'error': tr.error,
    }
```

---

*End of suggestions.*
