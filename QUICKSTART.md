# Quick Start Guide

Get started with the LLM Benchmark Suite in 3 simple steps!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Your Endpoints (Optional)

The benchmark suite comes pre-configured with vLLM and Nvidia NIM endpoints. If you need to change them:

Edit `src/config.py` and update the endpoint configurations with your own URLs and auth tokens.

## 3. Run Your First Benchmark

### Option A: Use the Command Line Tool (Recommended)

```bash
# Run a quick sequential benchmark (default 10 iterations)
python run_benchmark.py

# Run with custom settings
python run_benchmark.py --iterations 5 --max-tokens 256
```

### Option B: Use the Example Script

```bash
# Run the example (2 iterations for quick testing)
python example_usage.py
```

## Understanding the Output

After the benchmark completes, you'll find:

```
results/
├── benchmark_results_[timestamp].csv      # Detailed metrics for each request
├── benchmark_summary_[timestamp].csv      # Aggregated statistics
└── visualizations/                        # 8 comparison charts
    ├── latency_comparison_[timestamp].png
    ├── ttft_comparison_[timestamp].png
    ├── tpot_comparison_[timestamp].png
    ├── throughput_comparison_[timestamp].png
    └── ... (4 more charts)
```

## Key Metrics Explained

| Metric | What It Means | Better Value |
|--------|---------------|--------------|
| **TTFT** (Time to First Token) | How long until streaming starts | Lower |
| **TPOT** (Time per Output Token) | Average time to generate each token | Lower |
| **Total Latency** | End-to-end request time | Lower |
| **Throughput** | Tokens generated per second | Higher |

## Common Commands

```bash
# Quick test (5 iterations)
python run_benchmark.py --iterations 5

# Test with higher concurrency
python run_benchmark.py --mode concurrent --concurrent 5

# Longer responses
python run_benchmark.py --max-tokens 1024

# Save to custom location
python run_benchmark.py --output-dir my_results

# Skip visualizations (faster)
python run_benchmark.py --no-visualizations
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize prompts in `src/prompts.py`
- Add more endpoints in `src/config.py`
- Analyze results in the CSV files

## Troubleshooting

**Getting timeout errors?**
```bash
python run_benchmark.py --timeout 180
```

**Want fewer test prompts?**
Edit `src/prompts.py` and comment out some prompts in `get_test_prompts()`

**Need help?**
Check the [README.md](README.md) for detailed troubleshooting section.

---

Happy benchmarking! 🚀
