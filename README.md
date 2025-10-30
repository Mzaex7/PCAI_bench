# 1. Quick Start Guide

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

# 2. Usage

## Sequential Mode

```bash
# Run 20 iterations sequentially
python run_benchmark.py --mode sequential --iterations 20

# Custom output directory
python run_benchmark.py --output-dir my_benchmark_results

# Adjust generation parameters
python run_benchmark.py --max-tokens 1024 --temperature 0.9
```

## Concurrent Mode - load testing

```bash
# Run with 5 concurrent requests
python run_benchmark.py --mode concurrent --concurrent 5

# High concurrency test
python run_benchmark.py --mode concurrent --concurrent 10 --iterations 5
```

## Additional Options

```bash
# Increase timeout for slower endpoints
python run_benchmark.py --timeout 180

# Disable streaming mode
python run_benchmark.py --no-stream

# Skip visualization generation (faster)
python run_benchmark.py --no-visualizations
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `sequential` | Prompt execution mode: `sequential` (one prompt at a time per endpoint) or `concurrent` (multiple prompts in parallel per endpoint). **Note**: Endpoints are always tested in parallel. |
| `--iterations` | int | `10` | Number of iterations per prompt |
| `--concurrent` | int | `1` | Number of concurrent prompts per endpoint (concurrent mode only) |
| `--timeout` | int | `120` | Request timeout in seconds |
| `--max-tokens` | int | `512` | Maximum tokens to generate |
| `--temperature` | float | `0.7` | Sampling temperature |
| `--output-dir` | str | `results` | Output directory for results |
| `--no-visualizations` | flag | `False` | Skip generating charts |
| `--no-stream` | flag | `False` | Disable streaming mode |


# 3. Understanding the Output

After the benchmark completes, you'll find:
```
results/
├── benchmark_results_[timestamp].csv      # Detailed metrics for each request
├── benchmark_summary_[timestamp].csv      # Aggregated statistics
└── visualizations/                        # 8 comparison charts
    ├── latency_comparison_[timestamp].png
    ├── ttft_comparison_[timestamp].png
    ├── tpot_comparison_[timestamp].png
    ├── throughput_comparison_[timestamp].png
    └── ... (4 more charts)
```

## Visualizations (PNG)
- `latency_comparison_[timestamp].png`
- `latency_by_length_[timestamp].png`
- `ttft_comparison_[timestamp].png`
- `tpot_comparison_[timestamp].png`
- `throughput_comparison_[timestamp].png`
- `token_analysis_[timestamp].png`
- `success_rate_[timestamp].png`
- `complexity_performance_[timestamp].png`

# 4. Key Metrics Explained

| Metric | What It Means | Better Value |
|--------|---------------|--------------|
| **TTFT** (Time to First Token) | How long until streaming starts | Lower |
| **TPOT** (Time per Output Token) | Average time to generate each token | Lower |
| **Total Latency** | End-to-end request time | Lower |
| **Throughput** | Tokens generated per second | Higher |


# 5. test prompts
Edit `src/prompts.py` and comment out some prompts in `get_test_prompts()`
- **18 Test Prompts** across different categories
- **3 Length Categories**: Short, Medium, Long
- **3 Complexity Levels**: Simple, Moderate, Complex
- **Multiple Categories**: Factual Q&A, Creative Writing, Technical, Coding, Analysis, Business Strategy, etc.
