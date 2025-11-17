# LLM Endpoint Benchmarking Suite

A framework for benchmarking LLM inference endpoints with comprehensive metrics and professional visualizations.

## Features

- **Comprehensive Metrics**: TTFT, TPOT, throughput, latency distributions
- **Parallel Testing**: Fair comparison across all endpoints simultaneously  
- **Flexible Execution**: Sequential and concurrent modes
- **Professional Visualizations**: Production-ready charts
- **Real-time Progress**: Live tracking with success rates

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Model Selection

```bash
# List all available models
python run_benchmark.py --list-models

# Run benchmark on specific model
python run_benchmark.py --models llama

# Compare multiple models
python run_benchmark.py --models llama qwen gemma deepseek

# Compare different deployments of same model
python run_benchmark.py --models llama-vllm llama-nim

# Run all configured models (default)
python run_benchmark.py
```

### Sequential Mode

```bash
# Basic usage
python run_benchmark.py --iterations 20

# Custom configuration
python run_benchmark.py --max-tokens 1024 --temperature 0.9 --output-dir results
```

### Concurrent Mode

```bash
# Load testing with 5 concurrent requests
python run_benchmark.py --mode concurrent --concurrent 5

# High concurrency test
python run_benchmark.py --mode concurrent --concurrent 10 --models llama-vllm
```

### Advanced Options

```bash
python run_benchmark.py --timeout 180 --no-stream --no-visualizations
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | str | sequential | Execution mode (sequential/concurrent) |
| `--models` | str+ | all | Space-separated model names |
| `--list-models` | flag | - | Display available models |
| `--iterations` | int | 10 | Iterations per prompt |
| `--concurrent` | int | 1 | Concurrent requests per endpoint |
| `--timeout` | int | 240 | Request timeout (seconds) |
| `--max-tokens` | int | 2048 | Maximum tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--output-dir` | str | results | Output directory |
| `--no-visualizations` | flag | - | Skip chart generation |
| `--no-stream` | flag | - | Disable streaming |


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


## Automatisch die neueste CSV finden und visualisieren
```bash
python generate_visualizations.py
```
## Oder spezifische CSV angeben
```bash
python generate_visualizations.py --csv results/benchmark_results_2024-11-07_143022.csv
```
## Mit custom Output-Verzeichnis
```bash
python generate_visualizations.py --csv results/my_results.csv --output-dir custom_charts
```

## Performance Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| **TTFT** | Time to First Token | Streaming responsiveness |
| **TPOT** | Time Per Output Token | Per-token consistency |
| **Throughput** | Tokens per second | Production capacity |
| **Latency** | End-to-end duration | Overall performance |
## Test Suite

The benchmark includes 18 curated prompts across:
- **3 Lengths**: Short, Medium, Long
- **3 Complexity Levels**: Simple, Moderate, Complex
- **10+ Categories**: Factual Q&A, coding, analysis, creative writing, etc.

Edit `src/prompts.py` to customize the test suite.

## Adding New Models

1. **Edit `src/config.py`**:

```python
YOUR_MODEL = EndpointConfig(
    name="model-deployment",
    url="https://your-endpoint.com/v1/chat/completions",
    model_name="org/model-name",
    auth_token="your-bearer-token"
)

ALL_MODELS = {
    "your-model": YOUR_MODEL,
}
```

2. **Test**:

```bash
python run_benchmark.py --list-models
python run_benchmark.py --models your-model --iterations 2
```

**Requirements**: OpenAI-compatible `/v1/chat/completions` endpoint.

## License

MIT License - See [LICENSE](LICENSE) file for details.

Copyright (c) 2025 HPE AI Infrastructure Team
