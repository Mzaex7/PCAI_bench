# 1. Quick Start Guide

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

# 2. Usage

## Selecting Models

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

## Sequential Mode

```bash
# Run 20 iterations sequentially
python run_benchmark.py --mode sequential --iterations 20

# Custom output directory
python run_benchmark.py --output-dir my_benchmark_results

# Adjust generation parameters
python run_benchmark.py --max-tokens 1024 --temperature 0.9

# Test specific model only
python run_benchmark.py --models qwen --iterations 5
```

## Concurrent Mode - load testing

```bash
# Run with 5 concurrent requests
python run_benchmark.py --mode concurrent --concurrent 5

# High concurrency test on specific model
python run_benchmark.py --mode concurrent --concurrent 10 --iterations 5 --models llama
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
| `--mode` | str | `sequential` | Prompt execution mode: `sequential` or `concurrent` |
| `--models` | str+ | all | Space-separated list of model names to benchmark |
| `--list-models` | flag | - | List all available models and exit |
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

# 6. Adding New Models

To add a new model for benchmarking:

1. **Edit `src/config.py`** and add your model configuration:

```python
# --- Qwen 2.5 ---
QWEN = EndpointConfig(
    name="Qwen-2.5-7B",
    url="https://your-qwen-endpoint.com/v1/chat/completions",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    auth_token="your-auth-token"
)
```

2. **Add to the model registry**:

```python
ALL_MODELS = {
    # Llama family
    "llama": LLAMA_VLLM,
    "llama-vllm": LLAMA_VLLM,
    "llama-nim": LLAMA_NIM,
    
    # Add your new model here:
    "qwen": QWEN,  # ← Add here with short name
}
```

3. **Test your new model**:

```bash
# List to verify it's available
python run_benchmark.py --list-models

# Run benchmark on your new model
python run_benchmark.py --models qwen --iterations 2

# Compare with other models
python run_benchmark.py --models llama qwen gemma
```

**Note**: The endpoint must be OpenAI-compatible (supports `/v1/chat/completions` API).

## Quick Template for New Models

```python
# In config.py

# --- Your Model Name ---
YOUR_MODEL = EndpointConfig(
    name="Display Name (Deployment)",
    url="https://your-endpoint-url/v1/chat/completions",
    model_name="organization/model-name",
    auth_token="your-bearer-token"
)

# Add to ALL_MODELS:
ALL_MODELS = {
    # ... existing models ...
    "your-model": YOUR_MODEL,
}
```
