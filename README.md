
# LLM Benchmark Suite

A comprehensive Python benchmarking suite for comparing the performance of multiple LLM deployments. This toolkit provides deep technical benchmarking with detailed metrics, visualization capabilities, and support for both sequential and concurrent testing modes.

## Features

### 🎯 Core Capabilities

- **Comprehensive Performance Metrics**
  - Time to First Token (TTFT) - Streaming latency measurement
  - Time per Output Token (TPOT) - Token generation speed
  - Total end-to-end latency
  - Throughput (tokens/second)
  - Input/output token counts
  - Success/failure rate tracking

- **Diverse Test Suite**
  - 18 carefully crafted prompts
  - Varying lengths: Short, Medium, Long
  - Multiple complexity levels: Simple, Moderate, Complex
  - Different categories: Factual Q&A, Creative Writing, Technical Explanation, Coding, Analysis, and more

- **Flexible Testing Modes**
  - Sequential mode: One prompt at a time per endpoint (all endpoints tested in parallel)
  - Concurrent mode: Multiple prompts in parallel per endpoint (all endpoints tested in parallel)
  - Configurable iteration counts and concurrency levels
  - **Parallel endpoint testing**: All endpoints are always tested simultaneously for faster results

- **Rich Visualizations**
  - Latency comparisons (box plots)
  - TTFT and TPOT distributions (violin plots)
  - Throughput comparisons (bar charts)
  - Token count analysis
  - Performance across prompt lengths and complexity levels
  - Success rate tracking

- **Production-Ready**
  - Robust error handling
  - Detailed CSV output for further analysis
  - Modular, well-documented codebase
  - Configurable timeouts and parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the benchmark suite**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Configuration

### Endpoint Configuration

Edit `src/config.py` to configure your LLM endpoints:

```python
VLLM_ENDPOINT = EndpointConfig(
    name="vLLM",
    url="https://your-vllm-endpoint.com/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    auth_token="your-auth-token",
    verify_ssl=False
)

NVIDIA_NIM_ENDPOINT = EndpointConfig(
    name="Nvidia_NIM",
    url="https://your-nim-endpoint.com/v1/chat/completions",
    model_name="meta/llama-3.1-8b-instruct",
    auth_token="your-auth-token",
    verify_ssl=False
)
```

The suite currently supports OpenAI-compatible API endpoints.

## Usage

### Basic Usage

Run a sequential benchmark with default settings (10 iterations):

```bash
python run_benchmark.py
```

### Advanced Usage

#### Sequential Mode (Recommended for baseline measurements)

```bash
# Run 20 iterations sequentially
python run_benchmark.py --mode sequential --iterations 20

# Custom output directory
python run_benchmark.py --output-dir my_benchmark_results

# Adjust generation parameters
python run_benchmark.py --max-tokens 1024 --temperature 0.9
```

#### Concurrent Mode (For load testing)

```bash
# Run with 5 concurrent requests
python run_benchmark.py --mode concurrent --concurrent 5

# High concurrency test
python run_benchmark.py --mode concurrent --concurrent 10 --iterations 5
```

#### Additional Options

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

## Output

### Directory Structure

After running a benchmark, you'll find:

```
results/
├── benchmark_results_YYYYMMDD_HHMMSS.csv      # Detailed results
├── benchmark_summary_YYYYMMDD_HHMMSS.csv      # Aggregated statistics
└── visualizations/
    ├── latency_comparison_YYYYMMDD_HHMMSS.png
    ├── latency_by_length_YYYYMMDD_HHMMSS.png
    ├── ttft_comparison_YYYYMMDD_HHMMSS.png
    ├── tpot_comparison_YYYYMMDD_HHMMSS.png
    ├── throughput_comparison_YYYYMMDD_HHMMSS.png
    ├── token_analysis_YYYYMMDD_HHMMSS.png
    ├── success_rate_YYYYMMDD_HHMMSS.png
    └── complexity_performance_YYYYMMDD_HHMMSS.png
```

### CSV Output

#### Detailed Results (`benchmark_results_*.csv`)

Contains per-request metrics:
- Endpoint and model information
- Prompt metadata (length, complexity, category)
- Performance metrics (latency, TTFT, TPOT, throughput)
- Token counts (input, output, total)
- Success/failure status
- Full response text (truncated)
- Timestamp and iteration info

#### Summary Statistics (`benchmark_summary_*.csv`)

Contains aggregated statistics per endpoint:
- Total/successful/failed request counts
- Success rate percentage
- Average, min, max values for all timing metrics
- Average token counts

### Visualizations

The suite generates 8 comprehensive charts:

1. **Latency Comparison** - Box plot comparing total latency
2. **Latency by Prompt Length** - Performance across different input sizes
3. **TTFT Comparison** - Violin plot of time to first token
4. **TPOT Comparison** - Violin plot of time per output token
5. **Throughput Comparison** - Bar chart of tokens/second
6. **Token Analysis** - Input and output token statistics
7. **Success Rate** - Request success/failure percentages
8. **Complexity Performance** - Latency across complexity levels

## Interpreting Results

### Key Metrics

**Time to First Token (TTFT)**
- Lower is better
- Critical for user-perceived responsiveness
- Important for streaming applications

**Time per Output Token (TPOT)**
- Lower is better
- Indicates token generation efficiency
- Affects overall throughput

**Total Latency**
- Lower is better
- End-to-end request duration
- Includes network overhead + processing time

**Throughput (tokens/second)**
- Higher is better
- Overall generation speed
- Calculated as: output_tokens / total_latency

### Benchmark Methodology

**Endpoint Execution:**
- All configured endpoints are **always tested in parallel**
- Each endpoint runs independently on its own hardware
- No mutual interference between endpoint tests
- Significantly faster than sequential endpoint testing

**Sequential Mode (for prompts):**
- Tests prompts one at a time within each endpoint
- Provides stable, repeatable baseline measurements
- Eliminates prompt-to-prompt resource contention
- Recommended for initial performance characterization

**Concurrent Mode (for prompts):**
- Tests multiple prompts simultaneously within each endpoint
- Reveals scaling behavior and bottlenecks
- May show degraded performance under load
- Useful for capacity planning and stress testing

## Architecture

### Module Overview

```
llm_benchmark_suite/
├── src/
│   ├── config.py          # Endpoint and benchmark configuration
│   ├── prompts.py         # Test prompt generation and categorization
│   ├── benchmark.py       # Core benchmarking logic and metrics
│   ├── csv_writer.py      # CSV output generation
│   └── visualize.py       # Visualization and chart generation
├── run_benchmark.py       # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Key Classes

- **`EndpointConfig`**: Defines LLM endpoint configuration
- **`BenchmarkConfig`**: Defines benchmark execution parameters
- **`LLMBenchmark`**: Core benchmarking engine with async request handling
- **`BenchmarkResult`**: Data class for individual request metrics
- **`BenchmarkCSVWriter`**: CSV output generation
- **`BenchmarkVisualizer`**: Chart and visualization generation

## Troubleshooting

### SSL Certificate Errors

The suite is configured to skip SSL verification (`verify=False`) as specified in the requirements. If you need to enable SSL verification, modify `verify_ssl=True` in endpoint configs and update the request handling in `benchmark.py`.

### Timeout Errors

If you're experiencing frequent timeouts:
- Increase timeout: `--timeout 300`
- Reduce max tokens: `--max-tokens 256`
- Check network connectivity to endpoints

### Memory Issues

For large-scale benchmarks:
- Reduce iterations: `--iterations 5`
- Run fewer prompts by modifying `prompts.py`
- Disable visualizations: `--no-visualizations`

### Authentication Issues

Ensure your auth tokens are valid and not expired. Update tokens in `src/config.py` if needed.

## Extending the Suite

### Adding New Prompts

Edit `src/prompts.py` and add to the `get_test_prompts()` function:

```python
{
    "prompt": "Your prompt text here",
    "length": PromptLength.MEDIUM.value,
    "complexity": PromptComplexity.MODERATE.value,
    "category": "your_category"
}
```

### Adding New Endpoints

Edit `src/config.py`:

```python
NEW_ENDPOINT = EndpointConfig(
    name="My_Endpoint",
    url="https://my-endpoint.com/v1/chat/completions",
    model_name="model-name",
    auth_token="token"
)

ENDPOINTS = [VLLM_ENDPOINT, NVIDIA_NIM_ENDPOINT, NEW_ENDPOINT]
```

### Custom Visualizations

Extend the `BenchmarkVisualizer` class in `src/visualize.py`:

```python
def plot_custom_metric(self, save: bool = True) -> str:
    # Your visualization code here
    pass
```

## Best Practices

1. **Run multiple iterations** (10-20) for statistical significance
2. **Use sequential mode first** to establish baselines
3. **Test during off-peak hours** for consistent results
4. **Monitor endpoint health** during benchmarks
5. **Save results with descriptive names** for comparison over time
6. **Document any configuration changes** between benchmark runs

## Technical Details

- **Streaming Support**: Uses SSE (Server-Sent Events) for real-time token streaming
- **Async Architecture**: Built on `aiohttp` for efficient concurrent testing
- **OpenAI-Compatible**: Works with any OpenAI API-compatible endpoint
- **Extensible**: Modular design allows easy customization

## License

This benchmark suite is provided as-is for performance testing purposes.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-27
