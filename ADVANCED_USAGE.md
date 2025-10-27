# Advanced Usage Guide

This guide covers advanced features and customization options for the LLM Benchmark Suite.

## Table of Contents

1. [Custom Benchmark Configurations](#custom-benchmark-configurations)
2. [Programmatic Usage](#programmatic-usage)
3. [Custom Prompts](#custom-prompts)
4. [Analyzing Specific Metrics](#analyzing-specific-metrics)
5. [Integrating with CI/CD](#integrating-with-cicd)
6. [Performance Tuning](#performance-tuning)

## Custom Benchmark Configurations

### Creating Custom Configurations

```python
from src.config import BenchmarkConfig

# Configuration for quick smoke tests
smoke_test_config = BenchmarkConfig(
    num_iterations=2,
    concurrent_requests=1,
    timeout=60,
    max_tokens=128,
    temperature=0.5,
    stream=True
)

# Configuration for stress testing
stress_test_config = BenchmarkConfig(
    num_iterations=50,
    concurrent_requests=10,
    timeout=300,
    max_tokens=1024,
    temperature=0.7,
    stream=True
)

# Configuration for latency-focused testing
latency_config = BenchmarkConfig(
    num_iterations=100,
    concurrent_requests=1,  # Sequential for pure latency measurement
    timeout=120,
    max_tokens=256,
    temperature=0.0,  # Deterministic for consistency
    stream=True
)
```

## Programmatic Usage

### Basic Programmatic Example

```python
import asyncio
from src.config import ENDPOINTS, BenchmarkConfig
from src.benchmark import LLMBenchmark
from src.csv_writer import BenchmarkCSVWriter
from src.visualize import BenchmarkVisualizer

async def run_custom_benchmark():
    # Create configuration
    config = BenchmarkConfig(
        num_iterations=5,
        concurrent_requests=2,
        timeout=120,
        max_tokens=512
    )
    
    # Initialize and run benchmark
    benchmark = LLMBenchmark(config)
    results = await benchmark.run_benchmark(ENDPOINTS, sequential=False)
    
    # Process results
    results_dict = benchmark.get_results()
    
    # Save and visualize
    writer = BenchmarkCSVWriter()
    writer.write_results(results_dict, "my_results")
    
    visualizer = BenchmarkVisualizer(results_dict, "my_visualizations")
    visualizer.generate_all_visualizations()

# Run it
asyncio.run(run_custom_benchmark())
```

### Testing Specific Endpoints

```python
from src.config import VLLM_ENDPOINT, NVIDIA_NIM_ENDPOINT

# Test only vLLM
async def test_vllm_only():
    config = BenchmarkConfig(num_iterations=10)
    benchmark = LLMBenchmark(config)
    results = await benchmark.run_benchmark([VLLM_ENDPOINT], sequential=True)
    return results

# Test only Nvidia NIM
async def test_nim_only():
    config = BenchmarkConfig(num_iterations=10)
    benchmark = LLMBenchmark(config)
    results = await benchmark.run_benchmark([NVIDIA_NIM_ENDPOINT], sequential=True)
    return results
```

### Filtering Prompts

```python
from src.prompts import get_test_prompts, PromptLength, PromptComplexity

# Get only short prompts
short_prompts = [p for p in get_test_prompts() if p['length'] == 'short']

# Get only complex prompts
complex_prompts = [p for p in get_test_prompts() if p['complexity'] == 'complex']

# Get specific category
coding_prompts = [p for p in get_test_prompts() if p['category'] == 'coding']

# Test with filtered prompts
async def test_short_prompts_only():
    config = BenchmarkConfig(num_iterations=5)
    benchmark = LLMBenchmark(config)
    
    # You'll need to modify the benchmark.py to accept custom prompts
    # Or create a custom test loop
```

## Custom Prompts

### Adding Domain-Specific Prompts

Edit `src/prompts.py` and add your prompts:

```python
def get_custom_prompts():
    """Custom prompts for specific use case."""
    return [
        {
            "prompt": "Your domain-specific prompt here",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "domain_specific"
        },
        # Add more prompts...
    ]
```

### Creating Prompt Templates

```python
def generate_parameterized_prompts(params):
    """Generate prompts from templates."""
    prompts = []
    
    template = "Write a {style} story about {topic} in {length} words."
    
    for style in params['styles']:
        for topic in params['topics']:
            for length in params['lengths']:
                prompts.append({
                    "prompt": template.format(style=style, topic=topic, length=length),
                    "length": "medium",
                    "complexity": "moderate",
                    "category": "creative_writing"
                })
    
    return prompts

# Use it
params = {
    'styles': ['dramatic', 'humorous', 'mysterious'],
    'topics': ['time travel', 'artificial intelligence', 'space exploration'],
    'lengths': [100, 200, 500]
}

custom_prompts = generate_parameterized_prompts(params)
```

## Analyzing Specific Metrics

### Custom Analysis Script

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_ttft(csv_file):
    """Analyze Time to First Token in detail."""
    df = pd.read_csv(csv_file)
    df_success = df[df['success'] == True]
    
    # TTFT by endpoint
    ttft_stats = df_success.groupby('endpoint_name')['time_to_first_token'].agg([
        'mean', 'median', 'std', 'min', 'max'
    ])
    
    print("TTFT Statistics by Endpoint:")
    print(ttft_stats)
    
    # TTFT percentiles
    for endpoint in df_success['endpoint_name'].unique():
        endpoint_data = df_success[df_success['endpoint_name'] == endpoint]['time_to_first_token']
        p50 = endpoint_data.quantile(0.50)
        p90 = endpoint_data.quantile(0.90)
        p95 = endpoint_data.quantile(0.95)
        p99 = endpoint_data.quantile(0.99)
        
        print(f"\n{endpoint} TTFT Percentiles:")
        print(f"  P50: {p50:.3f}s")
        print(f"  P90: {p90:.3f}s")
        print(f"  P95: {p95:.3f}s")
        print(f"  P99: {p99:.3f}s")

def analyze_throughput_by_length(csv_file):
    """Analyze throughput across different prompt lengths."""
    df = pd.read_csv(csv_file)
    df_success = df[df['success'] == True]
    
    throughput_by_length = df_success.groupby(['endpoint_name', 'prompt_length'])['tokens_per_second'].mean().unstack()
    
    print("Average Throughput by Prompt Length:")
    print(throughput_by_length)
    
    # Visualize
    throughput_by_length.plot(kind='bar', figsize=(12, 6))
    plt.title('Throughput by Prompt Length')
    plt.ylabel('Tokens/Second')
    plt.xlabel('Endpoint')
    plt.legend(title='Prompt Length')
    plt.tight_layout()
    plt.savefig('throughput_analysis.png', dpi=300)
    print("Saved: throughput_analysis.png")

# Use it
analyze_ttft('results/benchmark_results_20251027_120000.csv')
analyze_throughput_by_length('results/benchmark_results_20251027_120000.csv')
```

### Statistical Significance Testing

```python
from scipy import stats

def compare_endpoints_statistically(csv_file, metric='total_latency'):
    """Perform statistical comparison between endpoints."""
    df = pd.read_csv(csv_file)
    df_success = df[df['success'] == True]
    
    endpoints = df_success['endpoint_name'].unique()
    
    if len(endpoints) != 2:
        print("This function requires exactly 2 endpoints")
        return
    
    endpoint1_data = df_success[df_success['endpoint_name'] == endpoints[0]][metric]
    endpoint2_data = df_success[df_success['endpoint_name'] == endpoints[1]][metric]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(endpoint1_data, endpoint2_data)
    
    print(f"Statistical Comparison: {metric}")
    print(f"  {endpoints[0]} mean: {endpoint1_data.mean():.4f}")
    print(f"  {endpoints[1]} mean: {endpoint2_data.mean():.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Difference is statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Difference is NOT statistically significant")

# Use it
compare_endpoints_statistically('results/benchmark_results_20251027_120000.csv', 'total_latency')
compare_endpoints_statistically('results/benchmark_results_20251027_120000.csv', 'time_to_first_token')
compare_endpoints_statistically('results/benchmark_results_20251027_120000.csv', 'tokens_per_second')
```

## Integrating with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/llm_benchmark.yml
name: LLM Performance Benchmark

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run benchmark
      run: |
        python run_benchmark.py --iterations 20 --output-dir results
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: results/
    
    - name: Check performance regression
      run: |
        python scripts/check_regression.py results/benchmark_summary_*.csv
```

### Performance Regression Detection

Create `scripts/check_regression.py`:

```python
import sys
import pandas as pd
import glob

def check_regression(current_results, threshold_increase=0.1):
    """Check if performance has regressed compared to baseline."""
    
    # Load current results
    current_df = pd.read_csv(current_results)
    
    # Load baseline (you'd store this somewhere)
    baseline_df = pd.read_csv('baseline/benchmark_summary.csv')
    
    regression_found = False
    
    for endpoint in current_df['endpoint_name'].unique():
        current_latency = current_df[current_df['endpoint_name'] == endpoint]['avg_total_latency'].values[0]
        baseline_latency = baseline_df[baseline_df['endpoint_name'] == endpoint]['avg_total_latency'].values[0]
        
        increase = (current_latency - baseline_latency) / baseline_latency
        
        print(f"{endpoint}:")
        print(f"  Baseline: {baseline_latency:.3f}s")
        print(f"  Current:  {current_latency:.3f}s")
        print(f"  Change:   {increase*100:+.1f}%")
        
        if increase > threshold_increase:
            print(f"  ⚠️  REGRESSION DETECTED! (>{threshold_increase*100}% increase)")
            regression_found = True
        else:
            print(f"  ✓ Within acceptable range")
        print()
    
    return 1 if regression_found else 0

if __name__ == "__main__":
    summary_file = glob.glob('results/benchmark_summary_*.csv')[0]
    exit_code = check_regression(summary_file)
    sys.exit(exit_code)
```

## Performance Tuning

### Optimizing for Different Scenarios

```python
# Optimized for accuracy (minimize variance)
accuracy_config = BenchmarkConfig(
    num_iterations=50,
    concurrent_requests=1,
    temperature=0.0,  # Deterministic
    stream=True
)

# Optimized for speed (quick validation)
speed_config = BenchmarkConfig(
    num_iterations=3,
    concurrent_requests=1,
    max_tokens=128,
    temperature=0.7,
    stream=True
)

# Optimized for load testing
load_config = BenchmarkConfig(
    num_iterations=20,
    concurrent_requests=20,
    timeout=300,
    max_tokens=512,
    stream=True
)
```

### Batching Tests for Efficiency

```python
async def run_batched_tests():
    """Run multiple test configurations efficiently."""
    
    configs = [
        ("quick", BenchmarkConfig(num_iterations=5, max_tokens=128)),
        ("standard", BenchmarkConfig(num_iterations=10, max_tokens=512)),
        ("thorough", BenchmarkConfig(num_iterations=20, max_tokens=1024))
    ]
    
    all_results = {}
    
    for name, config in configs:
        print(f"\nRunning {name} test...")
        benchmark = LLMBenchmark(config)
        results = await benchmark.run_benchmark(ENDPOINTS)
        all_results[name] = results
        
        # Save results
        writer = BenchmarkCSVWriter()
        writer.write_results(
            benchmark.get_results(),
            f"results/{name}"
        )
    
    return all_results

# Run all tests
asyncio.run(run_batched_tests())
```

### Memory-Efficient Processing

For very large benchmarks:

```python
async def run_memory_efficient_benchmark():
    """Process results in chunks to save memory."""
    
    config = BenchmarkConfig(num_iterations=100)
    benchmark = LLMBenchmark(config)
    
    # Process in chunks
    chunk_size = 10
    
    for i in range(0, config.num_iterations, chunk_size):
        chunk_config = BenchmarkConfig(
            num_iterations=chunk_size,
            concurrent_requests=config.concurrent_requests,
            timeout=config.timeout
        )
        
        chunk_benchmark = LLMBenchmark(chunk_config)
        chunk_results = await chunk_benchmark.run_benchmark(ENDPOINTS)
        
        # Save immediately
        writer = BenchmarkCSVWriter()
        writer.write_results(
            chunk_benchmark.get_results(),
            f"results/chunk_{i}"
        )
        
        # Clear memory
        del chunk_benchmark
        del chunk_results
```

## Tips and Tricks

### 1. Warmup Runs

Always run a few warmup requests before benchmarking:

```python
async def warmup(endpoint, num_requests=3):
    """Run warmup requests to stabilize performance."""
    print(f"Warming up {endpoint.name}...")
    warmup_config = BenchmarkConfig(num_iterations=num_requests)
    benchmark = LLMBenchmark(warmup_config)
    await benchmark.run_benchmark([endpoint], sequential=True)
    print(f"Warmup complete for {endpoint.name}")

# Use it
for endpoint in ENDPOINTS:
    asyncio.run(warmup(endpoint))
```

### 2. Monitoring During Benchmarks

```python
import psutil
import time

async def benchmark_with_monitoring():
    """Run benchmark while monitoring system resources."""
    
    start_cpu = psutil.cpu_percent(interval=1)
    start_memory = psutil.virtual_memory().percent
    
    config = BenchmarkConfig(num_iterations=10)
    benchmark = LLMBenchmark(config)
    
    start_time = time.time()
    results = await benchmark.run_benchmark(ENDPOINTS)
    duration = time.time() - start_time
    
    end_cpu = psutil.cpu_percent(interval=1)
    end_memory = psutil.virtual_memory().percent
    
    print(f"\nSystem Resource Usage:")
    print(f"  CPU: {start_cpu:.1f}% → {end_cpu:.1f}%")
    print(f"  Memory: {start_memory:.1f}% → {end_memory:.1f}%")
    print(f"  Duration: {duration:.2f}s")
    
    return results
```

### 3. Custom Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_debug.log'),
        logging.StreamHandler()
    ]
)

# Now your benchmarks will have detailed logs
```

---

For more information, see the main [README.md](README.md) or the [QUICKSTART.md](QUICKSTART.md) guide.
