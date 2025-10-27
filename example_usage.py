
#!/usr/bin/env python3
"""
Example usage script demonstrating how to use the benchmark suite programmatically.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ENDPOINTS, BenchmarkConfig
from benchmark import LLMBenchmark
from csv_writer import BenchmarkCSVWriter
from visualize import BenchmarkVisualizer


async def run_simple_benchmark():
    """Run a simple benchmark with minimal configuration."""
    
    print("="*80)
    print("Simple Benchmark Example")
    print("="*80)
    
    # Create a minimal configuration
    config = BenchmarkConfig(
        num_iterations=2,  # Just 2 iterations for quick testing
        concurrent_requests=1,
        timeout=60,
        max_tokens=256,
        temperature=0.7,
        stream=True
    )
    
    # Initialize benchmark
    benchmark = LLMBenchmark(config)
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = await benchmark.run_benchmark(ENDPOINTS, sequential=True)
    
    # Get results
    results_dict = benchmark.get_results()
    
    # Print quick summary
    print("\n" + "="*80)
    print("Quick Summary")
    print("="*80)
    
    for endpoint_name in set(r['endpoint_name'] for r in results_dict):
        endpoint_results = [r for r in results_dict if r['endpoint_name'] == endpoint_name]
        successful = [r for r in endpoint_results if r['success']]
        
        if successful:
            avg_latency = sum(r['total_latency'] for r in successful) / len(successful)
            avg_ttft = sum(r['time_to_first_token'] for r in successful if r['time_to_first_token']) / len(successful)
            avg_throughput = sum(r['tokens_per_second'] for r in successful if r['tokens_per_second']) / len(successful)
            
            print(f"\n{endpoint_name}:")
            print(f"  Success Rate: {len(successful)}/{len(endpoint_results)} ({100*len(successful)/len(endpoint_results):.1f}%)")
            print(f"  Avg Latency:  {avg_latency:.3f}s")
            print(f"  Avg TTFT:     {avg_ttft:.3f}s")
            print(f"  Avg Throughput: {avg_throughput:.2f} tokens/s")
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    csv_writer = BenchmarkCSVWriter()
    results_file = csv_writer.write_results(results_dict, "results")
    summary_file = csv_writer.write_summary(results_dict, "results")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    visualizer = BenchmarkVisualizer(results_dict, output_dir="results/visualizations")
    visualizer.generate_all_visualizations()
    
    print("\n✓ Example benchmark completed!")


if __name__ == "__main__":
    try:
        asyncio.run(run_simple_benchmark())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
