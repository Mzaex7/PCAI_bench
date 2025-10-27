
#!/usr/bin/env python3
"""
Main script to run LLM benchmark suite.

This script orchestrates the complete benchmarking process:
1. Runs benchmarks on configured endpoints
2. Saves results to CSV
3. Generates visualization charts
"""

import asyncio
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ENDPOINTS, BenchmarkConfig
from benchmark import LLMBenchmark
from csv_writer import BenchmarkCSVWriter
from visualize import BenchmarkVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LLM Benchmark Suite - Compare performance of multiple LLM deployments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequential benchmark (endpoints tested in parallel, prompts sequential)
  python run_benchmark.py
  
  # Run concurrent benchmark with 5 parallel prompts per endpoint
  python run_benchmark.py --concurrent 5
  
  # Run 20 iterations with custom output directory
  python run_benchmark.py --iterations 20 --output-dir my_results
  
  # Run sequential mode explicitly
  python run_benchmark.py --mode sequential --iterations 5

Note: All endpoints are ALWAYS tested in parallel (simultaneously).
      The --mode flag controls whether prompts within each endpoint run
      sequentially (one at a time) or concurrently (multiple at once).
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sequential', 'concurrent'],
        default='sequential',
        help='Prompt execution mode: sequential (one prompt at a time per endpoint) or concurrent (multiple prompts in parallel per endpoint). Note: Endpoints are always tested in parallel.'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations to run for each prompt (default: 10)'
    )
    
    parser.add_argument(
        '--concurrent',
        type=int,
        default=1,
        help='Number of concurrent prompts per endpoint (only used in concurrent mode, default: 1)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results and visualizations (default: results)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization charts'
    )
    
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable streaming mode'
    )
    
    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        num_iterations=args.iterations,
        concurrent_requests=args.concurrent if args.mode == 'concurrent' else 1,
        timeout=args.timeout,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=not args.no_stream
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LLM BENCHMARK SUITE                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration:
  Prompt Mode:        {args.mode}
  Endpoint Mode:      Parallel (all endpoints tested simultaneously)
  Iterations:         {config.num_iterations}
  Concurrent Prompts: {config.concurrent_requests}
  Timeout:            {config.timeout}s
  Max Tokens:         {config.max_tokens}
  Temperature:        {config.temperature}
  Streaming:          {config.stream}
  Output Directory:   {config.output_dir}
  
Endpoints to test:  {len(ENDPOINTS)} (in parallel)
""")
    
    for idx, endpoint in enumerate(ENDPOINTS, 1):
        print(f"  {idx}. {endpoint.name}")
        print(f"     Model: {endpoint.model_name}")
        print(f"     URL: {endpoint.url}")
        print()
    
    input("Press Enter to start benchmark...")
    
    # Initialize benchmark
    benchmark = LLMBenchmark(config)
    
    # Run benchmark
    try:
        sequential = (args.mode == 'sequential')
        results = await benchmark.run_benchmark(ENDPOINTS, sequential=sequential)
        
        if not results:
            print("\n⚠ No results generated. Exiting.")
            return 1
        
        # Get results as dictionaries
        results_dict = benchmark.get_results()
        
        # Write results to CSV
        print(f"\n{'='*80}")
        print("Saving Results")
        print(f"{'='*80}\n")
        
        csv_writer = BenchmarkCSVWriter()
        results_file = csv_writer.write_results(results_dict, config.output_dir)
        summary_file = csv_writer.write_summary(results_dict, config.output_dir)
        
        # Generate visualizations
        if not args.no_visualizations:
            visualizer = BenchmarkVisualizer(results_dict, 
                                            output_dir=os.path.join(config.output_dir, 'visualizations'))
            viz_files = visualizer.generate_all_visualizations()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE!")
        print(f"{'='*80}\n")
        
        successful = sum(1 for r in results_dict if r['success'])
        failed = len(results_dict) - successful
        
        print(f"Total Requests:     {len(results_dict)}")
        print(f"Successful:         {successful} ({100*successful/len(results_dict):.1f}%)")
        print(f"Failed:             {failed}")
        print(f"\nResults saved to:   {results_file}")
        print(f"Summary saved to:   {summary_file}")
        
        if not args.no_visualizations:
            print(f"Visualizations:     {os.path.join(config.output_dir, 'visualizations')}/")
        
        print("\n✓ Benchmark completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
