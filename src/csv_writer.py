
"""CSV Export for Benchmark Results.

Provides functionality to export benchmark results to CSV format with both
detailed per-request metrics and aggregated summary statistics.

Typical usage:
    writer = BenchmarkCSVWriter()
    filepath = writer.write_results(results, output_dir="results")
    summary_path = writer.write_summary(results, output_dir="results")
"""

import csv
import os
from typing import List, Dict
from datetime import datetime


class BenchmarkCSVWriter:
    """Handles CSV export of benchmark results.
    
    This class provides methods to export benchmark data in two formats:
    - Detailed: Individual metrics for each request
    - Summary: Aggregated statistics per endpoint
    """
    
    @staticmethod
    def write_results(results: List[Dict], output_dir: str = "results") -> str:
        """Export detailed benchmark results to CSV.
        
        Creates a timestamped CSV file containing individual metrics for each
        benchmark request including latency, throughput, token counts, and
        inter-token latency distributions.
        
        Args:
            results: List of benchmark result dictionaries.
            output_dir: Target directory for CSV file (default: "results").
            
        Returns:
            Absolute path to the created CSV file.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not results:
            print("No results to write")
            return filepath
        
        fieldnames = [
            'endpoint_name',
            'model_name',
            'prompt_text',
            'prompt_length',
            'prompt_complexity',
            'prompt_category',
            'success',
            'error_message',
            'total_latency',
            'time_to_first_token',
            'input_tokens',
            'output_tokens',
            'total_tokens',
            'tokens_per_second',
            'time_per_output_token',
            'avg_inter_token_latency',
            'min_inter_token_latency',
            'max_inter_token_latency',
            'p50_inter_token_latency',
            'p95_inter_token_latency',
            'p99_inter_token_latency',
            'response_text',
            'timestamp',
            'iteration',
            'concurrent_level'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\n✓ Results written to: {filepath}")
        return filepath
    
    @staticmethod
    def write_summary(results: List[Dict], output_dir: str = "results") -> str:
        """Export aggregated summary statistics to CSV.
        
        Creates a timestamped CSV file with endpoint-level statistics including
        success rates, average metrics, and percentile distributions.
        
        Args:
            results: List of benchmark result dictionaries.
            output_dir: Target directory for CSV file (default: "results").
            
        Returns:
            Absolute path to the created summary CSV file.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_summary_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not results:
            print("No results to summarize")
            return filepath
        
        endpoint_stats = {}
        
        for result in results:
            endpoint = result['endpoint_name']
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'total_latencies': [],
                    'ttfts': [],
                    'tpots': [],
                    'throughputs': [],
                    'input_tokens': [],
                    'output_tokens': [],
                    'avg_inter_token_latencies': [],
                    'min_inter_token_latencies': [],
                    'max_inter_token_latencies': [],
                    'p50_inter_token_latencies': [],
                    'p95_inter_token_latencies': [],
                    'p99_inter_token_latencies': [],
                }
            
            stats = endpoint_stats[endpoint]
            stats['total_requests'] += 1
            
            if result['success']:
                stats['successful_requests'] += 1
                stats['total_latencies'].append(result['total_latency'])
                
                if result['time_to_first_token']:
                    stats['ttfts'].append(result['time_to_first_token'])
                if result['time_per_output_token']:
                    stats['tpots'].append(result['time_per_output_token'])
                if result['tokens_per_second']:
                    stats['throughputs'].append(result['tokens_per_second'])
                if result['input_tokens']:
                    stats['input_tokens'].append(result['input_tokens'])
                if result['output_tokens']:
                    stats['output_tokens'].append(result['output_tokens'])
                
                if result.get('avg_inter_token_latency'):
                    stats['avg_inter_token_latencies'].append(result['avg_inter_token_latency'])
                if result.get('min_inter_token_latency'):
                    stats['min_inter_token_latencies'].append(result['min_inter_token_latency'])
                if result.get('max_inter_token_latency'):
                    stats['max_inter_token_latencies'].append(result['max_inter_token_latency'])
                if result.get('p50_inter_token_latency'):
                    stats['p50_inter_token_latencies'].append(result['p50_inter_token_latency'])
                if result.get('p95_inter_token_latency'):
                    stats['p95_inter_token_latencies'].append(result['p95_inter_token_latency'])
                if result.get('p99_inter_token_latency'):
                    stats['p99_inter_token_latencies'].append(result['p99_inter_token_latency'])
            else:
                stats['failed_requests'] += 1
        
        summary_data = []
        
        for endpoint, stats in endpoint_stats.items():
            summary = {
                'endpoint_name': endpoint,
                'total_requests': stats['total_requests'],
                'successful_requests': stats['successful_requests'],
                'failed_requests': stats['failed_requests'],
                'success_rate': f"{100 * stats['successful_requests'] / stats['total_requests']:.2f}%",
            }
            
            if stats['total_latencies']:
                summary['avg_total_latency'] = sum(stats['total_latencies']) / len(stats['total_latencies'])
                summary['min_total_latency'] = min(stats['total_latencies'])
                summary['max_total_latency'] = max(stats['total_latencies'])
            
            if stats['ttfts']:
                summary['avg_ttft'] = sum(stats['ttfts']) / len(stats['ttfts'])
                summary['min_ttft'] = min(stats['ttfts'])
                summary['max_ttft'] = max(stats['ttfts'])
            
            if stats['tpots']:
                summary['avg_tpot'] = sum(stats['tpots']) / len(stats['tpots'])
                summary['min_tpot'] = min(stats['tpots'])
                summary['max_tpot'] = max(stats['tpots'])
            
            if stats['throughputs']:
                summary['avg_throughput'] = sum(stats['throughputs']) / len(stats['throughputs'])
                summary['min_throughput'] = min(stats['throughputs'])
                summary['max_throughput'] = max(stats['throughputs'])
            
            if stats['input_tokens']:
                summary['avg_input_tokens'] = sum(stats['input_tokens']) / len(stats['input_tokens'])
            
            if stats['output_tokens']:
                summary['avg_output_tokens'] = sum(stats['output_tokens']) / len(stats['output_tokens'])
            
            if stats['avg_inter_token_latencies']:
                summary['avg_inter_token_latency'] = sum(stats['avg_inter_token_latencies']) / len(stats['avg_inter_token_latencies'])
            
            if stats['min_inter_token_latencies']:
                summary['min_inter_token_latency'] = min(stats['min_inter_token_latencies'])
            
            if stats['max_inter_token_latencies']:
                summary['max_inter_token_latency'] = max(stats['max_inter_token_latencies'])
            
            if stats['p50_inter_token_latencies']:
                summary['avg_p50_inter_token_latency'] = sum(stats['p50_inter_token_latencies']) / len(stats['p50_inter_token_latencies'])
            
            if stats['p95_inter_token_latencies']:
                summary['avg_p95_inter_token_latency'] = sum(stats['p95_inter_token_latencies']) / len(stats['p95_inter_token_latencies'])
            
            if stats['p99_inter_token_latencies']:
                summary['avg_p99_inter_token_latency'] = sum(stats['p99_inter_token_latencies']) / len(stats['p99_inter_token_latencies'])
            
            summary_data.append(summary)
        
        if summary_data:
            fieldnames = list(summary_data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"\u2713 Summary written to: {filepath}")
        
        return filepath
