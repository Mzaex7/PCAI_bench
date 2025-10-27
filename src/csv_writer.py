
"""
CSV writer for benchmark results.
"""

import csv
import os
from typing import List, Dict
from datetime import datetime


class BenchmarkCSVWriter:
    """Write benchmark results to CSV files."""
    
    @staticmethod
    def write_results(results: List[Dict], output_dir: str = "results") -> str:
        """
        Write benchmark results to a CSV file.
        
        Args:
            results: List of benchmark result dictionaries
            output_dir: Directory to save CSV file
            
        Returns:
            Path to the created CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not results:
            print("No results to write")
            return filepath
        
        # Define CSV columns
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
            'response_text',
            'timestamp',
            'iteration',
            'concurrent_level'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\n✓ Results written to: {filepath}")
        return filepath
    
    @staticmethod
    def write_summary(results: List[Dict], output_dir: str = "results") -> str:
        """
        Write a summary CSV with aggregated statistics per endpoint.
        
        Args:
            results: List of benchmark result dictionaries
            output_dir: Directory to save CSV file
            
        Returns:
            Path to the created summary CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_summary_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not results:
            print("No results to summarize")
            return filepath
        
        # Group results by endpoint
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
            else:
                stats['failed_requests'] += 1
        
        # Calculate summary statistics
        summary_data = []
        
        for endpoint, stats in endpoint_stats.items():
            summary = {
                'endpoint_name': endpoint,
                'total_requests': stats['total_requests'],
                'successful_requests': stats['successful_requests'],
                'failed_requests': stats['failed_requests'],
                'success_rate': f"{100 * stats['successful_requests'] / stats['total_requests']:.2f}%",
            }
            
            # Calculate averages
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
            
            summary_data.append(summary)
        
        # Write summary CSV
        if summary_data:
            fieldnames = list(summary_data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"✓ Summary written to: {filepath}")
        
        return filepath
