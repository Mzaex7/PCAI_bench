
"""
Main benchmarking script for LLM endpoint comparison.
Measures TTFT, TPOT, latency, throughput, and other performance metrics.
"""

import time
import asyncio
import aiohttp
import json
import warnings
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import urllib3

from config import EndpointConfig, BenchmarkConfig
from prompts import get_test_prompts

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


class ProgressTracker:
    """Enhanced progress tracking with rich terminal visualization."""
    
    def __init__(self, endpoints: List[str], total_requests_per_endpoint: int):
        """Initialize progress tracker."""
        self.endpoints = {name: {
            'completed': 0,
            'total': total_requests_per_endpoint,
            'success': 0,
            'failed': 0,
            'start_time': None,
            'latencies': [],
            'ttfts': [],
            'throughputs': [],
            'status': 'pending'  # pending, running, completed, error
        } for name in endpoints}
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.5  # Update display every 0.5 seconds
        self.first_display = True  # Track if this is the first display
        
    def _get_progress_bar(self, completed: int, total: int, width: int = 30) -> str:
        """Generate a visual progress bar."""
        if total == 0:
            return f"[{'█' * width}]"
        
        filled = int(width * completed / total)
        bar = '█' * filled + '░' * (width - filled)
        percentage = 100 * completed / total
        return f"[{bar}] {percentage:5.1f}%"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _calculate_eta(self, completed: int, total: int, elapsed: float) -> str:
        """Calculate estimated time of arrival."""
        if completed == 0 or completed == total:
            return "N/A"
        
        rate = completed / elapsed
        remaining = total - completed
        eta_seconds = remaining / rate if rate > 0 else 0
        return self._format_time(eta_seconds)
    
    def _get_status_symbol(self, status: str) -> str:
        """Get status symbol with color."""
        symbols = {
            'pending': '⏸',
            'running': '▶',
            'completed': '✓',
            'error': '✗'
        }
        return symbols.get(status, '?')
    
    def start_endpoint(self, endpoint_name: str):
        """Mark endpoint as started."""
        if endpoint_name in self.endpoints:
            self.endpoints[endpoint_name]['start_time'] = time.time()
            self.endpoints[endpoint_name]['status'] = 'running'
    
    def update_progress(self, endpoint_name: str, success: bool, 
                       latency: Optional[float] = None,
                       ttft: Optional[float] = None,
                       throughput: Optional[float] = None):
        """Update progress for an endpoint."""
        if endpoint_name not in self.endpoints:
            return
        
        ep = self.endpoints[endpoint_name]
        ep['completed'] += 1
        
        if success:
            ep['success'] += 1
            if latency:
                ep['latencies'].append(latency)
            if ttft:
                ep['ttfts'].append(ttft)
            if throughput:
                ep['throughputs'].append(throughput)
        else:
            ep['failed'] += 1
        
        if ep['completed'] >= ep['total']:
            ep['status'] = 'completed'
        
        # Throttle display updates
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval or ep['completed'] == ep['total']:
            self.display()
            self.last_update = current_time
    
    def display(self):
        """Display the current progress with rich formatting."""
        # Calculate total lines to display
        # 1 separator + 1 title + 1 separator + 1 blank + 
        # 1 overall + 1 success/fail + 1 blank + 1 header + 1 separator + 
        # N endpoints + 1 separator + 1 blank = 10 + N
        total_lines = 10 + len(self.endpoints)
        
        # Clear previous display
        if not self.first_display:
            # Move cursor up to the beginning of previous output
            sys.stdout.write(f'\033[{total_lines}A')
            # Clear from cursor to end of screen
            sys.stdout.write('\033[J')
        
        self.first_display = False
        
        elapsed = time.time() - self.start_time
        
        print("=" * 100)
        print(f"  🚀 LLM BENCHMARK PROGRESS  |  ⏱  Elapsed: {self._format_time(elapsed)}")
        print("=" * 100)
        
        # Calculate overall stats
        total_completed = sum(ep['completed'] for ep in self.endpoints.values())
        total_requests = sum(ep['total'] for ep in self.endpoints.values())
        total_success = sum(ep['success'] for ep in self.endpoints.values())
        total_failed = sum(ep['failed'] for ep in self.endpoints.values())
        
        # Overall progress bar
        overall_bar = self._get_progress_bar(total_completed, total_requests, width=40)
        print(f"\n  Overall: {overall_bar}  ({total_completed}/{total_requests} requests)")
        print(f"  Success: {total_success} ✓  |  Failed: {total_failed} ✗")
        print()
        
        # Header for endpoint table
        print(f"{'Endpoint':<20} {'Status':<8} {'Progress':<35} {'Success':<12} {'Avg TTFT':<12} {'Avg Thr.':<12} {'ETA':<10}")
        print("-" * 100)
        
        # Display each endpoint
        for name, ep in self.endpoints.items():
            status = self._get_status_symbol(ep['status'])
            progress_bar = self._get_progress_bar(ep['completed'], ep['total'], width=20)
            
            # Calculate averages
            avg_ttft = sum(ep['ttfts']) / len(ep['ttfts']) if ep['ttfts'] else 0
            avg_thr = sum(ep['throughputs']) / len(ep['throughputs']) if ep['throughputs'] else 0
            
            success_rate = f"{ep['success']}/{ep['completed']}"
            ttft_str = f"{avg_ttft:.3f}s" if avg_ttft > 0 else "N/A"
            thr_str = f"{avg_thr:.1f} t/s" if avg_thr > 0 else "N/A"
            
            # Calculate ETA for this endpoint
            ep_elapsed = time.time() - ep['start_time'] if ep['start_time'] else elapsed
            eta = self._calculate_eta(ep['completed'], ep['total'], ep_elapsed)
            
            # Truncate endpoint name if too long
            display_name = name[:18] + '..' if len(name) > 20 else name
            
            print(f"{display_name:<20} {status:<8} {progress_bar:<35} {success_rate:<12} {ttft_str:<12} {thr_str:<12} {eta:<10}")
        
        print("=" * 100)
        print()  # Extra line for clarity
        
        sys.stdout.flush()
    
    def final_summary(self):
        """Display final summary statistics."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 100)
        print("  🎉 BENCHMARK COMPLETED!")
        print("=" * 100)
        print(f"\n  Total Duration: {self._format_time(elapsed)}\n")
        
        print(f"{'Endpoint':<20} {'Requests':<12} {'Success':<10} {'Avg Latency':<15} {'Avg TTFT':<12} {'Avg Throughput':<15}")
        print("-" * 100)
        
        for name, ep in self.endpoints.items():
            display_name = name[:18] + '..' if len(name) > 20 else name
            
            avg_latency = sum(ep['latencies']) / len(ep['latencies']) if ep['latencies'] else 0
            avg_ttft = sum(ep['ttfts']) / len(ep['ttfts']) if ep['ttfts'] else 0
            avg_thr = sum(ep['throughputs']) / len(ep['throughputs']) if ep['throughputs'] else 0
            
            success_pct = f"{100*ep['success']/ep['total']:.1f}%" if ep['total'] > 0 else "N/A"
            
            print(f"{display_name:<20} {ep['total']:<12} {success_pct:<10} "
                  f"{avg_latency:.3f}s{'':<8} {avg_ttft:.3f}s{'':<5} {avg_thr:.1f} tokens/s")
        
        print("=" * 100 + "\n")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    endpoint_name: str
    model_name: str
    prompt_text: str
    prompt_length: str
    prompt_complexity: str
    prompt_category: str
    success: bool
    error_message: Optional[str]
    
    # Timing metrics
    total_latency: float  # Total time from request to complete response
    time_to_first_token: Optional[float]  # TTFT - time until streaming starts
    
    # Token metrics
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    
    # Throughput metrics
    tokens_per_second: Optional[float]  # Overall throughput
    time_per_output_token: Optional[float]  # TPOT - average time per token
    
    # Inter-token latency metrics
    inter_token_latencies: Optional[List[float]]  # List of latencies between consecutive tokens
    avg_inter_token_latency: Optional[float]  # Average inter-token latency
    min_inter_token_latency: Optional[float]  # Minimum inter-token latency
    max_inter_token_latency: Optional[float]  # Maximum inter-token latency
    p50_inter_token_latency: Optional[float]  # 50th percentile (median)
    p95_inter_token_latency: Optional[float]  # 95th percentile
    p99_inter_token_latency: Optional[float]  # 99th percentile
    
    # Response
    response_text: Optional[str]
    
    # Metadata
    timestamp: str
    iteration: int
    concurrent_level: int


class LLMBenchmark:
    """LLM endpoint benchmarking utility."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmarking utility."""
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.progress_tracker: Optional[ProgressTracker] = None
    
    async def _make_streaming_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: EndpointConfig,
        prompt: Dict,
        iteration: int,
        concurrent_level: int
    ) -> BenchmarkResult:
        """
        Make a streaming request to an LLM endpoint and measure performance.
        
        Args:
            session: aiohttp session
            endpoint: Endpoint configuration
            prompt: Prompt dictionary with text and metadata
            iteration: Current iteration number
            concurrent_level: Number of concurrent requests
            
        Returns:
            BenchmarkResult with all metrics
        """
        headers = {
            "Authorization": f"Bearer {endpoint.auth_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model_name,
            "messages": [
                {"role": "user", "content": prompt["prompt"]}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream
        }
        
        start_time = time.time()
        time_to_first_token = None
        output_tokens = 0
        response_text = ""
        input_tokens = None
        total_tokens = None
        error_message = None
        success = False
        
        # Inter-token latency tracking
        token_timestamps: List[float] = []
        last_token_time = None
        
        try:
            async with session.post(
                endpoint.url,
                headers=headers,
                json=payload,
                ssl=False,  # verify=False
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                
                if self.config.stream:
                    # Handle streaming response
                    first_token_received = False
                    
                    async for line in response.content:
                        if not line:
                            continue
                        
                        line = line.decode('utf-8').strip()
                        if not line.startswith('data: '):
                            continue
                        
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Record TTFT on first token
                            if not first_token_received:
                                time_to_first_token = time.time() - start_time
                                first_token_received = True
                            
                            # Extract token content
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    # Record token timestamp for inter-token latency
                                    current_time = time.time()
                                    token_timestamps.append(current_time)
                                    
                                    response_text += content
                                    output_tokens += 1
                                
                                # Extract usage information if available
                                if 'usage' in data:
                                    usage = data['usage']
                                    input_tokens = usage.get('prompt_tokens')
                                    total_tokens = usage.get('total_tokens')
                        
                        except json.JSONDecodeError:
                            continue
                else:
                    # Handle non-streaming response
                    data = await response.json()
                    time_to_first_token = time.time() - start_time
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        response_text = data['choices'][0].get('message', {}).get('content', '')
                    
                    if 'usage' in data:
                        usage = data['usage']
                        input_tokens = usage.get('prompt_tokens')
                        output_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens')
                
                success = True
        
        except asyncio.TimeoutError:
            error_message = "Request timeout"
        except aiohttp.ClientError as e:
            error_message = f"Client error: {str(e)}"
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
        
        # Calculate final metrics
        total_latency = time.time() - start_time
        
        # Calculate throughput metrics
        tokens_per_second = None
        time_per_output_token = None
        
        if output_tokens > 0 and total_latency > 0:
            tokens_per_second = output_tokens / total_latency
            time_per_output_token = total_latency / output_tokens
        
        # Calculate inter-token latencies
        inter_token_latencies = None
        avg_inter_token_latency = None
        min_inter_token_latency = None
        max_inter_token_latency = None
        p50_inter_token_latency = None
        p95_inter_token_latency = None
        p99_inter_token_latency = None
        
        if len(token_timestamps) > 1:
            # Calculate time differences between consecutive tokens
            inter_token_latencies = [
                token_timestamps[i] - token_timestamps[i-1] 
                for i in range(1, len(token_timestamps))
            ]
            
            if inter_token_latencies:
                avg_inter_token_latency = sum(inter_token_latencies) / len(inter_token_latencies)
                min_inter_token_latency = min(inter_token_latencies)
                max_inter_token_latency = max(inter_token_latencies)
                
                # Calculate percentiles
                sorted_latencies = sorted(inter_token_latencies)
                n = len(sorted_latencies)
                
                p50_idx = int(n * 0.50)
                p95_idx = int(n * 0.95)
                p99_idx = int(n * 0.99)
                
                p50_inter_token_latency = sorted_latencies[p50_idx] if p50_idx < n else sorted_latencies[-1]
                p95_inter_token_latency = sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1]
                p99_inter_token_latency = sorted_latencies[p99_idx] if p99_idx < n else sorted_latencies[-1]
        
        # If we couldn't get input tokens from usage, estimate from prompt
        if input_tokens is None:
            # Rough estimation: ~4 characters per token
            input_tokens = len(prompt["prompt"]) // 4
        
        if total_tokens is None and input_tokens and output_tokens:
            total_tokens = input_tokens + output_tokens
        
        return BenchmarkResult(
            endpoint_name=endpoint.name,
            model_name=endpoint.model_name,
            prompt_text=prompt["prompt"][:100] + "..." if len(prompt["prompt"]) > 100 else prompt["prompt"],
            prompt_length=prompt["length"],
            prompt_complexity=prompt["complexity"],
            prompt_category=prompt["category"],
            success=success,
            error_message=error_message,
            total_latency=total_latency,
            time_to_first_token=time_to_first_token,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            time_per_output_token=time_per_output_token,
            inter_token_latencies=inter_token_latencies,
            avg_inter_token_latency=avg_inter_token_latency,
            min_inter_token_latency=min_inter_token_latency,
            max_inter_token_latency=max_inter_token_latency,
            p50_inter_token_latency=p50_inter_token_latency,
            p95_inter_token_latency=p95_inter_token_latency,
            p99_inter_token_latency=p99_inter_token_latency,
            response_text=response_text[:200] + "..." if response_text and len(response_text) > 200 else response_text,
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
            concurrent_level=concurrent_level
        )
    
    async def _run_sequential_benchmark(
        self,
        endpoint: EndpointConfig,
        prompts: List[Dict]
    ) -> List[BenchmarkResult]:
        """
        Run sequential benchmark (one request at a time).
        
        Args:
            endpoint: Endpoint to test
            prompts: List of prompts to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        async with aiohttp.ClientSession() as session:
            for iteration in range(self.config.num_iterations):
                for prompt_idx, prompt in enumerate(prompts):
                    result = await self._make_streaming_request(
                        session, endpoint, prompt, iteration, concurrent_level=1
                    )
                    results.append(result)
                    
                    # Update progress tracker
                    if self.progress_tracker:
                        self.progress_tracker.update_progress(
                            endpoint.name,
                            result.success,
                            result.total_latency if result.success else None,
                            result.time_to_first_token if result.success else None,
                            result.tokens_per_second if result.success else None
                        )
        
        return results
    
    async def _run_concurrent_benchmark(
        self,
        endpoint: EndpointConfig,
        prompts: List[Dict]
    ) -> List[BenchmarkResult]:
        """
        Run concurrent benchmark (multiple requests simultaneously).
        
        Args:
            endpoint: Endpoint to test
            prompts: List of prompts to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        async with aiohttp.ClientSession() as session:
            for iteration in range(self.config.num_iterations):
                # Create tasks for concurrent execution
                tasks = []
                for prompt in prompts:
                    task = self._make_streaming_request(
                        session, endpoint, prompt, iteration, 
                        concurrent_level=self.config.concurrent_requests
                    )
                    tasks.append(task)
                
                # Run concurrent requests in batches
                for i in range(0, len(tasks), self.config.concurrent_requests):
                    batch = tasks[i:i + self.config.concurrent_requests]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    
                    # Update progress tracker for each result in batch
                    if self.progress_tracker:
                        for result in batch_results:
                            self.progress_tracker.update_progress(
                                endpoint.name,
                                result.success,
                                result.total_latency if result.success else None,
                                result.time_to_first_token if result.success else None,
                                result.tokens_per_second if result.success else None
                            )
        
        return results
    
    async def _benchmark_single_endpoint(
        self,
        endpoint: EndpointConfig,
        prompts: List[Dict],
        sequential: bool
    ) -> Tuple[EndpointConfig, List[BenchmarkResult]]:
        """
        Benchmark a single endpoint (helper for parallel execution).
        
        Args:
            endpoint: Endpoint to test
            prompts: List of prompts
            sequential: Whether to run prompts sequentially
            
        Returns:
            Tuple of (endpoint, results)
        """
        # Mark endpoint as started in progress tracker
        if self.progress_tracker:
            self.progress_tracker.start_endpoint(endpoint.name)
        
        if sequential or self.config.concurrent_requests == 1:
            results = await self._run_sequential_benchmark(endpoint, prompts)
        else:
            results = await self._run_concurrent_benchmark(endpoint, prompts)
        
        return endpoint, results
    
    async def run_benchmark(
        self,
        endpoints: List[EndpointConfig],
        sequential: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite with parallel endpoint testing.
        
        Args:
            endpoints: List of endpoints to test (will be tested in parallel)
            sequential: If True, run prompts sequentially within each endpoint;
                       if False, run prompts concurrently within each endpoint
            
        Returns:
            List of all benchmark results
        """
        print(f"\n{'='*100}")
        print(f"  🚀 LLM ENDPOINT BENCHMARK")
        print(f"{'='*100}\n")
        print(f"Configuration:")
        print(f"  • Prompt mode: {'Sequential' if sequential else f'Concurrent (batch size: {self.config.concurrent_requests})'}")
        print(f"  • Endpoint mode: Parallel (all endpoints tested simultaneously)")
        print(f"  • Iterations per prompt: {self.config.num_iterations}")
        print(f"  • Request timeout: {self.config.timeout}s")
        print(f"  • Max tokens: {self.config.max_tokens}")
        print(f"  • Temperature: {self.config.temperature}")
        print(f"  • Streaming: {'Enabled' if self.config.stream else 'Disabled'}")
        print()
        
        prompts = get_test_prompts()
        total_requests_per_endpoint = len(prompts) * self.config.num_iterations
        
        print(f"Test Suite:")
        print(f"  • Prompts: {len(prompts)} (across {len(set(p['length'] for p in prompts))} lengths, {len(set(p['complexity'] for p in prompts))} complexity levels)")
        print(f"  • Total requests per endpoint: {total_requests_per_endpoint}")
        print(f"  • Endpoints: {len(endpoints)}")
        print()
        
        # Print endpoint details
        print("Endpoints:")
        for i, ep in enumerate(endpoints, 1):
            print(f"  {i}. {ep.name} - {ep.model_name}")
        print("\n")  # Two newlines before progress starts
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            [ep.name for ep in endpoints],
            total_requests_per_endpoint
        )
        
        # Run all endpoints in parallel
        tasks = [
            self._benchmark_single_endpoint(endpoint, prompts, sequential)
            for endpoint in endpoints
        ]
        
        # Gather results from all endpoints
        endpoint_results = await asyncio.gather(*tasks)
        
        # Combine all results
        all_results = []
        for endpoint, results in endpoint_results:
            all_results.extend(results)
        
        self.results = all_results
        
        # Display final summary
        if self.progress_tracker:
            self.progress_tracker.final_summary()
        
        return all_results
    
    def get_results(self) -> List[Dict]:
        """Get results as list of dictionaries."""
        return [asdict(result) for result in self.results]
