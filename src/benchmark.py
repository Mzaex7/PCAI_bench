
"""
Main benchmarking script for LLM endpoint comparison.
Measures TTFT, TPOT, latency, throughput, and other performance metrics.
"""

import time
import asyncio
import aiohttp
import json
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import urllib3

from config import EndpointConfig, BenchmarkConfig
from prompts import get_test_prompts

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


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
        total_requests = self.config.num_iterations * len(prompts)
        completed = 0
        
        async with aiohttp.ClientSession() as session:
            for iteration in range(self.config.num_iterations):
                for prompt_idx, prompt in enumerate(prompts):
                    result = await self._make_streaming_request(
                        session, endpoint, prompt, iteration, concurrent_level=1
                    )
                    results.append(result)
                    completed += 1
                    
                    # Print progress every 10% or at key milestones
                    if completed % max(1, total_requests // 10) == 0 or completed == total_requests:
                        print(f"  [{endpoint.name}] Progress: {completed}/{total_requests} requests ({100*completed/total_requests:.0f}%)")
        
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
        total_iterations = self.config.num_iterations
        
        async with aiohttp.ClientSession() as session:
            for iteration in range(total_iterations):
                # Create tasks for concurrent execution
                tasks = []
                for prompt in prompts:
                    task = self._make_streaming_request(
                        session, endpoint, prompt, iteration, 
                        concurrent_level=self.config.concurrent_requests
                    )
                    tasks.append(task)
                
                # Run concurrent requests in batches
                batch_count = 0
                for i in range(0, len(tasks), self.config.concurrent_requests):
                    batch = tasks[i:i + self.config.concurrent_requests]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    batch_count += 1
                
                # Print progress after each iteration
                completed = iteration + 1
                print(f"  [{endpoint.name}] Progress: {completed}/{total_iterations} iterations ({100*completed/total_iterations:.0f}%) - {len(results)} requests completed")
        
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
        print(f"[{endpoint.name}] Starting benchmark...")
        print(f"  Model: {endpoint.model_name}")
        print(f"  URL: {endpoint.url}\n")
        
        if sequential or self.config.concurrent_requests == 1:
            results = await self._run_sequential_benchmark(endpoint, prompts)
        else:
            results = await self._run_concurrent_benchmark(endpoint, prompts)
        
        # Print summary for this endpoint
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        avg_latency = sum(r.total_latency for r in results if r.success) / successful if successful > 0 else 0
        avg_ttft = sum(r.time_to_first_token for r in results if r.success and r.time_to_first_token) / successful if successful > 0 else 0
        avg_throughput = sum(r.tokens_per_second for r in results if r.success and r.tokens_per_second) / successful if successful > 0 else 0
        
        print(f"\n[{endpoint.name}] ✓ Completed!")
        print(f"    Success: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        print(f"    Failed: {failed}")
        print(f"    Avg Total Latency: {avg_latency:.3f}s")
        print(f"    Avg TTFT: {avg_ttft:.3f}s")
        print(f"    Avg Throughput: {avg_throughput:.2f} tokens/s")
        print()
        
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
        print(f"\n{'='*80}")
        print(f"Starting {'Sequential' if sequential else 'Concurrent'} Benchmark")
        print(f"{'='*80}\n")
        print(f"Configuration:")
        print(f"  - Prompt mode: {'Sequential' if sequential else f'Concurrent (batch size: {self.config.concurrent_requests})'}")
        print(f"  - Endpoint mode: Parallel (all endpoints tested simultaneously)")
        print(f"  - Iterations: {self.config.num_iterations}")
        print(f"  - Timeout: {self.config.timeout}s")
        print(f"  - Max tokens: {self.config.max_tokens}")
        print(f"  - Temperature: {self.config.temperature}")
        print(f"  - Streaming: {self.config.stream}")
        print()
        
        prompts = get_test_prompts()
        print(f"Loaded {len(prompts)} test prompts")
        print(f"Testing {len(endpoints)} endpoints in parallel\n")
        
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
        return all_results
    
    def get_results(self) -> List[Dict]:
        """Get results as list of dictionaries."""
        return [asdict(result) for result in self.results]
