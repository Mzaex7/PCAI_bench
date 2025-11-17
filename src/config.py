
"""Configuration Management for LLM Benchmarking.

This module centralizes all configuration for the benchmarking suite including:
- Endpoint configurations with authentication
- Benchmark execution parameters
- Model registry for easy selection
- Utility functions for model management

Typical usage:
    from config import BenchmarkConfig, get_models_by_names
    
    config = BenchmarkConfig(num_iterations=10, concurrent_requests=5)
    endpoints = get_models_by_names(["llama-vllm", "qwen-nim"])
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    """Configuration for an LLM inference endpoint.
    
    Attributes:
        name: Human-readable identifier for this endpoint.
        url: Full URL to the OpenAI-compatible chat completions endpoint.
        model_name: Model identifier as expected by the endpoint.
        auth_token: Bearer token for authentication.
        api_type: API specification (currently only "openai" supported).
        verify_ssl: Whether to verify SSL certificates (default: False).
        timeout: Request timeout in seconds (default: 120).
    """
    name: str
    url: str
    model_name: str
    auth_token: str
    api_type: str = "openai"
    verify_ssl: bool = False
    timeout: int = 120


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration.
    
    Attributes:
        num_iterations: Number of times to run each prompt (default: 10).
        concurrent_requests: Maximum concurrent requests per endpoint (default: 1).
        timeout: Maximum request duration in seconds (default: 120).
        output_dir: Directory for result files (default: "results").
        temperature: Sampling temperature for generation (default: 0.7).
        max_tokens: Maximum tokens to generate per request (default: 1024).
        stream: Enable streaming response mode (default: True).
    """
    num_iterations: int = 10
    concurrent_requests: int = 1
    timeout: int = 120
    output_dir: str = "results"
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = True


LLAMA_31_8B_VLLM = EndpointConfig(
    name="llama-vllm",
    url="https://mlis-bench-vllm.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    auth_token=""
)

LLAMA_31_8B_NIM = EndpointConfig(
    name="llama-nim",
    url="https://mlis-bench-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta/llama-3.1-8b-instruct",
    auth_token=""
)

LLAMA_31_8B_VLLM_V11 = EndpointConfig(
    name="llama-vllm-new",
    url="https://mlis-bench-vllm-latest.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    auth_token=""
)

LLAMA_31_8B_OLLAMA = EndpointConfig(
    name="llama-ollama",
    url="https://ollama.hpepcai.demo.local/v1/chat/completions",
    model_name="llama3.1:8b-instruct-fp16",
    auth_token="ollama"
)

QWEN_25_7B_VLLM = EndpointConfig(
    name="qwen-vllm",
    url="https://mlis-bench-qwen-vllm.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    auth_token=""
)

QWEN_25_7B_NIM = EndpointConfig(
    name="qwen-nim",
    url="https://mlis-bench-qwen-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="qwen/qwen-2.5-7b-instruct",
    auth_token=""
)

QWEN_25_7B_OLLAMA = EndpointConfig(
    name="qwen-ollama",
    url="https://ollama.hpepcai.demo.local/v1/chat/completions",
    model_name="qwen2.5:7b-instruct-fp16",
    auth_token="ollama"
)

GEMMA_3_1B_VLLM = EndpointConfig(
    name="gemma-vllm",
    url="https://mlis-bench-gemma-vllm.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="google/gemma-3-1b-it",
    auth_token=""
)

GEMMA_3_1B_NIM = EndpointConfig(
    name="gemma-nim",
    url="https://mlis-bench-gemma-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="google/gemma-3-1b-it",
    auth_token=""
)

GEMMA_3_1B_OLLAMA = EndpointConfig(
    name="gemma-ollama",
    url="https://ollama.hpepcai.demo.local/v1/chat/completions",
    model_name="gemma3:1b-it-fp16",
    auth_token="ollama"
)


ALL_MODELS = {
    "llama-vllm": LLAMA_31_8B_VLLM,
    "llama-nim": LLAMA_31_8B_NIM,
    "llama-vllm-new": LLAMA_31_8B_VLLM_V11,
    "llama-ollama": LLAMA_31_8B_OLLAMA,

    
    "qwen-vllm": QWEN_25_7B_VLLM,
    "qwen-nim": QWEN_25_7B_NIM,
    "qwen-ollama": QWEN_25_7B_OLLAMA,

    
    "gemma-vllm": GEMMA_3_1B_VLLM,
    "gemma-nim": GEMMA_3_1B_NIM,
    "gemma-ollama": GEMMA_3_1B_OLLAMA,

}

DEFAULT_MODELS = [LLAMA_31_8B_VLLM, LLAMA_31_8B_NIM]

def get_models_by_names(names: list[str]) -> list[EndpointConfig]:
    """Retrieve model configurations by their registry names.
    
    Args:
        names: List of model identifiers from ALL_MODELS keys.
        
    Returns:
        List of corresponding EndpointConfig objects.
        
    Raises:
        ValueError: If any model name is not found in the registry.
    
    Examples:
        >>> get_models_by_names(["llama-vllm", "qwen-nim"])
        [EndpointConfig(...), EndpointConfig(...)]
    """
    models = []
    for name in names:
        name_lower = name.lower()
        if name_lower not in ALL_MODELS:
            available = ", ".join(sorted(ALL_MODELS.keys()))
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        models.append(ALL_MODELS[name_lower])
    return models


def list_available_models() -> list[str]:
    """Get sorted list of all available model identifiers.
    
    Returns:
        Alphabetically sorted list of model registry keys.
    """
    return sorted(ALL_MODELS.keys())


def print_available_models():
    """Display formatted list of available models grouped by family.
    
    Prints model registry to console with grouping by model family and
    usage examples for CLI selection.
    """
    print("\n📋 Available Models:")
    print("=" * 70)
    
    # Group by model base name
    models_by_family = {}
    for key in ALL_MODELS.keys():
        # Extract family name (e.g., "llama31", "qwen25", etc.)
        family = key.split('-')[0]
        if family not in models_by_family:
            models_by_family[family] = []
        models_by_family[family].append(key)
    
    # Print grouped by family
    for family in sorted(models_by_family.keys()):
        family_models = sorted(models_by_family[family])
        
        # Create a nice header
        family_display = family.upper()
        print(f"\n  {family_display}:")
        
        for model in family_models:
            # Mark aliases with an asterisk
            if model in ["llama", "qwen", "gemma", "deepseek", "gpt"]:
                print(f"    • {model} *")
            else:
                print(f"    • {model}")
    
    print("\n  * = convenience alias (defaults to vLLM variant)")
    print("=" * 70)
    print("\nExamples:")
    print("  Compare deployments:  --models llama31-8b-vllm llama31-8b-nim")
    print("  Compare models:       --models llama qwen gemma")
    print("  Compare vLLM versions: --models llama31-8b-vllm-v060 llama31-8b-vllm-v061")
    print("=" * 70)


ENDPOINTS = DEFAULT_MODELS
get_endpoints_by_names = get_models_by_names
list_available_endpoints = list_available_models

