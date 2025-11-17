
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
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNDM3ODI1LCJpYXQiOjE3NjE5MDE4MjUsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMGRiNzhkMTMtZmEwOS00MzgyLWJjNGItOTQ1ODgxZmJhMjIyIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjE5MDE4MjU2ODciLCJ1aWQiOiI3ZGYyNzE4Zi03NmQyLTQxYmQtYmU4NS01ODdhOWMzMjQ1ZTkifX0sIm5iZiI6MTc2MTkwMTgyNSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MTkwMTgyNTY4NyJ9.rAq0t0h_e0r_iNQvSnnixqSYA-DfQnp-5_SI7ke1cBf_2ENkcueu92QoDNAk88MxDLoTSztEToi-H0N-E_a5TH1bvGnJXeqEAylMTmiC73vW3kqPzYn1k89hGVq9QW3i1jbOuTGAux3Z0QMBUV3XWFqZhdUCunys2FjGX8NT97nS7-uEYFlrhATHUntNXqF-yLSGvYzcN6iDeLgPfQwvvBYk2APfDCuNY-fTJB2vJ0M183B9AYNMP4FdDzLgeMW8ZGKUbp0w_or-bC9dRN7QAKRja7RKhnQS3yL1JMDGBa-YUN2SPi0BOoYmGtiKigy3da73CCjKovq24SRd7QoPyg"
)

LLAMA_31_8B_NIM = EndpointConfig(
    name="llama-nim",
    url="https://mlis-bench-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta/llama-3.1-8b-instruct",
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNDM3NzQzLCJpYXQiOjE3NjE5MDE3NDMsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiY2JmNWVlYzUtZTFiZi00NjZhLTg0YjItNGVhNDc1NzlhZGRhIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjE5MDE3NDMxMTQiLCJ1aWQiOiI0YmYzYjE5ZC1hOWQwLTRhOTEtYTlhNC02ZDk5ZjVkODQyNDMifX0sIm5iZiI6MTc2MTkwMTc0Mywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MTkwMTc0MzExNCJ9.U5CP37hW2YnK2peRctxCMZ1QBnRjcipXpZ0iq6SGv1qc3U3KvNc2vMi0WimF67XorUq7h-2M7mTZN_xSc3eFAHMAfzqXrgphpBnaZ-B0LRhZEKzHU160TEvgfBX6BxZr3AU-gat0Tysx4TByHd1EqK5O5Fakr91Uma4xZXNKAciKjZi7FfGqMW5rO_Hyr-hBNCmaTSaDWO15RLLkE1bva7-6B2SMZ9_WKj_1BCeVwSg8ip37GR35mw--horBlX83u0vzKGrfEGXvC9wngRXDFtPcOp4zReggSEQij9rf6ExHT_Y8f-Zaku_hNi-oUmb-mYjf2NPiSAz1vydVIsx1pg"
)

LLAMA_31_8B_VLLM_V11 = EndpointConfig(
    name="llama-vllm-new",
    url="https://mlis-bench-vllm-latest.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNDQyMzg1LCJpYXQiOjE3NjE5MDYzODUsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiM2M0MDM5YWYtOThlZi00MTRkLWJiNzUtZDFkNWJiMjI5N2M4Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjE5MDYzODU0NTIiLCJ1aWQiOiIyZGNhYTgyMS0zZDMwLTQyMTAtYmM0Ni03NzQ5NzhmMzMyNGYifX0sIm5iZiI6MTc2MTkwNjM4NSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MTkwNjM4NTQ1MiJ9.cxS4QgvDwC9fGNC-wnXJjpShWCLH-g8ejEJ9WowCOQ_armeyeI6wkikHPQK-mwfUkjFENhqXuQ2w8JwrooY0LBDXhUMKxBeA6-C5lNyYTHJB8c1XFrPLM35ICWfc32mNXdvNneHkOhqgfe7FN1thBshLwazJ7YMVzYCbNt_PMDvZKaWSVQmphcrkfDrXdI-35Yp5lNij97MLOxz7UkZeh4yDZRcx4Gxj6G8s5dYoSXaBhSU9um1npRpTgw3UDrfFphbM0i3iaz32LDVasstyFkVOkcrQ-A3UQz3Dtb0y_ej1XdcMZgzBbYOVW2P8vMHefb_ZL60OXN25EUI_hKo6fw"
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
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzg3OTUwODgxLCJpYXQiOjE3NjIwMzA4ODEsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiZTgzY2MxNTUtMGFjZC00ZDEyLTliOTMtMTQwNmY5NDIwN2M0Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjIwMzA4ODE5MTkiLCJ1aWQiOiJhNDNjNmNjMC03MWMyLTQxMjktOTgwMC1lMTM3ZDUyZTJmOWQifX0sIm5iZiI6MTc2MjAzMDg4MSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MjAzMDg4MTkxOSJ9.jxGjkJ2lFA8oUXH68nmZh6M9h59uo4HCnn38m4K80c-IcNmDaR48sHUeU4lqJySZ-BpRbiu0fYQa-CMPSIApAecw1BQU5vX8rJvAQdVqYUc8scTC5HhRXij7hy_dR2mrs_YJki2ekULAHb6CC_DkiZZp9bTB9D0qKRqXRJRd1qkpy-BUZApCiRWqM_1fu3d1DER49u6rnzE1m3uoKCScLBQ2_NVzHg6SZfe_xYI63Kqy58tNNXApJI8cM6FtWRv13SYOW0UHKs0yT3Q8xViORB_igpcLQ2h_j29yGRIXy_-aVJE7OZ5UlSZiqKDqK_VvAwS0k9HOvg3zb3T70J-BuQ"
)

QWEN_25_7B_NIM = EndpointConfig(
    name="qwen-nim",
    url="https://mlis-bench-qwen-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="qwen/qwen-2.5-7b-instruct",
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNTYyMjg5LCJpYXQiOjE3NjIwMjYyODksImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiYWU0NWFjOGQtNGFiNC00NjQyLTgzYTEtMmFjMzBmNjc2YTZlIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjIwMjYyODk4NTciLCJ1aWQiOiIzNDk0NWQ5NS03Nzc1LTRkNmEtYTY1Zi1iMzRkMGQ3ODc0ZTMifX0sIm5iZiI6MTc2MjAyNjI4OSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MjAyNjI4OTg1NyJ9.TtomwU1zn5pTxa6BccTN7U5r5iZIpf_aUD5896LsmG1vPzl7FwCzjkpOSW7ClGtAFELuh96Tkbc4NaIU9KwznCJwZs9WDAYOk-WQpBNf3HSfInJDTxApyPLPyrz2psg7H-vW4NA9-f3ijrSfrJj-i1-NzltQ3AtdD7kUmhpxXrXUBSziTnmoo0MNHuqydNeGpPPYwvZhFTCPPZ7rqsuoLnJ038TS5X6IH2WH5iMYXKyrAwA8rBTAQ4jZ_TZJAVVnzAloDurF9xxv3t8LAX00h4jo3YBG0EYhiXJPlFjahXhDNYv1WBBJCKI1CMIIsYAT8DiMy2rZI6ZEqz-FKVKDRw"
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
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNTY0NDU4LCJpYXQiOjE3NjIwMjg0NTgsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiZWU5MzliNGEtODBlZC00OTdlLThmNjktZTU2Yjg2MmQ4M2MxIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjIwMjg0NTg2MTUiLCJ1aWQiOiIxZmU3OWVlMC04ZTMwLTRmYWEtYjQ5Ni1hNTkwODY1NWZiY2MifX0sIm5iZiI6MTc2MjAyODQ1OCwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MjAyODQ1ODYxNSJ9.5n7n0NP1MIGE8mm6QrPBPT_vff9i50wPhwuABrGtFTykW4buNddb_spxkRET1EKLhxjiHUv1_awRnln4zlKLtCyRioR3QCKfRqMkYzxzfGoFTwafZ7vnVlFSlqa8Bw-_CPTWFU_-4-RCAlTEjsl8G5Kll4oCrsDQuB0rG3xmGXKzSPVm37NnmQWSsAKbIdKfo9ZC-SJYzYxyzaIEuQyD9eqYEE-SRcvLu-6aAGC-gHNxHDAsllBk_P3dtfDytBwK26yaxgBlXPNF74LfCyaTFXg1aCD9mY2uRLA9shOO8vJIpBWwnWuD-9atWYKTeellZTLs1drVIiizXodcFwDbCg"
)

GEMMA_3_1B_NIM = EndpointConfig(
    name="gemma-nim",
    url="https://mlis-bench-gemma-ngc.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="google/gemma-3-1b-it",
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNTYxODEzLCJpYXQiOjE3NjIwMjU4MTMsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiZDI4NjYwNWYtY2RlMy00ZmUxLWEwMzgtYzg5M2NlZTNhMDkyIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjIwMjU4MTMyNzUiLCJ1aWQiOiI5MjYyZGZlMi0zOWIwLTQwMzUtOWM0MC1hMDQyOWY5NTg0NDcifX0sIm5iZiI6MTc2MjAyNTgxMywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MjAyNTgxMzI3NSJ9.SLtIeGcCvYFk7dk34dXw3AjPxRwKjosBMpEuV4NvSCmA4402QmMUBmBeOS_tNym40V0dvM57N7I-xxFIpjhzJNeRgYAwuCkNcE6SKQo66DYYewXe5pfh7qcdZKSLioV7lmgUDtYqPu3xn1ZqsH72roUKzSeTsXOA3U3V_mTBgl8cUNTrolQz3R37WDPVRJIr_jNsudxXXivnfAry2ouLTQOGISGJC7Zy0zeK8XqSWTdPc4A2A-tj8O8W0OextEIm24EMzzuyljsS77_HhqUOFbZVawcv_masIbSe9LzKV_MtwJuUX89DOUkjV9MgBXLtdtV2on6F8tHlyxDUfb9WFQ"
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

