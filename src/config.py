
"""
Configuration module for LLM benchmarking suite.
Contains endpoint configurations and benchmark settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    """Configuration for an LLM endpoint."""
    name: str
    url: str
    model_name: str
    auth_token: str
    api_type: str = "openai"  # openai-compatible API
    verify_ssl: bool = False
    timeout: int = 120


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    num_iterations: int = 10
    concurrent_requests: int = 1
    timeout: int = 120
    output_dir: str = "results"
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = True


# ============================================================================
# MODEL ENDPOINT CONFIGURATIONS
# ============================================================================
# Each endpoint should have a unique identifier for precise comparisons.
# Use descriptive names that include model, deployment, version details.
# Format examples: llama31-8b-vllm-v060, qwen25-7b-nim, gemma2-9b-vllm-v061

# --- Llama 3.1 8B Instruct ---
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
    name="llama31-vllm-new",
    url="https://mlis-bench-vllm-latest.project-user-zeitler.serving.hpepcai.demo.local/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    auth_token="eyJhbGciOiJSUzI1NiIsImtpZCI6InNHSEFRZkMwdTVadnVRWm9pRVdsRFA4dkZvS29wb2hrSE9LSVl0RnQxX2sifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkzNDQyMzg1LCJpYXQiOjE3NjE5MDYzODUsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiM2M0MDM5YWYtOThlZi00MTRkLWJiNzUtZDFkNWJiMjI5N2M4Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NjE5MDYzODU0NTIiLCJ1aWQiOiIyZGNhYTgyMS0zZDMwLTQyMTAtYmM0Ni03NzQ5NzhmMzMyNGYifX0sIm5iZiI6MTc2MTkwNjM4NSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc2MTkwNjM4NTQ1MiJ9.cxS4QgvDwC9fGNC-wnXJjpShWCLH-g8ejEJ9WowCOQ_armeyeI6wkikHPQK-mwfUkjFENhqXuQ2w8JwrooY0LBDXhUMKxBeA6-C5lNyYTHJB8c1XFrPLM35ICWfc32mNXdvNneHkOhqgfe7FN1thBshLwazJ7YMVzYCbNt_PMDvZKaWSVQmphcrkfDrXdI-35Yp5lNij97MLOxz7UkZeh4yDZRcx4Gxj6G8s5dYoSXaBhSU9um1npRpTgw3UDrfFphbM0i3iaz32LDVasstyFkVOkcrQ-A3UQz3Dtb0y_ej1XdcMZgzBbYOVW2P8vMHefb_ZL60OXN25EUI_hKo6fw"
)

# --- Qwen 2.5 7B (add your configs here) ---
# QWEN_25_7B_VLLM = EndpointConfig(
#     name="qwen25-7b-vllm",
#     url="your-qwen-url",
#     model_name="Qwen/Qwen2.5-7B-Instruct",
#     auth_token="your-token"
# )
#
# QWEN_25_7B_NIM = EndpointConfig(
#     name="qwen25-7b-nim",
#     url="your-qwen-url",
#     model_name="Qwen/Qwen2.5-7B-Instruct",
#     auth_token="your-token"
# )

# --- Gemma 2 9B (add your configs here) ---
# GEMMA_2_9B_VLLM = EndpointConfig(
#     name="gemma2-9b-vllm",
#     url="your-gemma-url",
#     model_name="google/gemma-2-9b-it",
#     auth_token="your-token"
# )
#
# GEMMA_2_9B_NIM = EndpointConfig(
#     name="gemma2-9b-nim",
#     url="your-gemma-url",
#     model_name="google/gemma-2-9b-it",
#     auth_token="your-token"
# )

# --- DeepSeek R1 Distill Qwen 7B (add your configs here) ---
# DEEPSEEK_R1_DISTILL_QWEN_7B_VLLM = EndpointConfig(
#     name="deepseek-r1-distill-qwen7b-vllm",
#     url="your-deepseek-url",
#     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     auth_token="your-token"
# )
#
# DEEPSEEK_R1_DISTILL_QWEN_7B_NIM = EndpointConfig(
#     name="deepseek-r1-distill-qwen7b-nim",
#     url="your-deepseek-url",
#     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     auth_token="your-token"
# )

# --- GPT-OSS 20B (add your configs here) ---
# GPT_OSS_20B_VLLM = EndpointConfig(
#     name="gpt-oss-20b-vllm",
#     url="your-gpt-oss-url",
#     model_name="gpt-oss-20b-model",
#     auth_token="your-token"
# )
#
# GPT_OSS_20B_NIM = EndpointConfig(
#     name="gpt-oss-20b-nim",
#     url="your-gpt-oss-url",
#     model_name="gpt-oss-20b-model",
#     auth_token="your-token"
# )


# ============================================================================
# MODEL REGISTRY - Select models by name
# ============================================================================
# Use lowercase identifiers with hyphens for CLI selection
# Format: modelname-size-deployment-version (e.g., llama31-8b-vllm-v060)

ALL_MODELS = {
    # Llama 3.1 8B family
    "llama31-8b-vllm": LLAMA_31_8B_VLLM,
    "llama31-8b-nim": LLAMA_31_8B_NIM,
    
    # Convenience aliases
    "llama": LLAMA_31_8B_VLLM,  # Default to vLLM
    "llama-vllm": LLAMA_31_8B_VLLM,
    "llama-nim": LLAMA_31_8B_NIM,
    
    # Add your vLLM version comparisons:
    # "llama31-8b-vllm-v060": LLAMA_31_8B_VLLM_V060,
    # "llama31-8b-vllm-v061": LLAMA_31_8B_VLLM_V061,
    # "llama31-8b-vllm-v062": LLAMA_31_8B_VLLM_V062,
    
    # Qwen family (uncomment when ready)
    # "qwen25-7b-vllm": QWEN_25_7B_VLLM,
    # "qwen25-7b-nim": QWEN_25_7B_NIM,
    # "qwen": QWEN_25_7B_VLLM,  # Default alias
    
    # Gemma family (uncomment when ready)
    # "gemma2-9b-vllm": GEMMA_2_9B_VLLM,
    # "gemma2-9b-nim": GEMMA_2_9B_NIM,
    # "gemma": GEMMA_2_9B_VLLM,  # Default alias
    
    # DeepSeek family (uncomment when ready)
    # "deepseek-r1-distill-qwen7b-vllm": DEEPSEEK_R1_DISTILL_QWEN_7B_VLLM,
    # "deepseek-r1-distill-qwen7b-nim": DEEPSEEK_R1_DISTILL_QWEN_7B_NIM,
    # "deepseek": DEEPSEEK_R1_DISTILL_QWEN_7B_VLLM,  # Default alias
    
    # GPT-OSS family (uncomment when ready)
    # "gpt-oss-20b-vllm": GPT_OSS_20B_VLLM,
    # "gpt-oss-20b-nim": GPT_OSS_20B_NIM,
    # "gpt-oss": GPT_OSS_20B_VLLM,  # Default alias
}

# Default models to test (if no selection is made)
DEFAULT_MODELS = [LLAMA_31_8B_VLLM, LLAMA_31_8B_NIM]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_models_by_names(names: list[str]) -> list[EndpointConfig]:
    """
    Get model configurations by their names.
    
    Args:
        names: List of model names (keys from ALL_MODELS)
        
    Returns:
        List of EndpointConfig objects
        
    Raises:
        ValueError: If a model name is not found
    
    Examples:
        >>> get_models_by_names(["llama", "qwen"])
        >>> get_models_by_names(["llama-vllm", "llama-nim"])
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
    """Get list of all available model names, grouped by family."""
    return sorted(ALL_MODELS.keys())


def print_available_models():
    """Print nicely formatted list of available models with grouping."""
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


# Backward compatibility aliases
ENDPOINTS = DEFAULT_MODELS
get_endpoints_by_names = get_models_by_names
list_available_endpoints = list_available_models

