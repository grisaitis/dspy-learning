"""Factory for creating language models."""

import os
import dspy


# Together AI model registry
TOGETHER_MODELS = {
    "llama-3.3-70b": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "deepseek-r1": "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen-2.5-72b": "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
}


def create_lm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    cache: bool = True,
    **kwargs,
) -> dspy.LM:
    """Create a DSPy language model using Together AI.

    Args:
        model: Model name (either short name from TOGETHER_MODELS or full Together AI path)
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        cache: Whether to enable LiteLLM caching
        **kwargs: Additional arguments to pass to dspy.LM

    Returns:
        Configured dspy.LM instance

    Example:
        >>> lm = create_lm("llama-3.3-70b", temperature=0.0)
        >>> dspy.configure(lm=lm)
    """
    # Resolve model name
    model_path = TOGETHER_MODELS.get(model, model)

    # Configure LiteLLM settings
    if cache:
        # Enable caching to reduce costs
        import litellm
        from litellm.caching import Cache

        litellm.cache = Cache()

    # Create DSPy LM
    lm = dspy.LM(
        model=model_path,
        api_key=os.getenv("TOGETHER_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return lm
