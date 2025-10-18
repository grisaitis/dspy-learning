"""Configuration for GSM8K baseline experiment."""

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Configuration for the GSM8K baseline experiment."""

    # Dataset settings
    dataset: str = "gsm8k"
    train_size: int = 20  # Start small for fast iteration
    val_size: int = 50

    # Model settings
    model: str = "llama-3.3-70b"  # Use Together AI free tier model
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 1000

    # Optimizer settings (for Phase 3)
    optimizer: str = "mipro_v2"
    num_candidates: int = 10  # Number of prompt candidates to try
    init_temperature: float = 1.0  # Temperature for prompt generation

    # Tracking settings
    mlflow_experiment: str = "gsm8k-baseline"
    run_name_baseline: str = "zero-shot-baseline"
    run_name_optimized: str = "mipro-optimized"

    # Reproducibility
    random_seed: int = 42
