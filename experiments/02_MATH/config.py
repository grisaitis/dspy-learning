from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    dataset: str = "gsm8k"
    train_size: int = 20
    val_size: int = 50

    model: str = "llama-3.3-70b"
    temperature: float = 0.0
    max_tokens: int = 1000

    optimizer: str = "mipro_v2"
    num_candidates: int = 10
    init_temperature: float = 1.0

    mlflow_experiment: str = "gsm8k-baseline"
    run_name_baseline: str = "zero-shot-baseline"
    run_name_optimized: str = "mipro-optimized"

    random_seed: int = 42
