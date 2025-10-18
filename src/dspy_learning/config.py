"""Base configuration for dspy-learning library."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for the dspy-learning library."""

    # API Keys
    together_api_key: str

    # MLflow settings
    mlflow_tracking_uri: str = "./mlruns"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings instance
settings = Settings()
