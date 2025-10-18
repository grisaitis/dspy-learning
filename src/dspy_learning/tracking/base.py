"""Base protocol for experiment tracking."""

from typing import Any, Protocol


class ExperimentTracker(Protocol):
    """Protocol for experiment tracking systems.

    This abstraction allows swapping between different tracking
    backends (MLflow, Weights & Biases, etc.) without changing
    experiment code.
    """

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters.

        Args:
            params: Dictionary of parameter names and values
        """
        ...

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (for time series)
        """
        ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file or directory as an artifact.

        Args:
            local_path: Path to local file or directory
            artifact_path: Optional destination path in artifact store
        """
        ...

    def end_run(self) -> None:
        """End the current tracking run."""
        ...
