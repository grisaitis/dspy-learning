"""MLflow experiment tracker implementation."""

from typing import Any
import mlflow


class MLflowTracker:
    """MLflow implementation of ExperimentTracker.

    Wraps MLflow's tracking API to match our protocol.
    """

    def __init__(self, experiment_name: str, run_name: str | None = None):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this specific run
        """
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values
        """
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric value to MLflow.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file or directory as an MLflow artifact.

        Args:
            local_path: Path to local file or directory
            artifact_path: Optional destination path in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically end run."""
        self.end_run()
