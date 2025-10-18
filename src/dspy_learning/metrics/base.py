"""Base protocol for metrics."""

from typing import Any, Protocol


class Metric(Protocol):
    """Protocol for evaluation metrics.

    Metrics should be callable functions that take an example,
    prediction, and optional trace, and return a score.
    """

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Any = None
    ) -> float:
        """Compute metric score.

        Args:
            example: The ground truth example
            prediction: The model's prediction
            trace: Optional execution trace for debugging

        Returns:
            Score (typically 0.0 for incorrect, 1.0 for correct)
        """
        ...
