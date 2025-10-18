"""Base protocol for dataset loaders."""

from typing import Protocol


class DatasetLoader(Protocol):
    """Protocol for dataset loaders.

    All dataset loaders should implement this interface to ensure
    consistent behavior across different datasets.
    """

    def load(
        self, train_size: int, val_size: int
    ) -> tuple[list, list]:
        """Load training and validation datasets.

        Args:
            train_size: Number of examples to load for training
            val_size: Number of examples to load for validation

        Returns:
            Tuple of (train_examples, val_examples)
        """
        ...
