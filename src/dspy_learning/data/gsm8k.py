"""GSM8K dataset loader."""

import dspy
from datasets import load_dataset


class GSM8KLoader:
    """Loader for the GSM8K (Grade School Math 8K) dataset.

    GSM8K is a dataset of grade school math word problems.
    Each example has a question and an answer with reasoning.
    """

    def load(self, train_size: int, val_size: int) -> tuple[list[dspy.Example], list[dspy.Example]]:
        """Load GSM8K dataset from HuggingFace.

        Args:
            train_size: Number of training examples to load
            val_size: Number of validation examples to load

        Returns:
            Tuple of (train_examples, val_examples) as dspy.Example objects
        """
        # Load from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main")

        # Convert to DSPy examples
        train_examples = [
            dspy.Example(
                question=item["question"],
                answer=item["answer"]
            ).with_inputs("question")
            for item in dataset["train"].select(range(train_size))
        ]

        # Use test split for validation
        val_examples = [
            dspy.Example(
                question=item["question"],
                answer=item["answer"]
            ).with_inputs("question")
            for item in dataset["test"].select(range(val_size))
        ]

        return train_examples, val_examples
