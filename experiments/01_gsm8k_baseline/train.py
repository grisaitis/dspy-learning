"""Training script for GSM8K baseline experiment.

This script:
1. Loads the GSM8K dataset
2. Runs a zero-shot baseline evaluation
3. Logs results to MLflow
4. (Future) Runs MIPROv2 optimization
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import dspy
from dspy_learning.config import settings
from dspy_learning.data.gsm8k import GSM8KLoader
from dspy_learning.metrics.exact_match import gsm8k_exact_match
from dspy_learning.models.factory import create_lm
from dspy_learning.tracking.mlflow import MLflowTracker

from config import ExperimentConfig


class MathQA(dspy.Signature):
    """Signature for math question answering.

    Input: A math word problem
    Output: The numerical answer
    """
    question = dspy.InputField(desc="A grade school math word problem")
    answer = dspy.OutputField(desc="The numerical answer")


def evaluate(module: dspy.Module, examples: list[dspy.Example], metric) -> dict:
    """Evaluate a module on examples.

    Args:
        module: DSPy module to evaluate
        examples: List of examples to evaluate on
        metric: Metric function to use

    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = len(examples)

    for example in examples:
        try:
            prediction = module(question=example.question)
            score = metric(example, prediction)
            correct += score
        except Exception as e:
            print(f"Error on example: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    """Main training loop."""

    # Load configuration
    config = ExperimentConfig()
    print(f"Configuration: {config.model_dump()}")

    # Set MLflow tracking URI
    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Create language model
    print(f"\nInitializing language model: {config.model}")
    lm = create_lm(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    dspy.configure(lm=lm)

    # Load dataset
    print(f"\nLoading GSM8K dataset (train={config.train_size}, val={config.val_size})")
    loader = GSM8KLoader()
    train_examples, val_examples = loader.load(
        train_size=config.train_size,
        val_size=config.val_size,
    )
    print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} val examples")

    # Show example
    print(f"\nExample problem:")
    print(f"Q: {train_examples[0].question}")
    print(f"A: {train_examples[0].answer}")

    # ========================================
    # Phase 1: Zero-Shot Baseline
    # ========================================

    print("\n" + "="*60)
    print("PHASE 1: Zero-Shot Baseline Evaluation")
    print("="*60)

    # Create baseline module (simple Predict)
    baseline_module = dspy.Predict(MathQA)

    # Evaluate baseline
    print(f"\nEvaluating baseline on {len(val_examples)} validation examples...")
    baseline_results = evaluate(baseline_module, val_examples, gsm8k_exact_match)

    print(f"\nBaseline Results:")
    print(f"  Accuracy: {baseline_results['accuracy']:.2%}")
    print(f"  Correct: {baseline_results['correct']}/{baseline_results['total']}")

    # Log to MLflow
    with MLflowTracker(config.mlflow_experiment, config.run_name_baseline) as tracker:
        # Log configuration
        tracker.log_params({
            "model": config.model,
            "temperature": config.temperature,
            "train_size": config.train_size,
            "val_size": config.val_size,
            "approach": "zero-shot",
        })

        # Log metrics
        tracker.log_metric("accuracy", baseline_results["accuracy"])
        tracker.log_metric("correct", baseline_results["correct"])
        tracker.log_metric("total", baseline_results["total"])

    print("\n✓ Baseline evaluation complete!")
    print(f"  Results logged to MLflow experiment: {config.mlflow_experiment}")

    # ========================================
    # Phase 2: Show MLflow UI command
    # ========================================

    print("\n" + "="*60)
    print("View results in MLflow UI:")
    print("  mlflow ui")
    print("  Then open: http://localhost:5000")
    print("="*60)


if __name__ == "__main__":
    main()
