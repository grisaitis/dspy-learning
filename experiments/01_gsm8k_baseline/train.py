import dspy
from dspy_learning.config import settings
from dspy_learning.data.gsm8k import GSM8KLoader
from dspy_learning.metrics.exact_match import gsm8k_exact_match
from dspy_learning.models.factory import create_lm
from dspy_learning.tracking.mlflow import MLflowTracker

from config import ExperimentConfig


EXPERIMENT_NAME = "DSPy-Optimization-GSM8K"


class MathQA(dspy.Signature):
    question = dspy.InputField(desc="A grade school math word problem")
    answer = dspy.OutputField(desc="The numerical answer")


def evaluate(
    module: dspy.Module,
    examples: list[dspy.Example],
    metric: dspy.Metric,
) -> dict:
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
    config = ExperimentConfig()
    print(f"Configuration: {config.model_dump()}")

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    print(f"\nInitializing language model: {config.model}")
    lm = create_lm(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    dspy.configure(lm=lm)

    print(f"\nLoading GSM8K dataset (train={config.train_size}, val={config.val_size})")
    loader = GSM8KLoader()
    train_examples, val_examples = loader.load(
        train_size=config.train_size,
        val_size=config.val_size,
    )
    print(
        f"Loaded {len(train_examples)} train examples, {len(val_examples)} val examples"
    )

    print("\nExample problem:")
    print(f"Q: {train_examples[0].question}")
    print(f"A: {train_examples[0].answer}")

    print("\n" + "=" * 60)
    print("PHASE 1: Zero-Shot Baseline Evaluation")
    print("=" * 60)

    baseline_module = dspy.Predict(MathQA)

    print(f"\nEvaluating baseline on {len(val_examples)} validation examples...")
    baseline_results = evaluate(baseline_module, val_examples, gsm8k_exact_match)

    print("\nBaseline Results:")
    print(f"  Accuracy: {baseline_results['accuracy']:.2%}")
    print(f"  Correct: {baseline_results['correct']}/{baseline_results['total']}")

    with MLflowTracker(config.mlflow_experiment, config.run_name_baseline) as tracker:
        tracker.log_params(
            {
                "model": config.model,
                "temperature": config.temperature,
                "train_size": config.train_size,
                "val_size": config.val_size,
                "approach": "zero-shot",
            }
        )

        tracker.log_metric("accuracy", baseline_results["accuracy"])
        tracker.log_metric("correct", baseline_results["correct"])
        tracker.log_metric("total", baseline_results["total"])

    print("\n✓ Baseline evaluation complete!")
    print(f"  Results logged to MLflow experiment: {config.mlflow_experiment}")

    print("\n" + "=" * 60)
    print("View results in MLflow UI:")
    print("  mlflow ui")
    print("  Then open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
