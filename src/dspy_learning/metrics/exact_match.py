"""Exact match metric for GSM8K."""

import re
import dspy


def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from GSM8K format.

    GSM8K answers are formatted like:
    "<<calculation>>\\n#### 42"

    We want to extract the final number after ####.

    Args:
        text: The answer text to parse

    Returns:
        The extracted numerical answer, or None if not found
    """
    # Look for the #### pattern
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        return match.group(1).replace(",", "")

    # Fallback: try to find any number in the text
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def gsm8k_exact_match(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Compute exact match accuracy for GSM8K.

    Args:
        example: Ground truth example with 'answer' field
        prediction: Model prediction with 'answer' field
        trace: Optional trace (unused)

    Returns:
        1.0 if answers match exactly, 0.0 otherwise
    """
    # Extract answers
    true_answer = extract_answer(example.answer)
    pred_answer = extract_answer(prediction.answer)

    # Handle extraction failures
    if true_answer is None or pred_answer is None:
        return 0.0

    # Compare
    return 1.0 if true_answer == pred_answer else 0.0
