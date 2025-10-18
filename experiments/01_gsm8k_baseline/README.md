# GSM8K Baseline Experiment

## Overview

First experiment to validate the dspy-learning infrastructure and get initial training results on GSM8K.

## Goals

1. **Phase 1**: Run zero-shot baseline evaluation
2. **Phase 2**: Run MIPROv2 optimization (future)
3. **Phase 3**: Compare baseline vs optimized performance

## Dataset

- **GSM8K** (Grade School Math 8K): Collection of grade school math word problems
- Training set: 20 examples (small for fast iteration)
- Validation set: 50 examples

## Model

- **Llama 3.3 70B Instruct Turbo** (Together AI free tier)
- Temperature: 0.0 (deterministic)
- Max tokens: 1000

## Running the Experiment

```bash
# Set up environment
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY

# Run training
cd experiments/01_gsm8k_baseline
python train.py

# View results in MLflow
mlflow ui
# Open http://localhost:5000
```

## Results

### Zero-Shot Baseline

- **Status**: Pending
- **Accuracy**: TBD
- **Notes**: To be filled after first run

### MIPROv2 Optimized

- **Status**: Not yet implemented
- **Accuracy**: TBD
- **Notes**: Phase 3 implementation

## Next Steps

1. ✓ Validate zero-shot baseline works
2. ⏸ Add MIPROv2 optimization
3. ⏸ Compare results and document learnings
