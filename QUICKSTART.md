# Quick Start Guide

## Initial Setup (One-time)

1. **Get Together AI API Key**
   - Sign up at https://api.together.xyz
   - Copy your API key from the dashboard

2. **Configure Environment**
   ```bash
   cd dspy-learning
   cp .env.example .env
   ```
   Edit `.env` and add your API key:
   ```
   TOGETHER_API_KEY=your_actual_key_here
   ```

## Running Your First Experiment

```bash
# Navigate to the experiment directory
cd experiments/01_gsm8k_baseline

# Run the training script
python train.py
```

**Expected output:**
- Configuration details
- Dataset loading (20 train, 50 validation examples)
- Zero-shot baseline evaluation
- Accuracy results
- MLflow logging confirmation

**Estimated time**: 2-5 minutes
**Estimated cost**: $0-2 (using free tier)

## Viewing Results

```bash
# From the dspy-learning root directory
mlflow ui
```

Then open your browser to: http://localhost:5000

You'll see:
- Experiment: `gsm8k-baseline`
- Run: `zero-shot-baseline`
- Metrics: accuracy, correct count, total
- Parameters: model, temperature, dataset sizes

## What's Next?

### Phase 2: Add MIPROv2 Optimization

The training script currently only runs zero-shot baseline. Next steps:

1. Implement optimizer wrapper in `src/dspy_learning/optimizers/mipro.py`
2. Update `train.py` to run MIPROv2 after baseline
3. Compare baseline vs optimized accuracy
4. Document results in experiment README

### Phase 3: Try Different Experiments

- Different models (DeepSeek-R1 for better math reasoning)
- Different datasets (HotPotQA, MultiArith)
- Different DSPy modules (ChainOfThought vs Predict)
- Different optimizers (BootstrapFewShot, COPRO)

## Troubleshooting

### "Together API key not found"
- Check that `.env` file exists in `dspy-learning/` root
- Verify `TOGETHER_API_KEY=...` is set correctly
- No quotes needed around the key

### "Module not found: dspy_learning"
- Make sure you're running from `experiments/01_gsm8k_baseline/`
- The script adds the correct path automatically

### "Rate limit exceeded"
- You've hit Together AI's free tier rate limits (60 req/min)
- Wait a few minutes and try again
- Consider using smaller validation set temporarily

### Low accuracy (<20%)
- This is expected for zero-shot on GSM8K
- Math problems are hard without optimization
- MIPROv2 optimization should improve significantly

## Project Structure Reference

```
dspy-learning/
├── .env                       # Your API keys (gitignored)
├── src/dspy_learning/         # Library code
│   ├── config.py             # Settings (reads .env)
│   ├── data/gsm8k.py         # Dataset loader
│   ├── metrics/exact_match.py # Evaluation metric
│   ├── models/factory.py     # LM factory
│   └── tracking/mlflow.py    # Experiment tracker
├── experiments/
│   └── 01_gsm8k_baseline/
│       ├── config.py         # Experiment config
│       ├── train.py          # Main script
│       └── README.md         # Results & notes
└── mlruns/                   # MLflow data (created on first run)
```

## Getting Help

- **DSPy Issues**: https://github.com/stanfordnlp/dspy
- **Together AI**: https://docs.together.ai/
- **MLflow**: https://mlflow.org/docs/

## Cost Monitoring

Together AI free tier limits:
- ~1M tokens/day
- ~60 requests/minute

For this experiment (20 train, 50 val):
- Zero-shot baseline: ~50-100 requests, ~50k tokens
- MIPROv2 (when implemented): ~200-500 requests, ~200k tokens

You can run many experiments per day on free tier!
