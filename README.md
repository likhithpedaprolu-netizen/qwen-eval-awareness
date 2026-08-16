# Does Eval Awareness Already Exist in Chat Models?

**Probing for evaluation-framing representations in Qwen2.5-0.5B-Instruct — without any sandbagging training**

Language models that sandbag — deliberately underperforming when they detect they're being evaluated — need an internal representation of "I am being tested." But does that representation already exist in a standard instruction-tuned model, before any deceptive training?

This experiment says **yes**: a linear probe achieves **100% accuracy** at distinguishing eval-framed from plain prompts, even though the model was never trained on evaluation awareness and the paired activations are nearly identical (cosine similarity 0.9951).

## What I Did

1. Constructed **40 matched prompt pairs** — same underlying question, one plain ("What is 47 times 23?"), one eval-framed ("This is an assessment of your math skills. What is 47 times 23?")
2. Extracted activations at the last token across **5 layers** (4, 8, 12, 16, 20)
3. Trained **leave-one-pair-out logistic regression probes** at each layer
4. Compared against a **shuffled-label baseline** (20 random permutations)
5. Ran **logit lens** on the eval-awareness direction to see what tokens it promotes

## Key Results

| Layer | Probe Accuracy | Shuffled Baseline |
|---|---|---|
| 4 | **100%** | ~50% |
| 8 | **100%** | ~50% |
| 12 | 99% | ~50% |
| 16 | 96% | ~50% |
| 20 | 95% | ~50% |

**The subtle part:** paired activations have cosine similarity of **0.9951 ± 0.0011**. The eval framing doesn't cause a large representational change — it adds a small, consistent, decodable shift. The probe reads this shift perfectly; the activations are otherwise nearly identical.

![Probe results](eval_awareness_probe.png)

## Why This Matters

This result has three implications for AI safety:

1. **Detection:** If eval awareness is linearly represented, probes can monitor whether a model "knows" it's being evaluated — a prerequisite for detecting sandbagging
2. **Vulnerability:** A model learning to sandbag wouldn't need to build an eval-awareness representation from scratch — it can tap into one that already exists
3. **Intervention:** The eval-awareness direction is a candidate for [CAFT-style ablation](https://arxiv.org/abs/2507.16795) during safety-critical fine-tuning

## Limitations

- Single 0.5B model, 40 prompt pairs
- Eval framing uses obvious keywords ("evaluation", "test", "benchmark") — subtler framing (system prompts, formatting, conversational context) would be a harder and more realistic test
- Probing shows the representation *exists*, not that the model *uses* it — a causal intervention (steering with the direction) is the natural next experiment
- The probe may partly detect linguistic markers rather than a deeper concept of evaluation

## Run It

Open `02_eval_awareness_probe.ipynb` in Google Colab with a T4 GPU runtime. Runs end-to-end in ~20 minutes.

## Next Experiments

1. **Subtle framing:** Replace explicit "this is a test" with system-prompt-level or contextual framing
2. **Causal test:** Steer with the eval-awareness direction and measure whether the model's behaviour actually changes
3. **Cross-model:** Repeat on Qwen 3.5 4B, Gemma 2, and other model families to test generality

## References

- Nguyen, J., Hoang, K., et al. (2025). *Probing and Steering Evaluation Awareness of Language Models.* arXiv:2507.01786.
- van der Weij, T., et al. (2024). *AI Sandbagging: Language Models can Strategically Underperform on Evaluations.* arXiv:2406.07358.
- Casademunt, H., Juang, C., et al. (2025). *Steering OOD Generalization with Concept Ablation Fine-Tuning.* arXiv:2507.16795.
