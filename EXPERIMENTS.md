# Reproduction experiments

This repository contains a local reproduction and extension of MiniOneRec on the
Amazon **Industrial and Scientific** category. The main additions include:

- atomic three-level Semantic ID tokens for Qwen2.5-0.5B;
- constrained beam search over the catalog prefix tree;
- aligned train/evaluation prompts and hierarchical A/AB/SID diagnostics;
- a hierarchical GRPO reward (`0.2 / 0.5 / 1.0` for A/AB/full-SID matches);
- a leakage-free SASRec baseline using the same temporal data split.

## Frozen test results

The frozen test set contains 4,533 samples. Large prediction files and model
checkpoints are intentionally excluded from Git.

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | Valid SID@5 |
|---|---:|---:|---:|---:|---:|
| SASRec | 0.0845 | 0.0725 | 0.1048 | 0.0791 | N/A |
| SFT | 0.1037 | 0.0808 | 0.1299 | 0.0893 | 1.0000 |
| SFT + hierarchical GRPO (500 steps) | 0.1037 | 0.0810 | 0.1288 | 0.0890 | 1.0000 |

The hierarchical reward increased the proportion of GRPO steps with a useful
within-group advantage from 11% under exact-match-only reward to 34.4% in the
500-step run. On the frozen test set, GRPO produced only a small ranking change;
the principal gains came from full-data SFT, prompt alignment, and constrained
decoding. This distinction is preserved to avoid overstating the RL result.

## Key diagnostics

- Prompt mismatch reduced HR@5 from about 9.95% to 8.12% in the aligned
  validation investigation.
- Prefix-tree decoding maintained a 100% valid Semantic ID rate.
- A/AB/full-SID hit rates, catalog coverage, prefix entropy, and first-token
  ranking were used to diagnose prefix collapse and reward sparsity.
- The SASRec implementation was retrained after correcting validation/test path
  reuse, preventing evaluation leakage.

