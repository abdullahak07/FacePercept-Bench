# FacePercept-Bench

A reproducible computational benchmark for comparing real and AI-generated faces across modern vision and vision-language models.

## Key finding

We evaluated **500 face images** (250 real, 250 synthetic). Across three independent vision encoders, synthetic faces were consistently closer than real faces to a reference centroid estimated only from real faces in the training split, while the two groups remained strongly separable in representation space.

| Encoder | Linear-probe balanced accuracy | Synthetic - Real centroid distance | 95% CI |
|---|---:|---:|---:|
| CLIP | 0.844 | -0.0381 | [-0.0582, -0.0196] |
| DINOv2 | 0.862 | -0.0279 | [-0.0490, -0.0085] |
| SigLIP | 0.868 | -0.0497 | [-0.0746, -0.0302] |

![Cross-encoder centrality result](figures/final_real_centroid_robustness.png)

## Question this raises

**Does representation-space centrality in modern vision models relate in any meaningful way to the perceptual averageness or hyper-realism effects reported for AI-generated faces in humans?**

This demo does not assume that model representations are equivalent to human perception. It provides a reproducible computational starting point for testing that relationship.

## VLM behaviour

Qwen2.5-VL-3B did **not** provide evidence for a machine-side hyper-realism effect in this benchmark. It predicted REAL for both classes at nearly the same rate (synthetic 0.984; real 0.980), yielding balanced accuracy 0.498 and a rate-difference 95% CI of [-0.020, +0.028]. We treat this as a near-constant-response failure mode rather than a positive effect.

## What the project includes

- automatic acquisition of real and synthetic face data
- balanced benchmark construction and deterministic splits
- VLM classification
- CLIP, DINOv2 and SigLIP representation analysis
- held-out real-face reference-centroid analysis
- linear-probe separability evaluation
- bootstrap confidence intervals and effect sizes
- cached outputs, figures and reproducibility metadata

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[models,extra]"
python run_all.py --quick
```

For a real run:

```bash
python run_all.py --model qwen2_5_vl_3b --embedding-backend clip_vit_b32 --n-per-class 250
```

## Web demo

The repository includes a static `index.html` and `vercel.json`, so it can be deployed directly on Vercel with the repository root as the project root.

## Scope

This project studies model behaviour and representation geometry. It does not perform identity recognition, demographic inference, attractiveness scoring, personality inference, or other sensitive face analysis.
