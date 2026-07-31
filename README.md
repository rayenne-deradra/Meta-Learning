# Cross-Domain Few-Shot Learning for Histopathology: MAML from Colorectal to Prostate Cancer

Model-Agnostic Meta-Learning (MAML) applied to a cross-domain, few-shot histopathology classification problem: a ResNet-18 backbone is meta-trained on a 9-class **colorectal** cancer dataset (CRC-NCT-HE-100K), then evaluated on how quickly it adapts to a binary benign-vs-cancer task on a **prostate** cancer dataset (SICAPv2) it has never seen during training.

## Motivation

Histopathology datasets are abundant for some cancer types and scarce for others, and manual annotation by pathologists is expensive and slow. This project asks whether a model trained to recognize tissue patterns in one cancer type can be adapted to a different cancer type using only a handful of labeled examples — a setting known as **few-shot, cross-domain classification**.

MAML is well suited to this because it learns an initialization that is explicitly optimized to be *easy to fine-tune*, rather than memorizing fixed class prototypes the way metric-based few-shot methods (e.g. Prototypical Networks) do. That distinction matters here: the target domain (prostate tissue) differs from the source domain (colorectal tissue) in staining, morphology, and class semantics, so an approach built for fast adaptation is a better fit than one built around fixed prototypes.

## Datasets

| Dataset | Role | Classes | Notes |
|---|---|---|---|
| [CRC-NCT-HE-100K](https://zenodo.org/records/1214456) | Meta-train / meta-validation | 9 (tumor, stroma, lymphocytes, mucosa, adipose, background, etc.) | 100,000 H&E-stained patches; split 70% / 15% for meta-train / meta-val |
| [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/1) | Meta-test (cross-domain) | 2 (benign / cancer, derived from Gleason-pattern masks) | Binary label built via mask-based thresholding; held-out 15% used for testing |

## Method

1. **Meta-train** on 5-way, 5-shot, 10-query episodes sampled from CRC-NCT.
2. **Meta-test** on 2-way (binary), 5-shot episodes sampled from SICAPv2 — the classifier head is replaced and the model adapts with a handful of inner-loop gradient steps per episode.
3. **Evaluate** adaptation quality against a random baseline, and visualize training dynamics, the confusion matrix, and qualitative test samples.

**Architecture:** ResNet-18 pretrained on ImageNet, split into a frozen backbone (`conv1`–`layer3`, provides stable low/mid-level features) and a meta-learned portion (`layer4` + linear head, the only parameters updated by MAML's inner and outer loops).

**Inner/outer loop:** implemented with [`higher`](https://github.com/facebookresearch/higher), which creates a differentiable functional copy of the model so gradients can flow back through the inner-loop adaptation steps to the meta-parameters (a gradient of gradients).

## Results

| Metric | Value |
|---|---|
| Meta-train best validation accuracy (CRC-NCT, 5-way) | 91.02% |
| Cross-domain test accuracy (SICAPv2, binary) | 65.30% |
| Random baseline (binary) | 50.00% |
| Gain over random baseline | +15.30% |

See `results/maml_crossdomain_results.png` for training curves, the confusion matrix, and sample test patches.

## Repository Structure

```
├── maml_crossdomain.ipynb        # Full pipeline: setup, data, model, training, cross-domain test, results
├── results/
│   └── maml_crossdomain_results.png
└── README.md
```

## Requirements

- Python, PyTorch, torchvision
- [`higher`](https://github.com/facebookresearch/higher) — differentiable inner-loop optimization for MAML
- numpy, matplotlib, seaborn, scikit-learn, Pillow

## Limitations and Future Work

Adapting a model meta-trained on colorectal tissue to a binary prostate-cancer task, with only 5 support images and no target-domain images seen during meta-training, is a substantially harder problem than same-domain few-shot classification. Directions for improvement include:

- Unfreezing more of the backbone (e.g. `layer3`) so meta-learning has more capacity to adapt domain-specific features, not just `layer4`.
- Narrowing the gap between inner-loop steps used at train vs. test time, or meta-training with a schedule of adaptation-step counts.
- Domain-appropriate stain augmentation (e.g. H&E color augmentation) during meta-training, so learned features are more robust to staining differences between datasets.
- Increasing the number of support shots at meta-test time if more labeled target-domain examples become available.

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
- Kather, J. N., et al. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* (CRC-NCT-HE-100K). Zenodo.
- Silva-Rodríguez, J., et al. (2020). *Going Deeper through the Gleason Scoring Scale: An Automatic End-to-End Classification, Grading, and Generation Framework for Prostate Biopsy Samples* (SICAPv2).

## Author

Rayenne Maissoune Deradra
