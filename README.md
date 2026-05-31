# Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling

:scroll: [arXiv](https://arxiv.org/pdf/2605.31371)
&nbsp; :computer: [Usage](#using-softsignum-or-softmuon-in-practice)

*TL;DR: We introduce SoftSignum and SoftMuon, smoothly relaxing sign-based optimization. Supported by a unified geometry-relaxation theory and non-convex convergence proofs, they resolve parameter heterogeneity and outperform baselines.*

This repository contains the official implementation for the paper ["Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling"](https://arxiv.org/pdf/2605.31371)

The repository provides:

- The official implementation of **SoftSignum** method.
- The official implementation of **SoftMuon** for different distributed training frameworks.

## Abstract:

Sign-based and LMO-inspired optimizers have recently attracted substantial attention in deep learning due to their strong performance and low memory footprint. However, their fixed-magnitude updates can hurt terminal convergence: they decouple update mechanisms from gradient magnitudes and fail to account for parameter heterogeneity, often leading to oscillation rather than convergence. We propose SoftSignum, a smooth relaxation of sign-based optimization that replaces the hard sign map with a temperature-controlled soft-sign transformation, enabling a parameter-wise transition from sign-like updates to magnitude-sensitive SGD-like steps. We complement it with an adaptive quantile-based temperature schedule and extend the same principle to matrix-valued optimizers, obtaining SoftMuon. We also develop a generalized geometry-relaxation framework based on strongly convex regularizers and Fenchel conjugates, proving convergence in stochastic non-convex setting. Experiments on diverse deep learning tasks, including LLM pretraining, show that SoftSignum and SoftMuon consistently improve over their hard sign-based counterparts and standard AdamW.

# Using SoftSignum or SoftMuon in practice

To use **SoftSignum** outside of this repository, you need only the following:

- [softsignum.py](./softsignum.py): the minimal single-file implementation;
- [the hyperparameters section](#hyperparameters).

To use **SoftMuon** outside of this repository, you need only the following:

- Choose a suitable implementation:
  - [softmuon.py](./softmuon/softmuon.py): based on Keller Jordan's reference Muon [implementation](https://github.com/KellerJordan/modded-nanogpt);
  - [d-softmuon.py](./softmuon/d_softmuon.py): based on toothacher17's Muon [implementation](https://github.com/toothacher17/Megatron-LM/tree/moonshot/distributedmuon-impl) for Megatron-LM;
  - [dion_softmuon.py](./softmuon/dion_softmuon.py): based on Microsoft's Muon [implementation](https://github.com/microsoft/dion/blob/main/dion/muon.py) used in the Dion project;
- [The section about hyperparameters](#hyperparameters).

---

Table of contents

- [Using SoftSignum or SoftMuon in practice](#using-softsignum-or-softmuon-in-practice)
- [Hyperparameters](#hyperparameters)
- [How to cite](#how-to-cite)

---

## Hyperparameters

**SoftSignum** and **SoftMuon** introduce only a small number of additional hyperparameters:

- the transition point $\alpha_{\text{sign}}$;
- the saturation tolerance $\varepsilon$;
- the number of Newton iterations $N_q$ used for quantile computation.

In our experiments, we use these standard values:

$\alpha_{\text{sign}} = 0.9,\quad \varepsilon = 10^{-4},\quad N_q = 10.$

These values provide a simple default configuration and allow **SoftSignum** and **SoftMuon** to be integrated into existing Signum and Muon pipelines without additional tuning.

Moreover, we investigate the robustness of our method with respect to the important hyperparameter $\alpha_{\text{sign}}$ and show that, when varying this hyperparameter over the range from 0.9 to 0.3, the final metrics remain stable.

# How to cite

```
@misc{feoktistov2026softsign,
      title={Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling}, 
      author={Dmitrii Feoktistov and Timofey Belinsky and Andrey Veprikov and Amir Zainullin and Aleksandr Beznosikov},
      year={2026},
      eprint={2605.31371},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

