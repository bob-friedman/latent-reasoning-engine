# latent-reasoning-engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21633843.svg)](https://doi.org/10.5281/zenodo.21633843)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bob-friedman/latent-reasoning-engine/blob/main/latent_gradient_steering.ipynb)

Welcome to the official repository for the **Latent Reasoning Engine**.

This project explores a novel inference-time compute procedure that unifies generation and verification into a single, cohesive latent reasoning stream. By transitioning from discrete autoregressive sampling to continuous gradient ascent on network hidden states, this architecture actively steers a large language model toward structurally sound reasoning before the text is generated.

## 📄 Core Research Report

The theoretical foundation and mathematical proofs of this project are detailed in the primary research document:

* **[SPARSITY_RESOLUTION.md](https://github.com/bob-friedman/latent-reasoning-engine/blob/main/SPARSITY_RESOLUTION.md)**: A comprehensive exploration of moving from statistical discrete sparsity mitigation to the Semantic Manifold Hypothesis, feature superposition, and ultimately, Latent Gradient Steering.

## 🚀 Experimental Notebook (Google Colab)

To validate the hypothesis that continuous latent gradient ascent can successfully steer generation, an interactive Jupyter Notebook is provided featuring an A/B double-blind experiment using a quantized `Mistral-7B-Instruct-v0.3` model.

**Important Note on Previous Versions:** The project had software bugs in the Colab script from the earliest version 1.0 through 1.3 (release version at github). This bug prevented the algorithm from properly performing a calculation to dynamically shift the model toward the positive exemplars. These issues have been resolved as of version 1.4.

The experiment may run directly in a web browser by clicking on the "Open in Colab" badge above, or download the `.ipynb` file to run it locally.

### Local Installation Requirements

To run the notebooks locally, the following dependencies are required:

`pip install -U bitsandbytes>=0.46.1 transformers accelerate torch`

## ⚙️ Usage Guide

### Inputs & Parameters

#### Optimization Hyper-parameters
* `max_tokens` (int, default: 250): The maximum number of new tokens to generate.
* `steps` (int, default: 8): The number of Adam optimizer gradient ascent steps performed per token.
* `lr` (float, default: 0.5): The learning rate for the Adam optimizer steering hidden states.
* `l2_weight` (float, default: 1.0): Regularization coefficient ($\lambda$) constraining drift from the initial manifold.
* `sim_scale` (float, default: 100.0): Multiplier for similarity gradients to balance them against the L2 manifold distance.
* `track_trajectories` (bool, default: True): Whether to log step-by-step optimization loss, similarity, and distance metrics.

### Outputs & Trajectory Metrics

`Control Output (Unsteered)`
Standard autoregressive text generated without gradient intervention. It serves as the baseline comparison.

`Treatment Output (Steered)`
Optimized text produced by projecting the gradient-guided hidden states. It integrates the structural cadence of the positive exemplars.

`Trajectory Logging`
When `track_trajectories=True`, the generator yields a nested list of dictionaries containing metrics for each Adam step of every token:
```json
[
  {
    "step": 1,
    "loss": -11.5020,
    "sim_good": 0.1768,
    "sim_bad": 0.1235,
    "manifold_dist": 0.0000
  },
  ...
  {
    "step": 8,
    "loss": -2.4988,
    "sim_good": 0.1906,
    "sim_bad": 0.1276,
    "manifold_dist": 10.1849
  }
]
```
Note: Because manifold_dist starts at zero, the total loss may initially increase as the vector begins to move. A successful trajectory is indicated by sim_good rising as the optimizer finds a stable balance against the L2 distance penalty.

`DPO Generation`
Direct Preference Optimization data is stored in a file named `dpo_dataset.jsonl` and formatted as sets of preference triplets. It is generated for use in the subsequent post-training of a large language model:
```json
{"prompt": "...", "chosen": "Treatment text...", "rejected": "Control text..."}
```

The guide for batch processing the generation of the DPO data: [BATCH_AUTOMATION.md](https://github.com/bob-friedman/latent-reasoning-engine/blob/main/batch_process/BATCH_AUTOMATION.md).

### Unit Tests
* `latent_gradient_steering.py`: The refactored production module containing the decoupled core functions.
* `test_latent_gradient_steering.py`: Unit tests verifying centroid calculation, greedy control generation, optimization correctness, and DPO generation.
  
A lightweight, fast-loading random GPT-2 model is configured for testing, requiring no GPU:
```bash
pytest test_latent_gradient_steering.py -v
```

## 🧠 Methodology Highlight: The Continuous Objective Function

Instead of evaluating discrete token probabilities, the engine dynamically calculates a gradient penalty on the pre-unembedding hidden state. At each token generation step, the hidden vector $h_t$ is optimized using Adam to maximize the following objective function:

$$\mathcal{J}(h_t) = \cos(h_t, \mathbf{c}_{\text{ideal}}) - 0.5 \cdot \cos(h_t, \mathbf{c}_{\text{corrupt}}) - \lambda \Vert h_t - h_{\text{initial}} \Vert_2$$

This mathematically forces the novel generation to conform to the established geometry of a sound argument, bypassing the standard limitations of purely autoregressive generation.

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgement

The conceptual development of the methodology and the drafting of the code and brief report benefited from discussions and iterative refinement with an AI language model, Gemini 3.1 Pro (Google). The author oversaw and reviewed the accuracy and robustness of all parts of this study.

## 📝 Citation

If using this code or methodology in your research, please cite this repository. The recommended format is:

*   Friedman, R. (2026) *Latent Reasoning Engine* (Version v1.4) [Method and software]. Zenodo. https://doi.org/10.5281/zenodo.21635950
