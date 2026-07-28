# latent-reasoning-engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bob-friedman/latent-reasoning-engine/blob/main/latent_gradient_steering.ipynb)

Welcome to the official repository for the **Latent Reasoning Engine**.

This project explores a novel inference-time compute procedure that unifies generation and verification into a single, cohesive latent reasoning stream. By transitioning from discrete autoregressive sampling to continuous gradient ascent on network hidden states, this architecture actively steers a large language model toward structurally sound reasoning before the text is generated.

## 📄 Core Research Report

The theoretical foundation and mathematical proofs of this project are detailed in the primary research document:

* **[SPARSITY_RESOLUTION.md](https://www.google.com/search?q=./SPARSITY_RESOLUTION.md)**: A comprehensive exploration of moving from statistical discrete sparsity mitigation to the Semantic Manifold Hypothesis, feature superposition, and ultimately, Latent Gradient Steering.

## 🚀 Experimental Notebooks (Google Colab)

To validate the hypothesis that continuous latent gradient ascent can successfully steer generation, interactive Jupyter Notebooks are provided featuring an A/B double-blind experiment using a quantized `Mistral-7B-Instruct-v0.3` model.

The experiment may run directly in a web browser by clicking on the "Open in Colab" badge above, or download the `.ipynb` file to run it locally.

### Local Installation Requirements

To run the notebooks locally, the following dependencies are required:

`pip install -U bitsandbytes>=0.46.1 transformers accelerate torch`

## 🧠 Methodology Highlight: The Continuous Objective Function

Instead of evaluating discrete token probabilities, the engine dynamically calculates a gradient penalty on the pre-unembedding hidden state. At each token generation step, the hidden vector $h_t$ is optimized using Adam to maximize the following objective function:

$$\mathcal{J}(h_t) = \cos(h_t, \mathbf{c}_{\text{ideal}}) - 0.5 \cdot \cos(h_t, \mathbf{c}_{\text{corrupt}}) - \lambda \Vert h_t - h_{\text{initial}} \Vert_2$$

This mathematically forces the novel generation to conform to the established geometry of a sound argument, bypassing the standard limitations of purely autoregressive generation.

## 📜 License

This project is licensed under the MIT License.
