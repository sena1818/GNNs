# Generative Neural Networks for the Sciences

Coursework for **Generative Neural Networks for the Sciences** at Heidelberg University (Winter Semester 2025/26).

This course covers generative models, probabilistic inference, and density estimation with applications in scientific computing.

## Setup

```bash
conda env create -f environment.yml
conda activate GNNS
```

## Repository Structure

```
.
├── tutorials/              # In-class tutorials
│   ├── T0_mnist_classifier/    # CNN basics on MNIST
│   ├── T1_vae/                 # Variational Autoencoder
│   └── T2_sbi/                 # Simulation-Based Inference
│
├── exercises/              # Weekly assignments
│   ├── ex01_density_forest/    # Density estimation with decision trees
│   ├── ex02_autoencoder/       # Autoencoder architectures
│   ├── ex03_normalizing_flows/ # RealNVP normalizing flows
│   ├── ex04_epidemiology/      # Epidemiological model inference (SBI)
│   ├── ex05_project_proposal/  # Final project proposal
│   └── ex06_sindy/             # SINDy: Sparse dynamics discovery
│
└── final_project/          # Diffusion Models for TSP
    ├── models/                 # GNN encoder + diffusion model
    ├── utils/                  # Decoding, visualization
    ├── experiments/            # Ablation & generalization scripts
    └── report/                 # LaTeX report
```

## Tutorials

| # | Topic | Key Concepts |
|---|-------|-------------|
| T0 | MNIST Classifier | CNN, cross-entropy loss, MPS acceleration |
| T1 | VAE | Reparameterization trick, ELBO, KL divergence |
| T2 | Simulation-Based Inference | SNPE, posterior estimation, Two Moons benchmark |

## Exercises

| # | Topic | Methods |
|---|-------|---------|
| 01 | Density Forest | Decision tree density estimation, KDE, Gaussian mixture |
| 02 | Autoencoder | Standard AE, convolutional AE, latent space visualization |
| 03 | Normalizing Flows | RealNVP, affine coupling layers, conditional generation |
| 04 | Epidemiology | SIR/SEIR models, simulation-based inference, posterior analysis |
| 05 | Project Proposal | Literature review, experimental design |
| 06 | SINDy | Sparse Identification of Nonlinear Dynamics, symbolic regression |

## Final Project

**Diffusion Models for Combinatorial Optimization: Solving TSP via Generative Modeling**

Reproducing and extending [DIFUSCO](https://arxiv.org/abs/2303.18138) (NeurIPS 2023) — using graph neural networks and discrete diffusion to solve the Travelling Salesman Problem.

Key experiments:
- GNN architecture ablation (Gated GCN vs GAT vs GCN)
- Cross-scale generalization (TSP-20/50/100)
- Decoding strategy comparison (greedy, beam search, 2-opt)

See [final_project/](final_project/) for details.

## Tech Stack

- Python 3.11, PyTorch 2.5.1 (MPS backend for Apple Silicon)
- torchvision, matplotlib, scipy, scikit-learn
- sbi, sbibm (simulation-based inference)
- PyTorch Geometric (final project)
