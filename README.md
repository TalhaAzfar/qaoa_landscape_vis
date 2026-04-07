# Quantum Optimization Visualizer

A lightweight Python toolkit for interactively exploring quantum variational landscapes, energy gradients, and optimization algorithms using Qiskit and Matplotlib.

## Features

- **QAOA Interactive GUI (`qaoa_gui.py`)**
  Feed the script a classical QUBO matrix, and it will construct the corresponding Cost Hamiltonian using Qiskit's native `QAOAAnsatz`. It then launches an interactive dual-panel dashboard allowing you to drag $\beta$ (Mixer) and $\gamma$ (Cost) sliders to track output bitstring frequencies and dynamically project your current coordinates onto a live 3D energy surface.

- **QAOA Covariance & Gradient Output (`covariance.py`)** 
  Natively mirrors `QAOAAnsatz` to statically sweep the Cost and Mixer parameters across a 3-qubit default problem. Employs Qiskit's Estimator using macroscopic finite-differences to calculate explicit partial gradients, mapping the 3D analytical Cost landscape while outputting a rigorous $5 \times 5$ cross-correlation Covariance matrix linking ($\beta$, $\gamma$, Cost, $\nabla \beta$, $\nabla \gamma$).

- **Math Documentation (`covariance.tex`)**
  A compiled LaTeX formulation detailing the mathematical layout of the utilized $5 \times 5$ Covariance matrix alongside an explicit formulation of the parameter-shift gradients being simulated.

## Requirements

You'll need `qiskit`, `numpy`, and `matplotlib` installed:
```bash
pip install qiskit matplotlib numpy
```

*(Note: Older Python 3.9 environments may throw Qiskit deprecation warnings. Upgrading to Python 3.10+ is recommended).*

## Getting Started

To launch the interactive QAOA graphical interface:
```bash
python qaoa_gui.py
```
*(When prompted in the terminal, simply enter `0` to build the default 3-qubit frustrated problem).*

To investigate the parameter-shift covariance mathematical simulation:
```bash
python covariance.py
```

