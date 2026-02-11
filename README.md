# Finite Time Lyapunov Exponents for Neural ODEs

This repository contains the code for the paper:  
**[Tracking Finite-Time Lyapunov Exponents to Robustify Neural ODEs](https://arxiv.org/abs/2602.09613)**

The code implements finite time Lyapunov exponents (FTLE) for low-dimensional neural ODEs. Various Jupyter notebooks track Lyapunov exponents for different nODE dynamics and demonstrate a modified training method (FTLE suppression) that improves robustness.

> **Getting Started:** `MLE_master.ipynb` is the best starting point.

---

### Method Comparison: Robustness via FTLE Suppression

| Standard Training | Training with FTLE suppression |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/e311cbd8-5fcb-4d96-afce-ffdd30731210" width="400" /> | <img src="https://github.com/user-attachments/assets/bd04aa81-f9a5-488c-850a-9529ea2e0744" width="400" /> |

---

### FTLE Dynamics Visualized

<p align="center">
  <b>Phase Space Evolution</b><br>
  <img src="https://github.com/user-attachments/assets/fa34d00a-376f-417e-9284-dcb0f1cf0cec" width="400" />
  <br><br>
  <b>Lyapunov Field Tracking</b><br>
  <img src="https://github.com/user-attachments/assets/bde6350f-6c31-4ce6-873a-40d64710f232" width="400" />
</p>

---
*Built using the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) package.*
