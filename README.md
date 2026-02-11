# Finite Time Lyapunov Exponents for neural ODEs

This repository contains code for the paper [Tracking Finite-Time Lyapunov Exponents to Robustify Neural ODEs](https://arxiv.org/abs/2602.09613)

The code implements finite time Lyapunov exponents (FTLE) for low-dimensional neural ODEs.


Various jupyter notebooks are included that track Lyapunov exponents for different nODE dynamics. Furthermore a modified training method is implemented and visually compared to the standard training. This is done via FTLE suppression and shows improved robustness.

MLE_master.ipynb is a good starting point


<img src="https://github.com/user-attachments/assets/fa34d00a-376f-417e-9284-dcb0f1cf0cec" width="400" />


Code is uses the torchdiffeq package https://github.com/rtqichen/torchdiffeq