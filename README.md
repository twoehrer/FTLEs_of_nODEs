# Finite Time Lyapunov Exponents for neural ODEs

The code implements finite time Lyapunov exponents (FTLE) for low-dimensional neural ODEs.

Various jupyter notebooks are included that track Lyapunov exponents for different nODE dynamics. Furthermore a modified training method is implemented and visually compared to the standard training. This is done via FTLE suppression and shows improved robustness.



<img src="./attacks_readme.png" width="60%" height="60%" >

Code is uses the torchdiffeq package https://github.com/rtqichen/torchdiffeq