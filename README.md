# Python Package for Granger Causality estimation (pyGC)

*Description:* This repository includes a python package to estimate Granger Causality (GC) from data, and it is structured as below:

```
pygc/
├── parametric.py
├── non_parametric.py
├── granger.py
├── tools.py
├── pySpec.py
```

Where,

- parametric.py: Contains funtions for parametric estimation of GC.
- non_parametric.py: Contains funtions for non-parametric estimation of GC.
- granger.py: Contains function to compute GC from the parameters estimated with parametric or non-parametric methods in the codes above.
- tools.py: Contains auxiliary functions.
- pySpec: Contains functions to compute power spectrum from data (with Fourier or Wavelet transforms).

#######################################################################################################################



