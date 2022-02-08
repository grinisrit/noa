# Quantitative Finance (QUANT)

We are building a derivative pricing library with a strong focus
on the differentiable programming paradigm. We implement efficient 
algorithms for sensitivity analysis of various pricing models 
suitable for fast calibration in realtime trading environments,
and benefiting from GPU acceleration for risk evaluation of large 
trading books. Yet our aim is to avoid as much as possible compromises 
on computational accuracy and underlying dynamics expressiveness needed
in complex modelling set-ups.  

We hope also that our approach will make it easier to integrate pricing 
components directly into algorithmic trading systems and machine learning 
pipelines, as well as portfolio and margin optimisation platforms.

## Usage

:warning: This component is under active development.

Hands-on the `noa::quant` component:

* [Introduction: the BSM model](bsm.ipynb) is great place to start 
providing even newcomers with the necessary option pricing background.
* [SABR and the volatility smile](sabr.ipynb) discusses 
Hagan's formula, its corrections, and the heat kernel asymptotic expansions, 
together with the Levenberg–Marquardt (LM) method for calibration.
* [Heston model for the volatility surface](heston.ipynb) describes recent 
advances in sensitivity analysis for the Heston characteristic function 
which enables efficient LM calibration.
* [Differentiable LSM algorithm](lsm.ipynb) presents an adjoint sensitivity
model for the Longstaff and Schwartz method over 
Andersen's exact Heston Monte-Carlo simulation scheme.
 

(c) 2022 GrinisRIT ltd.
