# Passage of Particles through Matter Simulations (PMS) 

High energy physics (HEP) beyond colliders challenges mathematical 
models and their implementation in a unique way. 
We have to take into account complex interactions with 
the media the experiment takes place in. 
This however opens up the door to answer subtle questions 
not only about the HEP process studied, but also the media itself.

For example, the detection of atmospheric muons passing 
through various obstacles leads to the development of 
imaging tools (muography) for their matter density, 
as well as a possibility to discriminate materials traversed, 
based on their atomic number.

Data collected from such experiments becomes significant, 
and the equipment is getting accessible beyond 
the physics labs for applications. 
We build a software suite that leverages machine learning 
and AI technologies to carry out analysis and inference 
on that data with performance and modelling complexity suitable 
not only for further scientific work, 
but also meeting industry requirements.

## Differentiable programming approach

This library implements adjoint sensitivity methods
over Backward Monte-Carlo schemes in particle physics simulations. 
A general introduction can be found in:

* R. Grinis. *Differentiable programming for particle physics simulations.* 
to appear in JETP (2021) [arXiv](https://arxiv.org/abs/2108.10245)

with [complementary material](differentiable_programming_pms.ipynb) 
and a [video workshop](https://www.youtube.com/watch?v=nJm_jbX6tJc)
recorded at the 
[QUARKS-2021](https://www.youtube.com/channel/UCXdL4IpBP3LqmUO2EqNCYxA) 
conference.

## Muography

Muography is currently a very active area of research. 
It presents great opportunities for applications in geology, 
civil engineering and nuclear security to mention a few.

A differentiable model for muon transport is in 
[preparation](muography.ipynb).

## Usage

:warning: This component is under active development.

The component implements energy loss and 
Coulomb scattering differential cross-sections 
calculations currently for Muons and Taus.
Some [examples](muon_dcs_calc.ipynb) are available as well as
[benchmarks](../../benchmark)
measuring `CPU/OpenMP` vs `CUDA` performance.

We also provide bindings to 
[PUMAS v1.1](https://github.com/niess/pumas). 
Usage examples can be found in
[functional tests](../../test/pms).

In the future, we plan to cover
a wider range of particles.


(c) 2022 GrinisRIT ltd. and contributors
