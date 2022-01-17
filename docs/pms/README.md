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
The computational physics algorithms are based on 
[pumas v1.0](https://github.com/niess/pumas/releases/tag/v1.0). 
We [document](muon_dcs_calc.ipynb) the main energy loss differential cross-sections 
calculations.

## Usage

:warning: This component is under active development.

The component `noa::pms::leptons` implements the creatiotraineen
of material's physics models for leptons
(essentially muons as of now) from configuration files available at
[noa-pms-models](https://github.com/grinisrit/noa-pms-models).
Examples can be found within the
[functional tests](../../test/pms) and
the [benchmarks](../../benchmark)
measuring `CPU/OpenMP` vs `CUDA` performance.

In the near future, we plan to cover
a wide range of particles.

## Acknowledgements

The semi-analytical BMC algorithm for Muons and Taus propagation relies on 
[pumas](https://github.com/niess/pumas).

To load the MDF settings we rely on 
[pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

(c) 2022 GrinisRIT ltd. and contributors
