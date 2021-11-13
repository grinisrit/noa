# Passage of Particles through Matter Simulations (PMS) 

This library implements the adjoint sensitivity methods over backward Monte-Carlo schemes.
The primary focus is on particle physics simulations.

:warning: This component is under active development.

An introduction for differentiable programming for particle physics 
simulations is presented in the 
[notebook](differentiable_programming_pms.ipynb). 

A [video workshop](https://www.youtube.com/watch?v=nJm_jbX6tJc)
has been recorded at the 
[QUARKS-2021](https://www.youtube.com/channel/UCXdL4IpBP3LqmUO2EqNCYxA) 
conference.

The component `noa::pms::leptons` implements the creation 
of material's physics models for leptons 
(essentially muons as of now) from configuration files available at
[noa-pms-models](https://github.com/grinisrit/noa-pms-models). 
The computational physics algorithms are based on 
[pumas v1.0](https://github.com/niess/pumas/releases/tag/v1.0). 
Examples can be found within the 
[functional tests](../../test/pms), 
the [benchmarks](../../benchmark) 
measuring `CPU/OpenMP` vs `CUDA` performance
and a [notebook](muon_dcs_calc.ipynb) 
documenting differential cross-sections calculations.
However, no transport functionality is available yet.

In `noa::pms::pumas` however, we are actively developing a whole frontend for 
[pumas v1.1](https://github.com/niess/pumas/releases/tag/v1.1)
which will also contain sensitivity analysis functionality. 

In the near future, we plan to cover 
a wide range of particles. 

## Acknowledgements

The semi-analytical BMC algorithm for Muons and Taus propagation relies on 
[pumas](https://github.com/niess/pumas).

To load the MDF settings we rely on 
[pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

(c) 2021 GrinisRIT ltd. and contributors
