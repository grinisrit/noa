# Passage of Particles through Matter Simulations (PMS) 

This library implements the adjoint sensitivity methods over backward Monte-Carlo schemes.
The primary focus is on particle physics simulations.

:warning: This component is under active development.

An introduction for differentiable programming for particle physics simulations is presented in the [notebook](differentiable_programming_pms.ipynb). Basic usage examples can be found in [functional tests](../../test/pms).

A [video workshop](https://www.youtube.com/watch?v=nJm_jbX6tJc)
has been recorded at the [QUARKS-2021](https://www.youtube.com/channel/UCXdL4IpBP3LqmUO2EqNCYxA) conference.

Physics model configuration files are available from 
[noa-pms-models](https://github.com/grinisrit/noa-pms-models). 
Currently, we support only Muons and Taus. In the near future, we plan to cover 
a wide range of particles. 

## Acknowledgements

The semi-analytical BMC algorithm for Muons and Taus propagation relies on 
[pumas](https://github.com/niess/pumas) version `1.0`.

To load the MDF settings we rely on 
[pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

(c) 2021 Roland Grinis, GrinisRIT ltd.
