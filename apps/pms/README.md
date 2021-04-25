# Passage of Particles through Matter Simulations (PMS) 

:warning: This component is under active development.

To build theses examples add `-DBUILD_PMS_APPS=ON` to the top level cmake command.

The materials configuration files for the physical models are available from [noa-pms-models](https://github.com/grinisrit/noa-pms-models).

To enable parallel execution of some of the algorithms you should link against `OpenMP`.

To load MDF configurations we rely on [pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

To build `CUDA` routines specify at `cmake` command line `-DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` (or the GPU architecture of your choice).

(c) 2021 Roland Grinis, GrinisRIT ltd.