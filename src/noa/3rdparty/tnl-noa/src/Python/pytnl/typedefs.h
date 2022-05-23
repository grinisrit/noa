#pragma once

// helper macros (the _NX variants are needed to expand macros in the arguments)
#define PYTNL_STRINGIFY(U) PYTNL_STRINGIFY_NX(U)
#define PYTNL_STRINGIFY_NX(U) #U

#define PYTNL_PPCAT(A, B) PYTNL_PPCAT_NX(A, B)
#define PYTNL_PPCAT_NX(A, B) A ## B

// the Python module name depends on the build type, this macro can be used to concatenate with the correct suffix
#ifdef PYTNL_MODULE_POSTFIX
   #define PYTNL_MODULE_NAME(name) PYTNL_PPCAT(name, PYTNL_MODULE_POSTFIX)
#else
   #define PYTNL_MODULE_NAME(name) name
#endif

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>

using RealType = double;
using DeviceType = TNL::Devices::Host;
using IndexType = int;

using Grid1D = TNL::Meshes::Grid<1, RealType, DeviceType, IndexType>;
using Grid2D = TNL::Meshes::Grid<2, RealType, DeviceType, IndexType>;
using Grid3D = TNL::Meshes::Grid<3, RealType, DeviceType, IndexType>;

using LocalIndexType = short int;
template< typename Topology >
using DefaultMeshTemplate = TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig<
                            Topology,
                            Topology::dimension,
                            RealType,
                            IndexType,
                            LocalIndexType > >;

using MeshOfEdges = DefaultMeshTemplate< TNL::Meshes::Topologies::Edge >;
using MeshOfTriangles = DefaultMeshTemplate< TNL::Meshes::Topologies::Triangle >;
using MeshOfQuadrangles = DefaultMeshTemplate< TNL::Meshes::Topologies::Quadrangle >;
using MeshOfTetrahedrons = DefaultMeshTemplate< TNL::Meshes::Topologies::Tetrahedron >;
using MeshOfHexahedrons = DefaultMeshTemplate< TNL::Meshes::Topologies::Hexahedron >;

using DistributedMeshOfEdges = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfEdges >;
using DistributedMeshOfTriangles = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfTriangles >;
using DistributedMeshOfQuadrangles = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfQuadrangles >;
using DistributedMeshOfTetrahedrons = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfTetrahedrons >;
using DistributedMeshOfHexahedrons = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfHexahedrons >;
