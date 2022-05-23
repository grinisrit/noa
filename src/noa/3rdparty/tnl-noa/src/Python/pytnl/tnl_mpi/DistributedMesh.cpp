// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "../typedefs.h"
#include "DistributedMesh.h"
#include "../tnl/Array.h"

void export_DistributedMeshes( py::module & m )
{
    // make sure that bindings for the local meshes are available
    py::module_::import(PYTNL_STRINGIFY(PYTNL_MODULE_NAME(tnl)));

    export_DistributedMesh< DistributedMeshOfEdges >( m, "DistributedMeshOfEdges" );
    export_DistributedMesh< DistributedMeshOfTriangles >( m, "DistributedMeshOfTriangles" );
    export_DistributedMesh< DistributedMeshOfQuadrangles >( m, "DistributedMeshOfQuadrangles" );
    export_DistributedMesh< DistributedMeshOfTetrahedrons >( m, "DistributedMeshOfTetrahedrons" );
    export_DistributedMesh< DistributedMeshOfHexahedrons >( m, "DistributedMeshOfHexahedrons" );

    // export VTKTypesArrayType
    using VTKTypesArrayType = typename DistributedMeshOfEdges::VTKTypesArrayType;
    export_Array< VTKTypesArrayType >(m, "VTKTypesArrayType");
}
