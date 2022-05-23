// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Mesh.h"

void export_Meshes( py::module & m )
{
    export_Mesh< MeshOfEdges >( m, "MeshOfEdges" );
    export_Mesh< MeshOfTriangles >( m, "MeshOfTriangles" );
    export_Mesh< MeshOfQuadrangles >( m, "MeshOfQuadrangles" );
    export_Mesh< MeshOfTetrahedrons >( m, "MeshOfTetrahedrons" );
    export_Mesh< MeshOfHexahedrons >( m, "MeshOfHexahedrons" );
}
