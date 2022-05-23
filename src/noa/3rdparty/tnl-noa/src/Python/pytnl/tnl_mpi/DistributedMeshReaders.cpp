// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "../tnl/MeshReaders.h"
#include "../typedefs.h"

#include <TNL/Meshes/Readers/PVTUReader.h>

void export_DistributedMeshReaders( py::module & m )
{
    using XMLVTK = TNL::Meshes::Readers::XMLVTK;
    using PVTUReader = TNL::Meshes::Readers::PVTUReader;

    // make sure that bindings for the parent class are available
    py::module_::import(PYTNL_STRINGIFY(PYTNL_MODULE_NAME(tnl)));

    py::class_< PVTUReader, XMLVTK >( m, "PVTUReader" )
        .def(py::init<std::string>())
        // loadMesh is not virtual in PVTUReader
        .def("loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfEdges >)
        .def("loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTriangles >)
        .def("loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfQuadrangles >)
        .def("loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTetrahedrons >)
        .def("loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfHexahedrons >)
    ;
}
