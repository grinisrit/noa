// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "MeshReaders.h"
#include "../typedefs.h"

void export_MeshReaders( py::module & m )
{
    using MeshReader = TNL::Meshes::Readers::MeshReader;
    using XMLVTK = TNL::Meshes::Readers::XMLVTK;

    // base class with trampolines for virtual methods
    py::class_< MeshReader, PyMeshReader >( m, "MeshReader" )
        .def(py::init<std::string>())
        // bindings against the actual class, NOT the trampoline
        .def("reset", &MeshReader::reset)
        .def("detectMesh", &MeshReader::detectMesh)
        .def("loadMesh", &MeshReader::template loadMesh< Grid1D >)
        .def("loadMesh", &MeshReader::template loadMesh< Grid2D >)
        .def("loadMesh", &MeshReader::template loadMesh< Grid3D >)
        .def("loadMesh", &MeshReader::template loadMesh< MeshOfEdges >)
        .def("loadMesh", &MeshReader::template loadMesh< MeshOfTriangles >)
        .def("loadMesh", &MeshReader::template loadMesh< MeshOfQuadrangles >)
        .def("loadMesh", &MeshReader::template loadMesh< MeshOfTetrahedrons >)
        .def("loadMesh", &MeshReader::template loadMesh< MeshOfHexahedrons >)
        .def("readPointData", &MeshReader::readPointData)
        .def("readCellData", &MeshReader::readCellData)
    ;

    py::class_< TNL::Meshes::Readers::VTKReader, MeshReader >( m, "VTKReader" )
        .def(py::init<std::string>())
    ;

    // base class for VTUReader, VTIReader and PVTUReader
    py::class_< XMLVTK, PyXMLVTK, MeshReader >( m, "XMLVTK" )
        .def(py::init<std::string>())
   ;

    py::class_< TNL::Meshes::Readers::VTUReader, XMLVTK >( m, "VTUReader" )
        .def(py::init<std::string>())
    ;

    py::class_< TNL::Meshes::Readers::VTIReader, XMLVTK >( m, "VTIReader" )
        .def(py::init<std::string>())
    ;
}
