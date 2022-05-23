// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "../tnl/MeshWriters.h"
#include "../typedefs.h"

#include <TNL/Meshes/Readers/MeshReader.h>

#include <TNL/Meshes/Writers/PVTUWriter.h>

template< template<typename> class WriterTemplate, typename LocalMesh, TNL::Meshes::VTK::FileFormat default_format >
void export_DistributedMeshWriter( py::module & m, const char* name )
{
    using Writer = WriterTemplate< LocalMesh >;
    using Mesh = TNL::Meshes::DistributedMeshes::DistributedMesh< LocalMesh >;

    // We cannot use MeshReader::VariantVector for Python bindings, because its variants are
    // std::vector<T> for T in std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t,
    // std::uint32_t, std::int64_t, std::uint64_t, float and double. Python types do not map
    // nicely to C++ types, integers even have unlimited precision, pybind11 even checks if given
    // Python value fits into the C++ type when selecting the alternative for a scalar type, and
    // for containers like std::vector it merely selects the first possible type. For reference, see
    // https://github.com/pybind/pybind11/issues/1625#issuecomment-723499161
    using VariantVector = mpark::variant< std::vector< IndexType >, std::vector< RealType > >;

    // Binding to Writer directly is not possible, because the writer has a std::ostream attribute
    // which would reference the streambuf created by the type caster from the Python file-like object.
    // However, the streambuf would be destroyed as soon as the writer is constructed and control
    // returned to Python, so the following invokations would use an invalid object and segfault.
    // To solve this, we use a transient wrapper struct PyWriter which holds the streambuf in its own
    // ostream attribute and is initialized by a py::object to avoid type casting.
    using PythonWriter = PyWriter< Writer, default_format >;
    py::class_< PythonWriter >( m, name )
        .def(py::init<py::object, TNL::Meshes::VTK::FileFormat>(), py::keep_alive<1, 2>(),
              py::arg("stream"), py::pos_only(), py::arg("format") = default_format)
        .def("writeMetadata", &Writer::writeMetadata, py::kw_only(), py::arg("cycle") = -1, py::arg("time") = -1)
        .def("writeVertices", static_cast< void (Writer::*)(const Mesh&) >(&Writer::template writeEntities< 0 >),
              py::arg("distributedMesh"))
        .def("writeVertices", static_cast< void (Writer::*)(const LocalMesh&, unsigned, unsigned) >(&Writer::template writeEntities< 0 >),
              py::arg("localMesh"), py::arg("GhostLevel") = 0, py::arg("MinCommonVertices") = 0)
        .def("writeCells", static_cast< void (Writer::*)(const Mesh&) >(&Writer::template writeEntities<>),
              py::arg("distributedMesh"))
        .def("writeCells", static_cast< void (Writer::*)(const LocalMesh&, unsigned, unsigned) >(&Writer::template writeEntities<>),
              py::arg("localMesh"), py::arg("GhostLevel") = 0, py::arg("MinCommonVertices") = 0)
        // INCONSISTENCY: the C++ methods writePPointData, writePCellData, writePDataArray do not
        // take the whole array as parameter, only the ValueType as a template parameter. Since
        // this does not map nicely to Python, we pass the whole array just like in the
        // VTKWriter and VTUWriter classes.
        // we use the VariantVector from MeshReader because we already have a caster for it
        .def("writePPointData", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       using value_type = typename std::decay_t<decltype(array)>::value_type;
                       writer.template writePPointData< value_type >( name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
        .def("writePCellData", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       using value_type = typename std::decay_t<decltype(array)>::value_type;
                       writer.template writePCellData< value_type >( name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
        .def("writePDataArray", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       using value_type = typename std::decay_t<decltype(array)>::value_type;
                       writer.template writePDataArray< value_type >( name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
        // NOTE: only the overload intended for sequential writing is exported, because we don't
        // have type casters for MPI_Comm (ideally, it would be compatible with the mpi4py objects)
        .def("addPiece", static_cast< std::string (Writer::*)(const std::string&, unsigned) >( &Writer::addPiece ),
              py::arg("mainFileName"), py::arg("subdomainIndex"))
    ;
}

void export_DistributedMeshWriters( py::module & m )
{
    constexpr TNL::Meshes::VTK::FileFormat default_format = TNL::Meshes::VTK::FileFormat::zlib_compressed;
    export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfEdges,        default_format >( m, "PVTUWriter_MeshOfEdges" );
    export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfTriangles,    default_format >( m, "PVTUWriter_MeshOfTriangles" );
    export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfQuadrangles,    default_format >( m, "PVTUWriter_MeshOfQuadrangles" );
    export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfTetrahedrons, default_format >( m, "PVTUWriter_MeshOfTetrahedrons" );
    export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfHexahedrons, default_format >( m, "PVTUWriter_MeshOfHexahedrons" );
}
