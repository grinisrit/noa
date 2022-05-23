// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "MeshWriters.h"
#include "../typedefs.h"

#include <TNL/Meshes/Readers/MeshReader.h>

#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/VTIWriter.h>

template< typename Writer, TNL::Meshes::VTK::FileFormat default_format >
void export_MeshWriter( py::module & m, const char* name )
{
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
        .def("writeVertices", &Writer::template writeEntities< 0 >)
        .def("writeCells", &Writer::template writeEntities<>)
        // we use the VariantVector from MeshReader because we already have a caster for it
        .def("writePointData", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       // we need a view for the std::vector
                       using vector_t = std::decay_t<decltype(array)>;
                       using view_t = TNL::Containers::ArrayView< std::add_const_t< typename vector_t::value_type >, TNL::Devices::Host, std::int64_t >;
                       view_t view( array.data(), array.size() );
                       writer.writePointData( view, name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
        .def("writeCellData", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       // we need a view for the std::vector
                       using vector_t = std::decay_t<decltype(array)>;
                       using view_t = TNL::Containers::ArrayView< std::add_const_t< typename vector_t::value_type >, TNL::Devices::Host, std::int64_t >;
                       view_t view( array.data(), array.size() );
                       writer.writeCellData( view, name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
        .def("writeDataArray", []( PythonWriter& writer, const VariantVector& array, std::string name, int numberOfComponents = 1 ) {
               using mpark::visit;
               visit( [&](auto&& array) {
                       // we need a view for the std::vector
                       using vector_t = std::decay_t<decltype(array)>;
                       using view_t = TNL::Containers::ArrayView< std::add_const_t< typename vector_t::value_type >, TNL::Devices::Host, std::int64_t >;
                       view_t view( array.data(), array.size() );
                       writer.writeDataArray( view, name, numberOfComponents );
                   },
                   array
               );
            },
            py::arg("array"), py::arg("name"), py::arg("numberOfComponents") = 1)
    ;
}

void export_MeshWriters( py::module & m )
{
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid1D >, TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_Grid1D" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid1D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_Grid1D" );
    export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid1D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTIWriter_Grid1D" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid2D >, TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_Grid2D" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid2D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_Grid2D" );
    export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid2D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTIWriter_Grid2D" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid3D >, TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_Grid3D" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid3D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_Grid3D" );
    export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid3D >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTIWriter_Grid3D" );

    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfEdges >,        TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_MeshOfEdges" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfEdges >,        TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfEdges" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfTriangles >,    TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_MeshOfTriangles" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfTriangles >,    TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfTriangles" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfQuadrangles >,  TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_MeshOfQuadrangles" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfQuadrangles >,  TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfQuadrangles" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfTetrahedrons >, TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_MeshOfTetrahedrons" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfTetrahedrons >, TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfTetrahedrons" );
    export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfHexahedrons >,  TNL::Meshes::VTK::FileFormat::binary          >( m, "VTKWriter_MeshOfHexahedrons" );
    export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfHexahedrons >,  TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfHexahedrons" );
}
