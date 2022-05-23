#include "../exceptions.h"
#include "../typedefs.h"

// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Array.h"
#include "Vector.h"

// external functions
void export_Object( py::module & m );
void export_String( py::module & m );
void export_Grid1D( py::module & m );
void export_Grid2D( py::module & m );
void export_Grid3D( py::module & m );
void export_VTKTraits( py::module & m );
void export_Meshes( py::module & m );
void export_MeshReaders( py::module & m );
void export_MeshWriters( py::module & m );
void export_SparseMatrices( py::module & m );

template< typename T >
using _array = TNL::Containers::Array< T, TNL::Devices::Host, IndexType >;

template< typename T >
using _vector = TNL::Containers::Vector< T, TNL::Devices::Host, IndexType >;

// Python module definition
PYBIND11_MODULE(PYTNL_MODULE_NAME(tnl), m)
{
    register_exceptions(m);

    export_Object(m);
    // TODO: TNL::File
    export_String(m);

    export_Array< _array<double> >(m, "Array");
    export_Vector< _array<double>, _vector<double> >(m, "Vector");
    export_Array< _array<int> >(m, "Array_int");
    export_Vector< _array<int>, _vector<int> >(m, "Vector_int");
    export_Array< _array<bool> >(m, "Array_bool");

    export_Grid1D(m);
    export_Grid2D(m);
    export_Grid3D(m);

    export_VTKTraits(m);

    export_Meshes(m);
    export_MeshReaders(m);
    export_MeshWriters(m);

    export_SparseMatrices(m);
}
