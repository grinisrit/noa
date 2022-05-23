#include "../exceptions.h"
#include "../typedefs.h"

// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include <TNL/MPI/Wrappers.h>
#include <TNL/MPI/ScopedInitializer.h>

// external functions
void export_DistributedMeshes( py::module & m );
void export_DistributedMeshReaders( py::module & m );
void export_DistributedMeshWriters( py::module & m );

#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>

// Python module definition
PYBIND11_MODULE(PYTNL_MODULE_NAME(tnl_mpi), m)
{
    register_exceptions(m);

    // MPI initialization and finalization
    // https://stackoverflow.com/q/64647846
    if( ! TNL::MPI::Initialized() ) {
        int argc = 0;
        char** argv = nullptr;
        TNL::MPI::Init( argc, argv );
    }
    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#module-destructors
    auto cleanup_callback = []() {
        if( TNL::MPI::Initialized() && ! TNL::MPI::Finalized() )
            TNL::MPI::Finalize();
    };
    m.add_object("_cleanup", py::capsule(cleanup_callback));

    // bindings for distributed data structures
    export_DistributedMeshes(m);
    export_DistributedMeshReaders(m);
    export_DistributedMeshWriters(m);

    // bindings for functions
    using TNL::Meshes::DistributedMeshes::distributeSubentities;
    m.def("distributeFaces", []( DistributedMeshOfTriangles& mesh ) {
          distributeSubentities< 1 >( mesh ); });
    m.def("distributeFaces", []( DistributedMeshOfQuadrangles& mesh ) {
          distributeSubentities< 1 >( mesh ); });
    m.def("distributeFaces", []( DistributedMeshOfTetrahedrons& mesh ) {
          distributeSubentities< 2 >( mesh ); });
    m.def("distributeFaces", []( DistributedMeshOfHexahedrons& mesh ) {
          distributeSubentities< 2 >( mesh ); });
}
