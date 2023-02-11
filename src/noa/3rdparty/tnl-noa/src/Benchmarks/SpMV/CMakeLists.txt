# CSR5 does not work properly yet:
#
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/9
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/10
#
# We can build it with TNL but it crashes with many CUDA errors. We should first check it
# with the original build.
#
#include( cmake/BuildCSR5.cmake )

# PETSc requires MPI
#if( TNL_BUILD_MPI AND TNL_USE_PETSC )
#   find_package( PETSc COMPONENTS CXX )
#endif()

#if( PETSC_FOUND )
#   message( "PETSC library found: ${PETSC_VERSION}")
#   set( PETSC_CXX_FLAGS -DHAVE_PETSC ${PETSC_DEFINITIONS} )
#   message( "   PETSC_INCLUDES  = ${PETSC_INCLUDES}" )
#   message( "   PETSC_CXX_FLAGS = ${PETSC_CXX_FLAGS}" )
#   message( "   PETSC_LIBRARIES = ${PETSC_LIBRARIES}" )
#endif()

if( TNL_BUILD_CUDA )
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cu )
   add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cu ${EXPLICIT_TEMPLATES} ReferenceFormats/LightSpMV-1.0/SpMV.cu ReferenceFormats/LightSpMV-1.0/SpMVCSR.cu )
   target_link_libraries( tnl-benchmark-spmv PUBLIC TNL::TNL_CUDA )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-spmv PUBLIC CUDA::cusparse )
else()
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cpp )
   add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cpp ${EXPLICIT_TEMPLATES} )
   target_link_libraries( tnl-benchmark-spmv PUBLIC TNL::TNL_CXX )
endif()
target_compile_options( tnl-benchmark-spmv PRIVATE ${PETSC_CXX_FLAGS} )
target_include_directories( tnl-benchmark-spmv PRIVATE ${PETSC_INCLUDES} )
target_link_libraries( tnl-benchmark-spmv ${PETSC_LIBRARIES} )

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
