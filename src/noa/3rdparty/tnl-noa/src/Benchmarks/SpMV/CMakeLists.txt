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
#if( ( ${BUILD_MPI} ) AND ( NOT ${WITH_PETSC} EQUAL "no" ) )
#   find_package( PETSc COMPONENTS CXX )
#endif()

#if( PETSC_FOUND )
#   message( "PETSC library found: ${PETSC_VERSION}")
#   set( PETSC_CXX_FLAGS -DHAVE_PETSC ${PETSC_DEFINITIONS} )
#   message( "   PETSC_INCLUDES  = ${PETSC_INCLUDES}" )
#   message( "   PETSC_CXX_FLAGS = ${PETSC_CXX_FLAGS}" )
#   message( "   PETSC_LIBRARIES = ${PETSC_LIBRARIES}" )
#endif()

if( BUILD_CUDA )
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cu )
   cuda_include_directories( ${CXX_BENCHMARKS_INCLUDE_DIRS} )
   cuda_add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cu ${EXPLICIT_TEMPLATES} ReferenceFormats/LightSpMV-1.0/SpMV.cu ReferenceFormats/LightSpMV-1.0/SpMVCSR.cu
                        OPTIONS ${CXX_BENCHMARKS_FLAGS} ${PETSC_CXX_FLAGS} )
   target_link_libraries( tnl-benchmark-spmv ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
else()
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cpp )
   add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cpp ${EXPLICIT_TEMPLATES} )
   target_compile_options( tnl-benchmark-spmv  PRIVATE ${CXX_BENCHMARKS_FLAGS} ${PETSC_CXX_FLAGS} )
   target_include_directories( tnl-benchmark-spmv PRIVATE ${CXX_BENCHMARKS_INCLUDE_DIRS} )
endif()
set_property( TARGET tnl-benchmark-spmv APPEND PROPERTY INCLUDE_DIRECTORIES ${PETSC_INCLUDES} )
target_link_libraries( tnl-benchmark-spmv ${PETSC_LIBRARIES} )

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
