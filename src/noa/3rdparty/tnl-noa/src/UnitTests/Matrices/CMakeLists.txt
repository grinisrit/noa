ADD_SUBDIRECTORY( Legacy )

set( COMMON_TESTS
            DenseMatrixTest
            DenseMatrixCopyTest
            TridiagonalMatrixTest
            MultidiagonalMatrixTest
            SparseMatrixTest_CSRScalar
            SparseMatrixTest_CSRVector
            SparseMatrixTest_CSRHybrid
            SparseMatrixTest_CSRLight
            SparseMatrixTest_CSRAdaptive
            SparseMatrixTest_Ellpack
            SparseMatrixTest_SlicedEllpack
            SparseMatrixTest_ChunkedEllpack
            SparseMatrixTest_BiEllpack
            SparseMatrixVectorProductTest_CSRScalar
            SparseMatrixVectorProductTest_CSRVector
            SparseMatrixVectorProductTest_CSRHybrid
            SparseMatrixVectorProductTest_CSRLight
            SparseMatrixVectorProductTest_CSRAdaptive
            SparseMatrixVectorProductTest_Ellpack
            SparseMatrixVectorProductTest_SlicedEllpack
            SparseMatrixVectorProductTest_ChunkedEllpack
            SparseMatrixVectorProductTest_BiEllpack
            SparseMatrixCopyTest
            BinarySparseMatrixTest_CSR
            BinarySparseMatrixTest_Ellpack
            BinarySparseMatrixTest_SlicedEllpack
            BinarySparseMatrixCopyTest
            SymmetricSparseMatrixTest_CSR
            LambdaMatrixTest
            SparseMatrixTest_SandboxMatrix
            SparseMatrixVectorProductTest_SandboxMatrix
            MatrixWrappingTest
)

set( CPP_TESTS
            StaticMatrixTest
)
set( CUDA_TESTS )
if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      add_executable( ${target} ${target}.cu )
      target_compile_options( ${target} PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()

if( TNL_BUILD_MPI )
   if( TNL_BUILD_CUDA )
      add_executable( DistributedMatrixTest DistributedMatrixTest.cu )
      target_compile_options( DistributedMatrixTest PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( DistributedMatrixTest PUBLIC TNL::TNL_CUDA )
   else()
      add_executable( DistributedMatrixTest DistributedMatrixTest.cpp )
      target_compile_options( DistributedMatrixTest PUBLIC ${CXX_TESTS_FLAGS} )
      target_link_libraries( DistributedMatrixTest PUBLIC TNL::TNL_CXX )
   endif()
   target_link_libraries( DistributedMatrixTest PUBLIC ${TESTS_LIBRARIES} )
   target_link_options( DistributedMatrixTest PUBLIC ${TESTS_LINKER_FLAGS} )

   # enable MPI support in TNL
   target_compile_definitions( DistributedMatrixTest PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( DistributedMatrixTest PUBLIC MPI::MPI_CXX )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedMatrixTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedMatrixTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
endif()
