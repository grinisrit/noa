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
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${TESTS_LIBRARIES} )
   target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      cuda_add_executable( ${target} ${target}.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} ${TESTS_LIBRARIES} )
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()

if( ${BUILD_MPI} )
   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
   else()
      ADD_EXECUTABLE( DistributedMatrixTest DistributedMatrixTest.cpp )
      TARGET_COMPILE_OPTIONS( DistributedMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   endif()
   target_link_libraries( DistributedMatrixTest ${TESTS_LIBRARIES} )
   target_link_options( DistributedMatrixTest PRIVATE ${TESTS_LINKER_FLAGS} )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedMatrixTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedMatrixTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMatrixTest${CMAKE_EXECUTABLE_SUFFIX}" )
endif()
