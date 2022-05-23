ADD_SUBDIRECTORY( ndarray )

set( CPP_TESTS
         ArrayTest
         ArrayViewTest
         StaticArrayTest
         StaticVectorTest
         StaticVectorOperationsTest
         StaticVectorOfStaticVectorsTest
         VectorTest
         VectorEvaluateAndReduceTest
         VectorBinaryOperationsTest
         VectorUnaryOperationsTest
         VectorVerticalOperationsTest
         VectorOfStaticVectorsTest
)
set( CUDA_TESTS
         ArrayTestCuda
         ArrayViewTestCuda
         VectorTestCuda
         VectorEvaluateAndReduceTestCuda
         VectorBinaryOperationsTestCuda
         VectorUnaryOperationsTestCuda
         VectorVerticalOperationsTestCuda
         # FIXME: fails due to unspecified launch failure in the CUDA reduction kernel for scalar product,
         #        see https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/issues/82
         #VectorOfStaticVectorsTestCuda
)

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
      CUDA_ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTestCuda.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
   else()
      ADD_EXECUTABLE( DistributedArrayTest DistributedArrayTest.cpp )
      TARGET_COMPILE_OPTIONS( DistributedArrayTest PRIVATE ${CXX_TESTS_FLAGS} )
   endif()
   target_link_libraries( DistributedArrayTest ${TESTS_LIBRARIES} )
   target_link_options( DistributedArrayTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorBinaryOperationsTest DistributedVectorBinaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorBinaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTest ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorBinaryOperationsTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorUnaryOperationsTest DistributedVectorUnaryOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorUnaryOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTest ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorUnaryOperationsTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( DistributedVectorVerticalOperationsTest DistributedVectorVerticalOperationsTest.cpp )
   TARGET_COMPILE_OPTIONS( DistributedVectorVerticalOperationsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedVectorVerticalOperationsTest ${TESTS_LIBRARIES} )
   target_link_options( DistributedVectorVerticalOperationsTest PRIVATE ${TESTS_LINKER_FLAGS} )

   if( BUILD_CUDA )
      CUDA_ADD_EXECUTABLE( DistributedVectorBinaryOperationsTestCuda DistributedVectorBinaryOperationsTestCuda.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorBinaryOperationsTestCuda ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorBinaryOperationsTestCuda PRIVATE ${TESTS_LINKER_FLAGS} )

      CUDA_ADD_EXECUTABLE( DistributedVectorUnaryOperationsTestCuda DistributedVectorUnaryOperationsTestCuda.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorUnaryOperationsTestCuda ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorUnaryOperationsTestCuda PRIVATE ${TESTS_LINKER_FLAGS} )

      CUDA_ADD_EXECUTABLE( DistributedVectorVerticalOperationsTestCuda DistributedVectorVerticalOperationsTestCuda.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedVectorVerticalOperationsTestCuda ${TESTS_LIBRARIES} )
      target_link_options( DistributedVectorVerticalOperationsTestCuda PRIVATE ${TESTS_LINKER_FLAGS} )
   endif( BUILD_CUDA )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedArrayTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedArrayTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedArrayTest${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorBinaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedVectorBinaryOperationsTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorUnaryOperationsTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedVectorUnaryOperationsTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedVectorVerticalOperationsTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedVectorVerticalOperationsTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorBinaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME DistributedVectorBinaryOperationsTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorBinaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )

      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorUnaryOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME DistributedVectorUnaryOperationsTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorUnaryOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )

      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME DistributedVectorVerticalOperationsTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME DistributedVectorVerticalOperationsTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedVectorVerticalOperationsTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
