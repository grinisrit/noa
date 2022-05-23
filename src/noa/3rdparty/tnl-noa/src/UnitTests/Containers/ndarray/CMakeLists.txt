set( CPP_TESTS  NDSubarrayTest SlicedNDArrayTest StaticNDArrayTest )
set( CUDA_TESTS  StaticNDArrayCudaTest )
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} NDArrayTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} NDArrayTest )
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
      CUDA_ADD_EXECUTABLE( DistributedNDArray_1D_test DistributedNDArray_1D_test.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArray_1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      CUDA_ADD_EXECUTABLE( DistributedNDArray_semi1D_test DistributedNDArray_semi1D_test.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_semi1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArray_semi1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      CUDA_ADD_EXECUTABLE( DistributedNDArrayOverlaps_1D_test DistributedNDArrayOverlaps_1D_test.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArrayOverlaps_1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      CUDA_ADD_EXECUTABLE( DistributedNDArrayOverlaps_semi1D_test DistributedNDArrayOverlaps_semi1D_test.cu
                           OPTIONS ${CUDA_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_semi1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArrayOverlaps_semi1D_test PRIVATE ${TESTS_LINKER_FLAGS} )
   else()
      ADD_EXECUTABLE( DistributedNDArray_1D_test DistributedNDArray_1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArray_1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArray_1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      ADD_EXECUTABLE( DistributedNDArray_semi1D_test DistributedNDArray_semi1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArray_semi1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArray_semi1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArray_semi1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      ADD_EXECUTABLE( DistributedNDArrayOverlaps_1D_test DistributedNDArrayOverlaps_1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArrayOverlaps_1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArrayOverlaps_1D_test PRIVATE ${TESTS_LINKER_FLAGS} )

      ADD_EXECUTABLE( DistributedNDArrayOverlaps_semi1D_test DistributedNDArrayOverlaps_semi1D_test.cpp )
      TARGET_COMPILE_OPTIONS( DistributedNDArrayOverlaps_semi1D_test PRIVATE ${CXX_TESTS_FLAGS} )
      TARGET_LINK_LIBRARIES( DistributedNDArrayOverlaps_semi1D_test ${TESTS_LIBRARIES} )
      target_link_options( DistributedNDArrayOverlaps_semi1D_test PRIVATE ${TESTS_LINKER_FLAGS} )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArray_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArray_1D_test COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedNDArray_1D_test_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArray_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArray_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArray_semi1D_test COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedNDArray_semi1D_test_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArray_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArrayOverlaps_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArrayOverlaps_1D_test COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedNDArrayOverlaps_1D_test_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArrayOverlaps_1D_test${CMAKE_EXECUTABLE_SUFFIX}" )

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArrayOverlaps_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME DistributedNDArrayOverlaps_semi1D_test COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME DistributedNDArrayOverlaps_semi1D_test_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedNDArrayOverlaps_semi1D_test${CMAKE_EXECUTABLE_SUFFIX}" )
endif()
