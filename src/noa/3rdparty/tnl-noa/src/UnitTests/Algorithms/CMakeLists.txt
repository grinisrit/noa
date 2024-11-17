ADD_SUBDIRECTORY( Segments )
ADD_SUBDIRECTORY( Sorting )

set( COMMON_TESTS
         containsTest
         MemoryOperationsTest
         MultireductionTest
         ParallelForTest
         staticForTest
         unrolledForTest
)

set( CPP_TESTS
         reduceTest
         scanTest
         SegmentedScanTest
)
set( CUDA_TESTS
         reduceTestCuda
         scanTestCuda
)
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
   add_executable( distributedScanTest distributedScanTest.cpp )
   target_compile_options( distributedScanTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( distributedScanTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( distributedScanTest PUBLIC ${TESTS_LINKER_FLAGS} )
   # enable MPI support in TNL
   target_compile_definitions( distributedScanTest PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( distributedScanTest PUBLIC MPI::MPI_CXX )

   if( TNL_BUILD_CUDA )
      add_executable( distributedScanTestCuda distributedScanTestCuda.cu )
      target_compile_options( distributedScanTestCuda PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( distributedScanTestCuda PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( distributedScanTestCuda PUBLIC ${TESTS_LINKER_FLAGS} )
      # enable MPI support in TNL
      target_compile_definitions( distributedScanTestCuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( distributedScanTestCuda PUBLIC MPI::MPI_CXX )
   endif()

   SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )
   ADD_TEST( NAME distributedScanTest COMMAND "mpirun" ${mpi_test_parameters})
   ADD_TEST( NAME distributedScanTest_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTest${CMAKE_EXECUTABLE_SUFFIX}" )

   if( TNL_BUILD_CUDA )
      SET( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
      ADD_TEST( NAME distributedScanTestCuda COMMAND "mpirun" ${mpi_test_parameters})
      ADD_TEST( NAME distributedScanTestCuda_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributedScanTestCuda${CMAKE_EXECUTABLE_SUFFIX}" )
   endif()
endif()
