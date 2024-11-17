set( CPP_TESTS  NDSubarrayTest SlicedNDArrayTest StaticNDArrayTest )
set( CUDA_TESTS  StaticNDArrayCudaTest )
if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} NDArrayTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} NDArrayTest )
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
   set( MPI_TESTS
            DistributedNDArray_1D_test
            DistributedNDArray_semi1D_test
            DistributedNDArrayOverlaps_1D_test
            DistributedNDArrayOverlaps_semi1D_test
   )
   if( TNL_BUILD_CUDA )
      foreach( target IN ITEMS ${MPI_TESTS} )
         add_executable( ${target} ${target}.cu )
         target_compile_options( ${target} PUBLIC ${CUDA_TESTS_FLAGS} )
         target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      endforeach()
   else()
      foreach( target IN ITEMS ${MPI_TESTS} )
         add_executable( ${target} ${target}.cpp )
         target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
         target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
      endforeach()
   endif()

   foreach( target IN ITEMS ${MPI_TESTS} )
      target_link_libraries( ${target} PUBLIC ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

      # enable MPI support in TNL
      target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )

      set( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" )
      add_test( NAME ${target} COMMAND "mpirun" ${mpi_test_parameters})
      add_test( NAME ${target}_nodistr COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" )
   endforeach()
endif()
