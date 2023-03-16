set( CPP_TESTS MPICommTest MPIUtilsTest )

if( TNL_BUILD_MPI )
foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

   # enable MPI support in TNL
   target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )

   foreach( np IN ITEMS 2 3 4 )
      set( mpirun_parameters -np ${np} -H localhost:${np} )
      add_test( NAME ${target}_np${np} COMMAND mpirun ${mpirun_parameters} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
   endforeach()
endforeach()
endif()
