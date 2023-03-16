ADD_EXECUTABLE( DirectionsTest DirectionsTest.cpp )
   TARGET_COMPILE_OPTIONS( DirectionsTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DirectionsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( DirectionsTest PUBLIC ${TESTS_LINKER_FLAGS} )

ADD_EXECUTABLE( CopyEntitesTest CopyEntitiesTest.cpp )
   TARGET_COMPILE_OPTIONS( CopyEntitesTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CopyEntitesTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( CopyEntitesTest PUBLIC ${TESTS_LINKER_FLAGS} )

# TODO: Fix distributed grid.
#ADD_EXECUTABLE( CutMeshFunctionTest CutMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutMeshFunctionTest PUBLIC ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutMeshFunctionTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
#   target_link_options( CutMeshFunctionTest PUBLIC ${TESTS_LINKER_FLAGS} )

ADD_TEST( NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( NAME CopyEntitesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitesTest${CMAKE_EXECUTABLE_SUFFIX} )
# ADD_TEST( NAME CutMeshFunctionTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )

if( TNL_BUILD_MPI )
   # TODO: Fix distributed grid.
   #ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
   #TARGET_COMPILE_OPTIONS( DistributedGridTest_1D PUBLIC ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( DistributedGridTest_1D PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   #target_link_options( DistributedGridTest_1D PUBLIC ${TESTS_LINKER_FLAGS} )

   #ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
   #TARGET_COMPILE_OPTIONS( DistributedGridTest_2D PUBLIC ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( DistributedGridTest_2D PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   #target_link_options( DistributedGridTest_2D PUBLIC ${TESTS_LINKER_FLAGS} )

   #ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
   #TARGET_COMPILE_OPTIONS( DistributedGridTest_3D PUBLIC ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( DistributedGridTest_3D PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   #target_link_options( DistributedGridTest_3D PUBLIC ${TESTS_LINKER_FLAGS} )

   #ADD_EXECUTABLE( CutDistributedGridTest CutDistributedGridTest.cpp )
   #TARGET_COMPILE_OPTIONS( CutDistributedGridTest PUBLIC ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( CutDistributedGridTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   #target_link_options( CutDistributedGridTest PUBLIC ${TESTS_LINKER_FLAGS} )

   #ADD_EXECUTABLE( CutDistributedMeshFunctionTest CutDistributedMeshFunctionTest.cpp )
   #TARGET_COMPILE_OPTIONS( CutDistributedMeshFunctionTest PUBLIC ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( CutDistributedMeshFunctionTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   #target_link_options( CutDistributedMeshFunctionTest PUBLIC ${TESTS_LINKER_FLAGS} )

if( TNL_BUILD_CUDA )
   add_executable( DistributedMeshTest DistributedMeshTest.cu )
   target_compile_options( DistributedMeshTest PUBLIC ${CUDA_TESTS_FLAGS} )
   target_link_libraries( DistributedMeshTest PUBLIC TNL::TNL_CUDA )
else()
   add_executable( DistributedMeshTest DistributedMeshTest.cpp )
   target_compile_options( DistributedMeshTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( DistributedMeshTest PUBLIC TNL::TNL_CXX )
endif()
target_link_libraries( DistributedMeshTest PUBLIC ${TESTS_LIBRARIES} )
target_link_options( DistributedMeshTest PUBLIC ${TESTS_LINKER_FLAGS} )

# TODO: Fix distributed grid.
#foreach( target IN ITEMS DistributedGridTest_1D DistributedGridTest_2D DistributedGridTest_3D CutDistributedGridTest DistributedMeshTest )
foreach( target IN ITEMS DistributedMeshTest )
   # enable MPI support in TNL
   target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
   # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
   target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )
endforeach()


# external libraries for tests which use mesh readers
find_package( ZLIB )
find_package( tinyxml2 QUIET )

if( ZLIB_FOUND AND tinyxml2_FOUND )
   # TODO: Fix distributed grid.
   # foreach( target IN ITEMS DistributedGridTest_1D DistributedGridTest_2D DistributedGridTest_3D DistributedMeshTest )
   foreach( target IN ITEMS DistributedMeshTest )
      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})

      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
   endforeach()
endif()


# TODO: Fix distributed grid
#SET (mpi_test_parameters_1d -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedGridTest_1D COMMAND "mpirun" ${mpi_test_parameters_1d})

#SET (mpi_test_parameters_2d -np 9 -H localhost:9 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedGridTest_2D COMMAND "mpirun" ${mpi_test_parameters_2d})

#SET (mpi_test_parameters_3d -np 27 -H localhost:27 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME DistributedGridTest_3D COMMAND "mpirun" ${mpi_test_parameters_3d})

#SET (mpi_test_parameters_CutDistributedGridTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME CutDistributedGridTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedGridTest})

# TODO: Fix
#SET (mpi_test_parameters_CutDistributedMeshFunctionTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME CutDistributedMeshFunctionTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedMeshFunctionTest})

add_test( NAME DistributedMeshTest_2x2 COMMAND mpirun -np 4 -H localhost:4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_3x3 COMMAND mpirun -np 9 -H localhost:9 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_4x4 COMMAND mpirun -np 16 -H localhost:16 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )

endif()
