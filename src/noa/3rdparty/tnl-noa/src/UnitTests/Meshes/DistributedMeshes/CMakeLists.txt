ADD_EXECUTABLE( DirectionsTest DirectionsTest.cpp )
   TARGET_COMPILE_OPTIONS( DirectionsTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DirectionsTest ${TESTS_LIBRARIES} )
   target_link_options( DirectionsTest PRIVATE ${TESTS_LINKER_FLAGS} )

ADD_EXECUTABLE( CopyEntitesTest CopyEntitiesTest.cpp )
   TARGET_COMPILE_OPTIONS( CopyEntitesTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CopyEntitesTest ${TESTS_LIBRARIES} )
   target_link_options( CopyEntitesTest PRIVATE ${TESTS_LINKER_FLAGS} )

# FIXME: DistributedGrid refucktoring
#ADD_EXECUTABLE( CutMeshFunctionTest CutMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutMeshFunctionTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutMeshFunctionTest ${TESTS_LIBRARIES} )
#   target_link_options( CutMeshFunctionTest PRIVATE ${TESTS_LINKER_FLAGS} )

ADD_TEST( NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( NAME CopyEntitesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitesTest${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( NAME CutMeshFunctionTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )

if( BUILD_MPI )
ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_1D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_1D ${TESTS_LIBRARIES} )
   target_link_options( DistributedGridTest_1D PRIVATE ${TESTS_LINKER_FLAGS} )

ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_2D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_2D ${TESTS_LIBRARIES} )
   target_link_options( DistributedGridTest_2D PRIVATE ${TESTS_LINKER_FLAGS} )

ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
   TARGET_COMPILE_OPTIONS( DistributedGridTest_3D PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( DistributedGridTest_3D ${TESTS_LIBRARIES} )
   target_link_options( DistributedGridTest_3D PRIVATE ${TESTS_LINKER_FLAGS} )

ADD_EXECUTABLE( CutDistributedGridTest CutDistributedGridTest.cpp )
   TARGET_COMPILE_OPTIONS( CutDistributedGridTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( CutDistributedGridTest ${TESTS_LIBRARIES} )
   target_link_options( CutDistributedGridTest PRIVATE ${TESTS_LINKER_FLAGS} )

#ADD_EXECUTABLE( CutDistributedMeshFunctionTest CutDistributedMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutDistributedMeshFunctionTest PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutDistributedMeshFunctionTest ${TESTS_LIBRARIES} )
#   target_link_options( CutDistributedMeshFunctionTest PRIVATE ${TESTS_LINKER_FLAGS} )

if( BUILD_CUDA )
   cuda_add_executable( DistributedMeshTest DistributedMeshTest.cu OPTIONS ${CUDA_TESTS_FLAGS})
else()
   add_executable( DistributedMeshTest DistributedMeshTest.cpp )
   target_compile_options( DistributedMeshTest PRIVATE ${CXX_TESTS_FLAGS} )
endif()
target_link_libraries( DistributedMeshTest ${TESTS_LIBRARIES} )
target_link_options( DistributedMeshTest PRIVATE ${TESTS_LINKER_FLAGS} )


# external libraries for tests which use mesh readers
find_package( ZLIB )
find_package( tinyxml2 QUIET )

if( ZLIB_FOUND AND tinyxml2_FOUND )
   foreach( target IN ITEMS DistributedGridTest_1D DistributedGridTest_2D DistributedGridTest_3D DistributedMeshTest )
      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} ${ZLIB_LIBRARIES})

      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} tinyxml2::tinyxml2)
   endforeach()
endif()


SET (mpi_test_parameters_1d -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_1D COMMAND "mpirun" ${mpi_test_parameters_1d})

SET (mpi_test_parameters_2d -np 9 -H localhost:9 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_2D COMMAND "mpirun" ${mpi_test_parameters_2d})

SET (mpi_test_parameters_3d -np 27 -H localhost:27 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME DistributedGridTest_3D COMMAND "mpirun" ${mpi_test_parameters_3d})

SET (mpi_test_parameters_CutDistributedGridTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}")
ADD_TEST( NAME CutDistributedGridTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedGridTest})

# TODO: Fix
#SET (mpi_test_parameters_CutDistributedMeshFunctionTest -np 12 -H localhost:12 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX}")
#ADD_TEST( NAME CutDistributedMeshFunctionTest COMMAND "mpirun" ${mpi_test_parameters_CutDistributedMeshFunctionTest})

add_test( NAME DistributedMeshTest_2x2 COMMAND mpirun -np 4 -H localhost:4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_3x3 COMMAND mpirun -np 9 -H localhost:9 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )
add_test( NAME DistributedMeshTest_4x4 COMMAND mpirun -np 16 -H localhost:16 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX} )

endif()
