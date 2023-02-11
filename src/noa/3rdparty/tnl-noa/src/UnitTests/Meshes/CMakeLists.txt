ADD_SUBDIRECTORY( DistributedMeshes )
ADD_SUBDIRECTORY( Grids )

ADD_EXECUTABLE( EntityTagsTest EntityTagsTest.cpp )
TARGET_COMPILE_OPTIONS( EntityTagsTest PUBLIC ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( EntityTagsTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
target_link_options( EntityTagsTest PUBLIC ${TESTS_LINKER_FLAGS} )

if( TNL_BUILD_CUDA )
   add_executable( MeshTest MeshTest.cu )
   target_compile_options( MeshTest PUBLIC ${CUDA_TESTS_FLAGS} )
   target_link_libraries( MeshTest PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
   target_link_options( MeshTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshIterationTest MeshIterationTest.cu )
   target_compile_options( MeshIterationTest PUBLIC ${CUDA_TESTS_FLAGS} )
   target_link_libraries( MeshIterationTest PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
   target_link_options( MeshIterationTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshOrderingTest MeshOrderingTest.cu )
   target_compile_options( MeshOrderingTest PUBLIC ${CUDA_TESTS_FLAGS} )
   target_link_libraries( MeshOrderingTest PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
   target_link_options( MeshOrderingTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshGeometryTest MeshGeometryTest.cu )
   target_compile_options( MeshGeometryTest PUBLIC ${CUDA_TESTS_FLAGS} )
   target_link_libraries( MeshGeometryTest PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
   target_link_options( MeshGeometryTest PUBLIC ${TESTS_LINKER_FLAGS} )
else()
   add_executable( MeshTest MeshTest.cpp )
   target_compile_options( MeshTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( MeshTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( MeshTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshIterationTest MeshIterationTest.cpp )
   target_compile_options( MeshIterationTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( MeshIterationTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( MeshIterationTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshOrderingTest MeshOrderingTest.cpp )
   target_compile_options( MeshOrderingTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( MeshOrderingTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( MeshOrderingTest PUBLIC ${TESTS_LINKER_FLAGS} )

   add_executable( MeshGeometryTest MeshGeometryTest.cpp )
   target_compile_options( MeshGeometryTest PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( MeshGeometryTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( MeshGeometryTest PUBLIC ${TESTS_LINKER_FLAGS} )
endif()

ADD_TEST( EntityTagsTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/EntityTagsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshIterationTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshIterationTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshOrderingTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshOrderingTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshGeometryTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshGeometryTest${CMAKE_EXECUTABLE_SUFFIX} )

# special tests needing external libraries
find_package( ZLIB )
find_package( tinyxml2 QUIET )

if( ZLIB_FOUND AND tinyxml2_FOUND )
   foreach( target IN ITEMS NetgenReaderTest VTKReaderTest VTUReaderTest VTIReaderTest FPMAReaderTest )
      add_executable(${target} ${target}.cpp)
      target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS} )
      target_link_libraries(${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES})
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})

      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)

      # configure path to the data directory
      target_compile_definitions(${target} PUBLIC "-DTNL_MESH_TESTS_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

      add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
   endforeach()
endif()
