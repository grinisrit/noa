ADD_SUBDIRECTORY( DistributedMeshes )

ADD_EXECUTABLE( EntityTagsTest EntityTagsTest.cpp )
TARGET_COMPILE_OPTIONS( EntityTagsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( EntityTagsTest ${TESTS_LIBRARIES} )
target_link_options( EntityTagsTest PRIVATE ${TESTS_LINKER_FLAGS} )

# Mesh cannot be compiled by nvcc < 9 due to bugs in the compiler
if( ${BUILD_CUDA} AND ${CUDA_VERSION_MAJOR} GREATER_EQUAL 9 )
   CUDA_ADD_EXECUTABLE( MeshTest MeshTest.cu
                        OPTIONS ${CUDA_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest ${TESTS_LIBRARIES} )
   target_link_options( MeshTest PRIVATE ${TESTS_LINKER_FLAGS} )

   CUDA_ADD_EXECUTABLE( MeshTraverserTest MeshTraverserTest.cu
                        OPTIONS ${CUDA_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTraverserTest ${TESTS_LIBRARIES} )
   target_link_options( MeshTraverserTest PRIVATE ${TESTS_LINKER_FLAGS} )

   CUDA_ADD_EXECUTABLE( MeshOrderingTest MeshOrderingTest.cu
                        OPTIONS ${CUDA_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest ${TESTS_LIBRARIES} )
   target_link_options( MeshOrderingTest PRIVATE ${TESTS_LINKER_FLAGS} )

   CUDA_ADD_EXECUTABLE( MeshGeometryTest MeshGeometryTest.cu
                        OPTIONS ${CUDA_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshGeometryTest ${TESTS_LIBRARIES} )
   target_link_options( MeshGeometryTest PRIVATE ${TESTS_LINKER_FLAGS} )
else()
   ADD_EXECUTABLE( MeshTest MeshTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest ${TESTS_LIBRARIES} )
   target_link_options( MeshTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( MeshTraverserTest MeshTraverserTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshTraverserTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTraverserTest ${TESTS_LIBRARIES} )
   target_link_options( MeshTraverserTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( MeshOrderingTest MeshOrderingTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshOrderingTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest ${TESTS_LIBRARIES} )
   target_link_options( MeshOrderingTest PRIVATE ${TESTS_LINKER_FLAGS} )

   ADD_EXECUTABLE( MeshGeometryTest MeshGeometryTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshGeometryTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshGeometryTest ${TESTS_LIBRARIES} )
   target_link_options( MeshGeometryTest PRIVATE ${TESTS_LINKER_FLAGS} )
endif()

ADD_TEST( EntityTagsTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/EntityTagsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshTraverserTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshTraverserTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshOrderingTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshOrderingTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshGeometryTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MeshGeometryTest${CMAKE_EXECUTABLE_SUFFIX} )

# special tests needing external libraries
find_package( ZLIB )
find_package( tinyxml2 QUIET )

if( ZLIB_FOUND AND tinyxml2_FOUND )
   foreach( target IN ITEMS NetgenReaderTest VTKReaderTest VTUReaderTest VTIReaderTest FPMAReaderTest )
      add_executable(${target} ${target}.cpp)
      target_compile_options(${target} PRIVATE ${CXX_TESTS_FLAGS} )
      target_link_libraries(${target} ${TESTS_LIBRARIES})
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )

      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} ${ZLIB_LIBRARIES})

      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} tinyxml2::tinyxml2)

      # configure path to the data directory
      target_compile_definitions(${target} PUBLIC "-DTNL_MESH_TESTS_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

      add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
   endforeach()
endif()
