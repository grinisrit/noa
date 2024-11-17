set( CPP_TESTS  SaveAndLoadMeshfunctionTest )
set( CUDA_TESTS )
if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} BoundaryMeshFunctionTest MeshFunctionTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} BoundaryMeshFunctionTest MeshFunctionTest )
endif()

# special tests needing external libraries
find_package( ZLIB )
find_package( tinyxml2 QUIET )

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

   target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
   target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})

   target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)

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
