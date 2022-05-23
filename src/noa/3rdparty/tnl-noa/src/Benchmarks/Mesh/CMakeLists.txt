if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-benchmark-mesh-cuda tnl-benchmark-mesh.cu )
   
   find_package( tinyxml2 QUIET )
   if( tinyxml2_FOUND )
      target_compile_definitions(tnl-benchmark-mesh-cuda PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(tnl-benchmark-mesh-cuda tinyxml2::tinyxml2)
   endif()

   find_package( ZLIB )
   if( ZLIB_FOUND )
      target_compile_definitions(tnl-benchmark-mesh-cuda PUBLIC "-DHAVE_ZLIB")
      target_include_directories(tnl-benchmark-mesh-cuda PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(tnl-benchmark-mesh-cuda ${ZLIB_LIBRARIES})
   endif()
   
   install( TARGETS tnl-benchmark-mesh-cuda RUNTIME DESTINATION bin )
endif()

ADD_EXECUTABLE( tnl-benchmark-mesh tnl-benchmark-mesh.cpp )

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   target_compile_definitions(tnl-benchmark-mesh PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(tnl-benchmark-mesh tinyxml2::tinyxml2)
endif()

find_package( ZLIB )
if( ZLIB_FOUND )
   target_compile_definitions(tnl-benchmark-mesh PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl-benchmark-mesh PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl-benchmark-mesh ${ZLIB_LIBRARIES})
endif()

install( TARGETS tnl-benchmark-mesh RUNTIME DESTINATION bin )
