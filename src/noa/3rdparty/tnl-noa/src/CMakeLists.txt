add_subdirectory( 3rdparty )

install( DIRECTORY TNL/ DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

# Note that it is important to start building examples as soon as possible,
# because they take the longest time and other stuff can be pipelined before
# they are finished (at least with Ninja).
if( ${BUILD_EXAMPLES} )
   add_subdirectory( Examples )
endif()

if( ${BUILD_BENCHMARKS} )
   add_subdirectory( Benchmarks )
endif()

if( ${BUILD_TOOLS} )
   add_subdirectory( Tools )
endif()

if( ${BUILD_TESTS} )
   add_subdirectory( UnitTests )
endif()
if( ${BUILD_MATRIX_TESTS} )
   add_subdirectory( UnitTests/Matrices )
endif()

if( ${BUILD_PYTHON} )
   add_subdirectory( Python )
endif()
