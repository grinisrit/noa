add_subdirectory( tnl )
if( BUILD_MPI )
   add_subdirectory( tnl_mpi )
endif()

install( DIRECTORY . DESTINATION "include/pytnl"
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )
