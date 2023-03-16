if( TNL_BUILD_MPI )
   find_package(Hypre)
   if( HYPRE_FOUND )
      foreach( source IN ITEMS hypre-ex5.c tnl-hypre-ex5.cpp tnl-hypre-PoissonEquation3D.cpp )
         string( REGEX REPLACE "\.cpp|\.c" "" target ${source} )
         add_executable( ${target} ${source} )
         target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )

         # enable MPI support in TNL
         target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
         # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
         target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )

         # enable Hypre support in TNL
         target_compile_definitions( ${target} PUBLIC "-DHAVE_HYPRE" )
         # add Hypre to the target
         target_include_directories( ${target} SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS} )
         target_link_libraries( ${target} PUBLIC ${HYPRE_LIBRARIES} -lm )

         install( TARGETS ${target} RUNTIME DESTINATION bin )
      endforeach()
   endif()
endif()
