if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-linear-solvers-cuda tnl-benchmark-linear-solvers.cu )
   target_link_libraries( tnl-benchmark-linear-solvers-cuda PUBLIC TNL::TNL_CUDA )
   target_compile_definitions( tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_CUSPARSE" )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-linear-solvers-cuda PUBLIC CUDA::cusparse )
   install( TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-linear-solvers-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-linear-solvers-cuda PUBLIC MPI::MPI_CXX )
   endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp )
   target_link_libraries( tnl-benchmark-linear-solvers PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-linear-solvers RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-linear-solvers PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-linear-solvers PUBLIC MPI::MPI_CXX )
   endif()
endif()
