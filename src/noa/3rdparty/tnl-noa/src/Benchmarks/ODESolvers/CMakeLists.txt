if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-ode-solvers-cuda tnl-benchmark-ode-solvers.cu )
   target_link_libraries( tnl-benchmark-ode-solvers-cuda PUBLIC TNL::TNL_CUDA )
   target_compile_definitions( tnl-benchmark-ode-solvers-cuda PUBLIC "-DHAVE_CUSPARSE" )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-ode-solvers-cuda PUBLIC CUDA::cusparse )
   install( TARGETS tnl-benchmark-ode-solvers-cuda RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-ode-solvers-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-ode-solvers-cuda PUBLIC MPI::MPI_CXX )
   endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-ode-solvers tnl-benchmark-ode-solvers.cpp )
   target_link_libraries( tnl-benchmark-ode-solvers PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-ode-solvers RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-ode-solvers PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-ode-solvers PUBLIC MPI::MPI_CXX )
   endif()
endif()
