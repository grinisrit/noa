if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-distributed-spmv-cuda tnl-benchmark-distributed-spmv.cu )
   target_link_libraries( tnl-benchmark-distributed-spmv-cuda PUBLIC TNL::TNL_CUDA )
   install( TARGETS tnl-benchmark-distributed-spmv-cuda RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-distributed-spmv-cuda PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-distributed-spmv-cuda PUBLIC MPI::MPI_CXX )
   endif()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-distributed-spmv tnl-benchmark-distributed-spmv.cpp )
   target_link_libraries( tnl-benchmark-distributed-spmv PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-distributed-spmv RUNTIME DESTINATION bin )

   if( TNL_BUILD_MPI )
      # enable MPI support in TNL
      target_compile_definitions( tnl-benchmark-distributed-spmv PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( tnl-benchmark-distributed-spmv PUBLIC MPI::MPI_CXX )
   endif()
endif()
