# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-ndarray tnl-benchmark-ndarray.cpp )
   target_link_libraries( tnl-benchmark-ndarray PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-ndarray RUNTIME DESTINATION bin )

   add_executable( tnl-benchmark-ndarray-boundary tnl-benchmark-ndarray-boundary.cpp )
   target_link_libraries( tnl-benchmark-ndarray-boundary PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-ndarray-boundary RUNTIME DESTINATION bin )
endif()

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-ndarray-cuda tnl-benchmark-ndarray-cuda.cu )
   target_link_libraries( tnl-benchmark-ndarray-cuda PUBLIC TNL::TNL_CUDA )
   install( TARGETS tnl-benchmark-ndarray-cuda RUNTIME DESTINATION bin )

   add_executable( tnl-benchmark-ndarray-boundary-cuda tnl-benchmark-ndarray-boundary-cuda.cu )
   target_link_libraries( tnl-benchmark-ndarray-boundary-cuda PUBLIC TNL::TNL_CUDA )
   install( TARGETS tnl-benchmark-ndarray-boundary-cuda RUNTIME DESTINATION bin )
endif()
