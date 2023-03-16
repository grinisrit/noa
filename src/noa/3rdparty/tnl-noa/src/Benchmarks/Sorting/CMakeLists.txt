if( DEFINED CUDA_SAMPLES_DIR )
   set( CUDA_SAMPLES_FLAGS "-I${CUDA_SAMPLES_DIR} -DHAVE_CUDA_SAMPLES")
   message( "   CUDA_SAMPLES_FLAGS = ${CUDA_SAMPLES_FLAGS}" )
else()
   message( WARNING "The CUDA_SAMPLES_DIR variable was not set, the sorting benchmark will be built without some reference algorithms.")
endif()

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-sort-cuda tnl-benchmark-sort.cu )
   target_link_libraries( tnl-benchmark-sort-cuda PUBLIC TNL::TNL_CUDA )
   # Source code of reference algorithms causes warnings which turn into errers
   # with CI/CD compiler flags. Therefore we use -Wno-error to turn it off.
   target_compile_options( tnl-benchmark-sort-cuda PRIVATE
         $<$<CUDA_COMPILER_ID:NVIDIA>:
               -Xcompiler ;
               -Wno-error=switch,-Wno-error=sign-compare
         >
         ${CUDA_SAMPLES_FLAGS}
   )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-sort-cuda PUBLIC CUDA::cusparse )
   install( TARGETS tnl-benchmark-sort-cuda RUNTIME DESTINATION bin )
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-sort tnl-benchmark-sort.cpp )
   target_link_libraries( tnl-benchmark-sort PUBLIC TNL::TNL_CXX )
   install( TARGETS tnl-benchmark-sort RUNTIME DESTINATION bin )
endif()
