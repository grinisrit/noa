if( BUILD_CUDA )
   cuda_add_executable( tnl-benchmark-distributed-spmv-cuda tnl-benchmark-distributed-spmv.cu )

   install( TARGETS tnl-benchmark-distributed-spmv-cuda RUNTIME DESTINATION bin )
endif()

add_executable( tnl-benchmark-distributed-spmv tnl-benchmark-distributed-spmv.cpp )

install( TARGETS tnl-benchmark-distributed-spmv RUNTIME DESTINATION bin )
