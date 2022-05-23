if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers-cuda tnl-benchmark-linear-solvers.cu
                        OPTIONS -DHAVE_CUSPARSE )
   TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers-cuda ${CUDA_cusparse_LIBRARY} )

   install( TARGETS tnl-benchmark-linear-solvers-cuda RUNTIME DESTINATION bin )
endif()

ADD_EXECUTABLE( tnl-benchmark-linear-solvers tnl-benchmark-linear-solvers.cpp )

install( TARGETS tnl-benchmark-linear-solvers RUNTIME DESTINATION bin )
