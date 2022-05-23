if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-benchmark-ode-solvers-cuda tnl-benchmark-ode-solvers.cu
                        OPTIONS -DHAVE_CUSPARSE )
   TARGET_LINK_LIBRARIES( tnl-benchmark-ode-solvers-cuda ${CUDA_cusparse_LIBRARY} )

   install( TARGETS tnl-benchmark-ode-solvers-cuda RUNTIME DESTINATION bin )
endif()

ADD_EXECUTABLE( tnl-benchmark-ode-solvers tnl-benchmark-ode-solvers.cpp )

install( TARGETS tnl-benchmark-ode-solvers RUNTIME DESTINATION bin )
