# TODO: Split the benchmark into several files for faster build

if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cu )
else()
    ADD_EXECUTABLE( tnl-benchmark-traversers tnl-benchmark-traversers.cpp )
endif()

install( TARGETS tnl-benchmark-traversers RUNTIME DESTINATION bin )
