if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-sort tnl-benchmark-sort.cu OPTIONS -Xcompiler -Wno-error=switch,-Wno-error=sign-compare ${CUDA_SAMPLES_FLAGS} )
    # Source code of reference algorithms contains warning which turn into errers with CI/CD compiler flags. Therefore we use -Wno-error to turn it off.
    TARGET_LINK_LIBRARIES( tnl-benchmark-sort ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
else()
    ADD_EXECUTABLE( tnl-benchmark-sort tnl-benchmark-sort.cpp )
endif()

install( TARGETS tnl-benchmark-sort RUNTIME DESTINATION bin )
