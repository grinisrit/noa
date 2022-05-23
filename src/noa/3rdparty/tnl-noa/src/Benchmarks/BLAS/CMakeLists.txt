if( BUILD_CUDA )
   #find_library( CUDADEVRT NAMES cudadevrt )
   cuda_add_executable( tnl-benchmark-blas tnl-benchmark-blas.cu )
   cuda_add_cublas_to_target( tnl-benchmark-blas )
    #target_link_libraries( tnl-benchmark-blas ${CUDADEVRT} )#${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudadevrt.a )
else()
    add_executable( tnl-benchmark-blas tnl-benchmark-blas.cpp )
endif()

find_library( CBLAS_LIBRARY NAMES cblas )

# fallback for Centos 7.5 - libcblas.so does not exist, link to libtatlas.so or libsatlas.so
# https://forums.centos.org/viewtopic.php?t=48543
find_library( TATLAS_LIBRARY NAMES tatlas
              PATH_SUFFIXES atlas )
find_library( SATLAS_LIBRARY NAMES satlas
              PATH_SUFFIXES atlas )

if( CBLAS_LIBRARY )
   target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
   target_link_libraries( tnl-benchmark-blas ${CBLAS_LIBRARY} )
elseif( TATLAS_LIBRARY )
   target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
   target_link_libraries( tnl-benchmark-blas ${TATLAS_LIBRARY} )
elseif( SATLAS_LIBRARY )
   target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
   target_link_libraries( tnl-benchmark-blas ${SATLAS_LIBRARY} )
else()
   # FIXME: We require the CBLAS interface, but CMake's FindBLAS cannot detect that,
   #        so this fails unless the BLAS implementation includes it in the same
   #        shared library file as the Fortran implementation (e.g. OpenBLAS does that).
   find_package( BLAS )
   if( BLAS_FOUND )
      target_compile_definitions( tnl-benchmark-blas PUBLIC "-DHAVE_BLAS" )
      target_link_libraries( tnl-benchmark-blas ${BLAS_LIBRARIES} )
   endif()
endif()

install( TARGETS tnl-benchmark-blas RUNTIME DESTINATION bin )
