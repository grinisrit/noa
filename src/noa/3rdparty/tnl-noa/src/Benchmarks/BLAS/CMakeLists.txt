# skip building host-only targets in CUDA-enabled CI jobs
if( TNL_BUILD_CPP_TARGETS )
   add_executable( tnl-benchmark-blas tnl-benchmark-blas.cpp )
   target_link_libraries( tnl-benchmark-blas PUBLIC TNL::TNL_CXX )
   set( BENCHMARKS tnl-benchmark-blas )
endif()

if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-blas-cuda tnl-benchmark-blas.cu )
   target_link_libraries( tnl-benchmark-blas-cuda PUBLIC TNL::TNL_CUDA )
   find_package( CUDAToolkit REQUIRED )
   target_link_libraries( tnl-benchmark-blas-cuda PUBLIC CUDA::cublas )
   set( BENCHMARKS ${BENCHMARKS} tnl-benchmark-blas-cuda )
endif()

find_library( CBLAS_LIBRARY NAMES cblas )

# fallback for Centos 7.5 - libcblas.so does not exist, link to libtatlas.so or libsatlas.so
# https://forums.centos.org/viewtopic.php?t=48543
find_library( TATLAS_LIBRARY NAMES tatlas
              PATH_SUFFIXES atlas )
find_library( SATLAS_LIBRARY NAMES satlas
              PATH_SUFFIXES atlas )

find_package( BLAS )

find_path(BLAS_INCLUDE_DIRS cblas.h
   /usr/include
   /usr/local/include
   $ENV{BLAS_HOME}/include
   /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers )

if(BLAS_INCLUDE_DIRS)
   foreach( target IN ITEMS ${BENCHMARKS} )
      target_include_directories( ${target} PUBLIC "${BLAS_INCLUDE_DIRS}" )
      if( CBLAS_LIBRARY )
            target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
            target_link_libraries( ${target} PUBLIC ${CBLAS_LIBRARY} )
      elseif( TATLAS_LIBRARY )
         target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
         target_link_libraries( ${target} PUBLIC ${TATLAS_LIBRARY} )
      elseif( SATLAS_LIBRARY )
         target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
         target_link_libraries( ${target} PUBLIC ${SATLAS_LIBRARY} )
      else()
         # FIXME: We require the CBLAS interface, but CMake's FindBLAS cannot detect that,
         #        so this fails unless the BLAS implementation includes it in the same
         #        shared library file as the Fortran implementation (e.g. OpenBLAS does that).
         if( BLAS_FOUND )
            target_compile_definitions( ${target} PUBLIC "-DHAVE_BLAS" )
            target_link_libraries( ${target} PUBLIC ${BLAS_LIBRARIES} )
         endif()
      endif()
      install( TARGETS ${target} RUNTIME DESTINATION bin )
   endforeach()
else()
   message( WARNING "Could not find BLAS headers (cblas.h) - benchmarks with BLAS will not be built.")
endif()

# find TBB (for C++17 parallel algorithms) https://stackoverflow.com/a/55989883
find_package( TBB )
if( TBB_FOUND )
   foreach( target IN ITEMS ${BENCHMARKS} )
      target_compile_definitions( ${target} PUBLIC -DHAVE_TBB )
      target_link_libraries( ${target} PUBLIC TBB::tbb )
   endforeach()
endif()

# find Thrust (provided by recent CUDA toolkits)
find_package( Thrust )
if( Thrust_FOUND )
   # https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
   thrust_create_target( ThrustHost HOST OMP DEVICE OMP )
   thrust_create_target( ThrustCuda HOST TBB DEVICE CUDA )
   foreach( target IN ITEMS ${BENCHMARKS} )
      target_compile_definitions( ${target} PUBLIC -DHAVE_THRUST )
   endforeach()
   if( TNL_BUILD_CPP_TARGETS )
      target_link_libraries( tnl-benchmark-blas PUBLIC ThrustHost )
   endif()
   if( TNL_BUILD_CUDA )
      target_link_libraries( tnl-benchmark-blas-cuda PUBLIC ThrustCuda )
   endif()
endif()
