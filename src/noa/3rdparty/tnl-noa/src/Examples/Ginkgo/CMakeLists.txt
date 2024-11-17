# workaround for https://github.com/ginkgo-project/ginkgo/issues/1096
if( NOT Ginkgo_FOUND )
   find_package( Ginkgo QUIET )
endif()

set( CPP_EXAMPLES
         tnl-ginkgo-converted-ellpack
         tnl-ginkgo-wrapped-csr
         tnl-ginkgo-wrapped-operator
         tnl-ginkgo-PoissonEquation2D
         tnl-ginkgo-PoissonEquation3D
)
set( CUDA_EXAMPLES
         tnl-ginkgo-converted-ellpack-cuda
         tnl-ginkgo-wrapped-csr-cuda
         tnl-ginkgo-wrapped-operator-cuda
         tnl-ginkgo-PoissonEquation2D-cuda
         tnl-ginkgo-PoissonEquation3D-cuda
)

if( Ginkgo_FOUND )
   foreach( target IN ITEMS ${CPP_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} TNL::TNL_CXX Ginkgo::ginkgo )
      install( TARGETS ${target} RUNTIME DESTINATION bin )
   endforeach()
   if( TNL_BUILD_CUDA )
      foreach( target IN ITEMS ${CUDA_EXAMPLES} )
         add_executable( ${target} ${target}.cu )
         target_link_libraries( ${target} TNL::TNL_CUDA Ginkgo::ginkgo )
         install( TARGETS ${target} RUNTIME DESTINATION bin )
      endforeach()
   endif()
endif()
