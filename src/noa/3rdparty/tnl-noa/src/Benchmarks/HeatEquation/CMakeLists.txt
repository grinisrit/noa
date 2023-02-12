if( TNL_BUILD_CUDA )
   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cu )
   target_link_libraries( tnl-benchmark-heat-equation PUBLIC TNL::TNL_CUDA )
   # NOTE: the following parameters are just for performance debugging
   # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#printing-code-generation-statistics
   #target_compile_options( tnl-benchmark-heat-equation PUBLIC --resource-usage )
   #set_target_properties( tnl-benchmark-heat-equation PROPERTIES CUDA_SEPARABLE_COMPILATION OFF )
else()
   add_executable( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )
   target_link_libraries( tnl-benchmark-heat-equation PUBLIC TNL::TNL_CXX )
endif()

install( TARGETS tnl-benchmark-heat-equation RUNTIME DESTINATION bin )
