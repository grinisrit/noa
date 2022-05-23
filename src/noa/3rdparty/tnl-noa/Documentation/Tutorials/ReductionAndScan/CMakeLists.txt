set( COMMON_EXAMPLES
     SumExample
     SumExampleWithFunctional
     ProductExample
     ScalarProductExample
     ScalarProductWithFunctionalExample
     MaximumNormExample
     ComparisonExample
     UpdateAndResidueExample
     MapReduceExample-1
     MapReduceExample-2
     MapReduceExample-3
     ReductionWithArgument
     ReductionWithArgumentWithFunctional
     inclusiveScanExample
     exclusiveScanExample
     inplaceInclusiveScanExample
     inplaceExclusiveScanExample
     SegmentedScanExample
)

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
   foreach( target IN ITEMS ${LONG_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      #add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      #set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
   foreach( target IN ITEMS ${LONG_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      #add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      #set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
endif()


IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunTutorialsReductionAndScanExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunTutorialsReductionAndScanExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
