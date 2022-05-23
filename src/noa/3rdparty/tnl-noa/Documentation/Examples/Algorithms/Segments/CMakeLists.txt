set( COMMON_EXAMPLES
   SegmentsExample_General
   SegmentsPrintingExample-1
   SegmentsPrintingExample-2
   SegmentsExample_CSR_constructor_1
   SegmentsExample_CSR_constructor_2
   SegmentsExample_CSR_getSerializationType
   SegmentsExample_CSR_getSegmentsType
   SegmentsExample_CSR_setSegmentsSizes
   SegmentsExample_CSR_getSegmentView
   SegmentsExample_CSR_forElements
   SegmentsExample_CSR_forSegments
   SegmentsExample_CSR_sequentialForSegments
   SegmentsExample_CSR_reduceSegments
   SegmentsExample_forElements
   SegmentsExample_forSegments-1
   SegmentsExample_forSegments-2
   SegmentsExample_reduceSegments
)

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
endif()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunSegmentsExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunSegmentsExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()