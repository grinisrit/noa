ADD_SUBDIRECTORY( Segments )

set( COMMON_EXAMPLES
   SortingExample
   SortingExample2
   SortingExample3
   ParallelForExample
   SequentialForExample
)

set( HOST_EXAMPLES
   staticForExample
   unrolledForExample
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

foreach( target IN ITEMS ${HOST_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
   set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
endforeach()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunAlgorithmsExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} ${HOST_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunAlgorithmsExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
