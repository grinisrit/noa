ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Pointers )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Solvers )

set( COMMON_EXAMPLES )

set( CUDA_EXAMPLES
   FileExampleCuda
)

set( HOST_EXAMPLES
   FileExample
   FileExampleSaveAndLoad
   FileNameExample
   FileNameExampleDistributedSystemNodeCoordinates
   FileNameExampleDistributedSystemNodeId
   ObjectExample_getType
   ParseObjectTypeExample
   StringExample
   StringExampleGetAllocatedSize
   StringExampleReplace
   StringExampleSplit
   StringExampleStrip
   TimerExample
   TimerExampleLogger )

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} ${CUDA_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} ${HOST_EXAMPLES})
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${target}.out )
   endforeach()
endif()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()