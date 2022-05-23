set( COMMON_EXAMPLES
   LambdaMatrixExample_Constructor
   LambdaMatrixExample_getCompressedRowLengths
   LambdaMatrixExample_getNonzeroElementsCount
   LambdaMatrixExample_Laplace
   LambdaMatrixExample_Laplace_2
   LambdaMatrixExample_getRow
   LambdaMatrixExample_reduceRows
   LambdaMatrixExample_reduceAllRows
   LambdaMatrixExample_forElements
   LambdaMatrixExample_forAllElements
   LambdaMatrixExample_forRows
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
   ADD_CUSTOM_TARGET( RunLambdaMatricesExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunLambdaMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()