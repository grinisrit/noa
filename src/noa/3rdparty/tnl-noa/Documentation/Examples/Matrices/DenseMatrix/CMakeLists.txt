set( COMMON_EXAMPLES
    DenseMatrixExample_Constructor_init_list
    DenseMatrixExample_setElements
    DenseMatrixExample_getCompressedRowLengths
    DenseMatrixExample_getElementsCount
    DenseMatrixExample_getConstRow
    DenseMatrixExample_getRow
    DenseMatrixExample_setElement
    DenseMatrixExample_addElement
    DenseMatrixExample_getElement
    DenseMatrixExample_reduceRows
    DenseMatrixExample_reduceAllRows
    DenseMatrixExample_forElements
    DenseMatrixExample_forAllElements
    DenseMatrixExample_forRows
    DenseMatrixViewExample_constructor
    DenseMatrixViewExample_getCompressedRowLengths
    DenseMatrixViewExample_getElementsCount
    DenseMatrixViewExample_getConstRow
    DenseMatrixViewExample_getRow
    DenseMatrixViewExample_setElement
    DenseMatrixViewExample_addElement
    DenseMatrixViewExample_getElement
    DenseMatrixViewExample_reduceRows
    DenseMatrixViewExample_reduceAllRows
    DenseMatrixViewExample_forElements
    DenseMatrixViewExample_forRows
    DenseMatrixViewExample_forAllElements
    DenseMatrixViewExample_wrap
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
   ADD_CUSTOM_TARGET( RunDenseMatricesExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunDenseMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()

