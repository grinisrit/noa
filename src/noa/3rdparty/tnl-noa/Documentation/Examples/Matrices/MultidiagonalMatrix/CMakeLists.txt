set( COMMON_EXAMPLES
   MultidiagonalMatrixExample_Constructor
    MultidiagonalMatrixExample_Constructor_init_list_1
    MultidiagonalMatrixExample_Constructor_init_list_2
    MultidiagonalMatrixExample_getSerializationType
    MultidiagonalMatrixExample_setElements
    MultidiagonalMatrixExample_getCompressedRowLengths
#    MultidiagonalMatrixExample_getElementsCount
    MultidiagonalMatrixExample_getConstRow
    MultidiagonalMatrixExample_getRow
# This example does not work with nvcc 10.1. Restore it here when it works.
#    MultidiagonalMatrixExample_setElement
    MultidiagonalMatrixExample_addElement
    MultidiagonalMatrixExample_getElement
    MultidiagonalMatrixExample_reduceRows
    MultidiagonalMatrixExample_reduceAllRows
    MultidiagonalMatrixExample_forElements
    MultidiagonalMatrixExample_forAllElements
    MultidiagonalMatrixExample_forRows
#    MultidiagonalMatrixViewExample_constructor
    MultidiagonalMatrixViewExample_getCompressedRowLengths
#    MultidiagonalMatrixViewExample_getElementsCount
    MultidiagonalMatrixViewExample_getConstRow
    MultidiagonalMatrixViewExample_getRow
    MultidiagonalMatrixViewExample_setElement
    MultidiagonalMatrixViewExample_addElement
    MultidiagonalMatrixViewExample_getElement
    MultidiagonalMatrixViewExample_reduceRows
    MultidiagonalMatrixViewExample_reduceAllRows
    MultidiagonalMatrixViewExample_forElements
    MultidiagonalMatrixViewExample_forAllElements
    MultidiagonalMatrixViewExample_forRows
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
   ADD_CUSTOM_TARGET( RunMultidiagonalMatricesExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunMultidiagonalMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()