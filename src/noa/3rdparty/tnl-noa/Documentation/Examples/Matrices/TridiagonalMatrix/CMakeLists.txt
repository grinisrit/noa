set( COMMON_EXAMPLES
    TridiagonalMatrixExample_Constructor_init_list_1
    TridiagonalMatrixExample_getSerializationType
    TridiagonalMatrixExample_setElements
    TridiagonalMatrixExample_getCompressedRowLengths
#   TridiagonalMatrixExample_getElementsCount
    TridiagonalMatrixExample_getConstRow
    TridiagonalMatrixExample_getRow
# This example does not work with nvcc 10.1. Restore it here when it works.
#   TridiagonalMatrixExample_setElement
    TridiagonalMatrixExample_addElement
    TridiagonalMatrixExample_getElement
    TridiagonalMatrixExample_reduceRows
    TridiagonalMatrixExample_reduceAllRows
    TridiagonalMatrixExample_forElements
    TridiagonalMatrixExample_forAllElements
    TridiagonalMatrixExample_forRows
#   TridiagonalMatrixViewExample_constructor
    TridiagonalMatrixViewExample_getCompressedRowLengths
#   TridiagonalMatrixViewExample_getElementsCount
    TridiagonalMatrixViewExample_getConstRow
    TridiagonalMatrixViewExample_getRow
    TridiagonalMatrixViewExample_setElement
    TridiagonalMatrixViewExample_addElement
    TridiagonalMatrixViewExample_getElement
    TridiagonalMatrixViewExample_reduceRows
    TridiagonalMatrixViewExample_reduceAllRows
    TridiagonalMatrixViewExample_forElements
    TridiagonalMatrixViewExample_forAllElements
    TridiagonalMatrixViewExample_forRows
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
   ADD_CUSTOM_TARGET( RunTridiagonalMatricesExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunTridiagonalMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
