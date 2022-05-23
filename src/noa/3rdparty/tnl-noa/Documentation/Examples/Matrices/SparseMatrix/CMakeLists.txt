set( COMMON_EXAMPLES
   SparseMatrixExample_Constructor_init_list_1
   SparseMatrixExample_Constructor_init_list_2
   SparseMatrixExample_Constructor_rowCapacities_vector
   SparseMatrixExample_Constructor_std_map
   SparseMatrixExample_getSerializationType
   SparseMatrixExample_setRowCapacities
   SparseMatrixExample_setElements
   SparseMatrixExample_setElements_map
   SparseMatrixExample_getCompressedRowLengths
   SparseMatrixExample_getConstRow
   SparseMatrixExample_getRow
   SparseMatrixExample_setElement
   SparseMatrixExample_addElement
   SparseMatrixExample_getElement
   SparseMatrixExample_reduceRows
   SparseMatrixExample_reduceAllRows
   SparseMatrixExample_forElements
   SparseMatrixExample_forAllElements
   SparseMatrixExample_forRows
   SparseMatrixViewExample_getSerializationType
   SparseMatrixViewExample_getCompressedRowLengths
   SparseMatrixViewExample_getConstRow
   SparseMatrixViewExample_getRow
   SparseMatrixViewExample_setElement
   SparseMatrixViewExample_addElement
   SparseMatrixViewExample_getElement
   SparseMatrixViewExample_reduceRows
   SparseMatrixViewExample_reduceAllRows
   SparseMatrixViewExample_forElements
   SparseMatrixViewExample_forRows
   SparseMatrixViewExample_forAllElements
   SparseMatrixViewExample_wrapCSR
   SparseMatrixViewExample_wrapEllpack
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
   ADD_CUSTOM_TARGET( RunSparseMatricesExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunSparseMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
