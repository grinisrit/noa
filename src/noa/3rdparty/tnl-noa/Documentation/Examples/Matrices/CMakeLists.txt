ADD_SUBDIRECTORY( DenseMatrix )
ADD_SUBDIRECTORY( LambdaMatrix )
ADD_SUBDIRECTORY( MultidiagonalMatrix )
ADD_SUBDIRECTORY( SparseMatrix )
ADD_SUBDIRECTORY( TridiagonalMatrix )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MatrixWriterReaderExample_cuda MatrixWriterReaderExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       OUTPUT MatrixWriterReaderExample.out )
ELSE( BUILD_CUDA )
   ADD_EXECUTABLE( MatrixWriterReaderExample MatrixWriterReaderExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                     OUTPUT MatrixWriterReaderExample.out )
ENDIF( BUILD_CUDA )

ADD_CUSTOM_TARGET( RunMatricesExamples ALL DEPENDS
   MatrixWriterReaderExample.out
)