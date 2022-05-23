IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayAllocation ArrayAllocation.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayAllocation > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayAllocation.out OUTPUT ArrayAllocation.out )
   CUDA_ADD_EXECUTABLE( ArrayIO ArrayIO.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayIO > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayIO.out OUTPUT ArrayIO.out )
   CUDA_ADD_EXECUTABLE( ArrayView-1 ArrayView-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayView-1 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayView-1.out OUTPUT ArrayView-1.out )
   CUDA_ADD_EXECUTABLE( ArrayView-2 ArrayView-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayView-2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayView-2.out OUTPUT ArrayView-2.out )
   CUDA_ADD_EXECUTABLE( ArrayViewForElements ArrayViewForElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewForElements > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayViewForElements.out OUTPUT ArrayViewForElements.out )
   CUDA_ADD_EXECUTABLE( contains contains.cu )
   ADD_CUSTOM_COMMAND( COMMAND contains > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/contains.out OUTPUT contains.out )
   CUDA_ADD_EXECUTABLE( ElementsAccessing-1 ElementsAccessing-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ElementsAccessing-1 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ElementsAccessing-1.out OUTPUT ElementsAccessing-1.out )
   CUDA_ADD_EXECUTABLE( ElementsAccessing-2 ElementsAccessing-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ElementsAccessing-2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ElementsAccessing-2.out OUTPUT ElementsAccessing-2.out )
   ADD_EXECUTABLE( StaticArrayExample StaticArrayExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND StaticArrayExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticArrayExample.out OUTPUT StaticArrayExample.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsArrays-cuda ALL DEPENDS
   ArrayAllocation.out
   ArrayIO.out
   ArrayView-1.out
   ArrayView-2.out
   contains.out
   ElementsAccessing-1.out
   ElementsAccessing-2.out
   ArrayViewForElements.out
   StaticArrayExample.out )
ENDIF()
