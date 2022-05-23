set( COMMON_TESTS
            Legacy_SparseMatrixTest
            # TODO: Uncomment the following when AdEllpack works
            #SparseMatrixTest_AdEllpack
            Legacy_SparseMatrixTest_BiEllpack
            Legacy_SparseMatrixTest_ChunkedEllpack
            Legacy_SparseMatrixTest_CSRScalar
            Legacy_SparseMatrixTest_CSRVector
            Legacy_SparseMatrixTest_CSRMultiVector
            Legacy_SparseMatrixTest_CSRLight
            Legacy_SparseMatrixTest_CSRLightWithoutAtomic
            Legacy_SparseMatrixTest_CSRAdaptive
            Legacy_SparseMatrixTest_Ellpack
            Legacy_SparseMatrixTest_SlicedEllpack
)

set( CPP_TESTS )
set( CUDA_TESTS )
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} ${TESTS_LIBRARIES} )
   target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      cuda_add_executable( ${target} ${target}.cu OPTIONS ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} ${TESTS_LIBRARIES} )
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()
