set( COMMON_TESTS
)

set( CPP_TESTS
   ComplexTest
)

set( CUDA_TESTS
)

if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} ${COMMON_TESTS} )
else()
   set( CPP_TESTS  ${CPP_TESTS} ${COMMON_TESTS} )
endif()

foreach( target IN ITEMS ${CPP_TESTS} )
   add_executable( ${target} ${target}.cpp )
   target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
   target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
   add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_TESTS} )
      add_executable( ${target} ${target}.cu )
      target_compile_options( ${target} PUBLIC ${CUDA_TESTS_FLAGS} )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA ${TESTS_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()


if( HAVE_GMP )
   ADD_EXECUTABLE( MultiPrecisionTest MultiPrecisionTest.cpp ../MultiPrecision.cpp )
   TARGET_COMPILE_OPTIONS( MultiPrecisionTest PUBLIC ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MultiPrecisionTest PUBLIC
                              TNL::TNL_CXX
                              ${TESTS_LIBRARIES}
                              ${GMP_LIBRARIES} )
   TARGET_LINK_OPTIONS( MultiPrecisionTest PUBLIC ${TESTS_LINKER_FLAGS} )
   ADD_TEST( MultiPrecisionTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MultiPrecisionTest${CMAKE_EXECUTABLE_SUFFIX} )
endif( HAVE_GMP )


# TODO: Fix the following tests
#ADD_EXECUTABLE( QuadTest QuadTest.cpp )
#TARGET_COMPILE_OPTIONS( QuadTest PUBLIC ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( QuadTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
#ADD_TEST( QuadTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/QuadTest${CMAKE_EXECUTABLE_SUFFIX} )

#ADD_EXECUTABLE( DoubleTest DoubleTest.cpp )
#TARGET_COMPILE_OPTIONS( DoubleTest PUBLIC ${CXX_TESTS_FLAGS} )
#TARGET_LINK_LIBRARIES( DoubleTest PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} )
#ADD_TEST( DoubleTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DoubleTest${CMAKE_EXECUTABLE_SUFFIX} )
