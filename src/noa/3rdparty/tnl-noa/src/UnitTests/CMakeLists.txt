ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Arithmetics )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Functions )
# Matrices are included from src/CMakeLists.txt
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( MPI )
ADD_SUBDIRECTORY( Pointers )
ADD_SUBDIRECTORY( Solvers )

set( CPP_TESTS  AssertTest
                base64Test
                CustomScalarTest
                FileNameTest
                MathTest
                ObjectTest
                StringTest
                TimerTest
                TypeInfoTest )
set( CUDA_TESTS  AssertCudaTest )
if( BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} AllocatorsTest FileTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} AllocatorsTest FileTest )
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

# special test needing external library
find_package( ZLIB )
if( ZLIB_FOUND )
   foreach( target IN ITEMS zlibCompressionTest )
      add_executable( ${target} ${target}.cpp )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_ZLIB" )
      target_compile_options( ${target} PRIVATE ${CXX_TESTS_FLAGS} )
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries( ${target} ${TESTS_LIBRARIES} ${ZLIB_LIBRARIES} )
      target_link_options( ${target} PRIVATE ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()
