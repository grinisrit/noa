ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Arithmetics )
ADD_SUBDIRECTORY( Containers )
#ADD_SUBDIRECTORY( Functions )
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
if( TNL_BUILD_CUDA )
   set( CUDA_TESTS  ${CUDA_TESTS} AllocatorsTest FileTest )
else()
   set( CPP_TESTS  ${CPP_TESTS} AllocatorsTest FileTest )
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

# special test needing external library
find_package( ZLIB )
if( ZLIB_FOUND )
   foreach( target IN ITEMS zlibCompressionTest )
      add_executable( ${target} ${target}.cpp )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_ZLIB" )
      target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX ${TESTS_LIBRARIES} ${ZLIB_LIBRARIES} )
      target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )
      add_test( ${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
   endforeach()
endif()


if( TNL_BUILD_MPI )
   find_package(Hypre)

   if( HYPRE_FOUND )
      foreach( target IN ITEMS HypreTest )
         add_executable( ${target} ${target}.cpp )
         target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )

         # enable MPI support in TNL
         target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
         # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
         target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )

         target_compile_definitions( ${target} PUBLIC "-DHAVE_HYPRE" )
         target_compile_options( ${target} PUBLIC ${CXX_TESTS_FLAGS} )
         target_include_directories( ${target} SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS} )
         target_link_libraries( ${target} PUBLIC ${TESTS_LIBRARIES} ${HYPRE_LIBRARIES} )
         target_link_options( ${target} PUBLIC ${TESTS_LINKER_FLAGS} )

         set( mpi_test_parameters -np 4 -H localhost:4 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" )
         add_test( NAME ${target} COMMAND "mpirun" ${mpi_test_parameters})
         add_test( ${target}_nodistr ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX} )
      endforeach()
   endif()
endif()
