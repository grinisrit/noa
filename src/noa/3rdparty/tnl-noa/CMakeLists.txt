# cmake 3.12.2 is required due to compatibility with CUDA 10
# (see the issue reported here: https://github.com/clab/dynet/issues/1457 )
# cmake 3.13 is required for add_link_options
cmake_minimum_required( VERSION 3.13 )

project( tnl )

set( tnlVersion "0.1" )

# declare all custom build options
option(OFFLINE_BUILD "Offline build (i.e. without downloading libraries such as pybind11)" OFF)
option(WITH_CUDA "Build with CUDA support" ON)
set(WITH_CUDA_ARCH "auto" CACHE STRING "Build for these CUDA architectures")
option(WITH_OPENMP "Build with OpenMP support" ON)
option(WITH_MPI "Build with MPI support" ON)
option(WITH_GMP "Build with GMP support" OFF)
option(WITH_COVERAGE "Enable code coverage reports from unit tests" OFF)
option(WITH_SYSTEM_GTEST "Use GTest installed in the local system and do not download the latest version" OFF)
option(BUILD_BENCHMARKS "Compile the 'src/Benchmarks' directory" OFF)
option(BUILD_EXAMPLES "Compile the 'src/Examples' directory" OFF)
option(BUILD_TOOLS "Compile the 'src/Tools' directory" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_MATRIX_TESTS "Build tests for matrix formats" OFF)
option(BUILD_PYTHON "Compile the Python bindings" OFF)
option(BUILD_DOC "Build examples included in the documentation" OFF)
set(CUDA_SAMPLES_PATH "none" CACHE STRING "Path to CUDA Samples - it is used only for some benchmarking.")

# install paths relative to the cmake's prefix
set( TNL_TARGET_INCLUDE_DIRECTORY "include/TNL" )
set( TNL_TARGET_DATA_DIRECTORY "share/TNL" )

# set cmake's include path so that we can include modules from
# the cmake directory in the TNL repository
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

# Note that in cmake 3.10 the FindOpenMP and FindMPI modules are broken - they do not work when
# CMAKE_EXECUTABLE_SUFFIX is not empty, see https://www.mail-archive.com/cmake@cmake.org/msg56886.html
# Hence, we find OpenMP and MPI before setting CMAKE_EXECUTABLE_SUFFIX.
find_package( OpenMP )
find_package( MPI )

# Settings for debug build
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( CMAKE_EXECUTABLE_SUFFIX "-dbg${CMAKE_EXECUTABLE_SUFFIX}" )  # suffix for executables
    set( CMAKE_DEBUG_POSTFIX "-dbg" )  # suffix for libraries
endif()

# check if the compiler is good enough
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   # GCC 5.0 is the first release with full C++11 support (due to libstdc++) as
   # well as full C++14 support: https://gcc.gnu.org/gcc-5/changes.html
   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
      message(FATAL_ERROR "Insufficient GCC version")
   endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   # Clang 3.4 has full C++14 support: http://clang.llvm.org/cxx_status.html
   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.4")
      message(FATAL_ERROR "Insufficient Clang version")
   endif()
endif()

# set C++ standard
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# set default build options
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Werror=vla" )
set( CMAKE_CXX_FLAGS_DEBUG "-g" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" )
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}" )
# pass -rdynamic only in Debug mode
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "" )
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE "" )
set( CMAKE_EXE_LINKER_FLAGS "" )
set( CMAKE_EXE_LINKER_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_EXE_LINKER_FLAGS_RELEASE "" )
set( CMAKE_SHARED_LINKER_FLAGS "" )
set( CMAKE_SHARED_LINKER_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_SHARED_LINKER_FLAGS_RELEASE "" )

if( ${WITH_CI_FLAGS} )
   # enforce (more or less) warning-free builds
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized" )
endif()

# set additional Debug/Release options using generator expressions
# (that way we can exclude some options for specific targets, see https://stackoverflow.com/a/59734798 for details)
add_compile_options(
   # GOTCHA: CMake uses semicolons as list item separator, spaces would lead to a single argument inside double-quotes on the command line
   "$<$<CONFIG:RELEASE>:-march=native;-mtune=native>"
)

if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   # disable some unimportant warnings
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs -Wno-unknown-pragmas" )
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
   # Intel's -Wall is very minimalistic, so add -w3 and disable some specific warnings
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w3 -diag-disable:remark" )
endif()

# disable GCC's infamous "maybe-uninitialized" warning (it produces mostly false positives)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized" )
endif()

# disable false Clang warning: https://stackoverflow.com/q/57645872
if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-self-assign-overloaded" )
endif()

# enable address sanitizer (does not work with MPI due to many false positives, does not work with nvcc at all)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   if( NOT ${WITH_MPI} AND NOT ${WITH_CUDA} )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer" )
      set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
   endif()
endif()

# enable link time optimizations (but not in continuous integration)
if( NOT DEFINED ENV{CI_JOB_NAME} )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      # LTO with GCC 9.1.0 and Debug build = internal compiler error
      # LTO with GCC 9.1.0 and nvcc 10.1 and Release build = fatal error: bytecode stream in file `blabla` generated with LTO version 7.1 instead of the expected 8.0
#      add_compile_options( "-flto" )
#      add_link_options( "-flto" )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
      add_compile_options( "-flto=thin" )
      add_link_options( "-flto=thin" )
   endif()
endif()

# force colorized output in continuous integration
if( DEFINED ENV{CI_JOB_NAME} OR ${CMAKE_GENERATOR} STREQUAL "Ninja" )
   message(STATUS "Continuous integration or Ninja detected -- forcing compilers to produce colorized output.")
   if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics" )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color" )
   endif()
endif()

# add the filesystem library to all targets
# https://en.cppreference.com/w/cpp/filesystem
link_libraries( stdc++fs )

# gtest has to be built before we add the MPI flags
if( ${BUILD_TESTS} OR ${BUILD_MATRIX_TESTS} )
   enable_testing()

   # let CTest write test results in the JUnit XML format
   set( CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --output-junit "${CMAKE_BINARY_DIR}/tests-report.xml" )

   if( ${WITH_SYSTEM_GTEST} OR ${OFFLINE_BUILD} )
      # find gtest installed in the local system
      find_package(GTest REQUIRED)
      if( GTEST_FOUND )
         include_directories( ${GTEST_INCLUDE_DIRS} )
      endif( GTEST_FOUND )
   else()
      # build gtest libs
      include( BuildGtest )
   endif()
   set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} -DHAVE_GTEST )
   set( CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -DHAVE_GTEST )
   set( TESTS_LIBRARIES ${TESTS_LIBRARIES} ${GTEST_LIBRARIES} )

   if( ${WITH_COVERAGE} AND CMAKE_BUILD_TYPE STREQUAL "Debug" )
      # set compiler flags needed for code coverage
      set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} --coverage )
      set( CUDA_TESTS_FLAGS ${CUDA_TESTS_FLAGS} -Xcompiler --coverage )
      set( TESTS_LINKER_FLAGS ${TESTS_LINKER_FLAGS} --coverage )
   endif()
endif()

####
# Check for OpenMP
#
if( OPENMP_FOUND AND ${WITH_OPENMP} )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}" )
endif()

####
# Check for MPI
#
if( MPI_CXX_FOUND AND ${WITH_MPI} )
   set( BUILD_MPI TRUE)
   # add the appropriate flags to all targets (will be hidden from the CMAKE_CXX_* variables)
   include_directories( ${MPI_CXX_INCLUDE_DIRS} )
   add_compile_options( ${MPI_CXX_COMPILE_OPTIONS} )
   add_compile_definitions( ${MPI_CXX_COMPILE_DEFINITIONS} )
   add_link_options( "SHELL:${MPI_CXX_LINK_FLAGS}" )
   link_libraries( ${MPI_CXX_LIBRARIES} )
   # enable MPI in TNL
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MPI" )
endif()

#####
# Check for CUDA
#
if( ${WITH_CUDA} )
    find_package( CUDA 9.0 )
    if( CUDA_FOUND )
        set( BUILD_CUDA TRUE)
        set(CUDA_SEPARABLE_COMPILATION ON)
        # Use the CUDA_HOST_COMPILER environment variable if the user specified it.
        if( NOT $ENV{CUDA_HOST_COMPILER} STREQUAL "" )
            message( "-- Setting CUDA_HOST_COMPILER to '$ENV{CUDA_HOST_COMPILER}'" )
            set( CUDA_HOST_COMPILER $ENV{CUDA_HOST_COMPILER} )
        else()
            if( EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
               message( "-- Setting CUDA_HOST_COMPILER to '${CUDA_TOOLKIT_ROOT_DIR}/bin/g++'" )
               set( CUDA_HOST_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
            else()
               message( "-- Setting CUDA_HOST_COMPILER to '${CMAKE_CXX_COMPILER}'" )
               set( CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} )
            endif()
        endif()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; -std=c++14 -DHAVE_CUDA --expt-relaxed-constexpr --expt-extended-lambda --default-stream per-thread)
        if( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; --generate-line-info)
        endif()
        # disable false compiler warnings
        #   reference for the -Xcudafe --diag_suppress and --display_error_number flags: https://stackoverflow.com/a/54142937
        #   incomplete list of tokens: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=code_is_unreachable -Xcudafe --diag_suppress=loop_not_reachable -Xcudafe --diag_suppress=implicit_return_from_non_void_function -Xcudafe --diag_suppress=unsigned_compare_with_zero -Xcudafe --display_error_number)
        # This diagnostic is just plain wrong in CUDA 9 and later, see https://github.com/kokkos/kokkos/issues/1470
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)
        # Select GPU architecture
        ## cmake bug: cuda_select_nvcc_arch_flags does not work with CMAKE_EXECUTABLE_SUFFIX
        ## see https://gitlab.kitware.com/cmake/cmake/issues/19636
        set( executable_suffix_backup "${CMAKE_EXECUTABLE_SUFFIX}" )
        set( CMAKE_EXECUTABLE_SUFFIX "" )
        if( WITH_CUDA_ARCH STREQUAL "all" )
           CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "All")
           LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
        elseif( WITH_CUDA_ARCH STREQUAL "auto" )
           CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
           LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
        elseif( NOT WITH_CUDA_ARCH STREQUAL "" )
            CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${WITH_CUDA_ARCH})
            LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
        else()
            message( FATAL_ERROR "\$WITH_CUDA_ARCH cannot be empty." )
        endif()
        set( CMAKE_EXECUTABLE_SUFFIX "${executable_suffix_backup}" )
    endif()
    if( NOT CUDA_SAMPLES_DIR STREQUAL "none" )
      set( CUDA_SAMPLES_FLAGS "-I${CUDA_SAMPLES_DIR} -DHAVE_CUDA_SAMPLES")
    endif()
endif()


find_package( DCMTK )
if( DCMTK_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_DCMTK_H" )
   include_directories( ${DCMTK_INCLUDE_DIRS} )
endif()

find_package( PNG )
if( PNG_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PNG_H" )
   include_directories( ${PNG_INCLUDE_DIRS} )
endif()

find_package( JPEG )
if( JPEG_FOUND )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_JPEG_H" )
   include_directories( ${JPEG_INCLUDE_DIRS} )
endif()

####
# Test for GMP
#
if( ${WITH_GMP} )
   if (GMP_INCLUDES AND GMP_LIBRARIES)
      set(GMP_FIND_QUIETLY TRUE)
   endif (GMP_INCLUDES AND GMP_LIBRARIES)

   find_path(GMP_INCLUDES
      NAMES
      gmp.h
      PATHS
      $ENV{GMPDIR}
      ${INCLUDE_INSTALL_DIR}
   )

   find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

   include(FindPackageHandleStandardArgs)
   find_package_handle_standard_args(GMP DEFAULT_MSG
                                      GMP_INCLUDES GMP_LIBRARIES)
   if( ${GMP_INCLUDES} STREQUAL "GMP_INCLUDES-NOTFOUND" OR ${GMP_LIBRARIES} STREQUAL "GMP_LIBRARIES-NOTFOUND" )
      message( "GMP was not found. Some tests for higher precision arithmetics will not be passed." )
   else()
      set( HAVE_GMP )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${GMP_INCLUDES} -DHAVE_GMP" )
      set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${GMP_LIBRARIES}" )
      mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES)
   endif()
endif()

# configure build paths
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )
include_directories( src )
include_directories( SYSTEM src/3rdparty )

# Add all subdirectories
add_subdirectory( src )

# Add subdirectories for examples included in the documentation
if( ${BUILD_DOC} )
   set( TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH "${CMAKE_SOURCE_DIR}/Documentation/output_snippets" )
   file(MAKE_DIRECTORY ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH})
   add_subdirectory( Documentation/Examples )
   add_subdirectory( Documentation/Tutorials )
endif()

# export compile_commands.json so it can be used by Clang tools
# https://clang.llvm.org/docs/JSONCompilationDatabase.html
set( CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "" )

# Print custom build options
message( "-- Build options:" )
message( "   OFFLINE_BUILD = ${OFFLINE_BUILD}" )
message( "   WITH_CUDA = ${WITH_CUDA}" )
message( "   WITH_CUDA_ARCH = ${WITH_CUDA_ARCH}" )
message( "   WITH_OPENMP = ${WITH_OPENMP}" )
message( "   WITH_MPI = ${WITH_MPI}" )
message( "   WITH_GMP = ${WITH_GMP}" )
message( "   WITH_COVERAGE = ${WITH_COVERAGE}" )
message( "   WITH_SYSTEM_GTEST= ${WITH_SYSTEM_GTEST}" )
message( "   BUILD_BENCHMARKS = ${BUILD_BENCHMARKS}" )
message( "   BUILD_EXAMPLES = ${BUILD_EXAMPLES}" )
message( "   BUILD_TOOLS = ${BUILD_TOOLS}" )
message( "   BUILD_TESTS = ${BUILD_TESTS}" )
message( "   BUILD_MATRIX_TESTS = ${BUILD_MATRIX_TESTS}" )
message( "   BUILD_PYTHON = ${BUILD_PYTHON}" )
message( "   BUILD_DOC = ${BUILD_DOC}" )
# Print compiler options
message( "-- Compiler options:" )
message( "   CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}" )
message( "   CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}" )
message( "   CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE}" )
message( "   CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}" )
message( "   CMAKE_EXE_LINKER_FLAGS_DEBUG = ${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
message( "   CMAKE_EXE_LINKER_FLAGS_RELEASE = ${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
message( "   CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}" )
message( "   CMAKE_SHARED_LINKER_FLAGS_DEBUG = ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
message( "   CMAKE_SHARED_LINKER_FLAGS_RELEASE = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
message( "   CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}" )
message( "   CUDA_SAMPLES_FLAGS = ${CUDA_SAMPLES_FLAGS}" )
message( "   GMP_LIBRARIES = ${GMP_LIBRARIES}" )

if( MPI_CXX_FOUND AND ${WITH_MPI} )
   message( "   MPI_CXX_COMPILE_OPTIONS = ${MPI_CXX_COMPILE_OPTIONS}" )
   message( "   MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}" )
   message( "   MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}" )
   message( "   MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}" )
   message( "   MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}" )
endif()
