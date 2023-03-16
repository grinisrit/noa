# compatibility with the CSR5 package

if( NOT DEFINED CUDA_SAMPLES_DIR )
    message( WARNING "CUDA_SAMPLES_DIR variable was not set and it is required by CSR5 benchmark - CSR5 benchmark is disabled.")
else()
    # Download and unpack CSR5 at configure time
    message( STATUS "CUDA_SAMPLES_DIR set to ${CUDA_SAMPLES_DIR}")
    configure_file(cmake/CSR5.cmake.in csr5-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Benchmarks/SpMV/csr5-download )
    if(result)
        message(WARNING "CMake step for CSR5 failed: ${result}")
    else()
        execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Benchmarks/SpMV/csr5-download )
        if(result)
            message( ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/src/Benchmarks/SpMV/csr5-download )
            message(WARNING "Build step for CSR5 failed: ${result}")
        else()
            set( CXX_BENCHMARKS_FLAGS ${CXX_BENCHMARKS_FLAGS} "-DHAVE_CSR5" )
            set( CXX_BENCHMARKS_INCLUDE_DIRS ${CXX_BENCHMARKS_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/src/Benchmarks/SpMV/csr5-src ${CUDA_SAMPLES_DIR}/common/inc)
            message( STATUS "CSR5 build was succesfull.")
        endif()
    endif()
endif()
