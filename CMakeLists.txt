####################################################################################################
#################                      NOA: main cmake file                       ##################
####################################################################################################

# Minimum cmake version
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Project
project(NOA
        VERSION 0.0.1
        DESCRIPTION
        "Differentiable Programming Algorithms in Modern C++"
        HOMEPAGE_URL https://github.com/grinisrit/noa
        LANGUAGES C CXX)

# User options
option(INSTALL_NOA "Install NOA to GNU locations" ON)
option(BUILD_NOA_TESTS "Build and run tests " ON)
option(BUILD_NOA_BENCHMARKS "Build and run benchmarks " OFF)
option(BUILD_NOA_CUDA "Build CUDA support" OFF)
option(BUILD_JNOA "Build JNI bindings" OFF)
option(BUILD_DOCS "Build documentation with doxygen" OFF)

# Require C++17
set(CMAKE_CXX_STANDARD 17)

# Warning flags
set(W_FLAGS "") #"-Wall;-Wextra;-Wpedantic")

# Native optimisations
set(NA_OPT_FLAGS "-march=native;-mtune=native")

# CUDA modern C++ extensions
set(MCXX_CUDA "--expt-relaxed-constexpr;--expt-extended-lambda")

# Cache path to root directory
set(NOA_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add custom modules
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Make sure LibTorch is installed 
include(check-torch)

# Set up CUDA if available
if (BUILD_NOA_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

# NOA library  
add_library(${PROJECT_NAME} INTERFACE)
# Alias for super-build pattern
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Header-only library
target_include_directories(
        ${PROJECT_NAME}
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Interface LibTorch
target_link_libraries(${PROJECT_NAME} INTERFACE torch)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_LINK_LIBRARIES torch)

if (BUILD_NOA_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (BUILD_NOA_BENCHMARKS)
    add_subdirectory(benchmark)
endif ()

if (BUILD_JNOA)
    add_subdirectory(jnoa)
endif ()

if (BUILD_DOCS)
    include(doxygen-docs)
endif ()


if (INSTALL_NOA)
    # locations are provided by GNUInstallDirs
    install(TARGETS
            ${PROJECT_NAME}
            EXPORT
            ${PROJECT_NAME}_Targets)

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
            ${PROJECT_NAME}ConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion)

    configure_package_config_file(
            ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
            ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake INSTALL_DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

    install(EXPORT ${PROJECT_NAME}_Targets
            FILE ${PROJECT_NAME}Targets.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

    install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
            ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            ${PROJECT_SOURCE_DIR}/cmake/pumas-materials.cmake
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/noa
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif ()
