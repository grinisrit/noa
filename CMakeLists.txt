####################################################################################################
#################                      NOA: main cmake file                       ##################
####################################################################################################

# Minimum cmake version
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Project
project(
	NOA 
	VERSION 0.0.1
	DESCRIPTION 
		"Bayesian Computation for Deep Learning over LibTorch"
	HOMEPAGE_URL https://github.com/grinisrit/noa
	LANGUAGES CXX)

# User options
option(INSTALL_NOA "Install NOA to GNU locations" ON)
option(BUILD_TESTING "Build and run tests " ON)
option(BUILD_BENCHMARKS "Build and run benchmarks " OFF)
option(BUILD_GHMC_APPS "Build GHMC apps " OFF)
option(BUILD_PMS_APPS "Build PMS apps " OFF)

# Require C++17
set(CMAKE_CXX_STANDARD 17)

# Compatibility with LibTorch cxx11 ABI
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Cache path to root directory
set(NOA_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add custom modules
include(FetchContent)
include(ExternalProject)
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Make sure LibTorch is installed 
include(check-torch)

# NOA library  
add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
	add_subdirectory(test)
endif()

if(BUILD_BENCHMARKS)
	add_subdirectory(benchmark)
endif()
#[[
if(INSTALL_NOA)
  	# locations are provided by GNUInstallDirs
  	install(
    	TARGETS ${PROJECT_NAME}
    	EXPORT ${PROJECT_NAME}_Targets)

  	include(CMakePackageConfigHelpers)
  	write_basic_package_version_file(
    	${PROJECT_NAME}ConfigVersion.cmake
    	VERSION ${PROJECT_VERSION}
    	COMPATIBILITY SameMajorVersion)

  	configure_package_config_file(
    	${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    	${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake INSTALL_DESTINATION
    	${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  	install(
    	EXPORT ${PROJECT_NAME}_Targets
    	FILE ${PROJECT_NAME}Targets.cmake
    	NAMESPACE ${PROJECT_NAME}::
    	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  	install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
				  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
				  ${PROJECT_SOURCE_DIR}/cmake/noa-pms-models.cmake
				  ${PROJECT_SOURCE_DIR}/cmake/conda-omp.cmake
			  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
			  
  	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/noa
          	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif()
]]
add_subdirectory(apps)
