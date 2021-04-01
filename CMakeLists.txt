####################################################################################################
#################                      GHMC: main cmake file                       #################
####################################################################################################

# Minimum cmake version
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Project
project(
	GHMC 
	VERSION 0.0.1
	DESCRIPTION 
		"Geometric Hamiltonian Monte Carlo algorithms over LibTorch"
	HOMEPAGE_URL https://github.com/grinisrit/ghmc
	LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)

# Add custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_library(${PROJECT_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# LibTorch
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE torch)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_LINK_LIBRARIES "torch")


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  	set(IS_TOPLEVEL_PROJECT TRUE)
else()
  	set(IS_TOPLEVEL_PROJECT FALSE)
endif()

option(INSTALL_LIBRARY "Install library to GNU locations" ${IS_TOPLEVEL_PROJECT})
option(BUILD_TESTING "Build and run tests " ${IS_TOPLEVEL_PROJECT})

if(BUILD_TESTING)
    enable_testing()
	add_subdirectory(test)
endif(BUILD_TESTING)

if(INSTALL_LIBRARY)
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
				  ${PROJECT_SOURCE_DIR}/cmake/pumas-materials.cmake
				  ${PROJECT_SOURCE_DIR}/cmake/conda-omp.cmake
			  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
			  
  	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ghmc
          	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif(INSTALL_LIBRARY)

option(BUILD_SAMPLER_APPS "Build sampler apps " OFF)
if(BUILD_SAMPLER_APPS)
	add_subdirectory(apps/sampler)
endif(BUILD_SAMPLER_APPS)

option(BUILD_PMS_APPS "Build PMS apps " OFF)
if(BUILD_PMS_APPS)
	include(pumas-materials)
	add_subdirectory(apps/pms)
endif(BUILD_PMS_APPS)