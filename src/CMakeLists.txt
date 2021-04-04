####################################################################################################
#####################                      NOA library                       #######################
####################################################################################################


# Source files
set(NOA_SRC 
    pms/mdf.cc)

add_library(${PROJECT_NAME} SHARED ${NOA_SRC})
# add alias so the project can be used with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Pugixml library
include(fetch-pugixml)

target_include_directories(${PROJECT_NAME}  
    PUBLIC
        $<BUILD_INTERFACE:${NOA_ROOT_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE 
        ${PUGIXML_SRC})


target_link_libraries(${PROJECT_NAME} PUBLIC torch)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -O3 -DPUGIXML_HEADER_ONLY)

set_target_properties(${PROJECT_NAME} 
    PROPERTIES 
        POSITION_INDEPENDENT_CODE 1
        INTERFACE_LINK_LIBRARIES "torch")
