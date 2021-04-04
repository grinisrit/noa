####################################################################################################
#####################                      NOA library                       #######################
####################################################################################################


# Source files
set(NOA_SRC 
    pms/mdf.cc)

add_library(noa SHARED ${NOA_SRC})

# Pugixml library
include(fetch-pugixml)

target_include_directories(noa  
    PUBLIC
        $<BUILD_INTERFACE:${NOA_ROOT_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE 
        ${PUGIXML_SRC})


target_link_libraries(noa PUBLIC torch)
target_compile_options(noa PRIVATE -Wall -Wextra -Wpedantic -O3 -DPUGIXML_HEADER_ONLY)

set_target_properties(noa 
    PROPERTIES 
        POSITION_INDEPENDENT_CODE 1
        INTERFACE_LINK_LIBRARIES "torch")
