include(ExternalProject)

if(NOT TARGET pumas_materials)
    ExternalProject_Add (pumas_materials
        GIT_REPOSITORY https://github.com/grinisrit/pumas-materials.git
        GIT_TAG v0.0.1
        SOURCE_DIR ${CMAKE_BINARY_DIR}/pumas-materials
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")
endif()