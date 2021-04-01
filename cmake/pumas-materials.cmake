include(ExternalProject)

ExternalProject_Add (Materials
    GIT_REPOSITORY git@github.com:grinisrit/pumas-materials.git
    GIT_TAG ghmcV0
    SOURCE_DIR ${CMAKE_BINARY_DIR}/materials
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)