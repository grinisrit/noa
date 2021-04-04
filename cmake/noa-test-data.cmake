ExternalProject_Add (Materials
    GIT_REPOSITORY git@github.com:grinisrit/noa-test-data.git
    GIT_TAG master
    SOURCE_DIR ${CMAKE_BINARY_DIR}/noa-test-data
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

