include(ExternalProject)

if(NOT TARGET test_data)
    ExternalProject_Add (test_data
        GIT_REPOSITORY https://github.com/grinisrit/noa-test-data.git
        GIT_TAG v0.0.2
        SOURCE_DIR ${CMAKE_BINARY_DIR}/noa-test-data
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")
endif()
