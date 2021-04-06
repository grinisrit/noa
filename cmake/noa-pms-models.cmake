include(ExternalProject)

if(NOT TARGET Materials)
    ExternalProject_Add (Materials
        GIT_REPOSITORY git@github.com:grinisrit/noa-pms-models.git
        GIT_TAG master
        SOURCE_DIR ${CMAKE_BINARY_DIR}/noa-pms-models
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")
endif()