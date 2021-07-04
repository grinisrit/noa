include(ExternalProject)

if(NOT TARGET pms_models)
    ExternalProject_Add (pms_models
        GIT_REPOSITORY https://github.com/grinisrit/noa-pms-models.git
        GIT_TAG v0.0.1
        SOURCE_DIR ${CMAKE_BINARY_DIR}/noa-pms-models
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")
endif()