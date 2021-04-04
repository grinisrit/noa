####################################################################################################
##########################                    NOA Apps                   ###########################
####################################################################################################


if(BUILD_GHMC_APPS OR BUILD_PMS_APPS)
    FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG        v2.2.2)

    FetchContent_GetProperties(gflags)

    if(NOT gflags_POPULATED)
        FetchContent_Populate(gflags)

        add_subdirectory(
            ${gflags_SOURCE_DIR}
            ${gflags_BINARY_DIR})
    endif()

endif()


if(BUILD_GHMC_APPS)
    add_subdirectory(ghmc)
endif()


if(BUILD_PMS_APPS)
    include(noa-pms-models)
    add_subdirectory(pms)
endif()