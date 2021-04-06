include(FetchContent)

if(NOT TARGET googletest)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.10.x)

    FetchContent_GetProperties(googletest)

    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)

        set(BUILD_GMOCK OFF CACHE BOOL "")
        set(INSTALL_GTEST OFF CACHE BOOL "")

        add_subdirectory(
            ${googletest_SOURCE_DIR}
            ${googletest_BINARY_DIR})
    endif()
endif()