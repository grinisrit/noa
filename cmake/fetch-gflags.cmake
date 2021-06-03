include(FetchContent)

if(NOT TARGET gflags)
    FetchContent_Declare(
        gflags
        GIT_REPOSITORY https://github.com/gflags/gflags.git
        GIT_TAG        v2.2.2)

    FetchContent_GetProperties(gflags)

    if(NOT gflags_POPULATED)
        FetchContent_Populate(gflags)

        set(BUILD_TESTING OFF CACHE BOOL "")
        set(GFLAGS_IS_SUBPROJECT TRUE CACHE BOOL "")
        set(INSTALL_HEADERS OFF CACHE BOOL "")
        set(INSTALL_SHARED_LIBS OFF CACHE BOOL "")
        set(INSTALL_STATIC_LIBS OFF CACHE BOOL "")
        set(GFLAGS_LIBRARY_INSTALL_DIR ${CMAKE_BINARY_DIR}/lib)

        add_subdirectory(
            ${gflags_SOURCE_DIR}
            ${gflags_BINARY_DIR})
    endif()

    # Compatibility with LibTorch cxx11 ABI
    target_compile_definitions(gflags_nothreads_static PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

endif()