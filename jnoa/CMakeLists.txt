####################################################################################################
########################              NOA bindings for KMath             ###########################
####################################################################################################

# Java Native Interface
find_package(JNI REQUIRED)

# JNoa library
add_library(jnoa SHARED jnoa.cc)

# JNI headers
target_include_directories(jnoa PRIVATE ${JNI_INCLUDE_DIRS})

# Link libraries
target_link_libraries(jnoa PRIVATE ${PROJECT_NAME} )
target_compile_options(jnoa PRIVATE -Wall -Wextra -Wpedantic -O3 -fPIC)
