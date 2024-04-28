cmake_minimum_required(VERSION 3.16)

#export PATH="/opt/homebrew/Cellar/llvm/18.1.4/bin${PATH:+:${PATH}}"

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(enzy)

set( CMAKE_CXX_STANDARD          20  )
set( CMAKE_CXX_STANDARD_REQUIRED ON  )
set( CMAKE_CXX_EXTENSIONS        OFF )

include_directories(../../../src/noa/3rdparty/rapidcsv)

find_package(Enzyme REQUIRED CONFIG PATHS /opt/homebrew/Cellar/enzyme/0.0.98)
message(STATUS "Found EnzymeAD: ${Enzyme_DIR}")

add_executable(enzyme_test main.cc)
target_link_libraries(enzyme_test PUBLIC ClangEnzymeFlags) 
target_compile_options(enzyme_test PUBLIC -O3)
target_compile_features(enzyme_test PRIVATE cxx_std_20)