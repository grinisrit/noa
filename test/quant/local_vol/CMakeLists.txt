find_package(Enzyme REQUIRED)
message(STATUS "Found EnzymeAD: ${Enzyme_DIR}")

add_executable(enzyme_test main.cc)
target_link_libraries(enzyme_test PUBLIC ClangEnzymeFlags) 
target_compile_options(enzyme_test PUBLIC -O3)
target_compile_features(enzyme_test PRIVATE cxx_std_20)