add_subdirectory(enzyme)

add_executable(local_vol_test main.cc)
target_link_libraries(local_vol_test PUBLIC ClangEnzymeFlags) 
target_compile_options(local_vol_test PUBLIC -O3)
target_compile_features(local_vol_test PRIVATE cxx_std_20)