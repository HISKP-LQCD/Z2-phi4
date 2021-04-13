

add_target_with_lib(main main.cpp)
add_target_with_lib(contraction contraction.cpp)
add_target_with_lib(conf_bin2asci conf_bin2asci.cpp)

add_executable(binning_contraction binning_contraction.cpp)
add_executable(merging_replicas merging_replicas.cpp)

