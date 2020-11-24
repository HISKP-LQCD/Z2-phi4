

add_library(
    updates STATIC
    updates.hpp
    ../modules/metropolis.cpp
    ../modules/cluster.cpp
) 
set_target_properties(updates PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(updates PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


add_library(
    lattice STATIC
    lattice.hpp
) 

set_target_properties(lattice PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(lattice PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



add_library(
    mutils STATIC
    mutils.hpp
    ../modules/mutils.cpp
) 

target_include_directories(mutils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


add_library(
    random STATIC
    random.hpp
    ../modules/ranlxd.cpp
) 
set_target_properties(random PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(random PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(
    utils STATIC
    utils.hpp
    ../modules/utils.cpp
    ../modules/endian.cpp
) 
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


add_library(
    geometry STATIC
    geometry.hpp
    ../modules/geometry/hopping.cpp
    ../modules/geometry/lex2c.cpp
) 

target_include_directories(geometry PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



add_library(
    IO_params STATIC
    IO_params.hpp
) 
set_target_properties(IO_params PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(IO_params PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


