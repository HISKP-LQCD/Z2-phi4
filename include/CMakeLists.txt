macro(target_link_libraries_kokkos TARGETNAME)
    target_link_libraries(${TARGETNAME}  PUBLIC Kokkos::kokkos)
    target_include_directories(${TARGETNAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") 
endmacro()

add_library(
    updates STATIC
    updates.hpp
    ../modules/metropolis.cpp
    ../modules/cluster.cpp
) 
set_target_properties(updates PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries_kokkos(updates )

add_library(
    write_viewer STATIC
    write_viewer.hpp
    ../modules/write_viewer.cpp
) 
target_link_libraries_kokkos(write_viewer )





add_library(
    DFT STATIC
    DFT.hpp
    ../modules/DFT.cpp
) 
IF( FFTW)
    target_link_libraries(DFT  PUBLIC fftw3)
elseif(CUFFT)
    target_link_libraries(DFT  PUBLIC cufft)
ENDIF()
target_link_libraries_kokkos(DFT)

#add_library(
#    lattice STATIC
#    lattice.hpp
#) 

#set_target_properties(lattice PROPERTIES LINKER_LANGUAGE CXX)
#target_link_libraries_kokkos(lattice)


add_library(
    mutils STATIC
    mutils.hpp
    ../modules/mutils.cpp
) 

target_include_directories(mutils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
#target_link_libraries_kokkos(mutils)

add_library(
    random STATIC
    random.hpp
    ../modules/ranlxd.cpp
) 
set_target_properties(random PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(random PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
#target_link_libraries_kokkos(random)

add_library(
    utils STATIC
    utils.hpp
    ../modules/utils.cpp
    ../modules/endian.cpp
) 
set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)
#target_include_directories(utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries_kokkos(utils)

add_library(
    geometry STATIC
    geometry.hpp
    ../modules/geometry/hopping.cpp
    #../modules/geometry/lex2c.cpp
) 
#target_link_libraries(geometry PUBLIC Kokkos::kokkos)
#target_include_directories(geometry PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries_kokkos(geometry)



add_library(
    IO_params STATIC
    IO_params.hpp
) 
set_target_properties(IO_params PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(IO_params PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
#target_link_libraries_kokkos(IO_params)


add_library(
    measurements STATIC
    measurements.hpp
    ../modules/measurements.cpp
) 
target_link_libraries_kokkos(measurements )
