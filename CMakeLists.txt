cmake_minimum_required(VERSION 3.13)
project(Z2-phi4 CXX)


# full path of g++  otherwise it does not work
#set(CMAKE_CXX_COMPILER /usr/bin/g++)

#set(CMAKE_CXX_FLAGS "-fopenmp   -pedantic -g   -lm -lgmp -lfftw3 ")
#set(CMAKE_CXX_FLAGS "-fopenmp   -pedantic  -g -O0 -pg -no-pie  -lm -lgmp -lfftw3 ")
#-g # then compile , then open gdb, write the commands : file $namefile; run  $options ; bt
# valgrind --leak-check=yes --track-origins=yes  ./program   #compile with -g to have line number


# no need to install Eigen with cmake and use find package
#SET( EIGEN3_INCLUDE_DIR /home/marco/library/eigen )
#IF( NOT EIGEN3_INCLUDE_DIR )
#    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
#ENDIF()
#INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#using the globally installed version of Eigen: sudo apt-get libeigen3-dev
# Eigen
#IF(NOT DEFINED EIGEN3_INCLUDE_DIR)
#  find_package(Eigen3 REQUIRED)
#ENDIF()
#INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )




add_subdirectory("${PROJECT_SOURCE_DIR}/external/kokkos" "external/kokkos")

#add_subdirectory(${PROJECT_SOURCE_DIR}/external/HighFive)


#include_directories(include)
add_subdirectory(include  include)
#include_directories(${CMAKE_SOURCE_DIR}/include)



include_directories(${CMAKE_SOURCE_DIR}/include)

#
macro(add_target_with_lib TARGETNAME)
  # Make the executable
  add_executable(${TARGETNAME} ${ARGN})
#  target_link_libraries(${TARGETNAME} PUBLIC lattice)
  target_link_libraries(${TARGETNAME} PUBLIC Kokkos::kokkos)
  target_link_libraries(${TARGETNAME} PUBLIC utils)
  target_link_libraries(${TARGETNAME} PUBLIC updates)
  target_link_libraries(${TARGETNAME} PUBLIC geometry)
  target_link_libraries(${TARGETNAME} PUBLIC random)
  target_link_libraries(${TARGETNAME} PUBLIC write_viewer)
#  target_link_libraries(${TARGETNAME} PUBLIC HighFive)
  target_include_directories(${TARGETNAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")
  target_include_directories(${TARGETNAME} PUBLIC "${CMAKE_BINARY_DIR}/include")
endmacro()

add_subdirectory(main)

