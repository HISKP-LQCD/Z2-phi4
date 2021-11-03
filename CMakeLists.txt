cmake_minimum_required(VERSION 3.13)
project(Z2-phi4  CXX)


# full path of g++  otherwise it does not work
#set(CMAKE_CXX_COMPILER /usr/bin/g++)

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

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} -DDEBUG  -g   -lm  ")
  set(Kokkos_ENABLE_DEBUG ON)
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON)
  MESSAGE("DEBUG mode ON")
  MESSAGE("Kokkos_ENABLE_DEBUG = ${Kokkos_ENABLE_DEBUG}")
  MESSAGE("Kokkos_ENABLE_DEBUG_BOUNDS_CHECK = ${Kokkos_ENABLE_DEBUG_BOUNDS_CHECK}")
ENDIF()

IF(FFTW)
  set(CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} -DFFTW   ")
  MESSAGE( "flags:  " ${CMAKE_CXX_FLAGS})
elseif(CUFFT)
  set(CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} -DcuFFT   ")
  MESSAGE( "flags:  " ${CMAKE_CXX_FLAGS})
  MESSAGE(FATAL_ERROR "You can't CUFFT" )
ENDIF()

MESSAGE("CMAKE_CXX_FLAGS   ${CMAKE_CXX_FLAGS}")
MESSAGE("CXXFLAGS   ${CXXFLAGS}")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/kokkos" "external/kokkos")
include_directories(${Kokkos_INCLUDE_DIRS_RET})
#target_link_libraries(myTarget kokkos)

#add_subdirectory(${PROJECT_SOURCE_DIR}/external/HighFive)


add_subdirectory(include  include)



######################### git version ################
######################################################
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    
## this in not the end, you need to  target_compile_definitions for the executable to build
## see the macro add_target_with_lib  
######################################################
######################################################


macro(add_target_with_lib TARGETNAME)
  # Make the executable
  add_executable(${TARGETNAME} ${ARGN})
#  target_link_libraries(${TARGETNAME} PUBLIC lattice)
  target_link_libraries(${TARGETNAME} PUBLIC Kokkos::kokkos)
  target_link_libraries(${TARGETNAME} PUBLIC utils)
  target_link_libraries(${TARGETNAME} PUBLIC updates)
  target_link_libraries(${TARGETNAME} PUBLIC geometry)
  #target_link_libraries(${TARGETNAME} PUBLIC random)
  target_link_libraries(${TARGETNAME} PUBLIC write_viewer)
  target_link_libraries(${TARGETNAME} PUBLIC measurements)
  target_link_libraries(${TARGETNAME} PUBLIC DFT)

#  target_link_libraries(${TARGETNAME} PUBLIC HighFive)
#  target_include_directories(${TARGETNAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")
#  target_include_directories(${TARGETNAME} PUBLIC "${CMAKE_BINARY_DIR}/include")
    
  target_compile_definitions(${TARGETNAME} PRIVATE
    "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
 
endmacro()

add_subdirectory(main)

