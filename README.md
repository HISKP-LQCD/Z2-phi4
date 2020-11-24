# $Z_2-\phi^4$ code 

# Download and Installation

In order to fetch the code:
```
$ git clone git@github.com:HISKP-LQCD/Z2-phi4.git
$ cd Z2-phi4
$ git submodule update --init --recursive
``` 
## Building

To build the program you may create a build directory and copy inside the build script do_cmake.sh

```
   mkdir build
   cd build
   cp ../do_cmake.sh .
   bash do_cmake.sh
   make
```
if you want to build the program somewere else you need to change the path inside the script do_cmake.sh (src_dir=where_I_create_the_build_directory).
If the compilation goes well a folder main will be created with the executable

## Running 

copy the example for the infile (example.in) and create the output directory. The output directory is specified in the example.in file ad  outpath
```
    cp ../../example.in  .
    mkdir data
   ./main -i example.in
```

## To do
 - [x] Random number generator: randlux serial, Mersenne twister parallel
 - [ ] Cluster update
   - add routines for $\mu=g=0$
 - [ ] Appending
 - [ ] Header to the confs
 - [ ] Measurements
 - [ ] parallelization ?
   - [x] OMP : Kokkos 
   - [ ] MPI ?
