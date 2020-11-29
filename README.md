# $Z_2-\phi^4$ code 

# Download and Installation

In order to fetch the code:
```
$ git clone git@github.com:HISKP-LQCD/Z2-phi4.git
$ cd Z2-phi4
$ git submodule update --init --recursive
``` 
## Building

To build the program you may create a build directory and copy inside one of the the build script  of the folder example_script. The script available are
```
  do_cmake_host.sh
```
for a host CPU openMP building and
```
  do_cmake_m100.sh
```
to build the this code on marconi100 (Nvidia volta 100).
To build on the host you have to 

```
   mkdir build
   cd build
   cp ../example_script/do_cmake_host.sh .
   bash do_cmake_host.sh
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
 - [x] Random number generator: 
    -  randlux serial Not used anymore
    -  Mersenne twister parallel Not working on GPU
    - [x] Kokkos random generator: Vigna, Sebastiano (2014). "An
      experimental exploration of Marsaglia's xorshift generators, scrambled" See:
      http://arxiv.org/abs/1402.6246
 - [ ] Cluster update
   - add routines for $\mu=g=0$, not really helpful
 - [ ] Appending
 - [ ] Header to the confs
 - [ ] Measurements
 - [ ] parallelization 
   - [x] OMP : Kokkos 
   - [ ] MPI 
