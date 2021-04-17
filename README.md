# $Z_2-\phi^4$ code 

# Download and Installation

In order to fetch the code:
```
$ git clone git@github.com:HISKP-LQCD/Z2-phi4.git
$ cd Z2-phi4
$ git submodule update --init --recursive
``` 
## Building

To build the program you may create a build directory and copy inside one of the the build script  of the folder example_script. One example 
```
  do_cmake_m100.sh
```
to build  on marconi100 (Nvidia volta 100).

```
   mkdir build
   cd build
   cp ../example_script/do_cmake_m100.sh .
   bash do_cmake_m100.sh
   make
```
if you want to build the program somewhere else you need to change the path inside the script do_cmake.sh (src_dir=where_I_create_the_build_directory).
If the compilation goes well a folder main will be created with the executable.
you can add option

```
-DCMAKE_CXX_FLAGS=-DTIMER
-DCMAKE_CXX_FLAGS=-DDEBUG
-DFFTW=ON
-DCUFFT=ON
```
in the building scritp to have more extra info on the timer

## Running 

copy the example for the infile (example.in) and create the output directory. The output directory is specified in the example.in file ad  outpath
```
    cp ../../example.in  .
    mkdir data
   ./main -i example.in
```

## Tips

for the GPU pascal of Qbig you need at least 25 local metropolis step to
saturate the bandwith 
```
metropolis_local_hits = 25
```


## Output

### Configurations

The configuration are written in binary files with an header, the function to write the configurations are in
```
modules/write_viewer.cpp
````
with the functions
```
void write_header(FILE *f_conf, cluster::IO_params params ,int iconf)
void write_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , const Viewphi &phi  )
```
The configuration will be written in the outpath folder specified in the infile
with a tipical name
```
T4_L4_msq00.130500_msq10.131350_l00.010000_l10.010000_mu0.020000_g0.000000_rep1_conf1
```
### Contraction

The contraction are  written in binary files with an header which differs from the header of the configuration. The header is written with 

```
@ modules/measurements.cpp
void write_header_measuraments(FILE *f_conf, cluster::IO_params params )
````

After the header there is an int (configuration number) and a block of data (sizeof(double) * T * number_of_correlators) corresponding to the contraction done with that configuration. The clock of data is written with the instructions
```
write (int) configuration_number 

for(int t=0; t<T; t++) {
write (double) correlator 1  at time t
write (double) correlator 2  at time t
write (double) correlator 3  at time t
...
}

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
 - [x] Header to the confs
 - [x] Measurements
 - [ ] parallelization 
   - [x] OMP : Kokkos 
   - [ ] MPI 
