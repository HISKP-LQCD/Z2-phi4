# $Z_2-\phi^4$ code 
## Building
Create your build directory and compile
```
   mkdir build
   cd build
   cmake ..
   make
```
if the compilation goes well a folder main will be created with the executable

## Running 
copy the example for the infile (example.in) and create the output directory. The output directory is specified in the example.in file ad  outpath
```
    cp ../../example.in  .
    mkdir data
   ./main -i example.in
```

## To do
 - [x] Random number generator randlux
 - [ ] Appending
 - [ ] writing the confs
 - [ ] Cluster update
 - [ ] Measuraments
 - [ ] parallelization ?
   - [ ] OMP ?
   - [ ] MPI ?
