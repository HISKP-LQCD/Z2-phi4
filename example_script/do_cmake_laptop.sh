#remove chache

rm -r CMakeFiles CMakeCache.txt

#load modules


src_dir=..
CXXFLAGS= " -fopenmp   -pedantic -g   -lm -lgmp -lfftw3 "
  cmake \
  ${src_dir}
