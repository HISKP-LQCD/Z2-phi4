#remove chache

rm -r CMakeFiles CMakeCache.txt

#load modules
module load spectrum_mpi/10.3.1--binary
module load cuda/10.1
module load fftw/3.3.8--spectrum_mpi--10.3.1--binary
module load gnu/8.4.0
module load zlib/1.2.11--gnu--8.4.0
module load szip/2.1.1--gnu--8.4.0
module load hdf5/1.12.0--spectrum_mpi--10.3.1--binary
module load python/3.8.2
module load cmake/3.17.1
module load essl/6.2.1--binary
module load boost/1.72.0--spectrum_mpi--10.3.1--binary


src_dir=${HOME}/Z2-phi4
#CXXFLAGS="-Xcompiler -O3 -Xcompiler -mtune=power9 -Xcompiler -mcpu=power9 -Xcompiler -g -Xcompiler -mno-float128   " \
  cmake \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ARCH_POWER9=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ARCH_VOLTA70=OFF \
  -DKokkos_ENABLE_CUDA=OFF\
  -DKokkos_ENABLE_CUDA_LAMBDA=OFF \
  ${src_dir}
#  -DKokkos_ARCH_SKX=ON \
