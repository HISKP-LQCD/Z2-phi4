#remove chache

rm -r CMakeFiles CMakeCache.txt

#load modules
module load Core/lmod/6.6
module load Core/settarg/6.6

#export CUDA_ROOT=/usr/local/cuda-9.2
#alias nvcc='/opt/cuda-9.2/bin/nvcc'

src_dir=${HOME}/Z2-phi4
#CXXFLAGS="-Xcompiler -O3 -Xcompiler -mtune=power9 -Xcompiler -mcpu=power9 -Xcompiler -g -Xcompiler -mno-float128 " \
CXXFLAGS="-Xcompiler -O3  -Xcompiler -g -Xcompiler -mno-avx2 " \
  cmake \
  -DCMAKE_CXX_COMPILER=${src_dir}/external/kokkos/bin/nvcc_wrapper \
  -DKokkos_ENABLE_OPENMP=ON\
  -DKokkos_ENABLE_SERIAL=ON\
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_PASCAL60=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_CXX_STANDARD=14 \
  ${src_dir}
#  -DKokkos_ARCH_SKX=ON \
#  -DKokkos_ARCH_VOLTA70=ON \
#  -DKokkos_ARCH_POWER9=ON \
#  -DCUDA_ROOT=/usr/local/cuda-9.2 \
