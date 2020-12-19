#remove chache

rm -r CMakeFiles CMakeCache.txt

#load modules
source load_modules.sh

src_dir=${HOME}/Z2-phi4
CXXFLAGS="-Xcompiler -O3 -Xcompiler -mtune=power9 -Xcompiler -mcpu=power9 -Xcompiler -g -Xcompiler -mno-float128 " \
  cmake \
  -DCMAKE_CXX_COMPILER=${src_dir}/external/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_POWER9=ON \
  -DKokkos_ENABLE_OPENMP=ON\
  -DKokkos_ENABLE_SERIAL=ON\
  -DKokkos_ARCH_VOLTA70=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_CXX_STANDARD=14 \
  ${src_dir}
#  -DKokkos_ARCH_SKX=ON \
