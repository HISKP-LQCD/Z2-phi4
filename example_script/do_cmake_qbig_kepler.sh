#remove chache

rm -r CMakeFiles CMakeCache.txt


module purge

module load foss/2021b
module load CMake/3.21.1-GCCcore-11.2.0
module load CUDA/11.5.0


src_dir=${HOME}/Z2-phi4
CXXFLAGS="-mtune=sandybridge -march=sandybridge -g" \
  cmake \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_COMPILER=${src_dir}/external/kokkos/bin/nvcc_wrapper \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
  -DKokkos_ENABLE_OPENMP=OFF\
  -DKokkos_ENABLE_SERIAL=ON\
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_KEPLER35=ON \
  ${src_dir}
