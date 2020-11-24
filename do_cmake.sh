rm -r CMakeFiles CMakeCache.txt

src_dir=..
CXXFLAGS="-O3 -mtune=native -march=native -g" \
  cmake \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DKokkos_ENABLE_OPENMP=ON \
  ${src_dir}
#  -DKokkos_ARCH_SKX=ON \
