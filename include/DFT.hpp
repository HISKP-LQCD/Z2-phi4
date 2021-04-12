#ifndef DFT_H
#define DFT_H

void  compute_FT(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip);
#ifdef KOKKOS_ENABLE_CUDA
#ifdef cuFFT
void  compute_cuFFT(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip);
#endif
#endif

#ifdef DEBUG
    void test_FT(cluster::IO_params params );
    #ifdef FFTW
        void test_FT_vs_FFTW(cluster::IO_params params);
    #endif    
#endif
    
#endif
