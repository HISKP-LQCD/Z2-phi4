#ifndef DFT_H
#define DFT_H

void  compute_FT(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip);
#ifdef DEBUG
    void test_FT(cluster::IO_params params );
    #ifdef FFTW
        void test_FT_vs_FFTW(cluster::IO_params params);
    #endif    
#endif
    
#endif