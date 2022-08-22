#define DFT_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include <complex>

#ifdef FFTW
#include <fftw3.h>
#endif

#ifdef cuFFT
#include <cufft.h>
#endif

//#ifndef cuFFT
void compute_FT(const Viewphi phi, cluster::IO_params params, Viewphip& phip) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm0 = sqrt(2 * params.data.kappa0);
    double norm1 = sqrt(2 * params.data.kappa1);
    int  L1 = params.data.L[1], L2 = params.data.L[2], L3 = params.data.L[3];

    //Viewphi phip("phip",2,params.data.L[0]*Vp);

    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    Kokkos::complex<double> I(0, 1);

    Kokkos::parallel_for("FT_loop", team_policy(T * Vp * 2, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int ii = teamMember.league_rank();
        //ii = comp+ 2*myt
        //myt=  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        double norm[2] = { norm0,norm1 };// need to be inside the loop for cuda<10
        const int p = ii / (4 * T);
        int res = ii - p * 4 * T;
        const int reim = res / (2 * T);
        res -= reim * 2 * T;
        const int t = res / 2;
        const int comp = res - 2 * t;

        const int px = p % Lp;
        const int pz = p / (Lp * Lp);
        const int py = (p - pz * Lp * Lp) / Lp;
#ifdef DEBUG
        if (p != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", p, px, py, Lp, pz, Lp, Lp);
            Kokkos::abort("DFT index p");
        }
        if (ii != comp + 2 * (t + T * (reim + p * 2))) {
            printf("error   in the FT\n");
            Kokkos::abort("DFT index comp");
        }
#endif
        const int xp = t + T * (reim + p * 2);
        phip(comp, xp) = 0;
        if (reim == 0) {
            //	for (size_t x=0;x<Vs;x++){
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const int x, Kokkos::complex<double>& inner) {

                size_t i0 = x + t * Vs;
                int ix = x % L1;
                int iz = x / (L1 * L2);
                int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
                if (x != ix + iy * L1 + iz * L1 * L2) {
                    printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                    Kokkos::abort("DFT index x re");
                }
#endif
                double wr = 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
                wr = cos(wr);


                inner += exp(I * phi(comp, i0)) * wr;
                }, phip(comp, xp));

        }
        else if (reim == 1) {
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const int x, Kokkos::complex<double>& inner) {
                size_t i0 = x + t * Vs;
                int ix = x % L1;
                int iz = x / (L1 * L2);
                int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
                if (x != ix + iy * L1 + iz * L1 * L2) {
                    printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                    Kokkos::abort("DFT index x im");
                }
#endif
                double wr = 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
                wr = sin(wr);

                inner += exp(I * phi(comp, i0)) * wr;
                }, phip(comp, xp));
        }

        phip(comp, xp) = phip(comp, xp) / ((double)Vs * norm[comp]);

    });


}




//void compute_FT_complex(const Viewphi phi, cluster::IO_params params ,  int iconf, complexphi &phip){
void compute_FT_complex(manyphi& phip, int i, const Viewphi& phi, cluster::IO_params params, int pow_n) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm0 = sqrt(2 * params.data.kappa0);
    double norm1 = sqrt(2 * params.data.kappa1);
    int  L1 = params.data.L[1], L2 = params.data.L[2], L3 = params.data.L[3];


    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;


    Kokkos::parallel_for("FT_loop", team_policy(T * Vp * 2 *2, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int ii = teamMember.league_rank();
        //ii = comp+ 2*t+ 2*T*p  
        //p=px+py*Lp+pz*Lp*Lp+  sign*Vp
        double norm[2] = { norm0,norm1 };// need to be inside the loop for cuda<10
        const int s = ii / (2 * T * Vp);
        int res = ii - s * (2 * T * Vp);
        const int p = res / (2 * T);
        res = res - p * (2 * T);
        const int t = res / (2);
        const int comp = res - t * (2);
        const int sign = -s * 2 + 1;// 0-->+ and  1-->-1

        const int px = p % Lp;
        const int pz = p / (Lp * Lp);
        const int py = (p - pz * Lp * Lp) / Lp;
#ifdef DEBUG
        if (p != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", p, px, py, Lp, pz, Lp, Lp);
            Kokkos::abort("DFT index p");
        }
        if (ii != comp + 2 * (t + T * (p + s * Vp))) {
            printf("error   in the FT\n");
            Kokkos::abort("DFT index comp");
        }
#endif
        const int xp = t + T * (p + s * Vp);
        phip(i, comp, xp) = 0;

        //	for (size_t x=0;x<Vs;x++){	
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const int x, Kokkos::complex<double>& inner) {

            size_t i0 = x + t * Vs;
            int ix = x % L1;
            int iz = x / (L1 * L2);
            int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
            if (x != ix + iy * L1 + iz * L1 * L2) {
                printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                Kokkos::abort("DFT index x re");
            }
            if (sign !=-1 && sign!=1 ) {
                printf("sign =%d\n",sign);
                Kokkos::abort("DFT sign  +p -p\n");
            }
#endif
            double wr = sign * 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
            // Kokkos::complex<double> ewr(0, 1);
            // ewr = exp( ewr * (-wr + pow_n * phi(comp, i0)));
            double theta=-wr + pow_n * phi(comp, i0);
            Kokkos::complex<double> ewr( cos(theta), sin(theta));
            inner += ewr;
            }, phip(i, comp, xp));

        phip(i, comp, xp) = phip(i, comp, xp) / ((double)Vs * norm[comp]);

    });
   

}


//void compute_FT_complex(const Viewphi phi, cluster::IO_params params ,  int iconf, complexphi &phip){
void compute_FT_complex_smearing(manyphi& phip, int i, const Viewphip& phi, cluster::IO_params params, int pow_n) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm0 = sqrt(2 * params.data.kappa0);
    double norm1 = sqrt(2 * params.data.kappa1);
    int  L1 = params.data.L[1], L2 = params.data.L[2], L3 = params.data.L[3];


    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;


    Kokkos::parallel_for("FT_loop", team_policy(T * Vp * 2, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int ii = teamMember.league_rank();
        //ii = comp+ 2*t+ 2*T*p 
        //p=px+py*Lp+pz*Lp*Lp+  minus_sign*Lp*Lp*
        double norm[2] = { norm0,norm1 };// need to be inside the loop for cuda<10
        int p = ii / (2 * T);
        int res = ii - p * 2 * T;
        const int t = res / 2;
        const int comp = res - 2 * t;

        const int sign = (p / Vp) * 2 - 1; //map (p / Vp)=0,1 --> -1,+1
        const int pp = p - sign * Vp;
        const int px = pp % Lp;
        const int pz = pp / (Lp * Lp);
        const int py = (pp - pz * Lp * Lp) / Lp;
#ifdef DEBUG
        if (pp != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", pp, px, py, Lp, pz, Lp, Lp);
            Kokkos::abort("DFT index p");
        }
        if (ii != comp + 2 * (t + T * (p))) {
            printf("error   in the FT\n");
            Kokkos::abort("DFT index comp");
        }
#endif
        const int xp = t + T * (p);
        phip(i, comp, xp) = 0;

        //	for (size_t x=0;x<Vs;x++){	
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const int x, Kokkos::complex<double>& inner) {

            size_t i0 = x + t * Vs;
            int ix = x % L1;
            int iz = x / (L1 * L2);
            int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
            if (x != ix + iy * L1 + iz * L1 * L2) {
                printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                Kokkos::abort("DFT index x re");
            }
#endif
            double wr = sign * 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
            Kokkos::complex<double> ewr(0, 1);
            ewr = exp(-ewr * (wr + pow_n * phi(comp, i0)));

            inner += ewr;
            }, phip(i, comp, xp));



        phip(i, comp, xp) = phip(i, comp, xp) / ((double)Vs * norm[comp]);

    });


}


void compute_smearing3FT(manyphi& phip, int i, const Viewphi phi, cluster::IO_params params) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm0 = sqrt(2 * params.data.kappa0);
    double norm1 = sqrt(2 * params.data.kappa1);
    int  L1 = params.data.L[1], L2 = params.data.L[2], L3 = params.data.L[3];

    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    Kokkos::complex<double> I(0, 1);

    Kokkos::parallel_for("FT_loop", team_policy(T * Vp * 2, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int ii = teamMember.league_rank();
        //ii = comp+ 2*myt  
        //myt=  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        double norm[2] = { norm0,norm1 };// need to be inside the loop for cuda<10
        int p = ii / (2 * T);
        int res = ii - p * 2 * T;
        const int t = res / 2;
        const int comp = res - 2 * t;

        const int sign = (p / Vp) * 2 - 1; //map (p / Vp)=0,1 --> -1,+1
        const int pp = p - sign * Vp;
        const int px = pp % Lp;
        const int pz = pp / (Lp * Lp);
        const int py = (pp - pz * Lp * Lp) / Lp;
#ifdef DEBUG
        if (pp != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", pp, px, py, Lp, pz, Lp, Lp);
            Kokkos::abort("DFT index p");
        }
        if (ii != comp + 2 * (t + T * (p))) {
            printf("error   in the FT\n");
            Kokkos::abort("DFT index comp");
        }
#endif
        const int xp = t + T * (p);
        phip(i, comp, xp) = 0;

        //	for (size_t x=0;x<Vs;x++){	
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const int x, Kokkos::complex<double>& inner) {

            int ix = x % L1;
            int iz = x / (L1 * L2);
            int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
            if (x != ix + iy * L1 + iz * L1 * L2) {
                printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                Kokkos::abort("DFT index x re");
            }
#endif
            double wr = sign * 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
            Kokkos::complex<double> ewr;
            ewr.real() = 0; ewr.imag() = 1;
            ewr = exp(-ewr * wr);

            Kokkos::complex<double> neighbourSum = 0;
            //x
            int xp1 = ((ix + 1) % L1) + iy * L1 + iz * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            xp1 = ((ix + L1 - 1) % L1) + iy * L1 + iz * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            //y 
            xp1 = ix + ((iy + 1) % L2) * L1 + iz * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            xp1 = ix + ((iy + L2 - 1) % L2) * L1 + iz * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            //z
            xp1 = ix + iy * L1 + ((iz + 1) % L3) * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            xp1 = ix + iy * L1 + ((iz + L3 - 1) % L3) * L1 * L2 + t * Vs;
            neighbourSum += exp(I * phi(comp, xp1));
            // sum also the center
            neighbourSum += exp(I * phi(comp, xp1));

            ewr *= neighbourSum * neighbourSum * neighbourSum;

            inner += ewr;
            }, phip(i, comp, xp));



        phip(i, comp, xp) = phip(i, comp, xp) / ((double)Vs * norm[comp]);

    });


}



#ifdef SCRATCHPAD
// not enough memory in kepler
void compute_FT_scratchpad(manyphi& phip, int i, const Viewphi phi, cluster::IO_params params, int pow_n) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm0 = sqrt(2 * params.data.kappa0);
    double norm1 = sqrt(2 * params.data.kappa1);
    int  L1 = params.data.L[1], L2 = params.data.L[2], L3 = params.data.L[3];


    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    typedef Kokkos::View< double*,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        ScratchViewType;
    int scratch_size = ScratchViewType::shmem_size(Vs);

    Kokkos::parallel_for("FT_loop",
        team_policy(T * Vp, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type & teamMember) {
        const int ii = teamMember.league_rank();

        double norm[2] = { norm0,norm1 };// need to be inside the loop for cuda<10
        const int p = ii / (2 * T);
        int res = ii - p * 2 * T;
        const int t = res / 2;
        const int comp = res - 2 * t;

        const int px = p % Lp;
        const int pz = p / (Lp * Lp);
        const int py = (p - pz * Lp * Lp) / Lp;
#ifdef DEBUG
        if (p != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", p, px, py, Lp, pz, Lp, Lp);
            Kokkos::abort("DFT index p");
        }
        if (ii != comp + 2 * (t + T * (p))) {
            printf("error   in the FT\n");
            Kokkos::abort("DFT index comp");
        }
#endif
        const int xp = t + T * (p);
        phip(i, comp, xp) = 0;

        ScratchViewType s_x(teamMember.team_scratch(0), scratch_size);
        if (teamMember.team_rank() == 0) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, Vs), [&](size_t ix) {
                s_x(ix) = phi(comp, ix);
                });
        }
        teamMember.team_barrier();
        //	for (size_t x=0;x<Vs;x++){	
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const size_t x, Kokkos::complex<double>& inner) {

            size_t i0 = x + t * Vs;
            int ix = x % L1;
            int iz = x / (L1 * L2);
            int iy = (x - iz * L1 * L2) / L1;
#ifdef DEBUG
            if (x != ix + iy * L1 + iz * L1 * L2) {
                printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, L1, iz, L1, L2);
                Kokkos::abort("DFT index x re");
            }
#endif
            double wr = 6.28318530718 * (px * ix / (double(L1)) + py * iy / (double(L2)) + pz * iz / (double(L3)));
            Kokkos::complex<double> ewr;
            ewr.real() = 0; ewr.imag() = 1;
            ewr = exp(-ewr * wr);
            for (int n = 0; n < pow_n;n++)
                ewr *= s_x(i0);

            inner += ewr;
            }, phip(i, comp, xp));



        phip(i, comp, xp) = phip(i, comp, xp) / ((double)Vs * norm[comp]);

    });

#endif // SCRATCHPAD


#ifdef DEBUG
    void test_FT(cluster::IO_params params) {
        size_t V = params.data.V;
        Viewphi  phi("phi", 2, V);
        printf("checking FT of constant field:\n");
        double kappa0 = params.data.kappa0;
        double kappa1 = params.data.kappa1;

        Kokkos::parallel_for("init_const_phi", V, KOKKOS_LAMBDA(size_t x) {
            phi(0, x) = 0;// the FT routines convert in to phisical phi 
            phi(1, x) = 0;
        });
        // Viewphip  phip_test("phip_test", 2, params.data.L[0] * Vp);
        // Viewphip::HostMirror h_phip_test("h_phip_test", 2, params.data.L[0] * Vp);
        manyphi phip_test("phip_test", 1, 2, params.data.L[0] * Vp * 2);
        manyphi::HostMirror h_phip_test("h_phip_test", 1, 2, params.data.L[0] * Vp * 2);
        // compute_FT(phi, params, phip_test);
        compute_FT_complex(phip_test, 0, phi, params, 1);

        Kokkos::parallel_for("init_const_phi", params.data.L[0] * Vp * 2, KOKKOS_LAMBDA(size_t x) {
            phip_test(0, 0, x) *= sqrt(2. * kappa0);// the FT routines convert in to phisical phi 
            phip_test(0, 1, x) *= sqrt(2. * kappa1);
        });
        // Deep copy device views to host views.
        Kokkos::deep_copy(h_phip_test, phip_test);

        int T = params.data.L[0];
        for (int t = 0; t < T; t++) {
            for (int x = 1; x < Vp * 2; x++) {
                if(x!=Vp){
                    int id = t + x * T;
                    if (fabs(h_phip_test(0, 0, id).real()) > 1e-11 || fabs(h_phip_test(0, 1, id).real()) > 1e-11) {
                        printf("error FT of a constant field do not gives delta_{p,0}: \n");
                        printf("h_phip_test(0,%d)=%.12g \n", x, h_phip_test(0, 0, id).real());
                        printf("h_phip_test(1,%d)=%.12g \n", x, h_phip_test(0, 1, id).real());
                        printf("id=t+T*p    id=%d   t=%d  p=%d\n ", id, t, x);
                        // exit(1);
                    }
                }
            }
            if (fabs(h_phip_test(0, 0, t).real() - 1) > 1e-11 || fabs(h_phip_test(0, 1, t).real() - 1) > 1e-11) {
                printf("error FT of a constant field do not gives delta_{p,0}: \n");
                printf("h_phip_test(0,%d)=%.12g \n", t, h_phip_test(0, 0, t).real());
                printf("h_phip_test(1,%d)=%.12g \n", t, h_phip_test(0, 1, t).real());
                printf("id=t+T*p    id=%d   t=%d  p=0\n ", t, t);
                // exit(1);
            }
            if (fabs(h_phip_test(0, 0, t+T * Vp ).real() - 1) > 1e-11 || fabs(h_phip_test(0, 1, t+T * Vp).real() - 1) > 1e-11) {
                printf("error FT of a constant field do not gives delta_{p,0}: \n");
                printf("h_phip_test(0,%d)=%.12g \n", t, h_phip_test(0, 0, t+T * Vp).real());
                printf("h_phip_test(1,%d)=%.12g \n", t, h_phip_test(0, 1, t+T * Vp).real());
                printf("id=t+T*p    id=%d   t=%d  p=0\n ", t+T * Vp, t);
                // exit(1);
            }
        }
        printf("\tpassed\n");
        // // this test can not be done for the complex field since the field the phase
        // printf("checking FT of delta_x,0 field:\n");
        // Kokkos::parallel_for("init_phi", V, KOKKOS_LAMBDA(size_t x) {
        //     if (x == 0) {
        //         phi(0, x) = 1;
        //         phi(1, x) = 1;
        //     }
        //     else {
        //         phi(0, x) = 0;// the FT routines convert in to phisical phi 
        //         phi(1, x) = 0;
        //     }
        // });
        // // compute_FT(phi, params, phip_test);
        // compute_FT_complex(phip_test, 0, phi, params, 1);
        // Kokkos::parallel_for("init_phi", params.data.L[0] * Vp, KOKKOS_LAMBDA(size_t x) {
        //     phip_test(0, 0, x) *= Vs * sqrt(2. * kappa0);// the FT routines convert in to phisical phi 
        //     phip_test(0, 1, x) *= Vs * sqrt(2. * kappa1);
        // });
        // // Deep copy device views to host views.
        // Kokkos::deep_copy(h_phip_test, phip_test);
        // for (size_t t = 0; t < 1; t++) {
        //     for (size_t x = 0; x < Vp; x++) {
        //         size_t id = t + x * T;
        //         if (x % 2 == 0) {//real part
        //             if (fabs(h_phip_test(0, 0, id).real() - 1) > 1e-11 || fabs(h_phip_test(0, 1, id).real() - 1) > 1e-11) {
        //                 printf("error FT of a delta_{x,0} field do not gives const: \n");
        //                 printf("h_phip_test(0,%ld)=%.12g \n", x, h_phip_test(0, 0, id).real());
        //                 printf("h_phip_test(1,%ld)=%.12g \n", x, h_phip_test(0, 1, id).real());
        //                 printf("id=t+T*p    id=%ld   t=%ld  p=%ld\n ", id, t, x);
        //                 //       exit(1);
        //             }
        //         }
        //         if (x % 2 == 1) {//imag part
        //             if (fabs(h_phip_test(0, 0, id).real()) > 1e-11 || fabs(h_phip_test(0, 1, id).real()) > 1e-11) {
        //                 printf("error FT of a delta_{x,0} field do not gives const: \n");
        //                 printf("h_phip_test(0,%ld)=%.12g \n", x, h_phip_test(0, 0, id).real());
        //                 printf("h_phip_test(1,%ld)=%.12g \n", x, h_phip_test(0, 1, id).real());
        //                 printf("id=t+T*p    id=%ld   t=%ld  p=%ld\n ", id, t, x);
        //                 //     exit(1);
        //             }
        //         }
        //     }
        // }
        // printf("\tpassed\n");
    }

#ifdef FFTW
    void test_FT_vs_FFTW(cluster::IO_params params) {

        size_t V = params.data.V;
        size_t Vs = V / params.data.L[0];
        int T = params.data.L[0];
        double kappa0 = params.data.kappa0;
        double kappa1 = params.data.kappa1;

        Viewphi  phi("phi", 2, V);
        printf("checking FT vs FFTW\n");
        //kokkos
        Viewphi::HostMirror  h_phi = Kokkos::create_mirror_view(phi);
        Kokkos::parallel_for("init_phi", V, KOKKOS_LAMBDA(size_t x) {
            phi(0, x) = x * sqrt(2. * kappa0);// the FT routines convert in to phisical phi 
            phi(1, x) = x * sqrt(2. * kappa1);
        });
        // Deep copy device views to host views.
        Kokkos::deep_copy(h_phi, phi);

        manyphi phip_test("phip", 1, 2, params.data.L[0] * Vp / 2);
        manyphi::HostMirror h_phip_test = Kokkos::create_mirror_view(phip_test);
        // compute_FT(phi, params , phip_test);
        compute_FT_complex(phip_test, 0, phi, params, 1);
        // Deep copy device views to host views.
        Kokkos::deep_copy(h_phip_test, phip_test);

        //FFTW
        fftw_plan p;
        fftw_complex* in;
        fftw_complex* out;
        int n[3];

        n[0] = params.data.L[1];
        n[1] = params.data.L[2];
        n[2] = params.data.L[3];

        in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Vs);
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Vs);

        //FFTW_FORWARD=e^{-ipx}   FFTW_BACKWARD=e^{+ipx}
        p = fftw_plan_dft(3, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        for (int t = 0; t < T;t++) {
            for (int x = 0; x < Vs; x++) {
                in[x][0] = h_phi(0, x + t * Vs);// h_phi should be avaluated at x+0*L^3 but it is the same
                in[x][1] = 0;
            }
            fftw_execute(p);
            for (int x = 0; x < Vs; x++) {
                out[x][0] /= ((double)Vs) * sqrt(2. * params.data.kappa0);// h_phi should be avaluated at x+0*L^3 but it is the same
                out[x][1] /= ((double)Vs) * sqrt(2. * params.data.kappa0);
                // printf("%g\t",out[x][1]);
               // out[x][1]/=-1;
                // printf("%g\n",out[x][1]);
            }
            for (int px = 0; px < Lp; px++) {
                for (int py = 0; py < Lp; py++) {
                    for (int pz = 0; pz < Lp; pz++) {
                        int p = px + py * params.data.L[1] + pz * params.data.L[1] * params.data.L[2];
                        int lp = px + py * Lp + pz * Lp * Lp;
                        if (fabs(out[p][0] - h_phip_test(0, 0, t + T * (0 + lp)).real()) > 1e-6) {
                            printf("error: FT does not produce the same result of FFTW (real part):");
                            printf("t=%d p=%d px=%d  py=%d  pz=%d\t", t, p, px, py, pz);
                            printf("real: FFTW=%.15g    FT=%.15g \n", out[p][0], h_phip_test(0, 0, t + T * (0 + lp)).real());
                            Kokkos::abort("");
                        }
                        if (fabs(out[p][1] - h_phip_test(0, 0, t + T * (0 + lp)).imag()) > 1e-6) {
                            printf("error: FT does not produce the same result of FFTW (imag part):");
                            printf("t=%d p=%d px=%d  py=%d  pz=%d\t", t, p, px, py, pz);
                            printf(" imag: FFTW=%.15g    FT=%.15g \n", out[p][1], h_phip_test(0, 0, t + T * (0 + lp)).imag());
                            Kokkos::abort("");
                        }


                    }
                }
            }
        }


        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
        printf("\tpassed\n");

    }
#endif  //FFTW
#endif


#ifdef KOKKOS_ENABLE_CUDA
#ifdef cuFFT
    void compute_cuFFT(const Viewphi phi, cluster::IO_params params, int iconf, Viewphi::HostMirror & h_phip) {
        int T = params.data.L[0];
        size_t Vs = params.data.V / T;
        double kappa0 = params.data.kappa0;
        double kappa1 = params.data.kappa1;


        Viewphi Kphi("Kphip", 2, params.data.L[0] * Vp);
        cufftHandle plan;
        cufftReal* idata;
        cufftComplex* odata;


        cudaMalloc((void**)&idata, sizeof(cufftReal) * Vs);
        cudaMalloc((void**)&odata, sizeof(cufftComplex) * Vs);
        int L[3] = { params.data.L[1], params.data.L[2],params.data.L[3] };
        cufftPlanMany(&plan, 3, L,
            NULL, 1, Vs, // *inembed, istride, idist
            NULL, 1, Vs, // *onembed, ostride, odist
            CUFFT_R2C, 2 * T);

        for (int comp = 0;comp < 2;comp++) {
            for (int t = 0;t < T;t++) {
                // lets hope I can initialise cuda things inside kokkos
                Kokkos::parallel_for("Kokkos_to_cuFFT", Vs, KOKKOS_LAMBDA(size_t x) {
                    idata[x] = phi(comp, x + t * Vs);
                    //printf("in: %g   %g\n",idata[x],phi(0,x));
                });

                /* Create a 3D FFT plan. */
                //cufftPlan3d(&plan, params.data.L[1], params.data.L[2],params.data.L[3], CUFFT_R2C);

                /* Use the CUFFT plan to transform the signal out of place. */
                if (cufftExecR2C(plan, idata, odata) != CUFFT_SUCCESS) {
                    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                }

                if (cudaThreadSynchronize() != cudaSuccess) {
                    fprintf(stderr, "Cuda error: Failed to synchronize\n");
                }

                Kokkos::parallel_for("cuFFT_to_Kokkos", Vp, KOKKOS_LAMBDA(size_t pp) {
                    int L[3] = { params.data.L[1], params.data.L[2],params.data.L[3] };

                    int reim = pp % 2;
                    int p = (pp - reim) / 2;
                    const int px = p % Lp;
                    const int pz = p / (Lp * Lp);
                    const int py = (p - pz * Lp * Lp) / Lp;
                    int pcuff = (px + py * (L[1] / 2 + 1) + pz * (L[1] / 2 + 1) * (L[2]));
                    int ip = t + pp * T;
                    double normFT[2] = { Vs * sqrt(2 * kappa0),Vs * sqrt(2 * kappa1) };
                    if (reim == 0)
                        Kphi(comp, ip) = odata[pcuff].x / normFT[comp];
                    else if (reim == 1)
                        Kphi(comp, ip) = -odata[pcuff].y / normFT[comp];
#ifdef DEBUG
                    if (p != px + py * Lp + pz * Lp * Lp)
                        printf("index problem if cuFFT  p=%d  !=  (%d,%d,%d)\n", p, px, py, pz);
                    if (pp != reim + p * 2)
                        printf("index problem if cuFFT  pp=%d  !=  %d+%d*2\n", pp, reim, p);
#endif
                });


#ifdef DEBUG
                Viewphi phip("phip", 2, params.data.L[0] * Vp);
                compute_FT(phi, params, phip);
                Kokkos::parallel_for("check_phi_cuFFT", Vp, KOKKOS_LAMBDA(size_t pp) {
                    int L[3] = { params.data.L[1], params.data.L[2],params.data.L[3] };
                    int reim = pp % 2;
                    int p = (pp - reim) / 2;
                    const int px = p % Lp;
                    const int pz = p / (Lp * Lp);
                    const int py = (p - pz * Lp * Lp) / Lp;
                    int pcuff = (px + py * (L[1] / 2 + 1) + pz * (L[1] / 2 + 1) * (L[2]));
                    int ip = t + pp * T;
                    if (fabs(Kphi(0, ip) - phip(0, ip)) > 1e-6)
                        printf("p=%d= (%d,%d,%d)  reim=%d pp=%ld ip=%d  t=%d pcuff=%d    cuFFT =%g DFT =%g\n", p, px, py, pz, reim, pp, ip, t, pcuff, Kphi(comp, ip), phip(comp, ip));
                });
#endif
            }
        }
        /* Destroy the CUFFT plan. */
        cufftDestroy(plan);
        cudaFree(idata); cudaFree(odata);
        // Deep copy device views to host views.
        Kokkos::deep_copy(h_phip, Kphi);
    }
#endif
#endif
