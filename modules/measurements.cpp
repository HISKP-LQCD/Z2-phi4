#define mesuraments_C 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include "measurements.hpp"
#include <complex>

inline void  compute_contraction_p1(int t, Viewphi::HostMirror h_phip, cluster::IO_params params, FILE* f_G2t, int iconf);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double* compute_magnetisations_serial(Viewphi::HostMirror phi, cluster::IO_params params) {

    double* m = (double*)calloc(2, sizeof(double));
    for (size_t x = 0; x < params.data.V; x++) {
        m[0] += sqrt(phi(0, x) * phi(0, x));
        m[1] += sqrt(phi(1, x) * phi(1, x));
    }
    m[0] *= sqrt(2 * params.data.kappa0) / ((double)params.data.V);
    m[1] *= sqrt(2 * params.data.kappa1) / ((double)params.data.V);
    return m;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double* compute_magnetisations(Viewphi phi, cluster::IO_params params) {

    size_t V = params.data.V; //you can not use params.data.X  on the device
    double* mr = (double*)calloc(2, sizeof(double));
    Kokkos::complex<double>  mrc[2];
    mrc[0].real() = 0; mrc[0].imag() = 0;
    mrc[1].real() = 0; mrc[1].imag() = 0;
    Kokkos::complex<double> I(0, 1);
    for (int comp = 0; comp < 2; comp++) {
        Kokkos::parallel_reduce("magnetization", V, KOKKOS_LAMBDA(const size_t x, Kokkos::complex<double>&inner) {
            inner += exp(I * phi(comp, x));
        }, mrc[comp]);
    }

    mr[0] = sqrt((mrc[0] * conj(mrc[0])).real()) / ((double)params.data.V);
    mr[1] = sqrt((mrc[1] * conj(mrc[1])).real()) / ((double)params.data.V);

    return mr;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void compute_energy(Viewphi phi, cluster::IO_params params, FILE* file) {

    size_t V = params.data.V; //better not use params.data.X  on the device since params.data contains std::string that do not exist in CUDA
    std::array<double, 3>  E = { 0,0,0 };

    int L[4] = { params.data.L[0], params.data.L[1], params.data.L[2], params.data.L[3] };
    const int V2 = L[1] * L[2];
    const int V3 = V2 * L[3];

    for (int comp = 0; comp < 2; comp++) {
        Kokkos::parallel_reduce("magnetization", V, KOKKOS_LAMBDA(const size_t x, double& inner) {
            Kokkos::complex<double> neighbourSum(0, 0);
            Kokkos::complex<double> I(0, 1);
            // direction  0
            int xp = x / (V3);
            xp = x + (-xp + (xp + 1) % L[0]) * (V3);
            neighbourSum = exp(I * phi(comp, xp));
            // direction z
            xp = (x % (V3)) / (V2);
            xp = x + (-xp + (xp + 1) % L[3]) * (V2);
            neighbourSum += exp(I * phi(comp, xp));
            // direction 1
            xp = (x % (L[1]));
            xp = x + (-xp + (xp + 1) % L[1]);
            neighbourSum += exp(I * phi(comp, xp));
            // direction 2
            xp = (x % (V2)) / L[1];
            xp = x + (-xp + (xp + 1) % L[2]) * L[1];
            neighbourSum += exp(I * phi(comp, xp));

            inner += (exp(-I * phi(comp, x)) * neighbourSum).real();
        }, E[comp]);
        E[comp] = E[comp] / ((double)params.data.V);
    }

    Kokkos::parallel_reduce("g_term", V, KOKKOS_LAMBDA(const size_t x, double& inner) {
        Kokkos::complex<double> I(0, 1);
        inner += (exp(-I * phi(1, x) + 3. * I * phi(0, x))).real();
    }, E[2]);
    E[2] = E[2] / ((double)params.data.V);

    Kokkos::complex<double>  phip[2] = { 0,0 };
    for (int comp = 0; comp < 2; comp++) {
        Kokkos::parallel_reduce("chi0", V, KOKKOS_LAMBDA(const size_t x, Kokkos::complex<double>&inner) {
            Kokkos::complex<double> I(0, 1);
            inner += exp(I * phi(comp, x));
        }, phip[comp]);
        phip[comp] = phip[comp] / ((double)params.data.V);
    }
    double chi20 = (phip[0] * conj(phip[0])).real();
    double chi21 = (phip[1] * conj(phip[1])).real();
    Kokkos::complex<double>  phip1[2] = { 0,0 };
    Kokkos::MDRangePolicy< Kokkos::Rank<4> > mdrange_policy({ 0,0,0,0 }, { L[1],L[2],L[3],L[0] });
    for (int comp = 0; comp < 2; comp++) {
        Kokkos::parallel_reduce("chip", mdrange_policy, KOKKOS_LAMBDA(const size_t x, const size_t y,
            const size_t z, const size_t t, Kokkos::complex<double>&inner) {
            Kokkos::complex<double> I(0, 1);
            double px = twoPI * x / ((double)L[1]);
            size_t ix = x + y * L[1] + z * L[1] * L[2] + t * L[1] * L[2] * L[3];
            inner += exp(I * (px + phi(comp, ix)));
        }, phip1[comp]);
        phip1[comp] = phip1[comp] / ((double)params.data.V);
    }
    double chi20p1 = (phip1[0] * conj(phip1[0])).real();
    double chi21p1 = (phip1[1] * conj(phip1[1])).real();

    double chi40 = (phip[0] * phip[0] * conj(phip[0] * phip[0])).real();
    double chi41 = (phip[1] * phip[1] * conj(phip[1] * phip[1])).real();
    double chi42 = (phip[1] * phip[1] * conj(phip[0] * phip[0])).real();
    double chi43 = (conj(phip[1]) * phip[1] * phip[0] * phip[0]).real();

    fprintf(file, "%.15g   %.15g  %.15g  ", E[0], E[1], E[2]);
    fprintf(file, "%.15g   %.15g  ", chi20, chi21);
    fprintf(file, "%.15g   %.15g  ", chi20p1, chi21p1);
    fprintf(file, "%.15g   %.15g  %.15g  %.15g\n", chi40, chi41, chi42, chi43);

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void check_spin(Viewphi phi, cluster::IO_params params) {

    size_t V = params.data.V; //you can not use params.data.X  on the device

    for (int comp = 0; comp < 2; comp++) {
        Kokkos::parallel_for("check spin", V, KOKKOS_LAMBDA(const size_t x) {
            if (phi(comp, x) != -1 && phi(comp, x) != 1) {
                printf("field is not +-1 at phi(comp=%i,x=%ld)=%g\n", comp, x, (double)phi(comp, x));
                Kokkos::abort("");
            }
        });
    }

}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void write_header_measuraments(FILE* f_conf, cluster::IO_params params, int ncorr) {

    fwrite(&params.data.L, sizeof(int), 4, f_conf);

    fwrite(params.data.formulation.c_str(), sizeof(char) * 100, 1, f_conf);

    fwrite(&params.data.msq0, sizeof(double), 1, f_conf);
    fwrite(&params.data.msq1, sizeof(double), 1, f_conf);
    fwrite(&params.data.lambdaC0, sizeof(double), 1, f_conf);
    fwrite(&params.data.lambdaC1, sizeof(double), 1, f_conf);
    fwrite(&params.data.muC, sizeof(double), 1, f_conf);
    fwrite(&params.data.gC, sizeof(double), 1, f_conf);

    fwrite(&params.data.metropolis_local_hits, sizeof(int), 1, f_conf);
    fwrite(&params.data.metropolis_global_hits, sizeof(int), 1, f_conf);
    fwrite(&params.data.metropolis_delta, sizeof(double), 1, f_conf);

    fwrite(&params.data.cluster_hits, sizeof(int), 1, f_conf);
    fwrite(&params.data.cluster_min_size, sizeof(double), 1, f_conf);

    fwrite(&params.data.seed, sizeof(int), 1, f_conf);
    fwrite(&params.data.replica, sizeof(int), 1, f_conf);


    //number of correlators
    //int ncorr=33;//number of correlators
    fwrite(&ncorr, sizeof(int), 1, f_conf);

    size_t size = params.data.L[0] * ncorr;  // number of double of each block
    fwrite(&size, sizeof(size_t), 1, f_conf);

}
/////////////////////////////////////////////////
namespace sample {  // namespace helps with name resolution in reduction identity 
    template< class ScalarType, int N >
    struct array_type {
        ScalarType the_array[N];

        KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
            array_type() {
            for (int i = 0; i < N; i++) { the_array[i] = 0; }
        }
        KOKKOS_INLINE_FUNCTION   // Copy Constructor
            array_type(const array_type& rhs) {
            for (int i = 0; i < N; i++) {
                the_array[i] = rhs.the_array[i];
            }
        }
        KOKKOS_INLINE_FUNCTION   // add operator
            array_type& operator += (const array_type& src) {
            for (int i = 0; i < N; i++) {
                the_array[i] += src.the_array[i];
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION   // volatile add operator 
            void operator += (const volatile array_type& src) volatile {
            for (int i = 0; i < N; i++) {
                the_array[i] += src.the_array[i];
            }
        }
    };
    //momentum 0,1 for x,y,z  re_im
    //typedef array_type<double,7> ValueType;  // used to simplify code below
    typedef array_type<double, 2>  array2;
    typedef array_type<double, 7>  array7;

    /////////////////////two component 
    template< class ScalarType, int N >
    struct two_component {
        ScalarType the_array[2][N];

        KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
            two_component() {
            for (int i = 0; i < N; i++) {
                the_array[0][i] = 0;
                the_array[1][i] = 0;
            }
        }
        KOKKOS_INLINE_FUNCTION   // Copy Constructor
            two_component(const two_component& rhs) {
            for (int i = 0; i < N; i++) {
                the_array[0][i] = rhs.the_array[0][i];
                the_array[1][i] = rhs.the_array[1][i];
            }
        }
        KOKKOS_INLINE_FUNCTION   // add operator
            two_component& operator += (const two_component& src) {
            for (int i = 0; i < N; i++) {
                the_array[0][i] += src.the_array[0][i];
                the_array[1][i] += src.the_array[1][i];
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION   // volatile add operator 
            void operator += (const volatile two_component& src) volatile {
            for (int i = 0; i < N; i++) {
                the_array[0][i] += src.the_array[0][i];
                the_array[1][i] += src.the_array[1][i];
            }
        }
    };
    typedef two_component<double, 7>  two_component7;
    typedef two_component<double, 8>  two_component8;
    typedef two_component<double, 128>  two_component128;
    typedef two_component<double, Vp>  two_componentVp;

}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<int N>
    struct reduction_identity< sample::array_type<double, N> > {
        KOKKOS_FORCEINLINE_FUNCTION static sample::array_type<double, N> sum() {
            return sample::array_type<double, N>();
        }
    };
    template<int N>
    struct reduction_identity< sample::two_component<double, N> > {
        KOKKOS_FORCEINLINE_FUNCTION static sample::two_component<double, N> sum() {
            return sample::two_component<double, N>();
        }
    };

}

/*
void compute_FT_old(const Viewphi phi, cluster::IO_params params, Viewphi::HostMirror& h_phip) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;
    double norm[2] = { sqrt(2. * params.data.kappa0),sqrt(2. * params.data.kappa1) };

    for (int t = 0; t < T; t++) {
        //for(int comp=0; comp<2; comp++){
        //    for (int p =0 ; p< Vp;p++)
        //        h_phip(comp,t+p*T)=0;
        //}
        sample::two_componentVp pp;
        Kokkos::parallel_reduce("FT_Vs_loop", Vs, KOKKOS_LAMBDA(const size_t x, sample::two_componentVp & upd) {
            size_t i0 = x + t * Vs;
            int ix = x % params.data.L[1];
            int iz = x / (params.data.L[1] * params.data.L[2]);
            int iy = (x - iz * params.data.L[1] * params.data.L[2]) / params.data.L[1];
#ifdef DEBUG
            if (x != ix + iy * params.data.L[1] + iz * params.data.L[1] * params.data.L[2]) {
                printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, params.data.L[1], iz, params.data.L[1], params.data.L[2]);
                Kokkos::abort("FT_Vs_loop");
            }
#endif
            for (int pz = 0; pz < Lp;pz++) {
                for (int py = 0; py < Lp;py++) {
                    for (int px = 0; px < Lp;px++) {
                        int re = (px + py * Lp + pz * Lp * Lp) * 2;
                        int im = re + 1;
                        double wr = 6.28318530718 * (px * ix / (double(params.data.L[1])) + py * iy / (double(params.data.L[2])) + pz * iz / (double(params.data.L[3])));
                        double wi = sin(wr);
                        wr = cos(wr);
                        for (int comp = 0; comp < 2; comp++) {
                            upd.the_array[comp][re] += phi(comp, i0) * wr;
                            upd.the_array[comp][im] += phi(comp, i0) * wi;
                        }
                    }
                }
            }


        }, Kokkos::Sum<sample::two_componentVp>(pp));
        //  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        for (int comp = 0; comp < 2; comp++) {
            for (int reim_p = 0; reim_p < Vp;reim_p++) // reim_p= (reim+ p*2)= 0,..,127
                h_phip(comp, t + reim_p * T) = pp.the_array[comp][reim_p] / ((double)Vs * norm[comp]);
        }
    }
}

void compute_FT_good(const Viewphi phi, cluster::IO_params params, Viewphi::HostMirror& h_phip) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;

    Viewphi phip("phip", 2, params.data.L[0] * Vp);

    //    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    //    typedef Kokkos::TeamPolicy<>::member_type  member_type;
        //for(int t=0; t<T; t++) {
    Kokkos::parallel_for("FT_loop", T * Vp * 2, KOKKOS_LAMBDA(const size_t ii) {
        //ii = comp+ 2*myt
        //myt=  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        double norm[2] = { sqrt(2. * params.data.kappa0),sqrt(2. * params.data.kappa1) };// need to be inside the loop for cuda<10
        const int p = ii / (4 * T);
        int res = ii - p * 4 * T;
        const int reim = res / (2 * T);
        res -= reim * 2 * T;
        const int t = res / 2;
        const int comp = res - 2 * t;

        const int px = p % Lp;
        const int pz = p / (Lp * Lp);
        const int py = (p - pz * Lp * Lp) / Lp;
        const int xp = t + T * (reim + p * 2);
#ifdef DEBUG
        if (p != px + py * Lp + pz * Lp * Lp) {
            printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n", p, px, py, Lp, pz, Lp, Lp); Kokkos::abort("aborting");
        }
#endif
        phip(comp, xp) = 0;
        if (reim == 0) {
            for (size_t x = 0; x < Vs;x++) {
                size_t i0 = x + t * Vs;
                int ix = x % params.data.L[1];
                int iz = x / (params.data.L[1] * params.data.L[2]);
                int iy = (x - iz * params.data.L[1] * params.data.L[2]) / params.data.L[1];
#ifdef DEBUG
                if (x != ix + iy * params.data.L[1] + iz * params.data.L[1] * params.data.L[2]) {
                    printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, params.data.L[1], iz, params.data.L[1], params.data.L[2]);
                    Kokkos::abort("aborting");
                }
#endif
                double wr = 6.28318530718 * (px * ix / (double(params.data.L[1])) + py * iy / (double(params.data.L[2])) + pz * iz / (double(params.data.L[3])));
                wr = cos(wr);

                phip(comp, xp) += phi(comp, i0) * wr;// /((double) Vs *norm[comp]);
            }
        }
        else if (reim == 1) {
            for (size_t x = 0; x < Vs;x++) {
                size_t i0 = x + t * Vs;
                int ix = x % params.data.L[1];
                int iz = x / (params.data.L[1] * params.data.L[2]);
                int iy = (x - iz * params.data.L[1] * params.data.L[2]) / params.data.L[1];
#ifdef DEBUG
                if (x != ix + iy * params.data.L[1] + iz * params.data.L[1] * params.data.L[2]) {
                    printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, params.data.L[1], iz, params.data.L[1], params.data.L[2]);
                    Kokkos::abort("aborting");
                }
#endif
                double wr = 6.28318530718 * (px * ix / (double(params.data.L[1])) + py * iy / (double(params.data.L[2])) + pz * iz / (double(params.data.L[3])));
                wr = sin(wr);

                phip(comp, xp) += phi(comp, i0) * wr;// /((double) Vs *norm[comp]);
            }

        }


        phip(comp, xp) = phip(comp, xp) / ((double)Vs * norm[comp]);

    });
    // Deep copy device views to host views.
    Kokkos::deep_copy(h_phip, phip);


}


void compute_FT_tmp(const Viewphi phi, cluster::IO_params params, Viewphi::HostMirror& h_phip) {
    int T = params.data.L[0];
    size_t Vs = params.data.V / T;

    Viewphi phip("phip", 2, params.data.L[0] * Vp);

    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    //for(int t=0; t<T; t++) {
    Kokkos::parallel_for("FT_loop", team_policy(T, Kokkos::AUTO, 32), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int t = teamMember.league_rank();
        double norm[2] = { sqrt(2. * params.data.kappa0),sqrt(2. * params.data.kappa1) }; // with cuda 10 you can move it outside

        sample::two_componentVp pp;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, Vs), [&](const size_t x, sample::two_componentVp& upd) {
            size_t i0 = x + t * Vs;
            int ix = x % params.data.L[1];
            int iz = x / (params.data.L[1] * params.data.L[2]);
            int iy = (x - iz * params.data.L[1] * params.data.L[2]) / params.data.L[1];
#ifdef DEBUG
            if (x != ix + iy * params.data.L[1] + iz * params.data.L[1] * params.data.L[2]) {
                printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n", x, ix, iy, params.data.L[1], iz, params.data.L[1], params.data.L[2]);
                Kokkos::abort("aborting");
            }
#endif
            for (int pz = 0; pz < Lp;pz++) {
                for (int py = 0; py < Lp;py++) {
                    for (int px = 0; px < Lp;px++) {
                        int re = (px + py * Lp + pz * Lp * Lp) * 2;
                        int im = re + 1;
                        double wr = 6.28318530718 * (px * ix / (double(params.data.L[1])) + py * iy / (double(params.data.L[2])) + pz * iz / (double(params.data.L[3])));
                        double wi = sin(wr);
                        wr = cos(wr);
                        for (int comp = 0; comp < 2; comp++) {
                            upd.the_array[comp][re] += phi(comp, i0) * wr;
                            upd.the_array[comp][im] += phi(comp, i0) * wi;
                        }
                    }
                }
            }


            }, Kokkos::Sum<sample::two_componentVp>(pp));
        //  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        for (int comp = 0; comp < 2; comp++) {
            for (int reim_p = 0; reim_p < Vp;reim_p++) // reim_p= (reim+ p*2)= 0,..,127
                phip(comp, t + reim_p * T) = pp.the_array[comp][reim_p] / ((double)Vs * norm[comp]);
        }
    });
    // Deep copy device views to host views.
    Kokkos::deep_copy(h_phip, phip);


}

*/



KOKKOS_INLINE_FUNCTION int ctolex(int x3, int x2, int x1, int x0, int L, int L2, int L3) {
    return x3 + x2 * L + x1 * L2 + x0 * L3;
}

void smearing_field(Viewphip& sphi, Viewphi& phi, cluster::IO_params params) {
    size_t V = params.data.V;
    double rho = 0.5;
    int R = 3;
    int L = params.data.L[3];
    int L2 = params.data.L[3] * params.data.L[2];
    int L3 = params.data.L[3] * params.data.L[2] * params.data.L[1];
    if (params.data.L[3] != params.data.L[2] || params.data.L[3] != params.data.L[1]) {
        Kokkos::abort("smearing for Lx!=Ly not implemented");
    }


    Kokkos::parallel_for("smearing loop", V, KOKKOS_LAMBDA(int x) {
        //x=x3+ x2*L3+x1*L2*L3 + x0*L1*L2*L3  
        int x0 = x / L3;
        int res = x - x0 * L3;
        int x1 = res / L2;
        res -= x1 * L2;
        int x2 = res / L;
        int x3 = res - x2 * L;
#ifdef DEBUG
        if (x != ctolex(x3, x2, x1, x0, L, L2, L3)) {
            printf("error   %d   = %d  + %d  L+ %d L^2 + %d L^3\n", x, x3, x2, x1, x0);
            Kokkos::abort("DFT index p");
        }
#endif

        sphi(0, x) = -phi(0, x);
        sphi(1, x) = -phi(1, x);
        double w;
        Kokkos::complex<double> I(0, 1);
        for (int dx3 = 0; dx3 < R; dx3++) {
            for (int dx2 = 0; dx2 < R; dx2++) {
                for (int dx1 = 0; dx1 < R; dx1++) {
                    w = exp(-rho * (dx3 * dx3 + dx2 * dx2 + dx1 * dx1));
                    int xp = ctolex((x3 + dx3) % L, (x2 + dx2) % L, (x1 + dx1) % L, x0, L, L2, L3);
                    int xm = ctolex((x3 - dx3 + L) % L, (x2 - dx2 + L) % L, (x1 - dx1 + L) % L, x0, L, L2, L3);
                    sphi(0, x) += w * (exp(I * phi(0, xp)) + exp(I * phi(0, xm)));
                    sphi(1, x) += w * (exp(I * phi(1, xp)) + exp(I * phi(1, xm)));
#ifdef DEBUG
                    if (xp != (x3 + dx3 + L) % L + (x2 + dx2 + L) % L * L + (x1 + dx1 + L) % L * L2 + x0 * L3) {
                        printf("error   %d   = %d  + %d  L+ %d L^2 + %d L^3\n", x, x3, x2, x1, x0);
                        Kokkos::abort("DFT index p");
                    }
#endif
                }
            }
        }

    });
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void  parallel_measurement_complex(manyphi mphip, manyphi::HostMirror h_mphip, cluster::IO_params params, FILE* f_G2t, FILE* f_checks, int iconf) {
    int T = params.data.L[0];
    fwrite(&iconf, sizeof(int), 1, f_G2t);
    bool smeared_contractions = (params.data.smearing == "yes");
    bool FT_phin_contractions = (params.data.FT_phin == "yes");
    bool smearing3FT = (params.data.smearing3FT == "yes");
    //Viewphi phip("phip",2,params.data.L[0]*Vp);
    // use layoutLeft   to_write(t,c) -> t+x*T; so that there is no need of reordering to write
    Kokkos::View<double**, Kokkos::LayoutRight > to_write("to_write", Ncorr, T);
    Kokkos::View<double**, Kokkos::LayoutRight>::HostMirror h_write = Kokkos::create_mirror_view(to_write);

    auto phip = Kokkos::subview(mphip, 0, Kokkos::ALL, Kokkos::ALL);
    //auto s_phip= Kokkos::subview( mphip, 1, Kokkos::ALL, Kokkos::ALL );
    //auto phi2p = Kokkos::subview( mphip, 2, Kokkos::ALL, Kokkos::ALL );
    //auto phi3p = Kokkos::subview( mphip, 3, Kokkos::ALL, Kokkos::ALL );
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> s_phip;
    if (smeared_contractions) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 1, Kokkos::ALL, Kokkos::ALL);
        s_phip = tmp;
    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi2p;
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi3p;
    if (FT_phin_contractions) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 2, Kokkos::ALL, Kokkos::ALL);
        phi2p = tmp;
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp3(mphip, 3, Kokkos::ALL, Kokkos::ALL);
        phi3p = tmp3;
    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi_s3p;
    if (smearing3FT) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 4, Kokkos::ALL, Kokkos::ALL);
        phi_s3p = tmp;
    }

    Kokkos::parallel_for("measurement_t_loop", T, KOKKOS_LAMBDA(size_t t) {
        // for (int c = 0; c < Ncorr; c++)
        //     to_write(c, t) = 0;
        const int  p1[3] = { 1,Lp,Lp * Lp };
        const int  p11[3] = { 1 + Lp,Lp + Lp * Lp,1 + Lp * Lp };// (1,1,0),(0,1,1),(1,0,1)
        const int p111 = 1 + Lp + Lp * Lp;
        Kokkos::complex<double> phi1[2][3]; //phi[comp] [ P=(1,0,0),(0,1,0),(0,0,1)]
        Kokkos::complex<double> phi11[2][3]; //phi[comp] [p=(1,1,0),(0,1,1),(1,0,1) ]
        Kokkos::complex<double> phi1_t[2][3];
        Kokkos::complex<double> phi11_t[2][3];
        Kokkos::complex<double> phi111[2]; //p=(1,1,1)
        Kokkos::complex<double> phi111_t[2];
        Kokkos::complex<double> bb[3][3]; //back to back [00,11,01][x,y,z]
        Kokkos::complex<double> bb_t[3][3]; //back to back [00,11,01][x,y,z]
        Kokkos::complex<double> A1[3], A1_t[3];  // phi0, phi1, phi01
        Kokkos::complex<double> E1[3], E1_t[3];
        Kokkos::complex<double> E2[3], E2_t[3];
        Kokkos::complex<double> o2p1[4][3], o2p1_t[4][3];
        Kokkos::complex<double> o2p11[4][3], o2p11_t[4][3];
        Kokkos::complex<double> o2p111[4], o2p111_t[4];

        for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            //int tx2_5=((T*2)/5+t1)%T;

            int t2 = (2 + t1) % T;
            int t3 = (3 + t1) % T;
            int t4 = (4 + t1) % T;
            int t5 = (5 + t1) % T;
            int t10 = (10 + t1) % T;
            int t12 = (12 + t1) % T;
            int t16 = (16 + t1) % T;
            int t20 = (20 + t1) % T;

            Kokkos::complex<double> pp0 = (phip(0, t1) * conj(phip(0, tpt1)));
            Kokkos::complex<double> pp1 = (phip(1, t1) * conj(phip(1, tpt1)));

            Kokkos::complex<double> p0;
            p0.real() = phip(0, t1).real();   p0.imag() = phip(1, t1).real();
            Kokkos::complex<double> cpt;
            cpt.real() = phip(0, tpt1).real();   cpt.imag() = -phip(1, tpt1).real();

            to_write(0, t) += pp0.real();
            to_write(1, t) += pp1.real();
            to_write(2, t) += (pp0 * pp0).real();
            to_write(3, t) += (pp1 * pp1).real();
            to_write(4, t) += (pp0 * pp0 + pp1 * pp1 + 4 * pp0 * pp1
                - phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)
                - phip(1, t1) * phip(1, t1) * phip(0, tpt1) * phip(0, tpt1)).real();

            to_write(5, t) += (pp0 * pp0 * pp0).real();
            to_write(6, t) += (pp1 * pp1 * pp1).real();
            to_write(7, t) += real(p0 * cpt * p0 * cpt * p0 * cpt);

            to_write(8, t) += (phip(0, t1) * phip(0, t_8) * phip(0, tpt1) * phip(0, t_2)).real();
            to_write(9, t) += (phip(1, t1) * phip(1, t_8) * phip(1, tpt1) * phip(1, t_2)).real();
            to_write(10, t) += (phip(0, t1) * phip(1, t_8) * phip(1, tpt1) * phip(0, t_2)).real();

            to_write(11, t) += (phip(0, t1) * phip(1, t1) * phip(0, tpt1) * phip(1, tpt1)).real();
            to_write(12, t) += (phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
            to_write(13, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
            to_write(14, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, tpt1) * phip(0, tpt1)).real();



            to_write(15, t) += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t16)).real();
            to_write(16, t) += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t16)).real();
            to_write(17, t) += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t16)).real();

            to_write(18, t) += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t16)).real();
            to_write(19, t) += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t16)).real();
            to_write(20, t) += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t16)).real();

            to_write(21, t) += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t20)).real();
            to_write(22, t) += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t20)).real();
            to_write(23, t) += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t20)).real();

            to_write(24, t) += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t20)).real();
            to_write(25, t) += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t20)).real();
            to_write(26, t) += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t20)).real();

            to_write(27, t) += (phip(0, t1) * phip(0, t5) * phip(0, tpt1) * phip(0, t20)).real();
            to_write(28, t) += (phip(1, t1) * phip(1, t5) * phip(1, tpt1) * phip(1, t20)).real();
            to_write(29, t) += (phip(0, t1) * phip(1, t5) * phip(1, tpt1) * phip(0, t20)).real();


            to_write(30, t) += (phip(1, t1) * phip(0, t3) * phip(0, tpt1) * phip(1, t16)).real();

            to_write(31, t) += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t10)).real();
            to_write(32, t) += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t12)).real();

            for (int comp = 0; comp < 2; comp++) {

                for (int i = 0; i < 3; i++) {  // i = x,y,z
                    int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                    int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                    int t1_p11 = t1 + (p11[i]) * T;   //     
                    int tpt1_p11 = tpt1 + (p11[i]) * T;   //   

                    int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                    int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                    //int t1_mp11 = t1 + (p11[i]) * T + T * Vp;   //    
                    // int tpt1_mp11 = tpt1 + (p11[i]) * T + T * Vp;   //    


                    phi1[comp][i] = phip(comp, t1_p);
                    phi1_t[comp][i] = phip(comp, tpt1_p);

                    phi11[comp][i] = phip(comp, t1_p11);
                    phi11_t[comp][i] = phip(comp, tpt1_p11);


                    bb[comp][i] = (phip(comp, t1_p) * phip(comp, t1_mp));
                    bb_t[comp][i] = (phip(comp, tpt1_p) * phip(comp, tpt1_mp));


                    o2p1[comp][i] = (phi1[comp][i] * phip(comp, t1));
                    o2p1_t[comp][i] = (phi1_t[comp][i] * phip(comp, tpt1));


                    o2p11[comp][i] = (phi11[comp][i] * phip(comp, t1));
                    o2p11_t[comp][i] = (phi11_t[comp][i] * phip(comp, tpt1));


                }
                int t1_p111 = t1 + (p111)*T;
                int tpt1_p111 = tpt1 + (p111)*T;
                // int t1_mp111 = t1 + (p111)*T + T * Vp;
                // int tpt1_mp111 = tpt1 + (p111)*T + T * Vp;


                phi111[comp] = phip(comp, t1_p111);
                phi111_t[comp] = phip(comp, tpt1_p111);

                o2p111[comp] = (phi111[comp] * phip(comp, t1));
                o2p111_t[comp] = (phi111_t[comp] * phip(comp, tpt1));

            }
            for (int i = 0; i < 3; i++) {
                // int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                // int tpt1_p = t1 + (p1[i]) * T;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                // int tpt1_mp11 = tpt1 + (p11[i]) * T + T * Vp;   //    

                bb[2][i] = (phi1[0][i] * phip(1, t1_mp) + phi1[1][i] * phip(0, t1_mp)) / 1.41421356237;//sqrt(2);
                bb_t[2][i] = (phi1_t[0][i] * phip(1, tpt1_mp) + phi1_t[1][i] * phip(0, tpt1_mp)) / 1.41421356237;//sqrt(2);
                o2p1[2][i] = phi1[0][i] * phip(1, t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p1_t[2][i] = phi1_t[0][i] * phip(1, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
                o2p1[3][i] = phi1[1][i] * phip(0, t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p1_t[3][i] = phi1_t[1][i] * phip(0, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;


                o2p11[2][i] = phi11[0][i] * phip(1, t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p11_t[2][i] = (phi11_t[0][i]) * phip(1, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
                o2p11[3][i] = phi11[1][i] * phip(0, t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p11_t[3][i] = (phi11_t[1][i]) * phip(0, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
            }
            o2p111[2] = phi111[0] * phip(1, t1);//  +  phi[1][i]*h_phip(0,t1) ;
            o2p111_t[2] = (phi111_t[0]) * phip(1, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;

            o2p111[3] = phi111[0] * phip(1, t1);//  +  phi[1][i]*h_phip(0,t1) ;
            o2p111_t[3] = (phi111_t[0]) * phip(1, tpt1);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;

            for (int comp = 0; comp < 3; comp++) {
                A1[comp] = (bb[comp][0] + bb[comp][1] + bb[comp][2]) / 1.73205080757;//sqrt(3);
                A1_t[comp] = (bb_t[comp][0] + bb_t[comp][1] + bb_t[comp][2]) / 1.73205080757;//sqrt(3);
                E1[comp] = (bb[comp][0] - bb[comp][1]) / 1.41421356237;//sqrt(2);
                E1_t[comp] = (bb_t[comp][0] - bb_t[comp][1]) / 1.41421356237;//sqrt(2);
                E2[comp] = (bb[comp][0] + bb[comp][1] - 2. * bb[comp][2]) / 2.44948974278;// sqrt(6);
                E2_t[comp] = (bb_t[comp][0] + bb_t[comp][1] - 2. * bb_t[comp][2]) / 2.44948974278;//sqrt(6);
            }

            to_write(33, t) += (phi1[0][0] * conj(phi1_t[0][0])).real();//one_to_one_p1
            to_write(34, t) += (phi1[1][0] * conj(phi1_t[1][0])).real();//one_to_one_p1
            to_write(35, t) += (phi1[0][1] * conj(phi1_t[0][1])).real();//one_to_one_p1
            to_write(36, t) += (phi1[1][1] * conj(phi1_t[1][1])).real();//one_to_one_p1
            to_write(37, t) += (phi1[0][2] * conj(phi1_t[0][2])).real();//one_to_one_p1
            to_write(38, t) += (phi1[1][2] * conj(phi1_t[1][2])).real();//one_to_one_p1


            to_write(39, t) += (A1[0] * conj(A1_t[0])).real();
            to_write(40, t) += (A1[1] * conj(A1_t[1])).real();
            to_write(41, t) += (A1[2] * conj(A1_t[2])).real();

            to_write(42, t) += (E1[0] * conj(E1_t[0])).real();
            to_write(43, t) += (E1[1] * conj(E1_t[1])).real();
            to_write(44, t) += (E1[2] * conj(E1_t[2])).real();

            to_write(45, t) += (E2[0] * conj(E2_t[0])).real();
            to_write(46, t) += (E2[1] * conj(E2_t[1])).real();
            to_write(47, t) += (E2[2] * conj(E2_t[2])).real();

            to_write(48, t) += (A1[0] * conj(E1_t[0])).real();
            to_write(49, t) += (A1[1] * conj(E1_t[1])).real();
            to_write(50, t) += (A1[2] * conj(E1_t[2])).real();

            to_write(51, t) += (E1[0] * conj(A1_t[0])).real();
            to_write(52, t) += (E1[1] * conj(A1_t[1])).real();
            to_write(53, t) += (E1[2] * conj(A1_t[2])).real();

            to_write(54, t) += (A1[0] * conj(E2_t[0])).real();
            to_write(55, t) += (A1[1] * conj(E2_t[1])).real();
            to_write(56, t) += (A1[2] * conj(E2_t[2])).real();

            to_write(57, t) += (E2[0] * conj(A1_t[0])).real();
            to_write(58, t) += (E2[1] * conj(A1_t[1])).real();
            to_write(59, t) += (E2[2] * conj(A1_t[2])).real();

            to_write(60, t) += (A1[0] * conj(phip(0, tpt1) * phip(0, tpt1))).real(); // A1 o20 before was h_phip(comp,  !!!t!!!)
            to_write(61, t) += (A1[1] * conj(phip(1, tpt1) * phip(1, tpt1))).real();
            to_write(62, t) += (A1[2] * conj(phip(0, tpt1) * phip(1, tpt1))).real();

            to_write(63, t) += (phip(0, t1) * phip(0, t1) * conj(A1_t[0])).real();// o20 A1 // two_to_two_o20A1
            to_write(64, t) += (phip(1, t1) * phip(1, t1) * conj(A1_t[1])).real();
            to_write(65, t) += (phip(0, t1) * phip(1, t1) * conj(A1_t[2])).real();

            for (int comp = 0; comp < 3; comp++)
                for (int i = 0; i < 3; i++)
                    to_write(66 + i + comp * 3, t) += (o2p1[comp][i] * conj(o2p1_t[comp][i])).real();  //two_to_two_o2p1o2p1 

            for (int comp = 0; comp < 2; comp++)
                for (int i = 0; i < 3; i++)
                    to_write(75 + comp + i * 2, t) += (phi11[comp][i] * conj(phi11_t[comp][i])).real();//one_to_one_p11
                    ///+  conj(phi11[comp][i])*phi11_t[comp][i] no need to add the conj because we are taking the real part

            for (int comp = 0; comp < 3; comp++)
                for (int i = 0; i < 3; i++)
                    to_write(81 + i + comp * 3, t) += (o2p11[comp][i] * conj(o2p11_t[comp][i])).real();//two_to_two_o2p11o2p11


            to_write(90, t) += (phi111[0] * conj(phi111_t[0])).real();//one_to_one_p111
            to_write(91, t) += (phi111[1] * conj(phi111_t[1])).real();

            to_write(92, t) += (o2p111[0] * conj(o2p111_t[0])).real();//two_to_two_o2p111o2p111
            to_write(93, t) += (o2p111[1] * conj(o2p111_t[1])).real();
            to_write(94, t) += (o2p111[2] * conj(o2p111_t[2])).real();

            for (int comp = 0; comp < 3; comp++)
                for (int i = 0; i < 3; i++)
                    to_write(95 + i + comp * 3, t) += (phip(comp % 2, t1) * o2p1[comp][i] * conj(phip(comp % 2, tpt1) * o2p1_t[comp][i])).real();  //three_to_three_o3p1o3p1 

            for (int comp = 0; comp < 3; comp++)
                for (int i = 0; i < 3; i++)
                    to_write(104 + i + comp * 3, t) += (phip(comp % 2, t1) * o2p11[comp][i] * conj(phip(comp % 2, tpt1) * o2p11_t[comp][i])).real();//three_to_three_o3p11o3p11

            to_write(113, t) += (phip(0, t1) * o2p111[0] * conj(phip(0, tpt1) * o2p111_t[0])).real();//three_to_three_o3p111o3p111
            to_write(114, t) += (phip(1, t1) * o2p111[1] * conj(phip(1, tpt1) * o2p111_t[1])).real();
            to_write(115, t) += (phip(0, t1) * o2p111[2] * conj(phip(1, tpt1) * o2p111_t[2])).real();


            to_write(116, t) += (phip(0, t1) * A1[0] * conj(phip(0, tpt1) * A1_t[0])).real();//three_to_three_A1A1
            to_write(117, t) += (phip(1, t1) * A1[1] * conj(phip(1, tpt1) * A1_t[1])).real();
            to_write(118, t) += (phip(0, t1) * A1[2] * conj(phip(1, tpt1) * A1_t[2])).real();


            to_write(119, t) += (phip(1, t_2) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(0, t1) * phip(0, t1) * phip(0, t1)).real();   //phi0^3  phi0^3phi1 phi1   
            to_write(120, t) += (phip(1, t10) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(0, t1) * phip(0, t1) * phip(0, t1)).real();   //phi0^3  phi0^3phi1 phi1
            to_write(121, t) += (phip(1, t12) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(0, t1) * phip(0, t1) * phip(0, t1)).real();   //phi0^3  phi0^3phi1 phi1
            to_write(122, t) += (phip(1, t16) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(0, t1) * phip(0, t1) * phip(0, t1)).real();   //phi0^3  phi0^3phi1 phi1

            to_write(123, t) += (phip(0, t_2) * phip(0, t_2) * phip(0, t_2) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(1, t1)).real();   //phi0^3  phi0^3phi1 phi1   
            to_write(124, t) += (phip(0, t10) * phip(0, t10) * phip(0, t10) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(1, t1)).real();   //phi0^3  phi0^3phi1 phi1
            to_write(125, t) += (phip(0, t12) * phip(0, t12) * phip(0, t12) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(1, t1)).real();   //phi0^3  phi0^3phi1 phi1
            to_write(126, t) += (phip(0, t16) * phip(0, t16) * phip(0, t16) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1) * phip(1, t1)).real();   //phi0^3  phi0^3phi1 phi1

            to_write(127, t) += (phip(0, t1) * conj(phip(1, tpt1))).real();   //phi0 --> phi1 
                // no need to program
                //phi1(t1) phi0(t1+t)
                // you can get it symmetryising 
                //phi0(t1) phi1(t1+T-t)   =phi0(t1) phi1(t1-t)  (shift the sum in t1->t1+t) = phi1(t1) phi0(t1+t)
            to_write(128, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(phip(1, tpt1))).real();   //phi0^3 --> phi1 
            to_write(129, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(phip(0, tpt1))).real();   //phi0^3 --> phi0

            for (int i = 0; i < 3; i++)// i=x,y,z
                to_write(130 + i, t) += (phi1[0][i] * conj(phi1_t[1][i])).real();   //phi0[px] phi1[px]  
            for (int i = 0; i < 3; i++)
                to_write(133 + i, t) += (phip(0, t1) * o2p1[0][i] * conj(phi1_t[1][i])).real();   //3phi0 [px] --> phi1[px]
            for (int i = 0; i < 3; i++)// i=x,y,z
                to_write(136 + i, t) += (phip(0, t1) * o2p1[0][i] * conj(phi1_t[0][i])).real();   //3phi0 [px] --> phi0[px]


            for (int i = 0; i < 3; i++)// i=x,y,z
                to_write(139 + i, t) += (phi11[0][i] * conj(phi11_t[1][i])).real();   //phi0[pxy] phi1[pxy]  
            for (int i = 0; i < 3; i++)
                to_write(142 + i, t) += (phip(0, t1) * o2p11[0][i] * conj(phi11_t[1][i])).real();   //3phi0 [pxy] --> phi1[pxy]
            for (int i = 0; i < 3; i++)
                to_write(145 + i, t) += (phip(0, t1) * o2p11[0][i] * conj(phi11_t[0][i])).real();   //3phi0 [pxy] --> phi0[pxy]

            to_write(148, t) += (phi111[0] * conj(phi111_t[1])).real();   //phi0[pxyz] phi1[pxyz]  
            to_write(149, t) += (phip(0, t1) * o2p111[0] * conj(phi111_t[1])).real();   //3phi0[pxyz] phi1[pxyz]  
            to_write(150, t) += (phip(0, t1) * o2p111[0] * conj(phi111_t[0])).real();   //3phi0[pxyz] phi0[pxyz]  

            if (smeared_contractions) {
                to_write(151, t) += (s_phip(0, t1) * conj(s_phip(0, tpt1))).real(); // phi0-->phi0  smear-smear
                to_write(152, t) += (s_phip(0, t1) * conj(phip(0, tpt1))).real();
                to_write(153, t) += (s_phip(0, t1) * conj(phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1))).real();
                to_write(154, t) += (s_phip(0, t1) * conj(phip(1, tpt1))).real();

                to_write(155, t) += (s_phip(0, t1) * s_phip(0, t1) * conj(s_phip(0, tpt1) * s_phip(0, tpt1))).real();

                to_write(156, t) += (s_phip(0, t1) * s_phip(0, t1) * s_phip(0, t1) * conj(s_phip(0, tpt1) * s_phip(0, tpt1) * s_phip(0, tpt1))).real();
                to_write(157, t) += (s_phip(0, t1) * s_phip(0, t1) * s_phip(0, t1) * conj(phip(0, tpt1))).real();
                to_write(158, t) += (s_phip(0, t1) * s_phip(0, t1) * s_phip(0, t1) * conj(phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1))).real();
                to_write(159, t) += (s_phip(0, t1) * s_phip(0, t1) * s_phip(0, t1) * conj(phip(1, tpt1))).real();

                to_write(160, t) += (s_phip(0, t1) * s_phip(0, t1) * s_phip(0, t1) * conj(s_phip(0, tpt1))).real();// phi0^3-->phi0  smear-smear
                for (int i = 0; i < 3; i++) {  // i = x,y,z
                    int t1_p = t1 + (p1[i]) * T;   // 2,4 6    real part
                    int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    real part
                    to_write(161, t) += (s_phip(0, t1_p) * conj(s_phip(0, tpt1_p))).real();// phi0(p1)-->phi0(p1)  smear-smear
                }
            }
            if (FT_phin_contractions) {
                to_write(162, t) += (phi2p(0, t1) * conj(phi2p(0, tpt1))).real();// phi2--> phi2 
            }
            // GEVP row:< (phi0^3 p-p A1)   O  >
            to_write(163, t) += (phip(0, t1) * A1[0] * conj(phip(0, tpt1))).real();   //phi0^3 p-p A1 --> phi0 
            to_write(164, t) += (phip(0, t1) * A1[0] * conj(phip(1, tpt1))).real();   //phi0^3 p-p A1 --> phi1 
            to_write(165, t) += (phip(0, t1) * A1[0] * conj(phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1))).real();  //phi0^3 p-p A1 --> phi0^3

            // GEVP row:< (phi0^3 p1-A1)   O  >
            for (int i = 0; i < 3; i++) {
                to_write(166, t) += (phi1[0][i] * A1[0] * conj(phi1_t[0][i] * A1_t[0])).real(); //phi0^3 p1 A1 --> phi0^3 p1 A1
                to_write(167, t) += (phi1[0][i] * A1[0] * conj(phi1_t[0][i])).real();   //phi0^3 p1 A1 --> phi0 p1
                to_write(168, t) += (phi1[0][i] * A1[0] * conj(phi1_t[1][i])).real();   //phi0^3 p1 A1 --> phi1 p1
                to_write(169, t) += (phi1[0][i] * A1[0] * conj(phip(0, tpt1)) * o2p1_t[0][i]).real(); //o2p1_t //phi0^3 p1 A1 --> phi0^3 p1
            }
            Kokkos::complex<double> p5 = (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1));
            Kokkos::complex<double> p5_t = (phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1) * phip(0, tpt1));

            if (FT_phin_contractions) {
                to_write(170, t) += (phi3p(0, t1) * conj(phi3p(0, tpt1))).real();// phi30--> phi30
                to_write(171, t) += (phip(0, t1) * conj(phi3p(0, tpt1))).real();   //phi0 --> phi1 
                to_write(172, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(phi3p(0, tpt1))).real();   //phi0^3 --> phi1 
                to_write(173, t) += (phip(1, t1) * conj(phi3p(0, tpt1))).real();   //phi0^3 --> phi0
                to_write(174, t) += (p5 * conj(phi3p(0, tpt1))).real();   //phi0^3 --> phi0


            }
            to_write(175, t) += (p5 * conj(p5_t)).real();// phi50--> phi50
            to_write(176, t) += (phip(0, t1) * conj(p5_t)).real();   //phi0 --> phi5 
            to_write(177, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(p5_t)).real();   //phi0^3 --> phi5 
            to_write(178, t) += (phip(1, t1) * conj(p5_t)).real();   //phi0^3 --> phi05
            to_write(179, t) += (phip(0, t1) * A1[0] * conj(p5_t)).real();   //phi0^3 --> phi05

            if (FT_phin_contractions) {
                to_write(180, t) += (phip(0, t1) * A1[0] * conj(phi3p(0, tpt1))).real();   //phi0^3 --> phi05
                to_write(181, t) += (phip(0, t1) * phi2p(0, t1) * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(182, t) += (phip(0, t1) * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(183, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(184, t) += (phip(1, t1) * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(185, t) += (p5 * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(185, t) += (phi3p(0, t1) * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();
                to_write(186, t) += (phip(0, t1) * A1[0] * conj(phip(0, tpt1) * phi2p(0, tpt1))).real();

            }
            Kokkos::complex<double> c001 = (phip(0, t1) * phip(0, t1) * phip(1, t1));
            Kokkos::complex<double> c001_t = (phip(0, tpt1) * phip(0, tpt1) * phip(1, tpt1));
            to_write(187, t) += (c001 * conj(c001_t)).real();// phi50--> phi50
            to_write(188, t) += (phip(0, t1) * conj(c001_t)).real();   //phi0 --> phi5 
            to_write(189, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(c001_t)).real();   //phi0^3 --> phi5 
            to_write(190, t) += (phip(1, t1) * conj(c001_t)).real();   //phi0^3 --> phi05
            to_write(191, t) += (phip(0, t1) * A1[0] * conj(c001_t)).real();   //phi0^3 --> phi05
            to_write(192, t) += (p5 * conj(c001_t)).real();   //phi0^3 --> phi05
            Kokkos::complex<double> c011 = (phip(0, t1) * phip(1, t1) * phip(1, t1));
            Kokkos::complex<double> c011_t = (phip(0, tpt1) * phip(1, tpt1) * phip(1, tpt1));
            to_write(193, t) += (c011 * conj(c011_t)).real();// phi50--> phi50
            to_write(194, t) += (phip(0, t1) * conj(c011_t)).real();   //phi0 --> phi5 
            to_write(195, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(c011_t)).real();   //phi0^3 --> phi5 
            to_write(196, t) += (phip(1, t1) * conj(c011_t)).real();   //phi0^3 --> phi05
            to_write(197, t) += (phip(0, t1) * A1[0] * conj(c011_t)).real();   //phi0^3 --> phi05
            to_write(198, t) += (p5 * conj(c011_t)).real();   //phi0^3 --> phi05
            to_write(199, t) += (c001 * conj(c011_t)).real();   //phi0^3 --> phi05

            Kokkos::complex<double> c111 = (phip(1, t1) * phip(1, t1) * phip(1, t1));
            Kokkos::complex<double> c111_t = (phip(1, tpt1) * phip(1, tpt1) * phip(1, tpt1));
            to_write(200, t) += (c111 * conj(c111_t)).real();
            to_write(201, t) += (phip(0, t1) * conj(c111_t)).real();   //phi0 --> phi5 
            to_write(202, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(c111_t)).real();   //phi0^3 --> phi5 
            to_write(203, t) += (phip(1, t1) * conj(c111_t)).real();   //phi0^3 --> phi05
            to_write(204, t) += (phip(0, t1) * A1[0] * conj(c111_t)).real();   //phi0^3 --> phi05
            to_write(205, t) += (p5 * conj(c111_t)).real();   //phi0^3 --> phi05
            to_write(206, t) += (c001 * conj(c111_t)).real();   //phi0^3 --> phi05
            to_write(207, t) += (c011 * conj(c111_t)).real();   //phi0^3 --> phi05

            to_write(208, t) += (phip(0, tpt1) * conj(phip(0, tpt1))).real();   //phi0^3 --> phi05

            if (smearing3FT) {
                to_write(209, t) += (phi_s3p(0, t1) * conj(phi_s3p(0, tpt1))).real();   //phi0 --> phi5 
                to_write(210, t) += (phip(0, t1) * conj(phi_s3p(0, tpt1))).real();   //phi0 --> phi5 
                to_write(211, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * conj(phi_s3p(0, tpt1))).real();   //phi0^3 --> phi5 
                to_write(212, t) += (phip(1, t1) * conj(phi_s3p(0, tpt1))).real();   //phi0^3 --> phi05

            }
            to_write(213, t) += (phip(0, t1) * phip(0, t1) * conj(phip(0, tpt1) * phip(1, tpt1))).real();
            to_write(214, t) += (phip(1, t1) * phip(1, t1) * conj(phip(0, tpt1) * phip(1, tpt1))).real();

            ///////////////////// p1
            to_write(215, t) += ((o2p1[0][0] * conj(o2p1_t[1][0])).real()
                + (o2p1[0][1] * conj(o2p1_t[1][1])).real()
                + (o2p1[0][2] * conj(o2p1_t[1][2])).real()) / 3.; // phi0(p1) phi0-->phi1(p1) phi1

            to_write(216, t) += ((o2p1[0][0] * conj(o2p1_t[2][0])).real()
                + (o2p1[0][1] * conj(o2p1_t[2][1])).real()
                + (o2p1[0][2] * conj(o2p1_t[2][2])).real()) / 3;  // phi0(p1) phi0-->phi0(p1) phi1

            to_write(217, t) += ((o2p1[1][0] * conj(o2p1_t[2][0])).real()
                + (o2p1[1][1] * conj(o2p1_t[2][1])).real()
                + (o2p1[1][2] * conj(o2p1_t[2][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1

            to_write(218, t) += ((o2p1[0][0] * conj(o2p1_t[3][0])).real()
                + (o2p1[0][1] * conj(o2p1_t[3][1])).real()
                + (o2p1[0][2] * conj(o2p1_t[3][2])).real()) / 3;  // phi0(p1) phi0-->phi0(p1) phi1

            to_write(219, t) += ((o2p1[1][0] * conj(o2p1_t[3][0])).real()
                + (o2p1[1][1] * conj(o2p1_t[3][1])).real()
                + (o2p1[1][2] * conj(o2p1_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1

            to_write(220, t) += ((o2p1[2][0] * conj(o2p1_t[3][0])).real()
                + (o2p1[2][1] * conj(o2p1_t[3][1])).real()
                + (o2p1[2][2] * conj(o2p1_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1

            to_write(221, t) += ((o2p1[3][0] * conj(o2p1_t[3][0])).real()
                + (o2p1[3][1] * conj(o2p1_t[3][1])).real()
                + (o2p1[3][2] * conj(o2p1_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1


//////////////////////////p 11



            to_write(222, t) += ((o2p11[0][0] * conj(o2p11_t[1][0])).real()
                + (o2p11[0][1] * conj(o2p11_t[1][1])).real()
                + (o2p11[0][2] * conj(o2p11_t[1][2])).real()) / 3.; // phi0(p11) phi0-->phi1(p11) phi1

            to_write(223, t) += ((o2p11[0][0] * conj(o2p11_t[2][0])).real()
                + (o2p11[0][1] * conj(o2p11_t[2][1])).real()
                + (o2p11[0][2] * conj(o2p11_t[2][2])).real()) / 3;  // phi0(p11) phi0-->phi0(p11) phi1

            to_write(224, t) += ((o2p11[1][0] * conj(o2p11_t[2][0])).real()
                + (o2p11[1][1] * conj(o2p11_t[2][1])).real()
                + (o2p11[1][2] * conj(o2p11_t[2][2])).real()) / 3;  // phi1(p11) phi1-->phi0(p11) phi1

            to_write(225, t) += ((o2p11[0][0] * conj(o2p11_t[3][0])).real()
                + (o2p11[0][1] * conj(o2p11_t[3][1])).real()
                + (o2p11[0][2] * conj(o2p11_t[3][2])).real()) / 3;  // phi0(p1) phi0-->phi0(p1) phi1

            to_write(226, t) += ((o2p11[1][0] * conj(o2p11_t[3][0])).real()
                + (o2p11[1][1] * conj(o2p11_t[3][1])).real()
                + (o2p11[1][2] * conj(o2p11_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1

            to_write(227, t) += ((o2p11[2][0] * conj(o2p11_t[3][0])).real()
                + (o2p11[2][1] * conj(o2p11_t[3][1])).real()
                + (o2p11[2][2] * conj(o2p11_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1

            to_write(228, t) += ((o2p11[3][0] * conj(o2p11_t[3][0])).real()
                + (o2p11[3][1] * conj(o2p11_t[3][1])).real()
                + (o2p11[3][2] * conj(o2p11_t[3][2])).real()) / 3;  // phi1(p1) phi1-->phi0(p1) phi1
////////////////////////////// p 1 1 1
            to_write(229, t) += (o2p111[0] * conj(o2p111_t[1])).real();  // phi0(p111) phi0-->phi1(p111) phi1
            to_write(230, t) += (o2p111[0] * conj(o2p111_t[2])).real();  // phi0(p111) phi0-->phi0(p111) phi1
            to_write(231, t) += (o2p111[1] * conj(o2p111_t[2])).real();  // phi1(p111) phi1-->phi0(p111) phi1
            to_write(232, t) += (o2p111[0] * conj(o2p111_t[3])).real();  // phi0(p111) phi0-->phi1(p111) phi1
            to_write(233, t) += (o2p111[1] * conj(o2p111_t[3])).real();  // phi0(p111) phi0-->phi0(p111) phi1
            to_write(234, t) += (o2p111[2] * conj(o2p111_t[3])).real();  // phi1(p111) phi1-->phi0(p111) phi1
            to_write(235, t) += (o2p111[3] * conj(o2p111_t[3])).real();  // phi1(p111) phi1-->phi0(p111) phi1

            to_write(236, t) += (A1[0] * conj(A1_t[1])).real();  // phi0(p) phi0(-p)-->phi1(p) phi1(-p)
            to_write(237, t) += (A1[0] * conj(A1_t[2])).real();  // phi0(p) phi0(-p)-->phi0(p) phi1(-p)
            to_write(238, t) += (A1[1] * conj(A1_t[2])).real();  // phi1(p) phi1(-p)-->phi0(p) phi1(-p)

            ///// three body

            to_write(239, t) += (phi1[0][0] * conj(phip(1, tpt1) * o2p1_t[0][0])
                + phi1[0][1] * conj(phip(1, tpt1) * o2p1_t[0][1])
                + phi1[0][2] * conj(phip(1, tpt1) * o2p1_t[0][2])).real() / 3.;   //phi0p --> 001p 
            to_write(240, t) += (phip(0, t1) * o2p1[0][0] * conj(phip(1, tpt1) * o2p1_t[0][0])
                + phip(0, t1) * o2p1[0][1] * conj(phip(1, tpt1) * o2p1_t[0][1])
                + phip(0, t1) * o2p1[0][2] * conj(phip(1, tpt1) * o2p1_t[0][2])).real() / 3.;// 001p--> 001p
            to_write(241, t) += (phi1[1][0] * conj(phip(1, tpt1) * o2p1_t[0][0])
                + phi1[1][1] * conj(phip(1, tpt1) * o2p1_t[0][1])
                + phi1[1][2] * conj(phip(1, tpt1) * o2p1_t[0][2])).real() / 3.;   //phi0p --> 001p 
            to_write(242, t) += (phip(1, t1) * o2p1[0][0] * conj(phip(1, tpt1) * o2p1_t[0][0])
                + phip(1, t1) * o2p1[0][1] * conj(phip(1, tpt1) * o2p1_t[0][1])
                + phip(1, t1) * o2p1[0][2] * conj(phip(1, tpt1) * o2p1_t[0][2])).real() / 3.;// 001p--> 001p

            to_write(243, t) += ((phi1[0][0] * conj(phi1_t[1][0])
                + phi1[0][1] * conj(phi1_t[1][1])
                + phi1[0][2] * conj(phi1_t[1][2])
                ) * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 3.;   //phi0p --> 001p 
            to_write(244, t) += (phip(0, t1) * (
                o2p1[0][0] * conj(phi1_t[1][0])
                + o2p1[0][1] * conj(phi1_t[1][1])
                + o2p1[0][2] * conj(phi1_t[1][2])
                ) * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 3.;   //phi0p --> 001p
            to_write(245, t) += ((phi1[1][0] * conj(phi1_t[1][0])
                + phi1[1][1] * conj(phi1_t[1][1])
                + phi1[1][2] * conj(phi1_t[1][2])
                ) * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 3.;   //phi0p --> 001p 
            to_write(246, t) += (phip(1, t1) * (
                o2p1[0][0] * conj(phi1_t[1][0])
                + o2p1[0][1] * conj(phi1_t[1][1])
                + o2p1[0][2] * conj(phi1_t[1][2])
                ) * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 3.;// 001p--> 001p
            to_write(247, t) += (phip(0, t1) * phip(0, t1) * (
                phi1[1][0] * conj(phi1_t[1][0])
                + phi1[1][1] * conj(phi1_t[1][1])
                + phi1[1][2] * conj(phi1_t[1][2])
                ) * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 3.;// 001p--> 001p

//011
            to_write(248, t) += (phi1[0][0] * conj(phip(0, tpt1) * o2p1_t[1][0])
                + phi1[0][1] * conj(phip(0, tpt1) * o2p1_t[1][1])
                + phi1[0][2] * conj(phip(0, tpt1) * o2p1_t[1][2])).real() / 3.;   //phi0p --> 001p 
            to_write(249, t) += (phip(0, t1) * o2p1[0][0] * conj(phip(0, tpt1) * o2p1_t[1][0])
                + phip(0, t1) * o2p1[0][1] * conj(phip(0, tpt1) * o2p1_t[1][1])
                + phip(0, t1) * o2p1[0][2] * conj(phip(0, tpt1) * o2p1_t[1][2])).real() / 3.;// 001p--> 001p
            to_write(250, t) += (phi1[1][0] * conj(phip(0, tpt1) * o2p1_t[1][0])
                + phi1[1][1] * conj(phip(0, tpt1) * o2p1_t[1][1])
                + phi1[1][2] * conj(phip(0, tpt1) * o2p1_t[1][2])).real() / 3.;   //phi0p --> 001p 
            to_write(251, t) += (phip(1, t1) * o2p1[0][0] * conj(phip(0, tpt1) * o2p1_t[1][0])
                + phip(1, t1) * o2p1[0][1] * conj(phip(0, tpt1) * o2p1_t[1][1])
                + phip(1, t1) * o2p1[0][2] * conj(phip(0, tpt1) * o2p1_t[1][2])).real() / 3.;// 001p--> 011p
            to_write(252, t) += (phip(0, t1) * phip(0, t1) * (
                phi1[1][0] * conj(o2p1_t[1][0])
                + phi1[1][1] * conj(o2p1_t[1][1])
                + phi1[1][2] * conj(o2p1_t[1][2])
                ) * conj(phip(0, tpt1))).real() / 3.;// 001p--> 011p
            to_write(253, t) += (phip(0, t1) * o2p1[1][0] * conj(phip(0, tpt1) * o2p1_t[1][0])
                + phip(0, t1) * o2p1[1][1] * conj(phip(0, tpt1) * o2p1_t[1][1])
                + phip(0, t1) * o2p1[1][2] * conj(phip(0, tpt1) * o2p1_t[1][2])).real() / 3.;// 001p--> 001p



            to_write(254, t) += ((phi1[0][0] * conj(phi1_t[0][0])
                + phi1[0][1] * conj(phi1_t[0][1])
                + phi1[0][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p 
            to_write(255, t) += (phip(0, t1) * (
                o2p1[0][0] * conj(phi1_t[0][0])
                + o2p1[0][1] * conj(phi1_t[0][1])
                + o2p1[0][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p 
            to_write(256, t) += ((phi1[1][0] * conj(phi1_t[0][0])
                + phi1[1][1] * conj(phi1_t[0][1])
                + phi1[1][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p 
            to_write(257, t) += (phip(1, t1) * (
                o2p1[0][0] * conj(phi1_t[0][0])
                + o2p1[0][1] * conj(phi1_t[0][1])
                + o2p1[0][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p      
            to_write(258, t) += (phip(0, t1) * phip(0, t1) * (
                phi1[1][0] * conj(phi1_t[0][0])
                + phi1[1][1] * conj(phi1_t[0][1])
                + phi1[1][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p
            to_write(259, t) += (phip(0, t1) * (
                o2p1[1][0] * conj(phi1_t[0][0])
                + o2p1[1][1] * conj(phi1_t[0][1])
                + o2p1[1][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p      
            to_write(260, t) += (phip(1, t1) * phip(1, t1) * (
                phi1[0][0] * conj(phi1_t[0][0])
                + phi1[0][1] * conj(phi1_t[0][1])
                + phi1[0][2] * conj(phi1_t[0][2])) * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 3.;   //phi0p --> 001p      

// A1 6x6 GEVP
            to_write(261, t) += (phip(1, t1) * phip(1, t1) * conj(A1_t[0])).real();
            to_write(262, t) += (phip(0, t1) * phip(1, t1) * conj(A1_t[0])).real();
            to_write(263, t) += (phip(0, t1) * phip(0, t1) * conj(A1_t[1])).real();// o20 A1 // two_to_two_o20A1
            to_write(264, t) += (phip(0, t1) * phip(1, t1) * conj(A1_t[1])).real();
            to_write(265, t) += (phip(0, t1) * phip(0, t1) * conj(A1_t[2])).real();// o20 A1 // two_to_two_o20A1
            to_write(266, t) += (phip(1, t1) * phip(1, t1) * conj(A1_t[2])).real();


        }// end loop t1
        for (int c = 0; c < Ncorr; c++)
            to_write(c, t) /= ((double)T);
    });

    if (params.data.checks == "yes") {
        compute_checks_complex(h_mphip, params, f_checks, iconf);
    }
    // Deep copy device views to host views.
    Kokkos::deep_copy(h_write, to_write);
    Kokkos::View<double**, Kokkos::LayoutLeft >::HostMirror lh_write("lh_write", Ncorr, T);
    for (int c = 0; c < Ncorr; c++)
        for (int t = 0; t < T; t++)
            lh_write(c, t) = h_write(c, t);

    fwrite(&lh_write(0, 0), sizeof(double), Ncorr * T, f_G2t);
}
void  compute_checks_complex(manyphi::HostMirror h_mphip, cluster::IO_params params, FILE* f, int iconf) {
    int T = params.data.L[0];
    fwrite(&iconf, sizeof(int), 1, f);
    Kokkos::complex<double> phi[2][3]; //phi[comp] [ xyz, -x-y-z]
    Kokkos::complex<double> phi_t[2][3];
    Kokkos::complex<double> bb[3][3]; //back to back [00,11,01][x,y,z]
    Kokkos::complex<double> bb_t[3][3]; //back to back [00,11,01][x,y,z]
    std::vector<int>  p1 = { 1,Lp,Lp * Lp };

    auto h_phip = Kokkos::subview(h_mphip, 0, Kokkos::ALL, Kokkos::ALL);
    for (int t = 0; t < T; t++) {

        double two0_to_two0[3] = { 0,0,0 };
        double two0pmp_to_two0[3] = { 0,0,0 };
        double twopmpx_to_twopmpy[3] = { 0,0,0 };
        double one_to_one_im[2] = { 0,0 };
        double two_to_two_im[2] = { 0,0 };
        double three_to_three_im[2] = { 0,0 };

        for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

            for (int comp = 0; comp < 2; comp++) {
                for (int i = 0; i < 3; i++) {
                    int t1_p = t1 + (p1[i]) * T;   // 2,4 6    real part
                    int tpt1_p = (t + t1) % T + (p1[i]) * T;   //2,4 6    real part


                    phi[comp][i] = h_phip(comp, t1_p);
                    phi_t[comp][i] = h_phip(comp, tpt1_p);


                    bb[comp][i] = phi[comp][i] * conj(phi[comp][i]);
                    bb_t[comp][i] = phi_t[comp][i] * conj(phi_t[comp][i]);

                }

            }
            for (int i = 0; i < 3; i++) {
                bb[2][i] = (phi[0][i] * conj(phi[1][i]) + phi[1][i] * conj(phi[0][i])) / sqrt(2);
                bb_t[2][i] = (phi_t[0][i] * conj(phi_t[1][i]) + phi_t[1][i] * conj(phi_t[0][i])) / sqrt(2);
            }
            for (int i = 0; i < 3; i++) {
                two0_to_two0[i] += (bb[0][i] * bb_t[0][i]).real();
                two0pmp_to_two0[i] += (bb[0][i] * h_phip(0, (t + t1) % T) * h_phip(0, (t + t1) % T)).real();
                twopmpx_to_twopmpy[i] += (bb[0][i] * bb_t[0][(i + 1) % 3]).real();
            }
            for (int i = 0; i < 2; i++) {
                one_to_one_im[i] += (h_phip(i, t1) * h_phip(i, tpt1)).imag();
                two_to_two_im[i] += (h_phip(i, t1) * h_phip(i, t1) * h_phip(i, tpt1) * h_phip(i, tpt1)).imag();
                three_to_three_im[i] += (h_phip(i, t1) * h_phip(i, t1) * h_phip(i, t1)
                    * h_phip(i, tpt1) * h_phip(i, tpt1) * h_phip(i, tpt1)).imag();
            }


        }

        for (int comp = 0; comp < 3; comp++) {
            two0_to_two0[comp] /= ((double)T);
            two0pmp_to_two0[comp] /= ((double)T);
            twopmpx_to_twopmpy[comp] /= ((double)T);

        }

        fwrite(&two0_to_two0[0], sizeof(double), 1, f); // 0 c++  || 1 R    00 x
        fwrite(&two0_to_two0[1], sizeof(double), 1, f); // 1 c++  || 2 R    00 y
        fwrite(&two0_to_two0[2], sizeof(double), 1, f); // 2 c++  || 3 R    00 z

        fwrite(&two0pmp_to_two0[0], sizeof(double), 1, f); // 3 c++  || 4 R    00 x0
        fwrite(&two0pmp_to_two0[1], sizeof(double), 1, f); // 4 c++  || 5 R    00 y0
        fwrite(&two0pmp_to_two0[2], sizeof(double), 1, f); // 5 c++  || 6 R    00 z0

        fwrite(&twopmpx_to_twopmpy[0], sizeof(double), 1, f); // 6 c++  || 7 R   00 (px -px)(py -py)
        fwrite(&twopmpx_to_twopmpy[1], sizeof(double), 1, f); // 7 c++  || 8 R   00 (py -py)(pz -pz)
        fwrite(&twopmpx_to_twopmpy[2], sizeof(double), 1, f); // 8 c++  || 9 R   00 (pz -pz)(px -px)

        fwrite(one_to_one_im, sizeof(double), 2, f); //9,10
        fwrite(two_to_two_im, sizeof(double), 2, f); //11,12
        fwrite(three_to_three_im, sizeof(double), 2, f); //13,14

    }


}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



KOKKOS_FUNCTION Kokkos::complex<double> compute_o2p1(Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phip,
    const int& t1, const int T, const int& comp, const int& i) {

    Kokkos::complex<double> o2p1(0, 0);
    const int  p1[3] = { 1, Lp, Lp * Lp };
    int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
    if (comp < 2)
        o2p1 += phip(comp, t1_p) * phip(comp, t1);
    else
        o2p1 += phip(0, t1_p) * phip(1, t1);

    return o2p1;
}




void  parallel_measurement_complex_1(manyphi mphip, manyphi::HostMirror h_mphip, cluster::IO_params params, FILE* f_G2t, FILE* f_checks, int iconf) {
    int T = params.data.L[0];
    fwrite(&iconf, sizeof(int), 1, f_G2t);
    bool smeared_contractions = (params.data.smearing == "yes");
    bool FT_phin_contractions = (params.data.FT_phin == "yes");
    bool smearing3FT = (params.data.smearing3FT == "yes");
    //Viewphi phip("phip",2,params.data.L[0]*Vp);
    // use layoutLeft   to_write(t,c) -> t+x*T; so that there is no need of reordering to write
    Kokkos::View<double**, Kokkos::LayoutRight > to_write("to_write", Ncorr, T);
    Kokkos::View<double**, Kokkos::LayoutRight>::HostMirror h_write = Kokkos::create_mirror_view(to_write);

    auto phip = Kokkos::subview(mphip, 0, Kokkos::ALL, Kokkos::ALL);

    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> s_phip;
    if (smeared_contractions) {
        s_phip = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride>(mphip, 1, Kokkos::ALL, Kokkos::ALL);
    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi2p;
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi3p;
    if (FT_phin_contractions) {
        phi2p = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride>(mphip, 2, Kokkos::ALL, Kokkos::ALL);
        phi3p = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride>(mphip, 3, Kokkos::ALL, Kokkos::ALL);

    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi_s3p;
    if (smearing3FT) {
        phi_s3p = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride>(mphip, 4, Kokkos::ALL, Kokkos::ALL);
    }
    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;

    Kokkos::parallel_for("measurement_t_loop", team_policy(T, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
        const int t = teamMember.league_rank();
        const int  p1[3] = { 1,Lp,Lp * Lp };
        const int  p11[3] = { 1 + Lp,Lp + Lp * Lp,1 + Lp * Lp };// (1,1,0),(0,1,1),(1,0,1)
        // const int p111 = 1 + Lp + Lp * Lp;

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * conj(phip(0, tpt1))).real();
            }, to_write(0, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(1, t1) * conj(phip(1, tpt1))).real();
            }, to_write(1, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * phip(0, t1) * conj(phip(0, tpt1) * phip(0, tpt1))).real();
            }, to_write(2, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(1, t1) * phip(1, t1) * conj(phip(1, tpt1) * phip(1, tpt1))).real();
            }, to_write(3, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp0 = (phip(0, t1) * conj(phip(0, tpt1)));
            Kokkos::complex<double> pp1 = (phip(1, t1) * conj(phip(1, tpt1)));
            inner += (pp0 * pp0 + pp1 * pp1 + 4 * pp0 * pp1
                - phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)
                - phip(1, t1) * phip(1, t1) * phip(0, tpt1) * phip(0, tpt1)).real();
            }, to_write(4, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp0 = (phip(0, t1) * conj(phip(0, tpt1)));
            inner += (pp0 * pp0 * pp0).real();
            }, to_write(5, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp1 = (phip(1, t1) * conj(phip(1, tpt1)));
            inner += (pp1 * pp1 * pp1).real();
            }, to_write(6, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> p0;
            p0.real() = phip(0, t1).real();   p0.imag() = phip(1, t1).real();
            Kokkos::complex<double> cpt;
            cpt.real() = phip(0, tpt1).real();   cpt.imag() = -phip(1, tpt1).real();
            inner += real(p0 * cpt * p0 * cpt * p0 * cpt);
            }, to_write(7, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            inner += (phip(0, t1) * phip(0, t_8) * phip(0, tpt1) * phip(0, t_2)).real();
            }, to_write(8, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            inner += (phip(1, t1) * phip(1, t_8) * phip(1, tpt1) * phip(1, t_2)).real();
            }, to_write(9, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            inner += (phip(0, t1) * phip(1, t_8) * phip(1, tpt1) * phip(0, t_2)).real();
            }, to_write(10, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * phip(1, t1) * phip(0, tpt1) * phip(1, tpt1)).real();
            }, to_write(11, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
            }, to_write(12, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
            }, to_write(13, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            inner += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, tpt1) * phip(0, tpt1)).real();
            }, to_write(14, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t16)).real();
            }, to_write(15, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t16)).real();
            }, to_write(16, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t16)).real();
            }, to_write(17, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t16)).real();
            }, to_write(18, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t16)).real();
            }, to_write(19, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t16)).real();
            }, to_write(20, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t20)).real();
            }, to_write(21, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t20)).real();
            }, to_write(22, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t20)).real();
            }, to_write(23, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t20)).real();
            }, to_write(24, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t20)).real();
            }, to_write(25, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t20)).real();
            }, to_write(26, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(0, t5) * phip(0, tpt1) * phip(0, t20)).real();
            }, to_write(27, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(1, t1) * phip(1, t5) * phip(1, tpt1) * phip(1, t20)).real();
            }, to_write(28, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            inner += (phip(0, t1) * phip(1, t5) * phip(1, tpt1) * phip(0, t20)).real();
            }, to_write(29, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            inner += (phip(1, t1) * phip(0, t3) * phip(0, tpt1) * phip(1, t16)).real();
            }, to_write(30, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t2 = (2 + t1) % T;
            int t10 = (10 + t1) % T;
            inner += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t10)).real();
            }, to_write(31, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            int t2 = (2 + t1) % T;
            int t12 = (12 + t1) % T;
            inner += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t12)).real();
            }, to_write(32, t));
        for (int i = 0; i < 3; i++) {
            for (int comp = 0; comp < 2; comp++) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
                    int tpt1 = (t + t1) % T;
                    int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                    int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                    inner += (phip(comp, t1_p) * conj(phip(comp, tpt1_p))).real();
                    }, to_write(33 + comp + 2 * i, t));
            }
        }

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += phip(0, t1_p) * phip(0, t1_mp);
                A1_t += phip(0, tpt1_p) * phip(0, tpt1_mp);
            }
            inner += (A1 * conj(A1_t)).real() / 3.0;
            }, to_write(39, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += phip(1, t1_p) * phip(1, t1_mp);
                A1_t += phip(1, tpt1_p) * phip(1, tpt1_mp);
            }
            inner += (A1 * conj(A1_t)).real() / 3.0;
            }, to_write(40, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += phip(0, t1_p) * phip(1, t1_mp) + phip(0, t1_mp) * phip(1, t1_p);
                A1_t += phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(0, tpt1_mp) * phip(1, tpt1_p);
            }
            inner += (A1 * conj(A1_t)).real() / 6.0;
            }, to_write(41, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            for (int i = 0; i < 2; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += (1 - i * 2) * phip(0, t1_p) * phip(0, t1_mp);
                E1_t += (1 - i * 2) * phip(0, tpt1_p) * phip(0, tpt1_mp);
            }
            inner += (E1 * conj(E1_t)).real() / 2.0;
            }, to_write(42, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            for (int i = 0; i < 2; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += (1 - i * 2) * phip(1, t1_p) * phip(1, t1_mp);
                E1_t += (1 - i * 2) * phip(1, tpt1_p) * phip(1, tpt1_mp);
            }
            inner += (E1 * conj(E1_t)).real() / 2.0;
            }, to_write(43, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            for (int i = 0; i < 2; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += (1 - i * 2) * (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                E1_t += (1 - i * 2) * (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E1 * conj(E1_t)).real() / 4.0;
            }, to_write(44, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * phip(0, t1_p) * phip(0, t1_mp);
                E2_t += c[i] * phip(0, tpt1_p) * phip(0, tpt1_mp);
            }
            inner += (E2 * conj(E2_t)).real() / 6.0;
            }, to_write(45, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * phip(1, t1_p) * phip(1, t1_mp);
                E2_t += c[i] * phip(1, tpt1_p) * phip(1, tpt1_mp);
            }
            inner += (E2 * conj(E2_t)).real() / 6.0;
            }, to_write(46, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                E2_t += c[i] * (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E2 * conj(E2_t)).real() / 12.0;
            }, to_write(47, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(0, t1_p) * phip(0, t1_mp));
                E1_t += c[i] * (phip(0, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (A1 * conj(E1_t)).real() / 2.44948974278;// sqrt(6);
            }, to_write(48, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(1, t1_p) * phip(1, t1_mp));
                E1_t += c[i] * (phip(1, tpt1_p) * phip(1, tpt1_mp));
            }
            inner += (A1 * conj(E1_t)).real() / 2.44948974278;// sqrt(6);
            }, to_write(49, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                E1_t += c[i] * (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (A1 * conj(E1_t)).real() / 4.89897948557;// sqrt(3*8);
            }, to_write(50, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += c[i] * (phip(0, t1_p) * phip(0, t1_mp));
                A1_t += (phip(0, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E1 * conj(A1_t)).real() / 2.44948974278;// sqrt(6);
            }, to_write(51, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += c[i] * (phip(1, t1_p) * phip(1, t1_mp));
                A1_t += (phip(1, tpt1_p) * phip(1, tpt1_mp));
            }
            inner += (E1 * conj(A1_t)).real() / 2.44948974278;// sqrt(6);
            }, to_write(52, t));
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E1(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1,-1,0 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E1 += c[i] * (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                A1_t += (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E1 * conj(A1_t)).real() / 4.89897948557;// sqrt(8*2);
            }, to_write(53, t));



        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1, 1, -2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(0, t1_p) * phip(0, t1_mp));
                E2_t += c[i] * (phip(0, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (A1 * conj(E2_t)).real() / 4.24264068712;// sqrt(6*3);
            }, to_write(54, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1, 1, -2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(1, t1_p) * phip(1, t1_mp));
                E2_t += c[i] * (phip(1, tpt1_p) * phip(1, tpt1_mp));
            }
            inner += (A1 * conj(E2_t)).real() / 4.24264068712;// sqrt(6*3);
            }, to_write(55, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            Kokkos::complex<double> E2_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                A1 += (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                E2_t += c[i] * (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (A1 * conj(E2_t)).real() / 8.48528137424;// 6*sqrt(2);
            }, to_write(56, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * (phip(0, t1_p) * phip(0, t1_mp));
                A1_t += (phip(0, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E2 * conj(A1_t)).real() / 4.24264068712;// sqrt(6*3);
            }, to_write(57, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1,1,-2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * (phip(1, t1_p) * phip(1, t1_mp));
                A1_t += (phip(1, tpt1_p) * phip(1, tpt1_mp));
            }
            inner += (E2 * conj(A1_t)).real() / 4.24264068712;// sqrt(6*3);
            }, to_write(58, t));


        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> E2(0, 0);
            Kokkos::complex<double> A1_t(0, 0);
            double c[3] = { 1, 1, -2 };
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int tpt1_p = tpt1 + (p1[i]) * T;   //2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                int tpt1_mp = tpt1 + (p1[i]) * T + T * Vp;   //2,4 6    
                E2 += c[i] * (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
                A1_t += (phip(0, tpt1_p) * phip(1, tpt1_mp) + phip(1, tpt1_p) * phip(0, tpt1_mp));
            }
            inner += (E2 * conj(A1_t)).real() / 8.48528137424;// sqrt(6*3*4);
            }, to_write(59, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1 += (phip(0, t1_p) * phip(0, t1_mp));
            }
            inner += (A1 * conj(phip(0, tpt1) * phip(0, tpt1))).real() / 1.73205080757;// sqrt(3);
            }, to_write(60, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1 += (phip(1, t1_p) * phip(1, t1_mp));
            }
            inner += (A1 * conj(phip(1, tpt1) * phip(1, tpt1))).real() / 1.73205080757;// sqrt(3);
            }, to_write(61, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1 += (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
            }
            inner += (A1 * conj(phip(0, tpt1) * phip(1, tpt1))).real() / 2.44948974278;// sqrt(3*2);
            }, to_write(62, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = tpt1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = tpt1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1_t += (phip(0, t1_p) * phip(0, t1_mp));
            }
            inner += (conj(A1_t) * (phip(0, t1) * phip(0, t1))).real() / 1.73205080757;// sqrt(3);
            }, to_write(63, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = tpt1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = tpt1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1_t += (phip(1, t1_p) * phip(1, t1_mp));
            }
            inner += (conj(A1_t) * (phip(1, t1) * phip(1, t1))).real() / 1.73205080757;// sqrt(3);
            }, to_write(64, t));

        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1_t(0, 0);
            for (int i = 0; i < 3; i++) {
                int t1_p = tpt1 + (p1[i]) * T;   // 2,4 6    
                int t1_mp = tpt1 + (p1[i]) * T + T * Vp;   // 2,4 6   
                A1_t += (phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp));
            }
            inner += (conj(A1_t) * (phip(0, t1) * phip(1, t1))).real() / 2.44948974278;// sqrt(3*2);
            }, to_write(65, t));

        for (int comp = 0; comp < 3; comp++) {
            for (int i = 0; i < 3; i++) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
                    int tpt1 = (t + t1) % T;
                    Kokkos::complex<double> o2p1(compute_o2p1(phip, t1, T, comp, i));
                    Kokkos::complex<double> o2p1_t(compute_o2p1(phip, tpt1, T, comp, i));
                    inner += (o2p1 * conj(o2p1_t)).real();
                    }, to_write(66 + i + comp * 3, t));
            }
        }
        for (int comp = 0; comp < 2; comp++) {
            for (int i = 0; i < 3; i++) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
                    int tpt1 = (t + t1) % T;
                    int t1_p11 = t1 + (p11[i]) * T;   //     
                    int tpt1_p11 = tpt1 + (p11[i]) * T;
                    inner += (phip(comp, t1_p11) * conj(phip(comp, tpt1_p11))).real();
                    }, to_write(75 + i * 2 + comp, t));
            }
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, Ncorr), [&](const int c) {
            to_write(c, t) /= ((double)T);
            });
    });

    if (params.data.checks == "yes") {
        compute_checks_complex(h_mphip, params, f_checks, iconf);
    }
    // Deep copy device views to host views.
    Kokkos::deep_copy(h_write, to_write);
    Kokkos::View<double**, Kokkos::LayoutLeft >::HostMirror lh_write("lh_write", Ncorr, T);
    for (int c = 0; c < Ncorr; c++)
        for (int t = 0; t < T; t++)
            lh_write(c, t) = h_write(c, t);

    fwrite(&lh_write(0, 0), sizeof(double), Ncorr * T, f_G2t);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#ifdef NOTCOMPILE

template<int comp>
KOKKOS_FUNCTION Kokkos::complex<double> compute_A1(Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phip, const int& t1, const int& T) {
    Kokkos::complex<double> A1(0, 0);
    const int  p1[3] = { 1,Lp,Lp * Lp };
    for (int i = 0; i < 3; i++) {
        int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
        int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
        A1 += phip(comp, t1_p) * phip(comp, t1_mp);
    }
    return A1 / 1.73205080757;
}

template<>           // explicit specialization for comp = 2
KOKKOS_FUNCTION Kokkos::complex<double> compute_A1<2>(Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phip, const int& t1, const int& T) {
    Kokkos::complex<double> A1(0, 0);
    const int  p1[3] = { 1,Lp,Lp * Lp };
    for (int i = 0; i < 3; i++) {
        int t1_p = t1 + (p1[i]) * T;   // 2,4 6    
        int t1_mp = t1 + (p1[i]) * T + T * Vp;   // 2,4 6   
        A1 += phip(0, t1_p) * phip(1, t1_mp) + phip(1, t1_p) * phip(0, t1_mp);
    }
    return A1 / 2.44948974278;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void  parallel_measurement_complex(manyphi mphip, manyphi::HostMirror h_mphip, cluster::IO_params params, FILE* f_G2t, FILE* f_checks, int iconf) {


    int T = params.data.L[0];
    fwrite(&iconf, sizeof(int), 1, f_G2t);
    bool smeared_contractions = (params.data.smearing == "yes");
    bool FT_phin_contractions = (params.data.FT_phin == "yes");
    bool smearing3FT = (params.data.smearing3FT == "yes");
    //Viewphi phip("phip",2,params.data.L[0]*Vp);
    // use layoutLeft   to_write(t,c) -> t+x*T; so that there is no need of reordering to write
    Kokkos::View<double**, Kokkos::LayoutRight > to_write("to_write", Ncorr, T);
    Kokkos::View<double**, Kokkos::LayoutRight>::HostMirror h_write = Kokkos::create_mirror_view(to_write);

    auto phip = Kokkos::subview(mphip, 0, Kokkos::ALL, Kokkos::ALL);
    //auto s_phip= Kokkos::subview( mphip, 1, Kokkos::ALL, Kokkos::ALL );
    //auto phi2p = Kokkos::subview( mphip, 2, Kokkos::ALL, Kokkos::ALL );
    //auto phi3p = Kokkos::subview( mphip, 3, Kokkos::ALL, Kokkos::ALL );
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> s_phip;
    if (smeared_contractions) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 1, Kokkos::ALL, Kokkos::ALL);
        s_phip = tmp;
    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi2p;
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi3p;
    if (FT_phin_contractions) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 2, Kokkos::ALL, Kokkos::ALL);
        phi2p = tmp;
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp3(mphip, 3, Kokkos::ALL, Kokkos::ALL);
        phi3p = tmp3;
    }
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> phi_s3p;
    if (smearing3FT) {
        Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutStride> tmp(mphip, 4, Kokkos::ALL, Kokkos::ALL);
        phi_s3p = tmp;
    }
    using MDPolicyType_2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
    MDPolicyType_2D mdpolicy_2d({ {0, 0} }, { {Ncorr , T} });
    Kokkos::parallel_for("measurement_t_loop", mdpolicy_2d, KOKKOS_LAMBDA(int c, int t) {

        const int  p1[3] = { 1,Lp,Lp * Lp };

        switch (c) {
        case 0: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * conj(phip(0, tpt1))).real();
        } break;
        case 1: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(1, t1) * conj(phip(1, tpt1))).real();
        } break;
        case 2: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t1) * conj(phip(0, tpt1) * phip(0, tpt1))).real();
        } break;
        case 3: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t1) * conj(phip(1, tpt1) * phip(1, tpt1))).real();
        } break;
        case 4: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp0 = (phip(0, t1) * conj(phip(0, tpt1)));
            Kokkos::complex<double> pp1 = (phip(1, t1) * conj(phip(1, tpt1)));
            to_write(c, t) += (pp0 * pp0 + pp1 * pp1 + 4 * pp0 * pp1
                - phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)
                - phip(1, t1) * phip(1, t1) * phip(0, tpt1) * phip(0, tpt1)).real();
        } break;
        case 5: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp0 = (phip(0, t1) * conj(phip(0, tpt1)));
            to_write(c, t) += (pp0 * pp0 * pp0).real();
        } break;
        case 6: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> pp1 = (phip(1, t1) * conj(phip(1, tpt1)));
            to_write(c, t) += (pp1 * pp1 * pp1).real();
        } break;
        case 7: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> p0;
            p0.real() = phip(0, t1).real();   p0.imag() = phip(1, t1).real();
            Kokkos::complex<double> cpt;
            cpt.real() = phip(0, tpt1).real();   cpt.imag() = -phip(1, tpt1).real();
            real(p0 * cpt * p0 * cpt * p0 * cpt);
        } break;
        case 8: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t_8) * phip(0, tpt1) * phip(0, t_2)).real();
        } break;
        case 9: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t_8) * phip(1, tpt1) * phip(1, t_2)).real();
        } break;
        case 10: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t_2 = (T / 2 + t1) % T;
            int t_8 = (T / 8 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t_8) * phip(1, tpt1) * phip(0, t_2)).real();
        } break;
        case 11: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t1) * phip(0, tpt1) * phip(1, tpt1)).real();
        } break;
        case 12: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
        } break;
        case 13: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(1, tpt1) * phip(1, tpt1)).real();
        } break;
        case 14: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, t1) * phip(0, tpt1) * phip(0, tpt1)).real();
        } break;
        case 15: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t16)).real();
        } break;
        case 16: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t16)).real();
        } break;
        case 17: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t16)).real();
        } break;
        case 18: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t16)).real();
        } break;
        case 19: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t16)).real();
        } break;

        case 20: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t16)).real();
        } break;
        case 21: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t3) * phip(0, tpt1) * phip(0, t20)).real();

        } break;
        case 22: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t3) * phip(1, tpt1) * phip(1, t20)).real();
        } break;
        case 23: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t3) * phip(1, tpt1) * phip(0, t20)).real();
        } break;
        case 24: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t4) * phip(0, tpt1) * phip(0, t20)).real();
        } break;
        case 25: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t4) * phip(1, tpt1) * phip(1, t20)).real();
        } break;
        case 26: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t4 = (4 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t4) * phip(1, tpt1) * phip(0, t20)).real();
        } break;
        case 27: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(0, t5) * phip(0, tpt1) * phip(0, t20)).real();
        } break;
        case 28: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(1, t5) * phip(1, tpt1) * phip(1, t20)).real();
        } break;
        case 29: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t5 = (5 + t1) % T;
            int t20 = (20 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t5) * phip(1, tpt1) * phip(0, t20)).real();
        } break;

        case 30: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t3 = (3 + t1) % T;
            int t16 = (16 + t1) % T;
            to_write(c, t) += (phip(1, t1) * phip(0, t3) * phip(0, tpt1) * phip(1, t16)).real();
        } break;
        case 31: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t2 = (2 + t1) % T;
            int t10 = (10 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t10)).real();
        } break;
        case 32: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t2 = (2 + t1) % T;
            int t12 = (12 + t1) % T;
            to_write(c, t) += (phip(0, t1) * phip(1, t2) * phip(1, tpt1) * phip(0, t12)).real();
        } break;
        case 33: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[0]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[0]) * T;   //2,4 6    
            to_write(c, t) += (phip(0, t1_p) * conj(phip(0, tpt1_p))).real();
        } break;
        case 34: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[0]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[0]) * T;   //2,4 6    
            to_write(c, t) += (phip(1, t1_p) * conj(phip(1, tpt1_p))).real();
        } break;
        case 35: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[1]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[1]) * T;   //2,4 6    
            to_write(c, t) += (phip(0, t1_p) * conj(phip(0, tpt1_p))).real();
        } break;
        case 36: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[1]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[1]) * T;   //2,4 6    
            to_write(c, t) += (phip(1, t1_p) * conj(phip(1, tpt1_p))).real();
        } break;
        case 37: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[2]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[2]) * T;   //2,4 6    
            to_write(c, t) += (phip(0, t1_p) * conj(phip(0, tpt1_p))).real();
        } break;
        case 38: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            int t1_p = t1 + (p1[2]) * T;   // 2,4 6    
            int tpt1_p = tpt1 + (p1[2]) * T;   //2,4 6    
            to_write(c, t) += (phip(1, t1_p) * conj(phip(1, tpt1_p))).real();
        } break;
        case 39: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1 = compute_A1<0>(phip, t1, T);
            Kokkos::complex<double> A1_t = compute_A1<0>(phip, tpt1, T);
            to_write(c, t) += (A1 * conj(A1_t)).real();
        } break;

        case 40: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1 = compute_A1<1>(phip, t1, T);
            Kokkos::complex<double> A1_t = compute_A1<1>(phip, tpt1, T);
            to_write(c, t) += (A1 * conj(A1_t)).real();
        } break;
        case 41: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;
            Kokkos::complex<double> A1 = compute_A1<2>(phip, t1, T);
            Kokkos::complex<double> A1_t = compute_A1<2>(phip, tpt1, T);
            to_write(c, t) += (A1 * conj(A1_t)).real();
        } break;
        case 42: for (int t1 = 0; t1 < T; t1++) {


        } break;
        case 43: for (int t1 = 0; t1 < T; t1++) {


        } break;
        case 44: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;
        case 45: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;
        case 46: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;
        case 47: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;
        case 48: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;
        case 49: for (int t1 = 0; t1 < T; t1++) {
            int tpt1 = (t + t1) % T;

        } break;


        case 50: for (int t1 = 0; t1 < T; t1++) {} break;
        case 51: for (int t1 = 0; t1 < T; t1++) {} break;
        case 52: for (int t1 = 0; t1 < T; t1++) {} break;
        case 53: for (int t1 = 0; t1 < T; t1++) {} break;
        case 54: for (int t1 = 0; t1 < T; t1++) {} break;
        case 55: for (int t1 = 0; t1 < T; t1++) {} break;
        case 56: for (int t1 = 0; t1 < T; t1++) {} break;
        case 57: for (int t1 = 0; t1 < T; t1++) {} break;
        case 58: for (int t1 = 0; t1 < T; t1++) {} break;
        case 59: for (int t1 = 0; t1 < T; t1++) {} break;


        case 60: for (int t1 = 0; t1 < T; t1++) {} break;
        case 61: for (int t1 = 0; t1 < T; t1++) {} break;
        case 62: for (int t1 = 0; t1 < T; t1++) {} break;
        case 63: for (int t1 = 0; t1 < T; t1++) {} break;
        case 64: for (int t1 = 0; t1 < T; t1++) {} break;
        case 65: for (int t1 = 0; t1 < T; t1++) {} break;
        case 66: for (int t1 = 0; t1 < T; t1++) {} break;
        case 67: for (int t1 = 0; t1 < T; t1++) {} break;
        case 68: for (int t1 = 0; t1 < T; t1++) {} break;
        case 69: for (int t1 = 0; t1 < T; t1++) {} break;


        case 70: for (int t1 = 0; t1 < T; t1++) {} break;
        case 71: for (int t1 = 0; t1 < T; t1++) {} break;
        case 72: for (int t1 = 0; t1 < T; t1++) {} break;
        case 73: for (int t1 = 0; t1 < T; t1++) {} break;
        case 74: for (int t1 = 0; t1 < T; t1++) {} break;
        case 75: for (int t1 = 0; t1 < T; t1++) {} break;
        case 76: for (int t1 = 0; t1 < T; t1++) {} break;
        case 77: for (int t1 = 0; t1 < T; t1++) {} break;
        case 78: for (int t1 = 0; t1 < T; t1++) {} break;
        case 79: for (int t1 = 0; t1 < T; t1++) {} break;


        case 80: for (int t1 = 0; t1 < T; t1++) {} break;
        case 81: for (int t1 = 0; t1 < T; t1++) {} break;
        case 82: for (int t1 = 0; t1 < T; t1++) {} break;
        case 83: for (int t1 = 0; t1 < T; t1++) {} break;
        case 84: for (int t1 = 0; t1 < T; t1++) {} break;
        case 85: for (int t1 = 0; t1 < T; t1++) {} break;
        case 86: for (int t1 = 0; t1 < T; t1++) {} break;
        case 87: for (int t1 = 0; t1 < T; t1++) {} break;
        case 88: for (int t1 = 0; t1 < T; t1++) {} break;
        case 89: for (int t1 = 0; t1 < T; t1++) {} break;


        case 90: for (int t1 = 0; t1 < T; t1++) {} break;
        case 91: for (int t1 = 0; t1 < T; t1++) {} break;
        case 92: for (int t1 = 0; t1 < T; t1++) {} break;
        case 93: for (int t1 = 0; t1 < T; t1++) {} break;
        case 94: for (int t1 = 0; t1 < T; t1++) {} break;
        case 95: for (int t1 = 0; t1 < T; t1++) {} break;
        case 96: for (int t1 = 0; t1 < T; t1++) {} break;
        case 97: for (int t1 = 0; t1 < T; t1++) {} break;
        case 98: for (int t1 = 0; t1 < T; t1++) {} break;
        case 99: for (int t1 = 0; t1 < T; t1++) {} break;

        case 100: for (int t1 = 0; t1 < T; t1++) {} break;
        case 101: for (int t1 = 0; t1 < T; t1++) {} break;
        case 102: for (int t1 = 0; t1 < T; t1++) {} break;
        case 103: for (int t1 = 0; t1 < T; t1++) {} break;
        case 104: for (int t1 = 0; t1 < T; t1++) {} break;
        case 105: for (int t1 = 0; t1 < T; t1++) {} break;
        case 106: for (int t1 = 0; t1 < T; t1++) {} break;
        case 107: for (int t1 = 0; t1 < T; t1++) {} break;
        case 108: for (int t1 = 0; t1 < T; t1++) {} break;
        case 109: for (int t1 = 0; t1 < T; t1++) {} break;
        case 110: for (int t1 = 0; t1 < T; t1++) {} break;
        case 111: for (int t1 = 0; t1 < T; t1++) {} break;
        case 112: for (int t1 = 0; t1 < T; t1++) {} break;
        case 113: for (int t1 = 0; t1 < T; t1++) {} break;
        case 114: for (int t1 = 0; t1 < T; t1++) {} break;
        case 115: for (int t1 = 0; t1 < T; t1++) {} break;
        case 116: for (int t1 = 0; t1 < T; t1++) {} break;
        case 117: for (int t1 = 0; t1 < T; t1++) {} break;
        case 118: for (int t1 = 0; t1 < T; t1++) {} break;
        case 119: for (int t1 = 0; t1 < T; t1++) {} break;
        case 120: for (int t1 = 0; t1 < T; t1++) {} break;
        case 121: for (int t1 = 0; t1 < T; t1++) {} break;
        case 122: for (int t1 = 0; t1 < T; t1++) {} break;
        case 123: for (int t1 = 0; t1 < T; t1++) {} break;
        case 124: for (int t1 = 0; t1 < T; t1++) {} break;
        case 125: for (int t1 = 0; t1 < T; t1++) {} break;
        case 126: for (int t1 = 0; t1 < T; t1++) {} break;
        case 127: for (int t1 = 0; t1 < T; t1++) {} break;
        case 128: for (int t1 = 0; t1 < T; t1++) {} break;
        case 129: for (int t1 = 0; t1 < T; t1++) {} break;
        case 130: for (int t1 = 0; t1 < T; t1++) {} break;
        case 131: for (int t1 = 0; t1 < T; t1++) {} break;
        case 132: for (int t1 = 0; t1 < T; t1++) {} break;
        case 133: for (int t1 = 0; t1 < T; t1++) {} break;
        case 134: for (int t1 = 0; t1 < T; t1++) {} break;
        case 135: for (int t1 = 0; t1 < T; t1++) {} break;
        case 136: for (int t1 = 0; t1 < T; t1++) {} break;
        case 137: for (int t1 = 0; t1 < T; t1++) {} break;
        case 138: for (int t1 = 0; t1 < T; t1++) {} break;
        case 139: for (int t1 = 0; t1 < T; t1++) {} break;
        case 140: for (int t1 = 0; t1 < T; t1++) {} break;
        case 141: for (int t1 = 0; t1 < T; t1++) {} break;
        case 142: for (int t1 = 0; t1 < T; t1++) {} break;
        case 143: for (int t1 = 0; t1 < T; t1++) {} break;
        case 144: for (int t1 = 0; t1 < T; t1++) {} break;
        case 145: for (int t1 = 0; t1 < T; t1++) {} break;
        case 146: for (int t1 = 0; t1 < T; t1++) {} break;
        case 147: for (int t1 = 0; t1 < T; t1++) {} break;
        case 148: for (int t1 = 0; t1 < T; t1++) {} break;
        case 149: for (int t1 = 0; t1 < T; t1++) {} break;
        case 150: for (int t1 = 0; t1 < T; t1++) {} break;
        case 151: for (int t1 = 0; t1 < T; t1++) {} break;
        case 152: for (int t1 = 0; t1 < T; t1++) {} break;
        case 153: for (int t1 = 0; t1 < T; t1++) {} break;
        case 154: for (int t1 = 0; t1 < T; t1++) {} break;
        case 155: for (int t1 = 0; t1 < T; t1++) {} break;
        case 156: for (int t1 = 0; t1 < T; t1++) {} break;
        case 157: for (int t1 = 0; t1 < T; t1++) {} break;
        case 158: for (int t1 = 0; t1 < T; t1++) {} break;
        case 159: for (int t1 = 0; t1 < T; t1++) {} break;
        case 160: for (int t1 = 0; t1 < T; t1++) {} break;
        case 161: for (int t1 = 0; t1 < T; t1++) {} break;
        case 162: for (int t1 = 0; t1 < T; t1++) {} break;
        case 163: for (int t1 = 0; t1 < T; t1++) {} break;
        case 164: for (int t1 = 0; t1 < T; t1++) {} break;
        case 165: for (int t1 = 0; t1 < T; t1++) {} break;
        case 166: for (int t1 = 0; t1 < T; t1++) {} break;
        case 167: for (int t1 = 0; t1 < T; t1++) {} break;
        case 168: for (int t1 = 0; t1 < T; t1++) {} break;
        case 169: for (int t1 = 0; t1 < T; t1++) {} break;
        case 170: for (int t1 = 0; t1 < T; t1++) {} break;
        case 171: for (int t1 = 0; t1 < T; t1++) {} break;
        case 172: for (int t1 = 0; t1 < T; t1++) {} break;
        case 173: for (int t1 = 0; t1 < T; t1++) {} break;
        case 174: for (int t1 = 0; t1 < T; t1++) {} break;
        case 175: for (int t1 = 0; t1 < T; t1++) {} break;
        case 176: for (int t1 = 0; t1 < T; t1++) {} break;
        case 177: for (int t1 = 0; t1 < T; t1++) {} break;
        case 178: for (int t1 = 0; t1 < T; t1++) {} break;
        case 179: for (int t1 = 0; t1 < T; t1++) {} break;
        case 180: for (int t1 = 0; t1 < T; t1++) {} break;
        case 181: for (int t1 = 0; t1 < T; t1++) {} break;
        case 182: for (int t1 = 0; t1 < T; t1++) {} break;
        case 183: for (int t1 = 0; t1 < T; t1++) {} break;
        case 184: for (int t1 = 0; t1 < T; t1++) {} break;
        case 185: for (int t1 = 0; t1 < T; t1++) {} break;
        case 186: for (int t1 = 0; t1 < T; t1++) {} break;
        case 187: for (int t1 = 0; t1 < T; t1++) {} break;
        case 188: for (int t1 = 0; t1 < T; t1++) {} break;
        case 189: for (int t1 = 0; t1 < T; t1++) {} break;
        case 190: for (int t1 = 0; t1 < T; t1++) {} break;
        case 191: for (int t1 = 0; t1 < T; t1++) {} break;
        case 192: for (int t1 = 0; t1 < T; t1++) {} break;
        case 193: for (int t1 = 0; t1 < T; t1++) {} break;
        case 194: for (int t1 = 0; t1 < T; t1++) {} break;
        case 195: for (int t1 = 0; t1 < T; t1++) {} break;
        case 196: for (int t1 = 0; t1 < T; t1++) {} break;
        case 197: for (int t1 = 0; t1 < T; t1++) {} break;
        case 198: for (int t1 = 0; t1 < T; t1++) {} break;
        case 199: for (int t1 = 0; t1 < T; t1++) {} break;

        case 200: for (int t1 = 0; t1 < T; t1++) {} break;
        case 201: for (int t1 = 0; t1 < T; t1++) {} break;
        case 202: for (int t1 = 0; t1 < T; t1++) {} break;
        case 203: for (int t1 = 0; t1 < T; t1++) {} break;
        case 204: for (int t1 = 0; t1 < T; t1++) {} break;
        case 205: for (int t1 = 0; t1 < T; t1++) {} break;
        case 206: for (int t1 = 0; t1 < T; t1++) {} break;
        case 207: for (int t1 = 0; t1 < T; t1++) {} break;
        case 208: for (int t1 = 0; t1 < T; t1++) {} break;
        case 209: for (int t1 = 0; t1 < T; t1++) {} break;
        case 210: for (int t1 = 0; t1 < T; t1++) {} break;
        case 211: for (int t1 = 0; t1 < T; t1++) {} break;
        case 212: for (int t1 = 0; t1 < T; t1++) {} break;
        case 213: for (int t1 = 0; t1 < T; t1++) {} break;
        case 214: for (int t1 = 0; t1 < T; t1++) {} break;
        case 215: for (int t1 = 0; t1 < T; t1++) {} break;
        case 216: for (int t1 = 0; t1 < T; t1++) {} break;
        case 217: for (int t1 = 0; t1 < T; t1++) {} break;
        case 218: for (int t1 = 0; t1 < T; t1++) {} break;
        case 219: for (int t1 = 0; t1 < T; t1++) {} break;
        case 220: for (int t1 = 0; t1 < T; t1++) {} break;
        case 221: for (int t1 = 0; t1 < T; t1++) {} break;
        case 222: for (int t1 = 0; t1 < T; t1++) {} break;
        case 223: for (int t1 = 0; t1 < T; t1++) {} break;
        case 224: for (int t1 = 0; t1 < T; t1++) {} break;
        case 225: for (int t1 = 0; t1 < T; t1++) {} break;
        case 226: for (int t1 = 0; t1 < T; t1++) {} break;
        case 227: for (int t1 = 0; t1 < T; t1++) {} break;
        case 228: for (int t1 = 0; t1 < T; t1++) {} break;
        case 229: for (int t1 = 0; t1 < T; t1++) {} break;
        case 230: for (int t1 = 0; t1 < T; t1++) {} break;
        case 231: for (int t1 = 0; t1 < T; t1++) {} break;
        case 232: for (int t1 = 0; t1 < T; t1++) {} break;
        case 233: for (int t1 = 0; t1 < T; t1++) {} break;
        case 234: for (int t1 = 0; t1 < T; t1++) {} break;
        case 235: for (int t1 = 0; t1 < T; t1++) {} break;
        case 236: for (int t1 = 0; t1 < T; t1++) {} break;
        case 237: for (int t1 = 0; t1 < T; t1++) {} break;
        case 238: for (int t1 = 0; t1 < T; t1++) {} break;
        case 239: for (int t1 = 0; t1 < T; t1++) {} break;
        case 240: for (int t1 = 0; t1 < T; t1++) {} break;
        case 241: for (int t1 = 0; t1 < T; t1++) {} break;
        case 242: for (int t1 = 0; t1 < T; t1++) {} break;
        case 243: for (int t1 = 0; t1 < T; t1++) {} break;
        case 244: for (int t1 = 0; t1 < T; t1++) {} break;
        case 245: for (int t1 = 0; t1 < T; t1++) {} break;
        case 246: for (int t1 = 0; t1 < T; t1++) {} break;
        case 247: for (int t1 = 0; t1 < T; t1++) {} break;
        case 248: for (int t1 = 0; t1 < T; t1++) {} break;
        case 249: for (int t1 = 0; t1 < T; t1++) {} break;
        case 250: for (int t1 = 0; t1 < T; t1++) {} break;
        case 251: for (int t1 = 0; t1 < T; t1++) {} break;
        case 252: for (int t1 = 0; t1 < T; t1++) {} break;
        case 253: for (int t1 = 0; t1 < T; t1++) {} break;
        case 254: for (int t1 = 0; t1 < T; t1++) {} break;
        case 255: for (int t1 = 0; t1 < T; t1++) {} break;
        case 256: for (int t1 = 0; t1 < T; t1++) {} break;
        case 257: for (int t1 = 0; t1 < T; t1++) {} break;
        case 258: for (int t1 = 0; t1 < T; t1++) {} break;
        case 259: for (int t1 = 0; t1 < T; t1++) {} break;
        case 260: for (int t1 = 0; t1 < T; t1++) {} break;
        case 261: for (int t1 = 0; t1 < T; t1++) {} break;
        case 262: for (int t1 = 0; t1 < T; t1++) {} break;
        case 263: for (int t1 = 0; t1 < T; t1++) {} break;
        case 264: for (int t1 = 0; t1 < T; t1++) {} break;
        case 265: for (int t1 = 0; t1 < T; t1++) {} break;
        case 266: for (int t1 = 0; t1 < T; t1++) {} break;
        case 267: for (int t1 = 0; t1 < T; t1++) {} break;
        case 268: for (int t1 = 0; t1 < T; t1++) {} break;
        case 269: for (int t1 = 0; t1 < T; t1++) {} break;
        case 270: for (int t1 = 0; t1 < T; t1++) {} break;
        case 271: for (int t1 = 0; t1 < T; t1++) {} break;
        case 272: for (int t1 = 0; t1 < T; t1++) {} break;
        case 273: for (int t1 = 0; t1 < T; t1++) {} break;
        case 274: for (int t1 = 0; t1 < T; t1++) {} break;
        case 275: for (int t1 = 0; t1 < T; t1++) {} break;
        case 276: for (int t1 = 0; t1 < T; t1++) {} break;
        case 277: for (int t1 = 0; t1 < T; t1++) {} break;
        case 278: for (int t1 = 0; t1 < T; t1++) {} break;
        case 279: for (int t1 = 0; t1 < T; t1++) {} break;
        case 280: for (int t1 = 0; t1 < T; t1++) {} break;
        case 281: for (int t1 = 0; t1 < T; t1++) {} break;
        case 282: for (int t1 = 0; t1 < T; t1++) {} break;
        case 283: for (int t1 = 0; t1 < T; t1++) {} break;
        case 284: for (int t1 = 0; t1 < T; t1++) {} break;
        case 285: for (int t1 = 0; t1 < T; t1++) {} break;
        case 286: for (int t1 = 0; t1 < T; t1++) {} break;
        case 287: for (int t1 = 0; t1 < T; t1++) {} break;
        case 288: for (int t1 = 0; t1 < T; t1++) {} break;
        case 289: for (int t1 = 0; t1 < T; t1++) {} break;
        case 290: for (int t1 = 0; t1 < T; t1++) {} break;
        case 291: for (int t1 = 0; t1 < T; t1++) {} break;
        case 292: for (int t1 = 0; t1 < T; t1++) {} break;
        case 293: for (int t1 = 0; t1 < T; t1++) {} break;
        case 294: for (int t1 = 0; t1 < T; t1++) {} break;
        case 295: for (int t1 = 0; t1 < T; t1++) {} break;
        case 296: for (int t1 = 0; t1 < T; t1++) {} break;
        case 297: for (int t1 = 0; t1 < T; t1++) {} break;
        case 298: for (int t1 = 0; t1 < T; t1++) {} break;
        case 299: for (int t1 = 0; t1 < T; t1++) {} break;

        case 300: for (int t1 = 0; t1 < T; t1++) {} break;
        case 301: for (int t1 = 0; t1 < T; t1++) {} break;
        case 302: for (int t1 = 0; t1 < T; t1++) {} break;
        case 303: for (int t1 = 0; t1 < T; t1++) {} break;
        case 304: for (int t1 = 0; t1 < T; t1++) {} break;
        case 305: for (int t1 = 0; t1 < T; t1++) {} break;
        case 306: for (int t1 = 0; t1 < T; t1++) {} break;
        case 307: for (int t1 = 0; t1 < T; t1++) {} break;
        case 308: for (int t1 = 0; t1 < T; t1++) {} break;
        case 309: for (int t1 = 0; t1 < T; t1++) {} break;
        case 310: for (int t1 = 0; t1 < T; t1++) {} break;
        case 311: for (int t1 = 0; t1 < T; t1++) {} break;
        case 312: for (int t1 = 0; t1 < T; t1++) {} break;
        case 313: for (int t1 = 0; t1 < T; t1++) {} break;
        case 314: for (int t1 = 0; t1 < T; t1++) {} break;
        case 315: for (int t1 = 0; t1 < T; t1++) {} break;
        case 316: for (int t1 = 0; t1 < T; t1++) {} break;
        case 317: for (int t1 = 0; t1 < T; t1++) {} break;
        case 318: for (int t1 = 0; t1 < T; t1++) {} break;
        case 319: for (int t1 = 0; t1 < T; t1++) {} break;
        case 320: for (int t1 = 0; t1 < T; t1++) {} break;
        case 321: for (int t1 = 0; t1 < T; t1++) {} break;
        case 322: for (int t1 = 0; t1 < T; t1++) {} break;
        case 323: for (int t1 = 0; t1 < T; t1++) {} break;
        case 324: for (int t1 = 0; t1 < T; t1++) {} break;
        case 325: for (int t1 = 0; t1 < T; t1++) {} break;
        case 326: for (int t1 = 0; t1 < T; t1++) {} break;
        case 327: for (int t1 = 0; t1 < T; t1++) {} break;
        case 328: for (int t1 = 0; t1 < T; t1++) {} break;
        case 329: for (int t1 = 0; t1 < T; t1++) {} break;
        case 330: for (int t1 = 0; t1 < T; t1++) {} break;
        case 331: for (int t1 = 0; t1 < T; t1++) {} break;
        case 332: for (int t1 = 0; t1 < T; t1++) {} break;
        case 333: for (int t1 = 0; t1 < T; t1++) {} break;
        case 334: for (int t1 = 0; t1 < T; t1++) {} break;
        case 335: for (int t1 = 0; t1 < T; t1++) {} break;
        case 336: for (int t1 = 0; t1 < T; t1++) {} break;
        case 337: for (int t1 = 0; t1 < T; t1++) {} break;
        case 338: for (int t1 = 0; t1 < T; t1++) {} break;
        case 339: for (int t1 = 0; t1 < T; t1++) {} break;
        case 340: for (int t1 = 0; t1 < T; t1++) {} break;
        case 341: for (int t1 = 0; t1 < T; t1++) {} break;
        case 342: for (int t1 = 0; t1 < T; t1++) {} break;
        case 343: for (int t1 = 0; t1 < T; t1++) {} break;
        case 344: for (int t1 = 0; t1 < T; t1++) {} break;
        case 345: for (int t1 = 0; t1 < T; t1++) {} break;
        case 346: for (int t1 = 0; t1 < T; t1++) {} break;
        case 347: for (int t1 = 0; t1 < T; t1++) {} break;
        case 348: for (int t1 = 0; t1 < T; t1++) {} break;
        case 349: for (int t1 = 0; t1 < T; t1++) {} break;
        case 350: for (int t1 = 0; t1 < T; t1++) {} break;
        case 351: for (int t1 = 0; t1 < T; t1++) {} break;
        case 352: for (int t1 = 0; t1 < T; t1++) {} break;
        case 353: for (int t1 = 0; t1 < T; t1++) {} break;
        case 354: for (int t1 = 0; t1 < T; t1++) {} break;
        case 355: for (int t1 = 0; t1 < T; t1++) {} break;
        case 356: for (int t1 = 0; t1 < T; t1++) {} break;
        case 357: for (int t1 = 0; t1 < T; t1++) {} break;
        case 358: for (int t1 = 0; t1 < T; t1++) {} break;
        case 359: for (int t1 = 0; t1 < T; t1++) {} break;
        case 360: for (int t1 = 0; t1 < T; t1++) {} break;
        case 361: for (int t1 = 0; t1 < T; t1++) {} break;
        case 362: for (int t1 = 0; t1 < T; t1++) {} break;
        case 363: for (int t1 = 0; t1 < T; t1++) {} break;
        case 364: for (int t1 = 0; t1 < T; t1++) {} break;
        case 365: for (int t1 = 0; t1 < T; t1++) {} break;
        case 366: for (int t1 = 0; t1 < T; t1++) {} break;
        case 367: for (int t1 = 0; t1 < T; t1++) {} break;
        case 368: for (int t1 = 0; t1 < T; t1++) {} break;
        case 369: for (int t1 = 0; t1 < T; t1++) {} break;
        case 370: for (int t1 = 0; t1 < T; t1++) {} break;
        case 371: for (int t1 = 0; t1 < T; t1++) {} break;
        case 372: for (int t1 = 0; t1 < T; t1++) {} break;
        case 373: for (int t1 = 0; t1 < T; t1++) {} break;
        case 374: for (int t1 = 0; t1 < T; t1++) {} break;
        case 375: for (int t1 = 0; t1 < T; t1++) {} break;
        case 376: for (int t1 = 0; t1 < T; t1++) {} break;
        case 377: for (int t1 = 0; t1 < T; t1++) {} break;
        case 378: for (int t1 = 0; t1 < T; t1++) {} break;
        case 379: for (int t1 = 0; t1 < T; t1++) {} break;
        case 380: for (int t1 = 0; t1 < T; t1++) {} break;
        case 381: for (int t1 = 0; t1 < T; t1++) {} break;
        case 382: for (int t1 = 0; t1 < T; t1++) {} break;
        case 383: for (int t1 = 0; t1 < T; t1++) {} break;
        case 384: for (int t1 = 0; t1 < T; t1++) {} break;
        case 385: for (int t1 = 0; t1 < T; t1++) {} break;
        case 386: for (int t1 = 0; t1 < T; t1++) {} break;
        case 387: for (int t1 = 0; t1 < T; t1++) {} break;
        case 388: for (int t1 = 0; t1 < T; t1++) {} break;
        case 389: for (int t1 = 0; t1 < T; t1++) {} break;
        case 390: for (int t1 = 0; t1 < T; t1++) {} break;
        case 391: for (int t1 = 0; t1 < T; t1++) {} break;
        case 392: for (int t1 = 0; t1 < T; t1++) {} break;
        case 393: for (int t1 = 0; t1 < T; t1++) {} break;
        case 394: for (int t1 = 0; t1 < T; t1++) {} break;
        case 395: for (int t1 = 0; t1 < T; t1++) {} break;
        case 396: for (int t1 = 0; t1 < T; t1++) {} break;
        case 397: for (int t1 = 0; t1 < T; t1++) {} break;
        case 398: for (int t1 = 0; t1 < T; t1++) {} break;
        case 399: for (int t1 = 0; t1 < T; t1++) {} break;

        }

        for (int c = 0; c < Ncorr; c++)
            to_write(c, t) /= ((double)T);
    });

    if (params.data.checks == "yes") {
        compute_checks_complex(h_mphip, params, f_checks, iconf);
    }
    // Deep copy device views to host views.
    Kokkos::deep_copy(h_write, to_write);
    Kokkos::View<double**, Kokkos::LayoutLeft >::HostMirror lh_write("lh_write", Ncorr, T);
    for (int c = 0; c < Ncorr; c++)
        for (int t = 0; t < T; t++)
            lh_write(c, t) = h_write(c, t);

    fwrite(&lh_write(0, 0), sizeof(double), Ncorr * T, f_G2t);
}

#endif // NOTCOMPILE
