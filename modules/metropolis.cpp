#define METROPOLIS_C

#include "IO_params.hpp"
#include "lattice.hpp"
#include "updates.hpp"
#include <math.h>

#ifdef DEBUG
#include "geometry.hpp"
#endif

KOKKOS_INLINE_FUNCTION int ctolex(int x3, int x2, int x1, int x0, int L, int L2, int L3) {
    return x3 + x2 * L + x1 * L2 + x0 * L3;
}


KOKKOS_INLINE_FUNCTION void compute_neibourgh_sum(Kokkos::complex<double> neighbourSum[3], Viewphi phi, size_t x, const int* L) {
    // x=x+ y*L1 + z*L1*L2 + t*L1*L2*L3
    const int V2 = L[1] * L[2];
    const int V3 = V2 * L[3];
    Kokkos::complex<double> i(0, 1);

    // direction  0
    int xp = x / (V3);
    int xm = x + (-xp + (xp + L[0] - 1) % L[0]) * (V3);
    xp = x + (-xp + (xp + 1) % L[0]) * (V3);
    neighbourSum[0] = exp(i * phi(0, xp)) + exp(i * phi(0, xm));
    neighbourSum[1] = exp(i * phi(1, xp)) + exp(i * phi(1, xm));
    neighbourSum[3] = (exp(i * phi(0, xp)) - exp(i * phi(0, xm))) * (exp(i * phi(0, xp)) - exp(i * phi(0, xm)));

    // direction z
    xp = (x % (V3)) / (V2);
    xm = x + (-xp + (xp + L[3] - 1) % L[3]) * (V2);
    xp = x + (-xp + (xp + 1) % L[3]) * (V2);
    neighbourSum[0] += exp(i * phi(0, xp)) + exp(i * phi(0, xm));
    neighbourSum[1] += exp(i * phi(1, xp)) + exp(i * phi(1, xm));
    neighbourSum[3] += (exp(i * phi(0, xp)) - exp(i * phi(0, xm))) * (exp(i * phi(0, xp)) - exp(i * phi(0, xm)));

    // direction 1
    xp = (x % (L[1]));
    xm = x + (-xp + (xp + L[1] - 1) % L[1]);
    xp = x + (-xp + (xp + 1) % L[1]);
    neighbourSum[0] += exp(i * phi(0, xp)) + exp(i * phi(0, xm));
    neighbourSum[1] += exp(i * phi(1, xp)) + exp(i * phi(1, xm));
    neighbourSum[3] += (exp(i * phi(0, xp)) - exp(i * phi(0, xm))) * (exp(i * phi(0, xp)) - exp(i * phi(0, xm)));

    // direction 2
    xp = (x % (V2)) / L[1];
    xm = x + (-xp + (xp + L[2] - 1) % L[2]) * L[1];
    xp = x + (-xp + (xp + 1) % L[2]) * L[1];
    neighbourSum[0] += exp(i * phi(0, xp)) + exp(i * phi(0, xm));
    neighbourSum[1] += exp(i * phi(1, xp)) + exp(i * phi(1, xm));
    neighbourSum[3] += (exp(i * phi(0, xp)) - exp(i * phi(0, xm))) * (exp(i * phi(0, xp)) - exp(i * phi(0, xm)));

    neighbourSum[3] /= 2.;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double metropolis_update(Viewphi& phi, cluster::IO_params params, RandPoolType& rand_pool, ViewLatt sectors) {
    // const double kappa, const double lambda,
    // const double delta, const size_t nb_of_hits){
    const size_t V = params.data.V;
    double acc = .0;
    const double delta = params.data.metropolis_delta;
    size_t nb_of_hits = params.data.metropolis_local_hits;
    const double g = params.data.g;
    const bool split_g = params.data.split_g;
    // auto &phi=*field;

#ifdef DEBUG
    static int init_hop = 0;
    Kokkos::View<size_t**> hop;
    if (init_hop == 0) {
        ViewLatt eo_debug(V / 2);
        Kokkos::View<size_t**> tmp1("hop", V, 2 * dim_spacetime);
        hop = tmp1;
        Kokkos::View<size_t**> ipt("ipt", V, dim_spacetime);
        hopping(params.data.L, hop, eo_debug, ipt);
        init_hop = 1;
        printf("initialised hopping for test\n");
    }
    else
        init_hop = 2;
    int test = init_hop;
#endif

    for (int color = 0; color < 3; color++) {
        // for(int x=0; x< V; x++) {
#ifdef DEBUG
        int acc_color = 0; // this should not be necessary
        Kokkos::parallel_reduce("lattice metropolis loop", sectors.size[color], KOKKOS_LAMBDA(size_t xx, int& update) {
#else
        Kokkos::parallel_for("lattice metropolis loop", sectors.size[color], KOKKOS_LAMBDA(size_t xx) {
#endif
            size_t x = sectors.rbg(color, xx);
            double kappa[2] = { params.data.kappa0, params.data.kappa1 };

            // getting a generator from the pool
            gen_type rgen = rand_pool.get_state(xx);

            // compute the neighbour sum
            Kokkos::complex<double> neighbourSum[3];
            compute_neibourgh_sum(neighbourSum, phi, x, params.data.L);
            Kokkos::complex<double> I(0, 1);

#ifdef DEBUG
            if (test == 1) {
                if (x == 0) printf("neighbourSum test start\n");
                Kokkos::complex<double> neighbourSum1[2];
                neighbourSum1[0].real() = 0; neighbourSum1[0].imag() = 0;
                neighbourSum1[1].real() = 0; neighbourSum1[1].imag() = 0;

                for (size_t dir = 0; dir < dim_spacetime; dir++) { // dir = direction
                    neighbourSum1[0] += exp(I * phi(0, hop(x, dir + dim_spacetime))) + exp(I * phi(0, hop(x, dir)));
                    neighbourSum1[1] += exp(I * phi(1, hop(x, dir + dim_spacetime))) + exp(I * phi(1, hop(x, dir)));
                }
                if ((neighbourSum1[0] - neighbourSum[0]).real() > 1e-12 || (neighbourSum1[0] - neighbourSum[0]).imag() > 1e-12) {
                    printf("comp 0, pos=%ld with hop:   %.12g   manually: %.12g\n", x, neighbourSum[0].real(), neighbourSum1[0].real());
                    Kokkos::abort("error in computing the neighbourSum:\n");
                }
                if ((neighbourSum1[1] - neighbourSum[1]).real() > 1e-12 || (neighbourSum1[0] - neighbourSum[0]).imag() > 1e-12) {
                    printf("comp 1, pos=%ld with hop:   %.12g   manually: %.12g\n", x, neighbourSum[1].real(), neighbourSum1[1].real());
                    Kokkos::abort("error in computing the neighbourSum:\n");
                }
                if (x == 0) printf("neighbourSum test passed\n");
            }

#endif
            for (size_t hit = 0; hit < nb_of_hits; hit++) {
                // exp(I phi) --> exp(I (phi+d))
                double d = (rgen.drand() * 2. - 1.) * delta;
                // change of action
                // S0= -2k Re{  phi1^dag(x) \sum_mu phi1(x+mu) } 
                // S1= -2k Re{  phi1^dag(x) \sum_mu phi1(x+mu) }
                // Sg=  g Re{ \phi1^dag  phi_0^3 }
                double dS = (-2. * kappa[0] * exp(-I * phi(0, x)) * neighbourSum[0] * (exp(-I * d) - 1.)).real();
                if (split_g) {
                    dS += (g * neighbourSum[3] * exp(I * (phi(0, x) - phi(1, x))) * (exp(I * d) - 1.)).real();
                }
                else
                    dS += (g * exp(I * (3 * phi(0, x) - phi(1, x))) * (exp(I * 3 * d) - 1.)).real();

                //  accept reject step -------------------------------------
                if (rgen.drand() < exp(-dS)) {
                    phi(0, x) += d;
#ifdef DEBUG
                    update++;
#endif
                }
                // second component
                d = (rgen.drand() * 2. - 1.) * delta;
                dS = (-2. * kappa[1] * exp(-I * phi(1, x)) * neighbourSum[1] * (exp(-I * d) - 1.)).real();
                if (split_g)
                    dS += (g * neighbourSum[3] * exp(I * (phi(0, x) - phi(1, x))) * (exp(-I * d) - 1.)).real();
                else
                    dS += (g * exp(I * (3 * phi(0, x) - phi(1, x))) * (exp(-I * d) - 1.)).real();

                //  accept reject step -------------------------------------
                if (rgen.drand() < exp(-dS)) {
                    phi(1, x) += d;
#ifdef DEBUG
                    update++;
#endif
                }
            } // multi hit ends here



            // Give the state back, which will allow another thread to aquire it
            rand_pool.free_state(rgen);
#ifdef DEBUG
        }, acc_color);
        acc += acc_color;
#else
        }); // end parallel loop over lattice site of the same color
#endif // if DEBUG compute acc
    } // end loop over sectors

    return acc / ((double)2. * nb_of_hits); // the 2 accounts for updating the component indiv.
}

void modulo_2pi(Viewphi& phi, const int V) {
    Kokkos::parallel_for("modulo_2pi", V, KOKKOS_LAMBDA(size_t x) {
        phi(0, x) = phi(0, x) - twoPI * floor(phi(0, x) / twoPI);
        phi(1, x) = phi(1, x) - twoPI * floor(phi(1, x) / twoPI);
    });
}
