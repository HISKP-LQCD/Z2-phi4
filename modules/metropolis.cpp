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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double metropolis_update(Viewphi& phi, cluster::IO_params params, RandPoolType& rand_pool, ViewLatt even_odd) {
    // const double kappa, const double lambda,
    // const double delta, const size_t nb_of_hits){
    const size_t V = params.data.V;
    double acc = .0;
    size_t nb_of_hits = params.data.metropolis_local_hits;
    const double g = params.data.g;

    // auto &phi=*field;

#ifdef DEBUG
    static int init_hop = 0;
    ViewLatt hop;
    if (init_hop == 0) {
        ViewLatt eo_debug("even_odd", 2, V / 2);
        ViewLatt tmp1("hop", V, 2 * dim_spacetime);
        hop = tmp1;
        ViewLatt ipt("ipt", V, dim_spacetime);
        hopping(params.data.L, hop, eo_debug, ipt);
        init_hop = 1;
    }
    else
        init_hop = 2;
    int test = init_hop;
#endif

    for (int parity = 0; parity < 2; parity++) {
        // for(int x=0; x< V; x++) {
        // Kokkos::parallel_reduce( "lattice loop", V/2, KOKKOS_LAMBDA( size_t xx , double &update) {
        Kokkos::parallel_for(
            "lattice metropolis loop", V / 2, KOKKOS_LAMBDA(size_t xx) {
            size_t x = even_odd(parity, xx);
            double kappa[2] = { params.data.kappa0, params.data.kappa1 };

            // getting a generator from the pool
            gen_type rgen = rand_pool.get_state(xx);
            // computing phi^2 on x
            // auto phiSqr = phi[0][x]*phi[0][x] + phi[1][x]*phi[1][x];

            // compute the neighbour sum
            double neighbourSum[2] = { 0.0, 0.0 };
            // x=x3+ x2*L3+x1*L2*L3 + x0*L1*L2*L3
            const int V2 = params.data.L[3] * params.data.L[2];
            const int V3 = V2 * params.data.L[1];

            // direction  0
            int xp = x / (V3);
            int xm = x + (-xp + (xp + params.data.L[0] - 1) % params.data.L[0]) * (V3);
            xp = x + (-xp + (xp + 1) % params.data.L[0]) * (V3);
            neighbourSum[0] += phi(0, xp) + phi(0, xm);
            neighbourSum[1] += phi(1, xp) + phi(1, xm);
            // direction 1
            xp = (x % (V3)) / (V2);
            xm = x + (-xp + (xp + params.data.L[1] - 1) % params.data.L[1]) * (V2);
            xp = x + (-xp + (xp + 1) % params.data.L[1]) * (V2);
            neighbourSum[0] += phi(0, xp) + phi(0, xm);
            neighbourSum[1] += phi(1, xp) + phi(1, xm);

            // direction 3
            xp = (x % (params.data.L[3]));
            xm = x + (-xp + (xp + params.data.L[3] - 1) % params.data.L[3]);
            xp = x + (-xp + (xp + 1) % params.data.L[3]);
            neighbourSum[0] += phi(0, xp) + phi(0, xm);
            neighbourSum[1] += phi(1, xp) + phi(1, xm);

            // direction 2
            xp = (x % (V2)) / params.data.L[3];
            xm = x + (-xp + (xp + params.data.L[2] - 1) % params.data.L[2]) * params.data.L[3];
            xp = x + (-xp + (xp + 1) % params.data.L[2]) * params.data.L[3];
            neighbourSum[0] += phi(0, xp) + phi(0, xm);
            neighbourSum[1] += phi(1, xp) + phi(1, xm);

#ifdef DEBUG
            if (test == 1) {
                double neighbourSum1[2] = { 0, 0 };
                for (size_t dir = 0; dir < dim_spacetime; dir++) { // dir = direction
                    neighbourSum1[0] += phi(0, hop(x, dir + dim_spacetime)) + phi(0, hop(x, dir));
                    neighbourSum1[1] += phi(1, hop(x, dir + dim_spacetime)) + phi(1, hop(x, dir));
                }
                if (fabs(neighbourSum1[0] - neighbourSum[0]) > 1e-12 || fabs(neighbourSum1[1] - neighbourSum[1]) > 1e-12) {
                    printf("comp 0 with hop:   %.12g   manually: %.12g\n", neighbourSum[0], neighbourSum1[0]);
                    printf("comp 1 with hop:   %.12g   manually: %.12g\n", neighbourSum[0], neighbourSum1[0]);
                    Kokkos::abort("error in computing the neighbourSum:\n");
                }
            }
#endif

           
            // if spin flip phi0
            if (rgen.urand(0, 2) ){
                // change of action
                double dS = 4. * kappa[0] * phi(0,x) * neighbourSum[0];
                dS -= 2 * g * neighbourSum[0] * neighbourSum[0] * phi(0, x) * phi(1, x);
                //  accept reject step -------------------------------------
                if (rgen.drand() < exp(-dS)) {
                    phi(0, x) =-phi(0, x);
                }
            }
            // if spin flip phi1
            if (rgen.urand(0, 2)){
                // change of action
                double dS = 4. * kappa[1] * phi(1,x) * neighbourSum[1];
                dS -= 2 * g * neighbourSum[0] * neighbourSum[0] * phi(0, x) * phi(1, x);
                //  accept reject step -------------------------------------
                if (rgen.drand() < exp(-dS)) {
                    phi(1, x) =-phi(1, x);
                }
            }
            

            // Give the state back, which will allow another thread to aquire it
            rand_pool.free_state(rgen);
            //},acc);  // end lattice_even loop in parallel
        }); // end lattice_even loop in parallel

    } // end loop parity

    return acc / (2 * nb_of_hits); // the 2 accounts for updating the component indiv.
}
