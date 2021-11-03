#ifndef updates_H
#define updates_H

#include "IO_params.hpp"
#include <random>

double metropolis_update(Viewphi &field, cluster::IO_params params,  RandPoolType &rand_pool, ViewLatt even_odd );
double cluster_update(Viewphi &field, cluster::IO_params params , std::mt19937_64 host_rand , ViewLatt &hop);

#endif
