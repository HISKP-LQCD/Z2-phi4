#ifndef updates_H
#define updates_H

#include "IO_params.hpp"
#include <random>

double metropolis_update(Viewphi &field, cluster::IO_params params,  RandPoolType rand_pool,  ViewLatt &hop, ViewLatt &even_odd );
double cluster_update(double  ***field, cluster::IO_params params ,  ViewLatt &hop);

#endif
