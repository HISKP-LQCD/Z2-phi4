#ifndef updates_H
#define updates_H

#include "IO_params.hpp"
#include <random>

double metropolis_update(Viewphi &field, cluster::IO_params params,  std::mt19937 * x_rand);
double cluster_update(double  ***field, cluster::IO_params params );

#endif
