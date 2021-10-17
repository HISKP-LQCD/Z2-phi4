#ifndef updates_H
#define updates_H

#include "IO_params.hpp"
#include <random>

double metropolis_update(Viewphi &field, cluster::IO_params params,  RandPoolType &rand_pool, ViewLatt even_odd );
double cluster_update(Viewphi &field, cluster::IO_params params , RandPoolType &rand_pool, std::mt19937_64 host_rand , ViewLatt &hop);
void Langevin3rd_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool);
void Langevin3rd_paper_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool);
void Langevin_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool);

#endif
