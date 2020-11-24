#ifndef updates_H
#define updates_H

#include "IO_params.hpp"


double metropolis_update(double ***field, cluster::IO_params params);
double cluster_update(double  ***field, cluster::IO_params params );

#endif
