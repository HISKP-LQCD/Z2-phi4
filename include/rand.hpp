#ifndef rand_H
#define rand_H
 
#include <cstdlib>
#include <cmath>
#include <limits>

double generateGaussianNoise(double mu, double sigma); 
double *generate_correlatedNoise(int N, double *mu, double **sigma);

#endif
