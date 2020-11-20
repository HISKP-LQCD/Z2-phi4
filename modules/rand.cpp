#define rand_H
 
#include <cstdlib>
#include <cmath>
#include <limits>
#include "linear_fit.hpp"


double generateGaussianNoise(double mu, double sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;

	thread_local double z1;
/*	thread_local bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;
*/
	double u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}


double *generate_correlatedNoise(int N, double *mu, double **sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;
    int i,j;
    double *eta=(double*) malloc(sizeof(double)*N);
    double *r=(double*) calloc(N,sizeof(double));
    double **L;
    

    L=cholesky_decomposition(sigma, N);
    for (i=0;i<N;i++)
        eta[i]=generateGaussianNoise(0, 1.);
    
    for (i=0;i<N;i++){
        for (j=0;j<N;j++)
            r[i]+=L[i][j]*eta[j];
        r[i]+=mu[i];
        //r[i]=eta[i]*sqrt(sigma[i][i])+ mu[i];
    }
    for (i=0;i<N;i++)
        free(L[i]);
    free(L);free(eta);
    
        
   return r;     
    
}
