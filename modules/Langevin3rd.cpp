#define Langevin3rd_C

#include <math.h>
#include "updates.hpp"
#include "IO_params.hpp"
#include "lattice.hpp"

#ifdef DEBUG
#include "geometry.hpp"
#endif




KOKKOS_INLINE_FUNCTION double gauss_rand( double mean, double sigma, double rand1, double rand2){
    double tpi=8.*atan(1.);
    double phi=rand1*tpi;
    double r  =sqrt(-2.*log(1.-rand2)); /* map second number [0,1) -> (0,1] */
    return sigma*r*cos(phi)+mean; 
}

KOKKOS_INLINE_FUNCTION double force( cluster::IO_params params ,Viewphi phi, size_t x){
    // compute the neighbour sum
      double neighbourSum = 0.0;
      //x=x3+ x2*L3+x1*L2*L3 + x0*L1*L2*L3  
      // direction  0
      size_t xp=x /(params.data.L[3]* params.data.L[2]*params.data.L[1]);
      size_t xm=x+( -xp+ (xp+params.data.L[0]- 1)%params.data.L[0]  )*( params.data.L[3]* params.data.L[2]*params.data.L[1]);
      xp=x+(- xp+  (xp+1)%params.data.L[0]) *( params.data.L[3]* params.data.L[2]*params.data.L[1]) ;
      neighbourSum += phi(0, xp ) + phi(0,xm );
      // direction 1
      xp=(x %(params.data.L[3]* params.data.L[2]*params.data.L[1])) / (params.data.L[3]* params.data.L[2] );
      xm=x+( -xp+ (xp+params.data.L[1]- 1)%params.data.L[1]  )*( params.data.L[3]* params.data.L[2]);
      xp=x+(- xp+  (xp+1)%params.data.L[1]) *( params.data.L[3]* params.data.L[2]) ;
      neighbourSum += phi(0, xp ) + phi(0,xm );
      
      // direction 3
      xp=(x %(params.data.L[3]));
      xm=x+( -xp+ (xp+params.data.L[3]- 1)%params.data.L[3]  );
      xp=x+(- xp+  (xp+1)%params.data.L[3])  ;
      neighbourSum += phi(0, xp ) + phi(0,xm );
      // direction 2
      xp=(x %(params.data.L[3]*params.data.L[2]  ))/params.data.L[3];
      xm=x+( -xp+ (xp+params.data.L[2]- 1)%params.data.L[2]  )*params.data.L[3];
      xp=x+(- xp+  (xp+1)%params.data.L[2]) *params.data.L[3] ;
      neighbourSum += phi(0, xp ) + phi(0,xm );
      
      return -neighbourSum + (8.0+params.data.msq0)*phi(0, x ) + params.data.lambdaC0*phi(0, x )*phi(0, x )*phi(0, x )/6.0;
      
}



void Langevin3rd_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool){
    size_t V=params.data.V;
    Viewphi tmp("tmp_field", 3,V);
    double c1=exp(-params.data.Langevin3rd_gamma*params.data.Langevin3rd_eps);
    double c2= sqrt(1.-c1*c1);
        
    Kokkos::parallel_for( "lattice Langevin3d loop", V, KOKKOS_LAMBDA( size_t x ) {    
        
        gen_type rgen = rand_pool.get_state(x);
        
        // mom update
        tmp(2,x)=phi(2,x) +c1 * phi(1,x);
        tmp(2,x)+=gauss_rand( 0,  c2, rgen.drand(), rgen.drand());
        
        
        // partial_t phi=pi
        tmp(0,x)=phi(0,x)+ params.data.Langevin3rd_eps * phi(1,x);
        // partial_t pi=rho
        tmp(1,x)=phi(1,x)+ params.data.Langevin3rd_eps * phi(2,x);
        
        // partial_t phi= - F -\gamma pi + sqrt(2*gamma*eps) *eta
        tmp(2,x)-=params.data.Langevin3rd_eps*force(params ,phi,  x);
        
        
       
        
        rand_pool.free_state(rgen);
    });
    Kokkos::deep_copy(phi,tmp);
    
    
    
    
}



void Langevin3rd_paper_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool){
    size_t V=params.data.V;
    Viewphi tmp("tmp_field", 3,V);
    double gamma=params.data.Langevin3rd_gamma;
    double xi= params.data.Langevin3rd_gamma;
    double eps=params.data.Langevin3rd_eps;
        
    Kokkos::parallel_for( "lattice Langevin3d loop", V, KOKKOS_LAMBDA( size_t x ) {    
        
        gen_type rgen = rand_pool.get_state(x);
        
        // mom update
        double eta=gauss_rand( 0,  sqrt(2*xi), rgen.drand(), rgen.drand());
        
        
        // partial_t phi=pi
        tmp(0,x)=phi(0,x)+ eps * phi(1,x);
        // partial_t pi=
        tmp(1,x)=phi(1,x)+eps*(-force(params ,phi,  x) + gamma * phi(2,x) );
        
        // partial_t rho= 
        tmp(2,x)=phi(2,x) - eps * (gamma* phi(1,x) +xi*phi(2,x) +eta);
        
        rand_pool.free_state(rgen);
    });
    Kokkos::deep_copy(phi,tmp);
    
    
    
    
}

void Langevin_euler(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool){
    size_t V=params.data.V;
    Viewphi tmp("tmp_field", 1,V);
    
    Kokkos::parallel_for( "lattice Langevin3d loop", V, KOKKOS_LAMBDA( size_t x ) {    
        
        gen_type rgen = rand_pool.get_state(x);
       
        // partial_t phi= - F -\gamma pi + sqrt(2*gamma*eps) *eta
        tmp(0,x)=phi(0,x);

        tmp(0,x)-=params.data.Langevin_eps*force(params ,phi,  x);
        
        
        // noise
        //rgen.drand()= number in flat distribution [0,1)
        tmp(0,x)-=gauss_rand( 0,  sqrt(2*params.data.Langevin_eps ), rgen.drand(), rgen.drand());
       
       
        
        rand_pool.free_state(rgen);
    });
    
    Kokkos::parallel_for( "lattice copy loop", V, KOKKOS_LAMBDA( size_t x ) { 
        phi(0,x)=tmp(0,x);

    });
//     Kokkos::deep_copy(phi,tmp);
    
    
    
    
}
