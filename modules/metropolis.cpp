#define METROPOLIS_C

#include <math.h>
#include "updates.hpp"
#include "random.hpp"
#include "IO_params.hpp"
#include "lattice.hpp"
#include <random>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double metropolis_update(Viewphi &phi, cluster::IO_params params, RandPoolType &rand_pool  , ViewLatt hop, ViewLatt even_odd){
                         //const double kappa, const double lambda, 
                         //const double delta, const size_t nb_of_hits){
  double mu=params.data.mu;
  double g=params.data.g;
  double delta =params.data.metropolis_delta;
  double nb_of_hits=params.data.metropolis_local_hits;
  
  size_t V=params.data.V;
  double acc = .0;
  //auto &phi=*field; 
  
  
  for (int parity = 0; parity <2 ;parity ++){
  //for(int x=0; x< V; x++) {  
  //Kokkos::parallel_reduce( "lattice loop", V/2, KOKKOS_LAMBDA( size_t xx , double &update) {    
  Kokkos::parallel_for( "lattice loop", V/2, KOKKOS_LAMBDA( size_t xx ) {    
      size_t x=even_odd(parity,xx);
      double kappa[2] ={params.data.kappa0, params.data.kappa1};
      double lambda[2]={params.data.lambda0, params.data.lambda1};
      
      //getting a generator from the pool 
      gen_type rgen = rand_pool.get_state();
      // computing phi^2 on x
      //auto phiSqr = phi[0][x]*phi[0][x] + phi[1][x]*phi[1][x];

      // running over the four components, comp, of the phi field - Each 
      // component is updated individually with multiple hits
      for(size_t comp = 0; comp < 2; comp++){
        double phiSqr = phi(comp,x)*phi(comp,x);

        // The other component 
        int comp_n=(comp+1)%2;
        double phi_n = phi(comp_n,x);
        // compute the neighbour sum
        double neighbourSum = 0.0;
    	//x=x3+ x2*L3+x1*L2*L3 + x0*L1*L2*L3  
        // direction  0
        size_t xp=x /(params.data.L[3]* params.data.L[2]*params.data.L[1]);
        size_t xm=x+( -xp+ (xp+params.data.L[0]- 1)%params.data.L[0]  )*( params.data.L[3]* params.data.L[2]*params.data.L[1]);
        xp=x+(- xp+  (xp+1)%params.data.L[0]) *( params.data.L[3]* params.data.L[2]*params.data.L[1]) ;
        neighbourSum += phi(comp, xp ) + phi(comp,xm );
        // direction 1
        xp=(x %(params.data.L[3]* params.data.L[2]*params.data.L[1])) / (params.data.L[3]* params.data.L[2] );
        xm=x+( -xp+ (xp+params.data.L[1]- 1)%params.data.L[1]  )*( params.data.L[3]* params.data.L[2]);
        xp=x+(- xp+  (xp+1)%params.data.L[1]) *( params.data.L[3]* params.data.L[2]) ;
        neighbourSum += phi(comp, xp ) + phi(comp,xm );
        // direction 3
        xp=(x %(params.data.L[3]));
        xm=x+( -xp+ (xp+params.data.L[3]- 1)%params.data.L[3]  );
        xp=x+(- xp+  (xp+1)%params.data.L[3])  ;
        neighbourSum += phi(comp, xp ) + phi(comp,xm );
        // direction 2
        xp=(x %(params.data.L[3]*params.data.L[2]  ))/params.data.L[3];
        xm=x+( -xp+ (xp+params.data.L[2]- 1)%params.data.L[2]  )*params.data.L[3];
        xp=x+(- xp+  (xp+1)%params.data.L[2]) *params.data.L[3] ;
        neighbourSum += phi(comp, xp ) + phi(comp,xm );

        #ifdef DEBUG
            double neighbourSum1=0;
            for(size_t dir = 0; dir < dim_spacetime; dir++) // dir = direction
                    neighbourSum1 += phi(comp, hop(x,dir+dim_spacetime) ) + phi(comp, hop(x,dir) );
            if(fabs(neighbourSum1 - neighbourSum)>1e-12) {
                printf("error in computing the neighbourSum:\n");
                printf("with hop:   %.12g   manually: %.12g\n",neighbourSum,neighbourSum1);
                exit(1);
            }
        #endif
        //for(size_t dir = 0; dir < dim_spacetime; dir++) // dir = direction
        //    neighbourSum += phi(comp, hop(x,dir+dim_spacetime) ) + phi(comp, hop(x,dir) );
        // doing the multihit
        for(size_t hit = 0; hit < nb_of_hits; hit++){
            double r[2];
            //  getting two random double  in 0,1
            r[0]=rgen.drand();
            r[1]=rgen.drand();
            double deltaPhi = (r[0]*2. - 1.)*delta;
            double deltaPhiPhi = deltaPhi * phi(comp,x);
            double deltaPhideltaPhi = deltaPhi * deltaPhi;
            // change of action
            double dS = -2.*kappa[comp]*deltaPhi*neighbourSum + 
                     2.*deltaPhiPhi*(1. - 2.*lambda[comp]*(1. - phiSqr - deltaPhideltaPhi)) +
                     deltaPhideltaPhi*(1. - 2.*lambda[comp]*(1. )) +
                     lambda[comp]*(6.*deltaPhiPhi*deltaPhiPhi + deltaPhideltaPhi*deltaPhideltaPhi);
           
            dS+= mu *phi_n * phi_n *( 2.* deltaPhiPhi + deltaPhideltaPhi       ) ;
            
            dS+= comp * g * phi_n * phi_n * phi_n * deltaPhi;  //component 1
            dS+= comp_n * g * deltaPhi * phi_n *( deltaPhideltaPhi + 3. * phiSqr + 3 * phi(comp,x) * deltaPhi  );   //component 0

            
            //  accept reject step -------------------------------------
            if(r[1] < exp(-dS)) {
              phi(comp,x) += deltaPhi;
              phiSqr = phi(comp,x)*phi(comp,x);
             // update++; 
            }
        } // multi hit ends here
      } // loop components  ends here

    // Give the state back, which will allow another thread to aquire it
    rand_pool.free_state(rgen);
  //},acc);  // end lattice_even loop in parallel
  });  // end lattice_even loop in parallel

  }//end loop parity
  
  return acc/(2*nb_of_hits); // the 2 accounts for updating the component indiv.

} 
