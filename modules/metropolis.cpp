#define METROPOLIS_C

#include <math.h>
#include "updates.hpp"
#include "random.hpp"
#include "IO_params.hpp"
#include "lattice.hpp"
#include <random>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double metropolis_update(Viewphi &phi, cluster::IO_params params, std::mt19937_64 * x_rand  , ViewLatt &hop, ViewLatt &even_odd){
                         //const double kappa, const double lambda, 
                         //const double delta, const size_t nb_of_hits){
  double kappa[2] ={params.data.kappa0, params.data.kappa1};
  double lambda[2]={params.data.lambda0, params.data.lambda1};
  double mu=params.data.mu;
  double g=params.data.g;
  double delta =params.data.metropolis_delta;
  double nb_of_hits=params.data.metropolis_local_hits;
  
  int V=params.data.V;
  double acc = .0;
  //auto &phi=*field; 
  
  for (int parity = 0; parity <2 ;parity ++){
  //for(int x=0; x< V; x++) {  
  Kokkos::parallel_reduce( "lattice loop", V/2, KOKKOS_LAMBDA( size_t xx , double &update) {    
      size_t x=even_odd(parity,xx);
      // computing phi^2 on x
      //auto phiSqr = phi[0][x]*phi[0][x] + phi[1][x]*phi[1][x];

      // running over the four components, comp, of the phi field - Each 
      // component is updated individually with multiple hits
      for(size_t comp = 0; comp < 2; comp++){
        //auto& Phi = phi[comp][x]; // this reference gives a speedup???
        auto phiSqr = phi(comp,x)*phi(comp,x);
        // The other component 
        int comp_n=(comp+1)%2;
        auto phi_n = phi(comp,x);
        // compute the neighbour sum
        auto neighbourSum = 0.0;
        for(size_t dir = 0; dir < dim_spacetime; dir++) // dir = direction
            neighbourSum += phi(comp, hop(x,dir+dim_spacetime) ) + phi(comp, hop(x,dir) );
        // doing the multihit

        for(size_t hit = 0; hit < nb_of_hits; hit++){
            double r[2];
 printf( "here \n");
            //THIS DOES NOT WORK IN THE GPU
            r[0]=x_rand[x]()/((double)x_rand[x].max() );
            r[1]=x_rand[x]()/((double)x_rand[x].max() );
 printf( "IT IS NOT ARRIVING HERE \n");
            auto deltaPhi = (r[0]*2. - 1.)*delta;
            auto deltaPhiPhi = deltaPhi * phi(comp,x);
            auto deltaPhideltaPhi = deltaPhi * deltaPhi;
            // change of action
            auto dS = -2.*kappa[comp]*deltaPhi*neighbourSum + 
                     2.*deltaPhiPhi*(1. - 2.*lambda[comp]*(1. - phiSqr - deltaPhideltaPhi)) +
                     deltaPhideltaPhi*(1. - 2.*lambda[comp]*(1. )) +
                     lambda[comp]*(6.*deltaPhiPhi*deltaPhiPhi + deltaPhideltaPhi*deltaPhideltaPhi);
           
            dS+= mu *phi_n * phi_n *( 2* deltaPhiPhi + deltaPhideltaPhi       ) ;
           
            dS+= comp * g * phi_n * phi_n * phi_n * deltaPhi;  //component 1
            dS+= comp_n * g * deltaPhi * phi_n *( deltaPhideltaPhi + 3. * phiSqr + 3 * phi(comp,x) * deltaPhi  );   //component 0
           
            
            //  accept reject step -------------------------------------
            if(r[1] < exp(-dS)) {
              //phiSqr -= Phi*Phi;
              phi(comp,x) += deltaPhi;
              //phiSqr += Phi*Phi;
              update++; 
            }
        } // multi hit ends here
    } // loop over sites ends here
    //phi.update(parity); // communicate boundaries
  },acc);
  }//end loop parity

  return acc/(2*nb_of_hits); // the 2 accounts for updating the component indiv.

} 
