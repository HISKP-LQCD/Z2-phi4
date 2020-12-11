#define CLUSTER_C

#include <math.h>
#include <vector>
#include "updates.hpp"
#include "random.hpp"
#include "IO_params.hpp"
#include "lattice.hpp"
#include "random.hpp"
 

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
typedef enum cluster_state_t {
  CLUSTER_UNCHECKED=0,
  CLUSTER_FLIP 
} cluster_state_t;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*inline void check_neighbour(const size_t x_look, const size_t y, 
                            const double kappa0, const double kappa1,
                            double **phi,
                            size_t& cluster_size,
                            std::vector<cluster_state_t>& checked_points,
                            std::vector<size_t>& look){

  if(checked_points.at(y) == CLUSTER_UNCHECKED){
    double dS0 = -4.*kappa0 *  phi[0][x_look]*phi[0][y];
    double dS1 = -4.*kappa1 *  phi[1][x_look]*phi[1][y];
    double dS=dS0+dS1;
    double r[1];
    ranlxd( r,1);
    //if((dS0 < 0.0) && (dS1 < 0.0) && (1.-exp(dS)) > r[0]){
    if((dS0 < 0.0) &&  (1.-exp(dS0)) > r[0]){
      look.emplace_back(y); // y will be used as a starting point in next iter.
      checked_points.at(y) = CLUSTER_FLIP;
      cluster_size++;
    }
  }
}
*/
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
double cluster_update(double  ***field, cluster::IO_params params , ViewLatt &hop){ 
                      //std::vector<size_t>& look_1, std::vector<size_t>& look_2){

  auto &phi=*field; 
  std::vector<size_t> look_1(0, 0), look_2(0, 0); // lookuptables for the cluster  
  double kappa0=params.data.kappa0;
  double kappa1=params.data.kappa1;
  double min_size=params.data.cluster_min_size;  
  size_t V=params.data.V;
  // lookuptable to check which lattice points will be flipped
  std::vector<cluster_state_t>   checked_points(V, CLUSTER_UNCHECKED);
   
  // while-loop: until at least some percentage of the lattice is updated ------
  size_t cluster_size = 0;
  while(double(cluster_size)/V <= min_size){
 
    // Choose a random START POINT for the cluster: 0 <= xx < volume and check 
    // if the point is already part of another cluster - if so another start 
    // point is choosen
    double r[1];
    ranlxd( r,1);
    size_t xx = size_t(r[0]* V);
    while(checked_points.at(xx) == CLUSTER_FLIP){
      ranlxd( r,1);
      xx = size_t(r[0]* V);
    }
    checked_points.at(xx) = CLUSTER_FLIP;
    look_1.emplace_back(xx);
    cluster_size++; 
       //   cout << "  starting point  =    "<< xx <<"="<< look_1[0] << "  field =" << phi[0][xx] << endl;

    // run over both lookuptables until there are no more points to update -----
    while(look_1.size()){ 
      // run over first lookuptable and building up second lookuptable
      look_2.resize(0);
      for(const auto& x_look : look_1){ 
        for(size_t dir = 0; dir < dim_spacetime; dir++){ 
          // negative direction
          auto y = hop(x_look,dir);
       //   cout << "  dir= "<< dir << "  field =" << phi[0][y] << endl;
          check_neighbour(x_look, y, kappa0,kappa1, phi,  cluster_size,
                          checked_points, look_2);
          // positive direction
          y = hop(x_look,dir+dim_spacetime);
       //   cout << "  dir= -"<< dir << "  field =" << phi[0][y] << endl;

          check_neighbour(x_look, y, kappa0,kappa1, phi,  cluster_size,
                          checked_points, look_2);
        }
      }
      // run over second lookuptable and building up first lookuptable
      look_1.resize(0);
      for(const auto& x_look : look_2){ 
        for(size_t dir = 0; dir < dim_spacetime; dir++){ 
          // negative direction
          auto y = hop(x_look,dir);
       //             cout << "  dir= "<< dir << "  field =" << phi[0][y] << endl;

          check_neighbour(x_look, y, kappa0,kappa1, phi, cluster_size,
                          checked_points, look_1);
          // positive direction
          y = hop(x_look,dir+dim_spacetime);
         //           cout << "  dir= -"<< dir << "  field =" << phi[0][y] << endl;

          check_neighbour(x_look, y, kappa0,kappa1, phi, cluster_size,
                          checked_points, look_1);
        }
      }
    } // while loop to build the cluster ends here
  } // while loop to ensure minimal total cluster size ends here

  // perform the phi flip ------------------------------------------------------
  for (int x=0; x<V; x++){
     if(checked_points.at(x) == CLUSTER_FLIP){
        phi[0][x] = -phi[0][x];
        phi[1][x] = -phi[1][x];
    } 
  }
  
  return cluster_size;
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
inline void check_neighbour(const size_t x_look, const size_t y, 
                            const double kappa0, const double kappa1,
                            Viewphi::HostMirror &h_phi,
                            size_t& cluster_size,
                            std::vector<cluster_state_t>& checked_points,
                            std::vector<size_t>& look, double r){

  if(checked_points.at(y) == CLUSTER_UNCHECKED){
    double dS0 = -4.*kappa0 *  h_phi(0,x_look)*h_phi(0,y);
    double dS1 = -4.*kappa1 *  h_phi(1,x_look)*h_phi(1,y);
    double dS=dS0+dS1;
    
    //if((dS0 < 0.0) && (dS1 < 0.0) && (1.-exp(dS)) > r[0]){
    if( (dS1 < 0.0)  && (dS0 < 0.0) &&  (1.-exp(dS)) > r){
      look.emplace_back(y); // y will be used as a starting point in next iter.
      checked_points.at(y) = CLUSTER_FLIP;
      cluster_size++;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double cluster_update(Viewphi  &phi, cluster::IO_params params , RandPoolType rand_pool ,ViewLatt &hop ){ 
                      //std::vector<size_t>& look_1, std::vector<size_t>& look_2){
 
  Viewphi::HostMirror h_phi = Kokkos::create_mirror_view( phi );
  ViewLatt::HostMirror h_hop = Kokkos::create_mirror_view( hop );
  // Deep copy device views to host views.
  Kokkos::deep_copy( h_phi, phi );
  Kokkos::deep_copy( h_hop, hop );
  
  std::vector<size_t> look_1(0, 0), look_2(0, 0); // lookuptables for the cluster  
  double kappa0=params.data.kappa0;
  double kappa1=params.data.kappa1;
  double min_size=params.data.cluster_min_size;  
  size_t V=params.data.V;
  // lookuptable to check which lattice points will be flipped
  std::vector<cluster_state_t>   checked_points(V, CLUSTER_UNCHECKED);
   
  // while-loop: until at least some percentage of the lattice is updated ------
  size_t cluster_size = 0; int count=0;
  while(double(cluster_size)/V <= min_size){
 //printf("iteration %d \n",count++);
    gen_type rgen = rand_pool.get_state();
    // Choose a random START POINT for the cluster: 0 <= xx < volume and check 
    // if the point is already part of another cluster - if so another start 
    // point is choosen
    double r=rgen.drand();
    size_t xx = size_t(r* V);
    while(checked_points.at(xx) == CLUSTER_FLIP){
      xx = size_t(rgen.drand()* V);
    }
    checked_points.at(xx) = CLUSTER_FLIP;
    look_1.emplace_back(xx);
    cluster_size++; 

    // run over both lookuptables until there are no more points to update -----
    while(look_1.size()){ 
      // run over first lookuptable and building up second lookuptable
      look_2.resize(0);
      for(const auto& x_look : look_1){ 
        for(size_t dir = 0; dir < dim_spacetime; dir++){ 
          // negative direction
          auto y = h_hop(x_look,dir);
          
          check_neighbour(x_look, y, kappa0,kappa1, h_phi,  cluster_size,
                          checked_points, look_2, rgen.drand() );
          // positive direction
          y = h_hop(x_look,dir+dim_spacetime);

          check_neighbour(x_look, y, kappa0,kappa1, h_phi,  cluster_size,
                          checked_points, look_2, rgen.drand() );
        }
      }
      // run over second lookuptable and building up first lookuptable
      look_1.resize(0);
      for(const auto& x_look : look_2){ 
        for(size_t dir = 0; dir < dim_spacetime; dir++){ 
          // negative direction
          auto y = h_hop(x_look,dir);

          check_neighbour(x_look, y, kappa0,kappa1, h_phi, cluster_size,
                          checked_points, look_1, rgen.drand());
          // positive direction
          y = h_hop(x_look,dir+dim_spacetime);

          check_neighbour(x_look, y, kappa0,kappa1, h_phi, cluster_size,
                          checked_points, look_1, rgen.drand());
        }
      }
    } // while loop to build the cluster ends here
    rand_pool.free_state(rgen);
  } // while loop to ensure minimal total cluster size ends here

  // perform the phi flip ------------------------------------------------------
  for (int x=0; x<V; x++){
     if(checked_points.at(x) == CLUSTER_FLIP){
        h_phi(0,x) = -h_phi(0,x) ;
        h_phi(1,x) = -h_phi(1,x) ;
    } 
  }
  // Deep copy host views to device views.
  Kokkos::deep_copy( phi, h_phi );
  
  return cluster_size;
}
