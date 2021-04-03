#define DFT_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include <complex>


void compute_FT(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    
    Viewphi phip("phip",2,params.data.L[0]*Vp);
    
    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    
    Kokkos::parallel_for( "FT_loop", team_policy( T*Vp*2, Kokkos::AUTO), KOKKOS_LAMBDA ( const member_type &teamMember ) {
        const int ii = teamMember.league_rank();
        //ii = comp+ 2*myt  
        //myt=  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        double norm[2]={sqrt(2.*params.data.kappa0),sqrt(2.*params.data.kappa1)};// need to be inside the loop for cuda<10
        const int p=ii/(4*T);
        int res=ii-p*4*T;
        const int reim=res/(2*T);
        res-=reim*2*T;
        const int t=res/2;
        const int comp=res-2*t;
        
        const int px=p%Lp;
        const int pz=p /(Lp*Lp);
        const int py= (p- pz*Lp*Lp)/Lp;
        #ifdef DEBUG
        if (p!= px+ py*Lp+pz*Lp*Lp){ printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n",p,px,py,Lp,pz,Lp,Lp);exit(1);}
        if (ii!= comp+2*(t+T*(reim+p*2))){ printf("error   in the ");exit(1);}
        #endif
        const int xp=t+T*(reim+p*2);
        phip(comp,xp)=0;
        if (reim==0){
            Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, Vs ), [&] ( const size_t x, double &inner) {
                size_t i0= x+t*Vs;
                int ix=x%params.data.L[1];
                int iz=x /(params.data.L[1]*params.data.L[2]);
                int iy=(x- iz*params.data.L[1]*params.data.L[2])/params.data.L[1];
                #ifdef DEBUG
                if (x!= ix+ iy*params.data.L[1]+iz*params.data.L[1]*params.data.L[2]){ 
                    printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n",x,ix,iy,params.data.L[1],iz,params.data.L[1],params.data.L[2]);
                    exit(1);
                }
                #endif
                double wr=6.28318530718 *( px*ix/(double (params.data.L[1])) +    py*iy/(double (params.data.L[2]))   +pz*iz/(double (params.data.L[3]))   );
                wr=cos(wr);
                
                inner+=phi(comp,i0)*wr;// /((double) Vs *norm[comp]);
            }, phip(comp,xp) );
        }
        else if(reim==1){
            Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, Vs ), [&] ( const size_t x, double &inner) {
                size_t i0= x+t*Vs;
                int ix=x%params.data.L[1];
                int iz=x /(params.data.L[1]*params.data.L[2]);
                int iy=(x- iz*params.data.L[1]*params.data.L[2])/params.data.L[1];
                #ifdef DEBUG
                if (x!= ix+ iy*params.data.L[1]+iz*params.data.L[1]*params.data.L[2]){ 
                    printf("error   %ld   = %d  + %d  *%d+ %d*%d*%d\n",x,ix,iy,params.data.L[1],iz,params.data.L[1],params.data.L[2]);
                    exit(1);
                }
                #endif
                double wr=6.28318530718 *( px*ix/(double (params.data.L[1])) +    py*iy/(double (params.data.L[2]))   +pz*iz/(double (params.data.L[3]))   );
                wr=sin(wr);
                
                inner+=phi(comp,i0)*wr;// /((double) Vs *norm[comp]);
            }, phip(comp,xp) );
        }
        
        phip(comp,xp)=phip(comp,xp)/((double) Vs *norm[comp]);
        
    });
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_phip, phip ); 
    
    
}
