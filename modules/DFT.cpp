#define DFT_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include <complex>

#ifdef FFTW
#include <fftw3.h>
#endif

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


#ifdef DEBUG
void test_FT(cluster::IO_params params){
    size_t V=params.data.V;
    size_t Vs=V/params.data.L[0];
    Viewphi  phi("phi",2,V);
    printf("checking FT of constant field:\n");
    
    Kokkos::parallel_for( "init_phi", V, KOKKOS_LAMBDA( size_t x) { 
        phi(0,x)= sqrt(2.*params.data.kappa0);// the FT routines convert in to phisical phi 
        phi(1,x)= sqrt(2.*params.data.kappa1);
    });  
    Viewphi::HostMirror h_phip_test("h_phip_test",2,params.data.L[0]*Vp);
    compute_FT(phi, params ,   0, h_phip_test);
    int T=params.data.L[0];
    for (size_t t=0; t< T; t++) {
        for (size_t x=1; x< Vp; x++) { 
            size_t id=t+x*T;
            if (fabs(h_phip_test(0,id)) >1e-11 ||  fabs(h_phip_test(1,id)) >1e-11  ){
                printf("error FT of a constant field do not gives delta_{p,0}: \n");
                printf("h_phip_test(0,%ld)=%.12g \n",x,h_phip_test(0,id));
                printf("h_phip_test(1,%ld)=%.12g \n",x,h_phip_test(1,id));
                printf("id=t+T*p    id=%ld   t=%ld  p=%ld\n ",id,t,x);
                exit(1);
            }
        }
        if (fabs(h_phip_test(0,t)-1) >1e-11 ||  fabs(h_phip_test(1,t)-1) >1e-11  ){
            printf("error FT of a constant field do not gives delta_{p,0}: \n");
            printf("h_phip_test(0,%ld)=%.12g \n",t,h_phip_test(0,t));
            printf("h_phip_test(1,%ld)=%.12g \n",t,h_phip_test(1,t));
            printf("id=t+T*p    id=%ld   t=%ld  p=0\n ",t,t);
            exit(1);
        }
    }
    printf("\tpassed\n");
    printf("checking FT of delta_x,0 field:\n");
    Kokkos::parallel_for( "init_phi", V, KOKKOS_LAMBDA( size_t x) { 
        if(x==0){
            phi(0,x)=Vs* sqrt(2.*params.data.kappa0);// the FT routines convert in to phisical phi 
            phi(1,x)=Vs* sqrt(2.*params.data.kappa1);
        }
        else{
            phi(0,x)= 0;// the FT routines convert in to phisical phi 
            phi(1,x)= 0;
        }
    });  
    compute_FT(phi, params ,   0, h_phip_test);
    for (size_t t=0; t< 1; t++) {
        for (size_t x=0; x< Vp; x++) { 
            size_t id=t+x*T;
            if (x%2 ==0){//real part
                if(fabs(h_phip_test(0,id)-1) >1e-11 ||  fabs(h_phip_test(1,id)-1) >1e-11  ){
                    printf("error FT of a delta_{x,0} field do not gives const: \n");
                    printf("h_phip_test(0,%ld)=%.12g \n",x,h_phip_test(0,id));
                    printf("h_phip_test(1,%ld)=%.12g \n",x,h_phip_test(1,id));
                    printf("id=t+T*p    id=%ld   t=%ld  p=%ld\n ",id,t,x);
                    exit(1);
                }
            }
            if (x%2 ==1){//imag part
                if(fabs(h_phip_test(0,id)) >1e-11 ||  fabs(h_phip_test(1,id)) >1e-11  ){
                    printf("error FT of a delta_{x,0} field do not gives const: \n");
                    printf("h_phip_test(0,%ld)=%.12g \n",x,h_phip_test(0,id));
                    printf("h_phip_test(1,%ld)=%.12g \n",x,h_phip_test(1,id));
                    printf("id=t+T*p    id=%ld   t=%ld  p=%ld\n ",id,t,x);
                    exit(1);
                }
            }
        }
    } 
    printf("\tpassed\n");
}

#ifdef FFTW
void test_FT_vs_FFTW(cluster::IO_params params){
    
    size_t V=params.data.V;
    size_t Vs=V/params.data.L[0];
    int T=params.data.L[0];
    Viewphi  phi("phi",2,V);
    printf("checking FT vs FFTW\n");
    //kokkos
    Viewphi::HostMirror  h_phi = Kokkos::create_mirror_view( phi );
    Kokkos::parallel_for( "init_phi", V, KOKKOS_LAMBDA( size_t x) { 
        phi(0,x)=x* sqrt(2.*params.data.kappa0);// the FT routines convert in to phisical phi 
        phi(1,x)=x* sqrt(2.*params.data.kappa1);
    });  
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_phi, phi );
    Viewphi::HostMirror h_phip_test("h_phip_test",2,params.data.L[0]*Vp);
    compute_FT(phi, params ,   0, h_phip_test);
    //FFTW
    fftw_plan p;
    fftw_complex *in;
    fftw_complex *out;
    int n[3];
    
    n[0]=params.data.L[1];
    n[1]=params.data.L[2];
    n[2]=params.data.L[3];
    
    in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Vs);
    out=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Vs);
    
    //FFTW_FORWARD=e^{-ipx}   FFTW_BACKWARD=e^{+ipx}
    p=fftw_plan_dft(3,n,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
    for (int x=0; x< Vs; x++){
        in[x][0]=h_phi(0,x);// h_phi should be avaluated at x+0*L^3 but it is the same
    }
    fftw_execute(p);
    for (int x=0; x< Vs; x++){
        out[x][0]/=Vs*sqrt(2.*params.data.kappa0);// h_phi should be avaluated at x+0*L^3 but it is the same
    }
    for (int px=0; px< Lp; px++){
        for (int py=0; py< Lp; py++){
            for (int pz=0; pz< Lp; pz++){
                int p=px +py*params.data.L[1]+pz*params.data.L[1]*params.data.L[2];
                int lp=px +py*Lp+pz*Lp*Lp;
                if (fabs(out[p][0]-h_phip_test(0,T*( 0+lp*2) ) )>1e-10 ){
                    printf("error: FT does not produce the same result of FFTW:");
                    printf(" px=%d  py=%d  pz=%d\t",px,py,pz);
                    printf("FFTW=%.12g    FT=%.12g \n",out[p][0],h_phip_test(0,T*( 0+lp*2) ) );
                }
            }
        }
    }
    
    
    fftw_destroy_plan(p);
    fftw_free(out);
    fftw_free(in); 
    printf("\tpassed\n");
    
}
#endif  //FFTW
#endif
