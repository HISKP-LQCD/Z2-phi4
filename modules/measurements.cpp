#define write_viewer_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include "measurements.hpp"
#include <complex>

inline void  compute_contraction_p1( int t , Viewphi::HostMirror h_phip, cluster::IO_params params , FILE *f_G2t , int iconf);
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double *compute_magnetisations_serial( Viewphi::HostMirror phi,  cluster::IO_params params){

  double *m=(double*) calloc(2,sizeof(double));
  for(size_t x =0; x< params.data.V; x++){
    m[0] += sqrt(phi(0,x)*phi(0,x));
    m[1] += sqrt(phi(1,x)*phi(1,x));
  }
  m[0]*=sqrt(2*params.data.kappa0)/((double)params.data.V);
  m[1]*=sqrt(2*params.data.kappa1)/((double)params.data.V);
  return m;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double *compute_magnetisations( Viewphi phi,  cluster::IO_params params){

  size_t V=params.data.V; //you can not use params.data.X  on the device
  double *mr=(double*) calloc(2,sizeof(double));

  for (int comp=0; comp<2;comp++){
      Kokkos::parallel_reduce( "magnetization", V, KOKKOS_LAMBDA ( const size_t x, double &inner ) {
           inner+=sqrt(phi(comp,x)*phi(comp,x));
       }, mr[comp] );
  }

  mr[0]*=sqrt(2*params.data.kappa0)/((double)params.data.V);
  mr[1]*=sqrt(2*params.data.kappa1)/((double)params.data.V);
  
  return mr;
}
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void write_header_measuraments(FILE *f_conf, cluster::IO_params params , int ncorr){

     fwrite(&params.data.L, sizeof(int), 4, f_conf); 

     fwrite(params.data.formulation.c_str(), sizeof(char)*100, 1, f_conf); 

     fwrite(&params.data.msq0, sizeof(double), 1, f_conf); 
     fwrite(&params.data.msq1, sizeof(double), 1, f_conf); 
     fwrite(&params.data.lambdaC0, sizeof(double), 1, f_conf); 
     fwrite(&params.data.lambdaC1, sizeof(double), 1, f_conf); 
     fwrite(&params.data.muC, sizeof(double), 1, f_conf); 
     fwrite(&params.data.gC, sizeof(double), 1, f_conf); 

     fwrite(&params.data.metropolis_local_hits, sizeof(int), 1, f_conf); 
     fwrite(&params.data.metropolis_global_hits, sizeof(int), 1, f_conf); 
     fwrite(&params.data.metropolis_delta, sizeof(double), 1, f_conf); 
     
     fwrite(&params.data.cluster_hits, sizeof(int), 1, f_conf); 
     fwrite(&params.data.cluster_min_size, sizeof(double), 1, f_conf); 

     fwrite(&params.data.seed, sizeof(int), 1, f_conf); 
     fwrite(&params.data.replica, sizeof(int), 1, f_conf); 
     
     
     //number of correlators
     //int ncorr=33;//number of correlators
     fwrite(&ncorr, sizeof(int), 1, f_conf); 
     
     size_t size= params.data.L[0]*ncorr;  // number of double of each block
     fwrite(&size, sizeof(size_t), 1, f_conf); 

}
/////////////////////////////////////////////////
 namespace sample {  // namespace helps with name resolution in reduction identity 
   template< class ScalarType, int N >
   struct array_type {
     ScalarType the_array[N];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     array_type() { 
       for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     array_type(const array_type & rhs) { 
        for (int i = 0; i < N; i++ ){
           the_array[i] = rhs.the_array[i];
        }
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     array_type& operator += (const array_type& src) {
       for ( int i = 0; i < N; i++ ) {
          the_array[i]+=src.the_array[i];
       }
       return *this;
     } 
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator += (const volatile array_type& src) volatile {
       for ( int i = 0; i < N; i++ ) {
         the_array[i]+=src.the_array[i];
       }
     }
   };
   //momentum 0,1 for x,y,z  re_im
   //typedef array_type<double,7> ValueType;  // used to simplify code below
   typedef array_type<double,2>  array2;
   typedef array_type<double,7>  array7;
   
   /////////////////////two component 
   template< class ScalarType, int N >
   struct two_component {
     ScalarType the_array[2][N];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     two_component() { 
       for (int i = 0; i < N; i++ ) { 
           the_array[0][i] = 0; 
           the_array[1][i] = 0; 
       }
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     two_component(const two_component & rhs) { 
        for (int i = 0; i < N; i++ ){
           the_array[0][i] = rhs.the_array[0][i];
           the_array[1][i] = rhs.the_array[1][i];
        }
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     two_component& operator += (const two_component& src) {
       for ( int i = 0; i < N; i++ ) {
          the_array[0][i]+=src.the_array[0][i];
          the_array[1][i]+=src.the_array[1][i];
       }
       return *this;
     } 
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator += (const volatile two_component& src) volatile {
       for ( int i = 0; i < N; i++ ) {
         the_array[0][i]+=src.the_array[0][i];
         the_array[1][i]+=src.the_array[1][i];
       }
     }
   };
   typedef two_component<double,7>  two_component7;
   typedef two_component<double,8>  two_component8;
   typedef two_component<double,128>  two_component128;
   typedef two_component<double,Vp>  two_componentVp;
   
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<int N>
   struct reduction_identity< sample::array_type<double,N> > {
      KOKKOS_FORCEINLINE_FUNCTION static sample::array_type<double,N> sum() {
         return sample::array_type<double,N>();
      }
   };
   template<int N>
   struct reduction_identity< sample::two_component<double,N> > {
      KOKKOS_FORCEINLINE_FUNCTION static sample::two_component<double,N> sum() {
         return sample::two_component<double,N>();
      }
   };
   
}


void compute_FT_old(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    double norm[2]={sqrt(2.*params.data.kappa0),sqrt(2.*params.data.kappa1)};
    
    for(int t=0; t<T; t++) {
        //for(int comp=0; comp<2; comp++){
        //    for (int p =0 ; p< Vp;p++)
        //        h_phip(comp,t+p*T)=0;
        //}
        sample::two_componentVp pp;
        Kokkos::parallel_reduce( "FT_Vs_loop", Vs , KOKKOS_LAMBDA ( const size_t x, sample::two_componentVp & upd ) {
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
            for (int pz=0; pz<Lp;pz++){
                for (int py=0; py<Lp;py++){
                    for (int px=0; px<Lp;px++){
                        int re=(px+py*Lp+pz*Lp*Lp)*2;
                        int im=re+1;
                        double wr=6.28318530718 *( px*ix/(double (params.data.L[1])) +    py*iy/(double (params.data.L[2]))   +pz*iz/(double (params.data.L[3]))   );
                        double wi=sin(wr);
                        wr=cos(wr);
                        for(int comp=0; comp<2; comp++){
                            upd.the_array[comp][re]+= phi(comp,i0)*wr;
                            upd.the_array[comp][im]+= phi(comp,i0)*wi;
                        }
                    }
                }
            }
            
            
        }, Kokkos::Sum<sample::two_componentVp>(pp)  );
        //  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        for(int comp=0; comp<2; comp++){
            for (int reim_p =0 ; reim_p< Vp;reim_p++) // reim_p= (reim+ p*2)= 0,..,127
                h_phip(comp,t+reim_p*T)=pp.the_array[comp][reim_p]/((double) Vs *norm[comp]);
        }	
    }
}

void compute_FT_good(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    
    Viewphi phip("phip",2,params.data.L[0]*Vp);

//    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
//    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    //for(int t=0; t<T; t++) {
    Kokkos::parallel_for( "FT_loop",T*Vp*2,  KOKKOS_LAMBDA ( const size_t ii ) {
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
        const int xp=t+T*(reim+p*2);
        #ifdef DEBUG
            if (p!= px+ py*Lp+pz*Lp*Lp){ printf("error   %d   = %d  + %d  *%d+ %d*%d*%d\n",p,px,py,Lp,pz,Lp,Lp);exit(1);}
        #endif
        phip(comp,xp)=0;
        if (reim==0){
		for (size_t  x=0; x< Vs;x++){
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
		    
            phip(comp,xp)+=phi(comp,i0)*wr;// /((double) Vs *norm[comp]);
		}
	}
	else if(reim==1){
		for (size_t  x=0; x< Vs;x++){
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
		    
		    phip(comp,xp)+=phi(comp,i0)*wr;// /((double) Vs *norm[comp]);
		} 
	    
	}
            
                  
        phip(comp,xp)=phip(comp,xp)/((double) Vs *norm[comp]);
        	
    });
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_phip, phip ); 
    

}

 
void compute_FT_tmp(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi::HostMirror &h_phip){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    
    Viewphi phip("phip",2,params.data.L[0]*Vp);

    typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
    typedef Kokkos::TeamPolicy<>::member_type  member_type;
    //for(int t=0; t<T; t++) {
    Kokkos::parallel_for( "FT_loop", team_policy( T, Kokkos::AUTO, 32 ), KOKKOS_LAMBDA ( const member_type &teamMember ) {
        const int t = teamMember.league_rank();
    	double norm[2]={sqrt(2.*params.data.kappa0),sqrt(2.*params.data.kappa1)}; // with cuda 10 you can move it outside
	    
        sample::two_componentVp pp;
        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, Vs ), [&] ( const size_t x, sample::two_componentVp & upd ) {
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
            for (int pz=0; pz<Lp;pz++){
                for (int py=0; py<Lp;py++){
                    for (int px=0; px<Lp;px++){
                        int re=(px+py*Lp+pz*Lp*Lp)*2;
                        int im=re+1;
                        double wr=6.28318530718 *( px*ix/(double (params.data.L[1])) +    py*iy/(double (params.data.L[2]))   +pz*iz/(double (params.data.L[3]))   );
                        double wi=sin(wr);
                        wr=cos(wr);
                        for(int comp=0; comp<2; comp++){
                            upd.the_array[comp][re]+= phi(comp,i0)*wr;
                            upd.the_array[comp][im]+= phi(comp,i0)*wi;
                        }
                    }
                }
            }
            
            
        }, Kokkos::Sum<sample::two_componentVp>(pp)  );
        //  t +T*(reim+ p*2)
        //p=px+py*4+pz*16
        for(int comp=0; comp<2; comp++){
            for (int reim_p =0 ; reim_p< Vp;reim_p++) // reim_p= (reim+ p*2)= 0,..,127
                phip(comp,t+reim_p*T)=pp.the_array[comp][reim_p]/((double) Vs *norm[comp]);
        }	
    });
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_phip, phip ); 
    

}
 


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void  parallel_measurement(Viewphi phip,  Viewphi::HostMirror h_phip,  cluster::IO_params params , FILE *f_G2t , FILE *f_checks, int iconf){
    int T=params.data.L[0];
    fwrite(&iconf,sizeof(int),1,f_G2t);        
    //Viewphi phip("phip",2,params.data.L[0]*Vp);
    // use layoutLeft   to_write(t,c) -> t+x*T; so that there is no need of reordering to write
    Kokkos::View<double**,Kokkos::LayoutLeft > to_write("to_write",  Ncorr,T );
    Kokkos::View<double**,Kokkos::LayoutLeft>::HostMirror h_write=  Kokkos::create_mirror_view( to_write );   
    
    // Deep copy host views to device views.
    //Kokkos::deep_copy( phip, h_phip );
    
    if (params.data.checks== "yes"){
    	// Deep copy device views to host views.
    	Kokkos::deep_copy( h_phip, phip ); // deep_copy with two arguments is a fence
    }
    
    Kokkos::parallel_for( "measurement_t_loop",T, KOKKOS_LAMBDA( size_t t) {
        for(int c=0; c<Ncorr; c++) 
            to_write(c,t)=0;
        const int  p1[3]={1,Lp,Lp*Lp};
        const int  p11[3]={1+Lp,Lp+Lp*Lp,1+Lp*Lp};// (1,1,0),(0,1,1),(1,0,1)
        const int p111=1+Lp+Lp*Lp;
        Kokkos::complex<double> phi1[2][3]; //phi[comp] [ P=(1,0,0),(0,1,0),(0,0,1)]
        Kokkos::complex<double> phi11[2][3]; //phi[comp] [p=(1,1,0),(0,1,1),(1,0,1) ]
        Kokkos::complex<double> phi1_t[2][3];
        Kokkos::complex<double> phi11_t[2][3];
        Kokkos::complex<double> phi111[2]; //p=(1,1,1)
        Kokkos::complex<double> phi111_t[2];
        Kokkos::complex<double> bb[3][3]; //back to back [00,11,01][x,y,z]
        Kokkos::complex<double> bb_t[3][3]; //back to back [00,11,01][x,y,z]
        Kokkos::complex<double> A1[3],A1_t[3];  // phi0, phi1, phi01
        Kokkos::complex<double> E1[3],E1_t[3];
        Kokkos::complex<double> E2[3],E2_t[3];
        Kokkos::complex<double> o2p1[3][3],o2p1_t[3][3];
        Kokkos::complex<double> o2p11[3][3],o2p11_t[3][3];
        Kokkos::complex<double> o2p111[3],o2p111_t[3];
        
        for(int t1=0; t1<T; t1++) {
            int tpt1=(t+t1)%T;
            int t_8=(T/8+t1)%T;
            int t_2=(T/2+t1)%T;
            //int tx2_5=((T*2)/5+t1)%T;
            
            int t2=(2+t1)%T;
            int t3=(3+t1)%T;
            int t4=(4+t1)%T;
            int t5=(5+t1)%T;
            int t10=(10+t1)%T;
            int t12=(12+t1)%T;
            int t16=(16+t1)%T;
            int t20=(20+t1)%T;
            
            double pp0=phip(0,t1) *phip(0 , tpt1);
            double pp1=phip(1,t1) *phip(1 , tpt1);

            Kokkos::complex<double> p0;
            p0.real()=phip(0,t1);   p0.imag()=phip(1,t1); 
            Kokkos::complex<double> cpt ; 
            cpt.real()=phip(0,tpt1);   cpt.imag()= -phip(1,tpt1); 
            
            to_write(0,t)+=pp0;
            to_write(1,t)+=pp1; 
            to_write(2,t)+=pp0*pp0;
            to_write(3,t)+=pp1*pp1;
            to_write(4,t)+= pp0*pp0 + pp1*pp1 + 4*pp0*pp1
            - phip(0,t1) *phip(0 , t1)* phip(1,tpt1) *phip(1 , tpt1)
            - phip(1,t1) *phip(1 , t1)* phip(0,tpt1) *phip(0 , tpt1);
            
            to_write(5,t)+=pp0*pp0*pp0;
            to_write(6,t)+=pp1*pp1*pp1;
            to_write(7,t)+=  real(p0*cpt* p0*cpt *p0*cpt);
            
            to_write(8,t) +=phip(0,t1)*phip(0,t_8)* phip(0,tpt1)*phip(0,t_2 );
            to_write(9,t) +=phip(1,t1)*phip(1,t_8)* phip(1,tpt1)*phip(1,t_2 );
            to_write(10,t)+=phip(0,t1)*phip(1,t_8)* phip(1,tpt1)*phip(0,t_2 );
            
            to_write(11,t)+=phip(0,t1)*phip(1,t1)  *  phip(0,tpt1)*phip(1,tpt1); 
            to_write(12,t)+= phip(0,t1)*phip(0,t1)    *     phip(1,tpt1)*phip(1,tpt1);
            to_write(13,t)+= phip(0,t1)*phip(0,t1)*phip(0,t1)*phip(0,t1) *     phip(1,tpt1)*phip(1,tpt1)   ;
            to_write(14,t)+= phip(0,t1)*phip(0,t1)*phip(0,t1)*phip(0,t1) *     phip(0,tpt1)*phip(0,tpt1)   ;
            
            
            
            to_write(15,t) +=phip(0,t1)*phip(0,t3)* phip(0,tpt1)*phip(0,t16 );
            to_write(16,t) +=phip(1,t1)*phip(1,t3)* phip(1,tpt1)*phip(1,t16 );
            to_write(17,t)+=phip(0,t1)*phip(1,t3)* phip(1,tpt1)*phip(0,t16 );
            
            to_write(18,t) +=phip(0,t1)*phip(0,t4)* phip(0,tpt1)*phip(0,t16 );
            to_write(19,t) +=phip(1,t1)*phip(1,t4)* phip(1,tpt1)*phip(1,t16 );
            to_write(20,t)+=phip(0,t1)*phip(1,t4)* phip(1,tpt1)*phip(0,t16 );
            
            to_write(21,t) +=phip(0,t1)*phip(0,t3)* phip(0,tpt1)*phip(0,t20 );
            to_write(22,t) +=phip(1,t1)*phip(1,t3)* phip(1,tpt1)*phip(1,t20 );
            to_write(23,t)+=phip(0,t1)*phip(1,t3)* phip(1,tpt1)*phip(0,t20 );
            
            to_write(24,t) +=phip(0,t1)*phip(0,t4)* phip(0,tpt1)*phip(0,t20 );
            to_write(25,t) +=phip(1,t1)*phip(1,t4)* phip(1,tpt1)*phip(1,t20 );
            to_write(26,t) +=phip(0,t1)*phip(1,t4)* phip(1,tpt1)*phip(0,t20 );
            
            to_write(27,t) +=phip(0,t1)*phip(0,t5)* phip(0,tpt1)*phip(0,t20 );
            to_write(28,t)+=phip(1,t1)*phip(1,t5)* phip(1,tpt1)*phip(1,t20 );
            to_write(29,t)+=phip(0,t1)*phip(1,t5)* phip(1,tpt1)*phip(0,t20 );
            
            
            to_write(30,t)+=phip(1,t1)*phip(0,t3)* phip(0,tpt1)*phip(1,t16 );
            
            to_write(31,t)+=phip(0,t1)*phip(1,t2)* phip(1,tpt1)*phip(0,t10 );
            to_write(32,t)+=phip(0,t1)*phip(1,t2)* phip(1,tpt1)*phip(0,t12 );
            
            for (int comp=0; comp< 2;comp++){
                
                for(int i=0;i<3;i++){
                    int t1_p =t1+(  2*p1[i])*T;   // 2,4 6    real part
                    int t1_ip=t1_p+T;//t1+(1+ 2*p1[i])*T;   /// 3,5 7 imag part
                    int tpt1_p=tpt1+(2*p1[i])*T;   //2,4 6    real part
                    int tpt1_ip= tpt1_p+T; //(t+t1)%T+(1+ 2*p1[i])*T;   /// 3,5,6 imag
                    
                    int t1_p11 =t1+(  2*p11[i])*T;   //     real part
                    int t1_ip11=t1_p11+T;//t1+(1+ 2*p11[i])*T;   ///  imag part
                    int tpt1_p11=tpt1+(2*p11[i])*T;   //    real part
                    int tpt1_ip11=tpt1_p11+T;//(t+t1)%T+(1+ 2*p11[i])*T;   ///  imag
                    
                    
                    
                    phi1[comp][i].real()=phip(comp,t1_p);       phi1[comp][i].imag()=phip(comp,t1_ip);
                    phi1_t[comp][i].real()=phip(comp,tpt1_p);  phi1_t[comp][i].imag()=phip(comp,tpt1_ip);
                    
                    
                    phi11[comp][i].real()=phip(comp,t1_p11);      phi11[comp][i].imag()=phip(comp,t1_ip11);
                    phi11_t[comp][i].real()=phip(comp,tpt1_p11);  phi11_t[comp][i].imag()=phip(comp,tpt1_ip11);
                    
                    
                    
                    bb[comp][i]=phi1[comp][i]*conj(phi1[comp][i]);
                    bb_t[comp][i]=phi1_t[comp][i]*conj(phi1_t[comp][i]);
                    
                    
                    o2p1[comp][i]=phi1[comp][i]*phip(comp,t1);
                    o2p1_t[comp][i]=conj(phi1_t[comp][i])*phip(comp,(t1+t)%T);
                    
                    
                    o2p11[comp][i]=phi11[comp][i]*phip(comp,t1);
                    o2p11_t[comp][i]=conj(phi11_t[comp][i])*phip(comp,(t1+t)%T);
                    
                    
                }
                int t1_p111 =t1+(  2*p111)*T;   //     real part
                int t1_ip111=t1_p111+T;//  ;t1+(1+ 2*p111)*T;   ///  imag part
                int tpt1_p111=tpt1+(2*p111)*T;   //    real part
                int tpt1_ip111= tpt1_p111+T  ;//(t+t1)%T+(1+ 2*p111)*T;   ///  imag
                
                phi111[comp].real()=phip(comp,t1_p111);     phi111[comp].imag()=phip(comp,t1_ip111);
                phi111_t[comp].real()=phip(comp,tpt1_p111); phi111_t[comp].imag()=phip(comp,tpt1_ip111);
                 
                o2p111[comp]=phi111[comp] * phip(comp,t1);
                o2p111_t[comp]=conj(phi111_t[comp])*phip(comp,(t1+t)%T);
                
            }
            for(int i=0;i<3;i++){
                bb[2][i]=(phi1[0][i]*conj(phi1[1][i])+phi1[1][i]*conj(phi1[0][i])  )/1.41421356237;//sqrt(2);
                bb_t[2][i]=(phi1_t[0][i]*conj(phi1_t[1][i])+phi1_t[1][i]*conj(phi1_t[0][i])  )/1.41421356237;//sqrt(2);
                o2p1[2][i]=phi1[0][i]*phip(1,t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p1_t[2][i]=conj(phi1_t[0][i])*phip(1,(t1+t)%T);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
                
                o2p11[2][i]=phi11[0][i]*phip(1,t1);//  +  phi[1][i]*h_phip(0,t1) ;
                o2p11_t[2][i]=conj(phi11_t[0][i])*phip(1,(t1+t)%T);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
            }
            o2p111[2]=phi111[0]*phip(1,t1);//  +  phi[1][i]*h_phip(0,t1) ;
            o2p111_t[2]=conj(phi111_t[0])*phip(1,(t1+t)%T);//   +   conj(phi_t[1][i])*h_phip(0,(t1+t)%T)   ;
            
            for (int comp=0; comp< 3;comp++){
                A1[comp]=  (bb[comp][0]+bb[comp][1]+bb[comp][2])/1.73205080757;//sqrt(3);
                A1_t[comp]=(bb_t[comp][0]+bb_t[comp][1]+bb_t[comp][2] )/1.73205080757;//sqrt(3);
                E1[comp]=  (bb[comp][0]  -bb[comp][1] )/1.41421356237;//sqrt(2);
                E1_t[comp]=(bb_t[comp][0]-bb_t[comp][1] )/1.41421356237;//sqrt(2);
                E2[comp]=  (bb[comp][0]+bb[comp][1]-2.*bb[comp][2] )/2.44948974278;// sqrt(6);
                E2_t[comp]=(bb_t[comp][0]+bb_t[comp][1]-2.*bb_t[comp][2] )/2.44948974278;//sqrt(6);
            }
            
            to_write(33,t)+=( phi1[0][0]* conj(phi1_t[0][0]) + phi1_t[0][0]* conj(phi1[0][0]) ).real();//one_to_one_p1
            to_write(34,t)+=( phi1[1][0]* conj(phi1_t[1][0]) + phi1_t[1][0]* conj(phi1[1][0]) ).real();//one_to_one_p1
            to_write(35,t)+=( phi1[0][1]* conj(phi1_t[0][1]) + phi1_t[0][1]* conj(phi1[0][1]) ).real();//one_to_one_p1
            to_write(36,t)+=( phi1[1][1]* conj(phi1_t[1][1]) + phi1_t[1][1]* conj(phi1[1][1]) ).real();//one_to_one_p1
            to_write(37,t)+=( phi1[0][2]* conj(phi1_t[0][2]) + phi1_t[0][2]* conj(phi1[0][2]) ).real();//one_to_one_p1
            to_write(38,t)+=( phi1[1][2]* conj(phi1_t[1][2]) + phi1_t[1][2]* conj(phi1[1][2]) ).real();//one_to_one_p1
            
            
            to_write(39,t)+=(A1[0]*A1_t[0]).real();
            to_write(40,t)+=(A1[1]*A1_t[1]).real();
            to_write(41,t)+=(A1[2]*A1_t[2]).real();
                                               
            to_write(42,t)+=(E1[0]*E1_t[0]).real();
            to_write(43,t)+=(E1[1]*E1_t[1]).real();
            to_write(44,t)+=(E1[2]*E1_t[2]).real();
                                               
            to_write(45,t)+=(E2[0]*E2_t[0]).real();
            to_write(46,t)+=(E2[1]*E2_t[1]).real();
            to_write(47,t)+=(E2[2]*E2_t[2]).real();
                                               
            to_write(48,t)+=(A1[0]*E1_t[0]).real();
            to_write(49,t)+=(A1[1]*E1_t[1]).real();
            to_write(50,t)+=(A1[2]*E1_t[2]).real();
                                               
            to_write(51,t)+=(E1[0]*A1_t[0]).real();
            to_write(52,t)+=(E1[1]*A1_t[1]).real();
            to_write(53,t)+=(E1[2]*A1_t[2]).real();
                                               
            to_write(54,t)+=(A1[0]*E2_t[0]).real();
            to_write(55,t)+=(A1[1]*E2_t[1]).real();
            to_write(56,t)+=(A1[2]*E2_t[2]).real();
                                               
            to_write(57,t)+=(E2[0]*A1_t[0]).real();
            to_write(58,t)+=(E2[1]*A1_t[1]).real();
            to_write(59,t)+=(E2[2]*A1_t[2]).real();
                                               
            to_write(60,t)+=(A1[0]*phip(0,tpt1) *phip(0,tpt1)).real(); // A1 o20 before was h_phip(comp,  !!!t!!!)
            to_write(61,t)+=(A1[1]*phip(1,tpt1) *phip(1,tpt1)).real();
            to_write(62,t)+=(A1[2]*phip(0,tpt1) *phip(1,tpt1)).real();
                                               
            to_write(63,t)+=(phip(0,t) *phip(0,t) *A1_t[0]).real();// o20 A1 // two_to_two_o20A1
            to_write(64,t)+=(phip(1,t) *phip(1,t) *A1_t[1]).real();
            to_write(65,t)+=(phip(0,t) *phip(1,t)* A1_t[2]).real();                   
            
            for (int comp=0; comp< 3;comp++)
                for(int i=0;i<3;i++)
                    to_write(66+i+comp*3,t)+=(o2p1[comp][i]*o2p1_t[comp][i]).real();  //two_to_two_o2p1o2p1 
                    
            for (int comp=0; comp< 2;comp++)
                for(int i=0;i<3;i++)
                    to_write(75+comp+i*2,t)+=(phi11[comp][i]*conj(phi11_t[comp][i])   ).real();//one_to_one_p11
                    ///+  conj(phi11[comp][i])*phi11_t[comp][i] no need to add the conj because we are taking the real part
                
            for (int comp=0; comp< 3;comp++)
                for(int i=0;i<3;i++)
                    to_write(81+i+comp*3,t)+=(o2p11[comp][i]*o2p11_t[comp][i]).real() ;//two_to_two_o2p11o2p11
                    
                    
            to_write(90,t)+=(phi111[0]*conj(phi111_t[0])  ).real();//one_to_one_p111
            to_write(91,t)+=(phi111[1]*conj(phi111_t[1])  ).real();
            
            to_write(92,t)+=(o2p111[0]*o2p111_t[0]).real();//two_to_two_o2p111o2p111
            to_write(93,t)+=(o2p111[1]*o2p111_t[1]).real();
            to_write(94,t)+=(o2p111[2]*o2p111_t[2]).real();
            
            for (int comp=0; comp< 3;comp++)
                for(int i=0;i<3;i++)
                    to_write(95+i+comp*3,t)+=(phip(comp%2,t1)* o2p1[comp][i]* phip(comp%2,tpt1) * o2p1_t[comp][i]).real();  //three_to_three_o3p1o3p1 
                    
             for (int comp=0; comp< 3;comp++)
                for(int i=0;i<3;i++)
                    to_write(104+i+comp*3,t)+=(phip(comp%2,t1)*o2p11[comp][i]*  phip(comp%2,tpt1)*o2p11_t[comp][i]).real();//three_to_three_o3p11o3p11
             
            to_write(113,t)+=(phip(0,t1)*o2p111[0]*phip(0,tpt1)*o2p111_t[0]).real();//three_to_three_o3p111o3p111
            to_write(114,t)+=(phip(1,t1)*o2p111[1]*phip(1,tpt1)*o2p111_t[1]).real();
            to_write(115,t)+=(phip(0,t1)*o2p111[2]  *phip(1,tpt1)*o2p111_t[2]).real();
             
            
            to_write(116,t)+=(phip(0,t1)*A1[0]* phip(0,tpt1)*A1_t[0]).real();//three_to_three_A1A1
            to_write(117,t)+=(phip(1,t1)*A1[1]* phip(1,tpt1)*A1_t[1]).real();
            to_write(118,t)+=(phip(0,t1)*A1[2]* phip(1,tpt1)*A1_t[2]).real();
            
            
            to_write(119,t)+=(phip(1,t_2)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   * phip(0,t1)*phip(0,t1)*phip(0,t1)    );   //phi0^3  phi0^3phi1 phi1   
            to_write(120,t)+=(phip(1,t10)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   * phip(0,t1)*phip(0,t1)*phip(0,t1)    );   //phi0^3  phi0^3phi1 phi1
            to_write(121,t)+=(phip(1,t12)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   * phip(0,t1)*phip(0,t1)*phip(0,t1)    );   //phi0^3  phi0^3phi1 phi1
            to_write(122,t)+=(phip(1,t16)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   * phip(0,t1)*phip(0,t1)*phip(0,t1)    );   //phi0^3  phi0^3phi1 phi1
            
            to_write(123,t)+=(phip(0,t_2)*phip(0,t_2)*phip(0,t_2)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   *  phip(1,t1)   );   //phi0^3  phi0^3phi1 phi1   
            to_write(124,t)+=(phip(0,t10)*phip(0,t10)*phip(0,t10)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   *  phip(1,t1)   );   //phi0^3  phi0^3phi1 phi1
            to_write(125,t)+=(phip(0,t12)*phip(0,t12)*phip(0,t12)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   *  phip(1,t1)   );   //phi0^3  phi0^3phi1 phi1
            to_write(126,t)+=(phip(0,t16)*phip(0,t16)*phip(0,t16)*    phip(0,tpt1)*phip(0,tpt1)*phip(0,tpt1)*phip(1,tpt1)   *  phip(1,t1)   );   //phi0^3  phi0^3phi1 phi1
            
        }
        for(int c=0; c<Ncorr; c++) 
            to_write(c,t)/=((double) T);
    });
    
    if (params.data.checks== "yes"){
        compute_checks( h_phip,   params,f_checks, iconf);
    }
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_write, to_write ); 
    fwrite(&h_write(0,0),sizeof(double),Ncorr*T,f_G2t);
}
    
    
    
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void  compute_G2t_ASCI(const Viewphi &phi, cluster::IO_params params , FILE *f_G2t ){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;

    Viewphi phip("G2t",2,T);
    Viewphi::HostMirror h_phip = Kokkos::create_mirror_view( phip );


    for (int comp=0; comp<2;comp++){
       for(int t=0; t<T; t++) {
           h_phip(comp,t) = 0;
           Kokkos::parallel_reduce( "G2t_Vs_loop", Vs , KOKKOS_LAMBDA ( const size_t x, double &inner ) {
               size_t i0= x+t*Vs;
               inner+=phi(comp,i0);
           }, h_phip(comp,t)  );
           h_phip(comp,t)=h_phip(comp,t)/((double) Vs);
       }
    }

    // now we continue on the host 
    for(int t=0; t<T; t++) {
        double G2t0=0;
        double G2t1=0;
        double C2t=0;
        double C3t=0;
        for(int t1=0; t1<T; t1++) {
            int tpt1=(t+t1)%T;
            double pp0=h_phip(0,t1) *h_phip(0 , tpt1);
            double pp1=h_phip(1,t1) *h_phip(1 , tpt1);
            std::complex<double> p0 = h_phip(0,t1) + 1i* h_phip(1,t1);
            std::complex<double> cpt = h_phip(0,tpt1) - 1i* h_phip(1,tpt1);
            
            G2t0+=pp0;
            G2t1+=pp1; 
            C2t+= pp0*pp0 + pp1*pp1 + 4*pp0*pp1
                - h_phip(0,t1) *h_phip(0 , t1)* h_phip(1,tpt1) *h_phip(1 , tpt1)
                - h_phip(1,t1) *h_phip(1 , t1)* h_phip(0,tpt1) *h_phip(0 , tpt1);
            C3t+=  real(p0*cpt* p0*cpt *p0*cpt);
            
        } 
        double norm=2.*params.data.kappa0;
        G2t0*=norm/((double) T);
        G2t1*=2.*params.data.kappa1/((double) T);
        C2t*=norm*norm/((double) T);
        C3t*=norm*norm*norm/((double) T);
        
        fprintf(f_G2t,"%d \t %.12g \t %.12g  \t %.12g \t %.12g \n",t,G2t0,G2t1,C2t,C3t);
    }

    
}

void  compute_checks(Viewphi::HostMirror h_phip, cluster::IO_params params , FILE *f , int iconf){
    int T=params.data.L[0];
    fwrite(&iconf,sizeof(int),1,f); 
    std::complex<double> phi[2][3]; //phi[comp] [ xyz, -x-y-z]
    std::complex<double> phi_t[2][3];
    std::complex<double> bb[3][3]; //back to back [00,11,01][x,y,z]
    std::complex<double> bb_t[3][3]; //back to back [00,11,01][x,y,z]
    std::vector<int>  p1={1,Lp,Lp*Lp};
    
    for(int t=0; t<T; t++) {
        
        double two0_to_two0[3]={0,0,0};
        double two0pmp_to_two0[3]={0,0,0}; 
        double twopmpx_to_twopmpy[3]={0,0,0};
         
        for(int t1=0; t1<T; t1++) {
            
            
                for (int comp=0; comp< 2;comp++){
                for(int i=0;i<3;i++){
                    int t1_p =t1+(  2*p1[i])*T;   // 2,4 6    real part
                    int t1_ip=t1+(1+ 2*p1[i])*T;   /// 3,5 7 imag part
                    int tpt1_p=(t+t1)%T+(2*p1[i])*T;   //2,4 6    real part
                    int tpt1_ip=(t+t1)%T+(1+ 2*p1[i])*T;   /// 3,5,6 imag
                    
                    phi[comp][i]=h_phip(comp,t1_p) + 1i* h_phip(comp,t1_ip);
                    phi_t[comp][i]=h_phip(comp,tpt1_p) + 1i* h_phip(comp,tpt1_ip);
                     
                    
                    bb[comp][i]=phi[comp][i]*conj(phi[comp][i]);
                    bb_t[comp][i]=phi_t[comp][i]*conj(phi_t[comp][i]);
                    //cout<< "p-p    " << t <<phi[comp][i] *conj(phi[comp][i])*phi_t[comp][i] *conj(phi_t[comp][i])   <<endl;
                    //cout<< "p p    " << t <<phi[comp][i] *(phi[comp][i])*conj(phi_t[comp][i] * phi_t[comp][i])   <<endl;
                    
                }
                
            }
            for(int i=0;i<3;i++){
                bb[2][i]=(phi[0][i]*conj(phi[1][i])+phi[1][i]*conj(phi[0][i])  )/sqrt(2);
                bb_t[2][i]=(phi_t[0][i]*conj(phi_t[1][i])+phi_t[1][i]*conj(phi_t[0][i])  )/sqrt(2);
            }
            for (int i=0; i< 3;i++){
                  
                two0_to_two0[i]+=real(bb[0][i]*bb_t[0][i]);
                two0pmp_to_two0[i]+=real(bb[0][i]*h_phip(0,(t+t1)%T) *h_phip(0,(t+t1)%T) );
                twopmpx_to_twopmpy[i]+=real(bb[0][i]*bb_t[0][(i+1)%3]);
            }
            
            
        } 
        
        for (int comp=0; comp< 3;comp++){
            two0_to_two0[comp]/=((double) T);
            two0pmp_to_two0[comp]/=((double) T);
            twopmpx_to_twopmpy[comp]/=((double) T);
            
        }
        
        fwrite(&two0_to_two0[0],sizeof(double),1,f); // 0 c++  || 1 R    00 x
        fwrite(&two0_to_two0[1],sizeof(double),1,f); // 1 c++  || 2 R    00 y
        fwrite(&two0_to_two0[2],sizeof(double),1,f); // 2 c++  || 3 R    00 z
        
        fwrite(&two0pmp_to_two0[0],sizeof(double),1,f); // 3 c++  || 4 R    00 x0
        fwrite(&two0pmp_to_two0[1],sizeof(double),1,f); // 4 c++  || 5 R    00 y0
        fwrite(&two0pmp_to_two0[2],sizeof(double),1,f); // 5 c++  || 6 R    00 z0
        
        fwrite(&twopmpx_to_twopmpy[0],sizeof(double),1,f); // 6 c++  || 7 R   00 (px -px)(py -py)
        fwrite(&twopmpx_to_twopmpy[1],sizeof(double),1,f); // 7 c++  || 8 R   00 (py -py)(pz -pz)
        fwrite(&twopmpx_to_twopmpy[2],sizeof(double),1,f); // 8 c++  || 9 R   00 (pz -pz)(px -px)
        
    }
    
    
}
