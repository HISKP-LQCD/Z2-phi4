#define write_viewer_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include <complex>

 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double *compute_magnetisations_serial( Viewphi::HostMirror phi,  cluster::IO_params params){

  double *m=(double*) calloc(2,sizeof(double));
  for(int x =0; x< params.data.V; x++){
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
 //as slow as the seiral
 /* Kokkos::View<double *> m("m",2);
  Kokkos::View<double *>::HostMirror h_m = Kokkos::create_mirror_view( m );
  typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
  typedef Kokkos::TeamPolicy<>::member_type  member_type;
  
  Kokkos::parallel_for("magnetization", team_policy( 2 , Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember ) {
       const int comp = teamMember.league_rank();
       m(comp) = 0;
       Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, V ), [&] ( const size_t x, double &inner ) {
           inner+=sqrt(phi(comp,x)*phi(comp,x));
       }, m(comp) );
       
  });
  // Deep copy device views to host views.
   Kokkos::deep_copy( h_m, m ); 
  mr[0]=h_m(2)*sqrt(2*params.data.kappa0)/((double)params.data.V);
  mr[1]=h_m(1)*sqrt(2*params.data.kappa1)/((double)params.data.V);
*/
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
double  *compute_G2( Viewphi::HostMirror phi, cluster::IO_params params ){
    double  *G2=(double*) calloc(4,sizeof(double));
    double vev;
    double kappa0=params.data.kappa0;
    int V=params.data.V;
    int L0=params.data.L[0];
    int Vs=V/L0;
    double tmp1=0, tmp2=0;
    
    for (int x0=0; x0< L0;x0++){
        double w1=(cos(2.*3.1415926535*x0 /(double (L0)) ));
        double w2=(cos(4.*3.1415926535*x0 /(double (L0)) ));
        double ws1=(sin(2.*3.1415926535*x0 /(double (L0)) ));
        double ws2=(sin(4.*3.1415926535*x0 /(double (L0)) ));

        for (int xs=0; xs< Vs; xs++){
            int x=xs+ x0*Vs;    
            G2[0]+=phi(0,x);
            G2[1]+=phi(0,x)*w1;
            tmp1+=phi(0,x)*ws1;
            G2[2]+=phi(0,x)*w2;
            tmp2+=phi(0,x)*ws2;
        }
    }
        
    vev=G2[0]*sqrt(2*kappa0)/((double) V);
    G2[3]=vev;
    
    G2[0]*=G2[0]*2*kappa0;
    G2[0]/=((double) V*V);

    G2[1]=G2[1]*G2[1]+tmp1*tmp1;
    G2[1]*=2*kappa0/((double) V);

    G2[2]=G2[2]*G2[2]+tmp2*tmp2;
    G2[2]*=2*kappa0/((double) V);
    
    return G2;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void  compute_G2t_serial_host(Viewphi::HostMirror phi, cluster::IO_params params , FILE *f_G2t ){
    int L[dim_spacetime];
    double V=1;
    size_t Vs=1;
    for (int i=0; i<dim_spacetime;i++ ){
        L[i]=params.data.L[i];
        V*=L[i];
        Vs*=L[i];
    }
    Vs/=params.data.L[0];
    double  **G2t=(double**) malloc(sizeof(double*)*2);
    G2t[0]=(double*) calloc(L[0],sizeof(double));
    G2t[1]=(double*) calloc(L[0],sizeof(double));     
 

    for(int t=0; t<L[0]; t++) {
        double G2t0=0;
        double G2t1=0;
        for(int t1=0; t1<L[0]; t1++) {
            double phip[2][2]={{0,0},{0,0}};
            int tpt1=(t+t1)%L[0];
            for(int x=0; x<Vs; x++){
                size_t i0= x+t1*Vs;
                phip[0][0]+=phi(0,i0);
                phip[1][0]+=phi(1,i0);	
                i0= x+tpt1*Vs;
                phip[0][1]+=phi(0,i0);	
                phip[1][1]+=phi(1,i0);	
            }
            G2t0+=phip[0][0]*phip[0][1];
            G2t1+=phip[1][0]*phip[1][1];
            
        }
        G2t0*=2.*params.data.kappa0/(Vs*Vs*L[0]);
        G2t1*=2.*params.data.kappa1/(Vs*Vs*L[0]);
        fprintf(f_G2t,"%d \t %.12g \t %.12g \n",t,G2t0,G2t1);
    }
    free(G2t[0]);
    free(G2t[1]);
    free(G2t);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void write_header_measuraments(FILE *f_conf, cluster::IO_params params ){

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
     
     
     int ncorr=15;//number of correlators
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
   typedef array_type<double,7> ValueType;  // used to simplify code below
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< sample::ValueType > {
      KOKKOS_FORCEINLINE_FUNCTION static sample::ValueType sum() {
         return sample::ValueType();
      }
   };
}
 
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void  compute_G2t(const Viewphi &phi, cluster::IO_params params , FILE *f_G2t , int iconf){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    fwrite(&iconf,sizeof(int),1,f_G2t);        

    Viewphi phip("G2t",2,T);
    Viewphi::HostMirror h_phip = Kokkos::create_mirror_view( phip );

    double norm[2]={sqrt(2.*params.data.kappa0),sqrt(2.*params.data.kappa1)};

    for (int comp=0; comp<2;comp++){
       for(int t=0; t<T; t++) {
           h_phip(comp,t) = 0;
           Kokkos::parallel_reduce( "G2t_Vs_loop", Vs , KOKKOS_LAMBDA ( const size_t x, double &inner ) {
               size_t i0= x+t*Vs;
               inner+=phi(comp,i0);
           }, h_phip(comp,t)  );
           h_phip(comp,t)=h_phip(comp,t)/((double) Vs *norm[comp]);
       }
    }

 /* 
    sample::ValueType tr; 
    for (int comp=0; comp<2;comp++){
       for(int t=0; t<T; t++) {
           h_phip(comp,t) = 0;
           Kokkos::parallel_reduce( "G2t_Vs_loop", Vs , KOKKOS_LAMBDA ( const size_t x, sample::ValueType & upd ) {
               size_t i0= x+t*Vs;
               int ix=x%params.data.L[1];
               int iy=(x- ix)%(params.data.L[1]*params.data.L[2]);

               int iz=x /(Vs/params.data.L[3]);
               double twopi=6.28318530718;//2.*3.1415926535;
               upd.the_array[0]+=phi(comp,i0);
               upd.the_array[1]+=phi(comp,i0)*(cos(twopi*ix /(double (params.data.L[1])) ));//mom1 x
               upd.the_array[2]+=phi(comp,i0)*(sin(twopi*ix /(double (params.data.L[1])) ));
               
               upd.the_array[3]+=phi(comp,i0)*(cos(twopi*iy /(double (params.data.L[2])) ));//mom1 y
               upd.the_array[4]+=phi(comp,i0)*(sin(twopi*iy /(double (params.data.L[2])) ));
               
               upd.the_array[5]+=phi(comp,i0)*(cos(twopi*iz /(double (params.data.L[3])) ));//mom1 z
               upd.the_array[6]+=phi(comp,i0)*(sin(twopi*iz /(double (params.data.L[3])) ));
               
               
               
               
           }, Kokkos::Sum<sample::ValueType>(tr)  );
           h_phip(comp,t)=tr.the_array[0]/((double) Vs *norm[comp]);
       }
    }
    */

    // now we continue on the host 
    for(int t=0; t<T; t++) {
        double G2t0=0;
        double G2t1=0;
        double C2t0=0;
        double C2t1=0;
        double C2t=0;
        double C3t0=0;
        double C3t1=0;
        double C3t=0;
        double C40=0;
        double C41=0;
        double C401=0;
        double C201=0;
        double two0totwo1=0;
        double four0totwo1=0;
        double four0totwo0=0;
        for(int t1=0; t1<T; t1++) {
            int tpt1=(t+t1)%T;
            double pp0=h_phip(0,t1) *h_phip(0 , tpt1);
            double pp1=h_phip(1,t1) *h_phip(1 , tpt1);
            std::complex<double> p0 = h_phip(0,t1) + 1i* h_phip(1,t1);
            std::complex<double> cpt = h_phip(0,tpt1) - 1i* h_phip(1,tpt1);
            
            G2t0+=pp0;
            G2t1+=pp1; 
            C2t0+=pp0*pp0;
            C2t1+=pp1*pp1;
            C2t+= pp0*pp0 + pp1*pp1 + 4*pp0*pp1
                - h_phip(0,t1) *h_phip(0 , t1)* h_phip(1,tpt1) *h_phip(1 , tpt1)
                - h_phip(1,t1) *h_phip(1 , t1)* h_phip(0,tpt1) *h_phip(0 , tpt1);
                
            C3t0+=pp0*pp0*pp0;
            C3t1+=pp1*pp1*pp1;
            C3t+=  real(p0*cpt* p0*cpt *p0*cpt);
            
            C40+=h_phip(0,t1)*h_phip(0,(T/8+t1)%T )* h_phip(0,tpt1)*h_phip(0,(T/2+t1)%T );
            C41+=h_phip(1,t1)*h_phip(1,(T/8+t1)%T )* h_phip(1,tpt1)*h_phip(1,(T/2+t1)%T );
            C401+=h_phip(0,t1)*h_phip(1,(T/8+t1)%T )* h_phip(1,tpt1)*h_phip(0,(T/2+t1)%T );
            
            C201+=h_phip(0,t1)*h_phip(1,t1)  *  h_phip(0,tpt1)*h_phip(1,tpt1); 
            two0totwo1+= h_phip(0,t1)*h_phip(0,t1)    *     h_phip(1,tpt1)*h_phip(1,tpt1);
            four0totwo1+= h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1) *     h_phip(1,tpt1)*h_phip(1,tpt1)   ;
            four0totwo0+= h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1) *     h_phip(0,tpt1)*h_phip(0,tpt1)   ;

            
        } 
       
        G2t0/=((double) T);
        G2t1/=((double) T);
        C2t0/=((double) T);
        C2t1/=((double) T);
        C2t/=((double) T);
        C3t0/=((double) T);
        C3t1/=((double) T);
        C3t/=((double) T);
        C40/=((double) T);
        C40/=((double) T);
        C401/=((double) T);
        C201/=((double) T);
        two0totwo1/=((double) T);
        four0totwo1/=((double) T);
        four0totwo0/=((double) T);
        
        fwrite(&G2t0,sizeof(double),1,f_G2t); // 0 c++  || 1 R 
        fwrite(&G2t1,sizeof(double),1,f_G2t);
        fwrite(&C2t0,sizeof(double),1,f_G2t);
        fwrite(&C2t1,sizeof(double),1,f_G2t);
        fwrite(&C2t,sizeof(double),1,f_G2t); // 4 c++  || 5 R 
        fwrite(&C3t0,sizeof(double),1,f_G2t); // 5 c++  || 6 R 
        fwrite(&C3t1,sizeof(double),1,f_G2t);
        fwrite(&C3t,sizeof(double),1,f_G2t);
        fwrite(&C40,sizeof(double),1,f_G2t);   // 8 c++ || 9 R
        fwrite(&C41,sizeof(double),1,f_G2t); // 9 c++  || 10 R 
        fwrite(&C401,sizeof(double),1,f_G2t);// 10 c++  || 11 R 
        fwrite(&C201,sizeof(double),1,f_G2t);
        fwrite(&two0totwo1,sizeof(double),1,f_G2t);  //12 c++  || 13 R 
        fwrite(&four0totwo1,sizeof(double),1,f_G2t);
        fwrite(&four0totwo0,sizeof(double),1,f_G2t);// 14 c++  || 15 R 
    }

    
}

 
 
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void  compute_G2t_ASCI(const Viewphi &phi, cluster::IO_params params , FILE *f_G2t ){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;

    Viewphi phip("G2t",2,T);
    Viewphi::HostMirror h_phip = Kokkos::create_mirror_view( phip );
/*
    typedef Kokkos::TeamPolicy<>               team_policy;
    typedef Kokkos::TeamPolicy<>::member_type  member_type;

    for (int comp=0; comp<2;comp++){
    Kokkos::parallel_for( "G2t_loop", team_policy( L[0], Kokkos::AUTO, 32 ), KOKKOS_LAMBDA ( const member_type &teamMember ) {
       const int t = teamMember.league_rank();
       phip(comp,t) = 0;
       Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, Vs ), [&] ( const size_t x, double &inner ) {
           size_t i0= x+t*Vs;
           inner+=phi(comp,i0);
       }, phip(comp,t)  );
       phip(comp,t)=phip(comp,t)/((double) Vs);
    });
    }
    // Deep copy device views to host views.
    Kokkos::deep_copy( h_phip, phip ); 
*/

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
