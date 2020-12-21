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
void  compute_G2t(const Viewphi &phi, cluster::IO_params params , FILE *f_G2t ){
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
