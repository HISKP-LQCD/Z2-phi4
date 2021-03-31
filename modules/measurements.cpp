#define write_viewer_H 

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include <IO_params.hpp>
#include <complex>

void  compute_contraction_p1( int t , Viewphi::HostMirror h_phip, cluster::IO_params params , FILE *f_G2t , int iconf);
 
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
     
     
     int ncorr=51;//number of correlators
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
void  compute_G2t(Viewphi::HostMirror h_phip, cluster::IO_params params , FILE *f_G2t , int iconf){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    fwrite(&iconf,sizeof(int),1,f_G2t);        

    

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
        
        double C40_03t16 =0;
        double C41_03t16 =0;
        double C401_03t16=0;
        
        double C40_04t16 =0;
        double C41_04t16 =0;
        double C401_04t16=0;
        
        double C40_03t20 =0;
        double C41_03t20 =0;
        double C401_03t20=0;
        
        double C40_04t20 =0;
        double C41_04t20 =0;
        double C401_04t20=0;
        
        double C40_05t20 =0;
        double C41_05t20 =0;
        double C401_05t20=0;
        
        double C410_03t16=0;
        
        double C401_02t10=0;
        double C401_02t12=0;
        
        
        
        
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

            C40 +=h_phip(0,t1)*h_phip(0,t_8)* h_phip(0,tpt1)*h_phip(0,t_2 );
            C41 +=h_phip(1,t1)*h_phip(1,t_8)* h_phip(1,tpt1)*h_phip(1,t_2 );
            C401+=h_phip(0,t1)*h_phip(1,t_8)* h_phip(1,tpt1)*h_phip(0,t_2 );
            
            C201+=h_phip(0,t1)*h_phip(1,t1)  *  h_phip(0,tpt1)*h_phip(1,tpt1); 
            two0totwo1+= h_phip(0,t1)*h_phip(0,t1)    *     h_phip(1,tpt1)*h_phip(1,tpt1);
            four0totwo1+= h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1) *     h_phip(1,tpt1)*h_phip(1,tpt1)   ;
            four0totwo0+= h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1)*h_phip(0,t1) *     h_phip(0,tpt1)*h_phip(0,tpt1)   ;


            
            C40_03t16 +=h_phip(0,t1)*h_phip(0,t3)* h_phip(0,tpt1)*h_phip(0,t16 );
            C41_03t16 +=h_phip(1,t1)*h_phip(1,t3)* h_phip(1,tpt1)*h_phip(1,t16 );
            C401_03t16+=h_phip(0,t1)*h_phip(1,t3)* h_phip(1,tpt1)*h_phip(0,t16 );
                    
            C40_04t16 +=h_phip(0,t1)*h_phip(0,t4)* h_phip(0,tpt1)*h_phip(0,t16 );
            C41_04t16 +=h_phip(1,t1)*h_phip(1,t4)* h_phip(1,tpt1)*h_phip(1,t16 );
            C401_04t16+=h_phip(0,t1)*h_phip(1,t4)* h_phip(1,tpt1)*h_phip(0,t16 );
            
            C40_03t20 +=h_phip(0,t1)*h_phip(0,t3)* h_phip(0,tpt1)*h_phip(0,t20 );
            C41_03t20 +=h_phip(1,t1)*h_phip(1,t3)* h_phip(1,tpt1)*h_phip(1,t20 );
            C401_03t20+=h_phip(0,t1)*h_phip(1,t3)* h_phip(1,tpt1)*h_phip(0,t20 );
            
            C40_04t20 +=h_phip(0,t1)*h_phip(0,t4)* h_phip(0,tpt1)*h_phip(0,t20 );
            C41_04t20 +=h_phip(1,t1)*h_phip(1,t4)* h_phip(1,tpt1)*h_phip(1,t20 );
            C401_04t20 +=h_phip(0,t1)*h_phip(1,t4)* h_phip(1,tpt1)*h_phip(0,t20 );
            
            C40_05t20 +=h_phip(0,t1)*h_phip(0,t5)* h_phip(0,tpt1)*h_phip(0,t20 );
            C41_05t20+=h_phip(1,t1)*h_phip(1,t5)* h_phip(1,tpt1)*h_phip(1,t20 );
            C401_05t20+=h_phip(0,t1)*h_phip(1,t5)* h_phip(1,tpt1)*h_phip(0,t20 );
            
            
            C410_03t16+=h_phip(1,t1)*h_phip(0,t3)* h_phip(0,tpt1)*h_phip(1,t16 );
            
            C401_02t10+=h_phip(0,t1)*h_phip(1,t2)* h_phip(1,tpt1)*h_phip(0,t10 );
            C401_02t12+=h_phip(0,t1)*h_phip(1,t2)* h_phip(1,tpt1)*h_phip(0,t12 );
            
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
        C41/=((double) T);
        C401/=((double) T);
        C201/=((double) T);
        two0totwo1/=((double) T);
        four0totwo1/=((double) T);
        four0totwo0/=((double) T);
        
       
        C40_03t16 /=((double) T);
        C41_03t16 /=((double) T); 
        C401_03t16/=((double) T); 
        
        C40_04t16 /=((double) T); 
        C41_04t16 /=((double) T); 
        C401_04t16/=((double) T); 
        
        C40_03t20 /=((double) T); 
        C41_03t20 /=((double) T); 
        C401_03t20/=((double) T); 
        
        C40_04t20 /=((double) T); 
        C41_04t20 /=((double) T); 
        C401_04t20 /=((double) T); 
        
        C40_05t20 /=((double) T); 
        C41_05t20/=((double) T); 
        C401_05t20/=((double) T); 
        
        C410_03t16/=((double) T); 
        
        C401_02t10/=((double) T); 
        C401_02t12/=((double) T); 
        
        
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
        
        fwrite(&C40_03t16 ,sizeof(double),1,f_G2t); // 15 c++  || 16 R 
        fwrite(&C41_03t16 ,sizeof(double),1,f_G2t); 
        fwrite(&C401_03t16,sizeof(double),1,f_G2t); // 17 c++  || 18 R 
        
        fwrite(&C40_04t16 ,sizeof(double),1,f_G2t); // 18 c++  || 19 R 
        fwrite(&C41_04t16 ,sizeof(double),1,f_G2t); // 19 c++  || 20 R 
        fwrite(&C401_04t16,sizeof(double),1,f_G2t); // 20 c++  || 21 R 
        
        fwrite(&C40_03t20 ,sizeof(double),1,f_G2t); // 21 c++  || 22 R 
        fwrite(&C41_03t20 ,sizeof(double),1,f_G2t); // 22 c++  || 23 R 
        fwrite(&C401_03t20,sizeof(double),1,f_G2t); // 23 c++  || 24 R 
        
        fwrite(&C40_04t20 ,sizeof(double),1,f_G2t); // 24 c++  || 25 R 
        fwrite(&C41_04t20 ,sizeof(double),1,f_G2t); // 25 c++  || 26 R 
        fwrite(&C401_04t20 ,sizeof(double),1,f_G2t); // 26 c++  || 17 R 
        
        fwrite(&C40_05t20 ,sizeof(double),1,f_G2t); // 27 c++  || 28 R 
        fwrite(&C41_05t20,sizeof(double),1,f_G2t); // 28 c++  || 29 R 
        fwrite(&C401_05t20,sizeof(double),1,f_G2t); // 29 c++  || 30 R 
        
        fwrite(&C410_03t16,sizeof(double),1,f_G2t); // 30 c++  || 31 R 
        
        fwrite(&C401_02t10,sizeof(double),1,f_G2t); // 31 c++  || 32 R 
        fwrite(&C401_02t12,sizeof(double),1,f_G2t); // 32 c++  || 33 R 

        compute_contraction_p1(  t ,  h_phip,  params , f_G2t ,  iconf);
    }

    
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

void  compute_contraction_p1( int t , Viewphi::HostMirror h_phip, cluster::IO_params params , FILE *f_G2t , int iconf){
    int T=params.data.L[0];
    size_t Vs=params.data.V/T;
    
        
        double one_to_one_p[2][3]={{0,0,0},{0,0,0}};
        double two_to_two_A1[3]={0,0,0};
        double two_to_two_E1[3]={0,0,0};
        double two_to_two_E2[3]={0,0,0};
        double two_to_two_A1E1[3]={0,0,0};
        
        for(int t1=0; t1<T; t1++) {
            
            
            std::complex<double> phi[2][3*2]; //phi[comp] [ xyz, -x-y-z]
            std::complex<double> phi_t[2][3*2];
            std::complex<double> bb[3][3]; //back to back [00,11,01][x,y,z]
            std::complex<double> bb_t[3][3]; //back to back [00,11,01][x,y,z]
            std::complex<double> A1[3],A1_t[3];  // phi0, phi1, phi01
            std::complex<double> E1[3],E1_t[3];
            std::complex<double> E2[3],E2_t[3];
            for (int comp=0; comp< 2;comp++){
                std::vector<int>  p1={1,Lp,Lp*Lp};
                for(int i=0;i<3;i++){
                    int t1_p =t1+(  2*p1[i])*T;   // 2,4 6    real part
                    int t1_ip=t1+(1+ 2*p1[i])*T;   /// 3,5 7 imag part
                    int tpt1_p=(t+t1)%T+(2*p1[i])*T;   //2,4 6    real part
                    int tpt1_ip=(t+t1)%T+(1+ 2*p1[i])*T;   /// 3,5,6 imag
                    
                    phi[comp][i]=h_phip(comp,t1_p) + 1i* h_phip(comp,t1_ip);
                    phi_t[comp][i]=h_phip(comp,tpt1_p) + 1i* h_phip(comp,tpt1_ip);
                    
                    one_to_one_p[comp][i]+=real( phi[comp][i]* conj(phi_t[comp][i]) )+real( phi_t[comp][i]* conj(phi[comp][i]) );
                    
                    bb[comp][i]=phi[comp][i]*conj(phi[comp][i]);
                    bb_t[comp][i]=phi_t[comp][i]*conj(phi_t[comp][i]);
                    
                }
                
            }
            for(int i=0;i<3;i++){
                bb[2][i]=(phi[0][i]*conj(phi[1][i])+phi[1][i]*conj(phi[0][i])  )/sqrt(2);
                bb_t[2][i]=(phi_t[0][i]*conj(phi_t[1][i])+phi_t[1][i]*conj(phi_t[0][i])  )/sqrt(2);
            }
            for (int comp=0; comp< 3;comp++){
                A1[comp]=  (bb[comp][0]+bb[comp][1]+bb[comp][2])/sqrt(3);
                A1_t[comp]=(bb_t[comp][0]+bb_t[comp][1]+bb_t[comp][2] )/sqrt(3);
                E1[comp]=  (bb[comp][0]  -bb[comp][1] )/sqrt(2);
                E1_t[comp]=(bb_t[comp][0]-bb_t[comp][1] )/sqrt(2);
                E2[comp]=  (bb[comp][0]+bb[comp][1]-2.*bb[comp][2] )/sqrt(6);
                E2_t[comp]=(bb_t[comp][0]+bb_t[comp][1]-2.*bb_t[comp][2] )/sqrt(6);
                
                two_to_two_A1[comp]+=real(A1[comp]*A1_t[comp]);
                two_to_two_E1[comp]+=real(E1[comp]*E1_t[comp]);
                two_to_two_E2[comp]+=real(E2[comp]*E2_t[comp]);
                two_to_two_A1E1[comp]+=real(A1[comp]*E1_t[comp]+E1[comp]*A1_t[comp]);
                
            }
            
            
        } 
        for (int comp=0; comp< 2;comp++){
            for(int i=0;i<3;i++)
                one_to_one_p[comp][i]/=((double) T);
        }
        for (int comp=0; comp< 3;comp++){
            two_to_two_A1[comp]/=((double) T);
            two_to_two_E1[comp]/=((double) T);
            two_to_two_E2[comp]/=((double) T);
            two_to_two_A1E1[comp]/=((double) T);
        }
        
        fwrite(&one_to_one_p[0][0],sizeof(double),1,f_G2t); // 33 c++  || 34 R    00 x
        fwrite(&one_to_one_p[1][0],sizeof(double),1,f_G2t); // 34 c++  || 35 R    11 x
        fwrite(&one_to_one_p[0][1],sizeof(double),1,f_G2t); // 35 c++  || 36 R    00 y
        fwrite(&one_to_one_p[1][1],sizeof(double),1,f_G2t); // 36 c++  || 37 R    11 y
        fwrite(&one_to_one_p[0][2],sizeof(double),1,f_G2t); // 37 c++  || 38 R    00 z
        fwrite(&one_to_one_p[1][2],sizeof(double),1,f_G2t); // 38 c++  || 39 R    11 z
        
        fwrite(&two_to_two_A1[0],sizeof(double),1,f_G2t); // 39 c++  || 40 R    A1_0 *A1_0
        fwrite(&two_to_two_A1[1],sizeof(double),1,f_G2t); // 40 c++  || 41 R    A1_0 *A1_0
        fwrite(&two_to_two_A1[2],sizeof(double),1,f_G2t); // 41 c++  || 42 R    A1_0 *A1_0
        
        fwrite(&two_to_two_E1[0],sizeof(double),1,f_G2t); // 42 c++  || 43 R    A1_0 *A1_0
        fwrite(&two_to_two_E1[1],sizeof(double),1,f_G2t); // 43 c++  || 44 R    A1_0 *A1_0
        fwrite(&two_to_two_E1[2],sizeof(double),1,f_G2t); // 44 c++  || 45 R    A1_0 *A1_0
        
        fwrite(&two_to_two_E2[0],sizeof(double),1,f_G2t); // 45 c++  || 46 R    A1_0 *A1_0
        fwrite(&two_to_two_E2[1],sizeof(double),1,f_G2t); // 46 c++  || 47 R    A1_0 *A1_0
        fwrite(&two_to_two_E2[2],sizeof(double),1,f_G2t); // 47 c++  || 48 R    A1_0 *A1_0
        
        fwrite(&two_to_two_A1E1[0],sizeof(double),1,f_G2t); // 48 c++  || 49 R    A1_0 *A1_0
        fwrite(&two_to_two_A1E1[1],sizeof(double),1,f_G2t); // 49 c++  || 50 R    A1_0 *A1_0
        fwrite(&two_to_two_A1E1[2],sizeof(double),1,f_G2t); // 50 c++  || 51 R    A1_0 *A1_0
        
        
    
    
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
