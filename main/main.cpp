#define CONTROL

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>


#include "IO_params.hpp"
#include "mutils.hpp"
#include "lattice.hpp"
#include "geometry.hpp"
#include "updates.hpp"
#include <random>

#include "random.hpp"
#include "utils.hpp" 
#include "write_viewer.hpp"
//#include <highfive/H5File.hpp>

#include <Kokkos_Core.hpp>


static int endian;
std::string rng_file; 
static FILE *frng=NULL ;


 
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
void  compute_G2t(Viewphi phi, cluster::IO_params params , FILE *f_G2t ){
    int L[dim_spacetime]={params.data.L[0]};
    size_t Vs=params.data.V/params.data.L[0];

    Viewphi phip("G2t",2,L[0]);
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
       for(int t=0; t<L[0]; t++) {
       h_phip(comp,t) = 0;
       Kokkos::parallel_reduce( "G2t_Vs_loop", Vs , KOKKOS_LAMBDA ( const size_t x, double &inner ) {
           size_t i0= x+t*Vs;
	   inner+=phi(comp,i0);
       }, h_phip(comp,t)  );
       h_phip(comp,t)=h_phip(comp,t)/((double) Vs);
       }
    }

    // now we continue on the host 
    for(int t=0; t<L[0]; t++) {
        double G2t0=0;
        double G2t1=0;
        for(int t1=0; t1<L[0]; t1++) {
            int tpt1=(t+t1)%L[0];
            G2t0+=h_phip(0,t1) *h_phip(0 , tpt1);
            G2t1+=h_phip(1,t1) *h_phip(1 , tpt1); 
        } 
        G2t0*=2.*params.data.kappa0/((double) L[0]);
        G2t1*=2.*params.data.kappa1/((double) L[0]);

        fprintf(f_G2t,"%d \t %.12g \t %.12g \n",t,G2t0,G2t1);
    }

    
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
 double create_phi_update(const double delta, std::mt19937_64 * x_rand, size_t x){
  
  double r=x_rand[x]()/((double)x_rand[x].max() );;
  return (r*2-1)*delta;

} 
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*static void write_rng_state(int N, int state[])
{
   int i,iw;
   stdint_t istd[1];

   istd[0]=(stdint_t)(N);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,frng);

   for(i=0;i<N;i++)
   {
      istd[0]=(stdint_t)(state[i]);

      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      iw+=fwrite(istd,sizeof(stdint_t),1,frng);
   }

   error(iw!=(N+1),1,"write_rng_state ",
              "Incorrect write count");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static int* read_rng_state(void)
{
   int i,ir,N;
   int *state;
   stdint_t istd[1];

   ir=fread(istd,sizeof(stdint_t),1,frng);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   N=(int)(istd[0]);
   state=(int*) malloc(N*sizeof(int));

   for(i=0;i<N;i++)
   {
      ir+=fread(istd,sizeof(stdint_t),1,frng);

      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      state[i]=(int)(istd[0]);
   }

   error(ir!=(N+1),1,"read_rng_state ",
              "Incorrect read count");

   return state;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static void init_rng( cluster::IO_params params)
{
   int *state;
   int append=params.data.append;
   int seed=params.data.seed;
   int level=params.data.level;

   if (append)
   {
      frng=fopen(rng_file.c_str(),"rb");
      error(frng==NULL,1,"init_rng [smd2.c]",
           "Unable to open ranlux state file");

      state=read_rng_state();
      rlxd_reset(state);
      free(state);

      fclose(frng);
   }
   else
      rlxd_init(level,seed);   
}
*/


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    
    endian=endianness();
    //int hop[V][2*D];
    //int ipt[V][D];
    cluster::IO_params params(argc, argv);
    cout << "time " << params.data.L[0] << endl;
    cout << "volume " <<params.data.V << endl;
    cout << "seed " <<params.data.seed << endl;
    cout << "level " <<params.data.level << endl;
    cout << "output " <<params.data.outpath << endl;
    cout << "start mes " << params.data.start_measure << endl;
    size_t V=params.data.V;
    
    
    
    
    
    // init_rng( params);
    
    // starting kokkos
    Kokkos::initialize( argc, argv );{
    
    // seed the PRNG (MT19937) for each  lattice size, with seed , CPU only
/*    std::mt19937_64 *x_rand=(std::mt19937_64*) malloc(sizeof(std::mt19937_64)*V);
    std::mt19937_64  seed_generator(params.data.seed);
    for (size_t x=0;x < V;x++){
        std::mt19937_64 tmp_generator( seed_generator() );
        x_rand[x]=tmp_generator;
    }
*/
    cout << "Kokkos started:"<< endl; 
    cout << "   execution space:"<< typeid(Kokkos::DefaultExecutionSpace).name() << endl; 
    cout << "   host  execution    space:"<<  Kokkos::HostSpace::name << endl; 
    
    int layout_value=check_layout();
    // Create a random number generator pool (64-bit states or 1024-bit state)
    // Both take an 64 bit unsigned integer seed to initialize a Random_XorShift generator 
    // which is used to fill the generators of the pool.
    RandPoolType rand_pool(params.data.seed);
    //Kokkos::Random_XorShift1024_Pool<> rand_pool1024(5374857); 
    cout << "random pool initialised"<< endl; 
    
    ViewLatt    hop("hop",V,2*dim_spacetime);
    ViewLatt    even_odd("even_odd",2,V/2);
    ViewLatt    ipt("ipt",V,dim_spacetime);
    
    hopping( params.data.L, hop,even_odd,ipt);    
    cout << "hopping initialised"<< endl; 
        
    Viewphi  phi("phi",2,V);

    // Initialize phi vector on host.
/*    for (size_t x =0 ; x< V;x++){
        h_phi(0,x)=create_phi_update(1.,x_rand,x);
        h_phi(1,x)=create_phi_update(1.,x_rand,x);
    }*/
    // Initialize phi on the device
    Kokkos::parallel_for( "init_phi", V, KOKKOS_LAMBDA( size_t x) { 
        // get a random generatro from the pool
        gen_type rgen = rand_pool.get_state();
        phi(0,x)=(rgen.drand()*2.-1.);
        phi(1,x)=(rgen.drand()*2.-1.);
        // Give the state back, which will allow another thread to aquire it
        rand_pool.free_state(rgen);
    });   
    // Deep copy device views to host views.
    //  Kokkos::deep_copy( h_phi, phi );
   
/*    std::string mes_file = params.data.outpath + 
                              "/mes_T" + std::to_string(params.data.L[0]) +
                              ".X" + std::to_string(params.data.L[1]) +
                              ".Y" + std::to_string(params.data.L[2]) +
                              ".Z" + std::to_string(params.data.L[3]) +
                              ".msq" + std::to_string(params.data.msq0);*/
    std::string mes_file = params.data.outpath + 
                              "/mes_T" + std::to_string(params.data.L[0]) +
                              "_L" + std::to_string(params.data.L[1]) +
                              "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                              "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                              "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)   +
                              "_rep" + std::to_string(params.data.replica) ;
    std::string G2t_file = params.data.outpath + 
                              "/G2t_T" + std::to_string(params.data.L[0]) +
                              "_L" + std::to_string(params.data.L[1]) +
                              "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                              "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                              "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)   +
                              "_rep" + std::to_string(params.data.replica) ; 
                              
    cout << "Writing magnetization to: " << mes_file << endl;
    cout << "Writing G2t       to: " << G2t_file << endl;
    FILE *f_mes = fopen(mes_file.c_str(), "w+"); 
    FILE *f_G2t = fopen(G2t_file.c_str(), "w+"); 
    if (f_mes == NULL  || f_G2t == NULL  ) {
               printf("Error opening file %s or %s \n", mes_file.c_str(), G2t_file.c_str());
               exit(1);
    }    
           
    double time_update=0,time_mes=0,time_writing=0;
    // The update ----------------------------------------------------------------
    for(int ii = 0; ii < params.data.start_measure+params.data.total_measure; ii++) {
         //  cout << "Starting step   "<< ii <<endl; 
         // Timer 
         Kokkos::Timer timer1;
           
         // cluster update
  /*      double cluster_size = 0.0;
        for(size_t nb = 0; nb < params.data.cluster_hits; nb++)
            cluster_size += cluster_update(  &phi ,  params  );
        cluster_size /= params.data.cluster_hits;
        cluster_size /= (double) V;
        clock_t mid = clock(); // start time for one update step
     */   
        // metropolis update
        double acc = 0.0;
        for(int global_metro_hits = 0; global_metro_hits < params.data.metropolis_global_hits;         global_metro_hits++){
           acc += metropolis_update(phi,params, rand_pool, hop, even_odd);
        }
        acc /= params.data.metropolis_global_hits;
  //      cout << "Metropolis.acc=" << acc/V << endl ;

        clock_t end = clock(); // end time for one update step
        // Calculate time of Metropolis update
        double time = timer1.seconds();
        //printf("time metropolis (%g  s)\n",time);
        time_update+=time;

        //Measure every 
        if(ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.measure_every_X_updates == 0){
            Kokkos::Timer timer_2;
            // Deep copy device views to host views.
            //Kokkos::deep_copy( h_phi, phi );
//printf("time_deep copy %g \n",timer_2.seconds());
    //        cout << "measuring  " <<endl;
    //        double *ms=compute_magnetisations_serial( h_phi,   params);
//printf("time magnetization serial %g \n",timer_2.seconds());
            double *m=compute_magnetisations( phi,   params);
//printf("time magnetization device %g \n",timer_2.seconds());

//            double *G2=compute_G2( h_phi,   params);
//printf("time G2p %g \n",timer_2.seconds());
            compute_G2t( phi,   params,f_G2t);
//printf("time G2t %g \n",timer_2.seconds());
//            compute_G2t_serial_host( h_phi, params,  f_G2t);
            //fprintf(f_mes,"%.15g   %.15g   %.15g   %.15g   %.15g  %.15g\n",m[0], m[1], G2[0], G2[1], G2[2], G2[3]);
            fprintf(f_mes,"%.15g   %.15g \n",m[0], m[1]);
           // cout << "    phi0 norm=" << m[0]  << "    phi1 norm=" << m[1]  << endl;
            free(m);//free(G2);
           
            time = timer_2.seconds();
//            printf("time measurament (%g  s)\n",time);
            time_mes+=time;
//printf("time write free etc %g \n",timer_2.seconds());

        }
        // write the configuration to disk
        if(params.data.save_config == "yes" && ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.save_config_every_X_updates == 0){
            Kokkos::Timer timer3;
            // Deep copy device views to host views.
      //      cout << "saving conf  " <<endl;
            std::string conf_file = params.data.outpath + 
                              "/T" + std::to_string(params.data.L[0]) +
                              "_L" + std::to_string(params.data.L[1]) +
                              "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                              "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                              "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)  + 
                              "_rep" + std::to_string(params.data.replica) + 
                              "_conf" + std::to_string(ii);
            cout << "Writing configuration to: " << conf_file << endl;
            FILE *f_conf = fopen(conf_file.c_str(), "wb"); 
            if (f_conf == NULL) {
               printf("Error opening file %s!\n", conf_file.c_str());
               exit(1);
            }
            write_viewer(f_conf, layout_value, V, phi ); 
            time = timer3.seconds();
            //printf("time writing (%g  s)\n",time);
            time_writing+=time;
        }    
    }

    
    
    /*
   // save rng 
   cout << "writing the rng state to: "<< rng_file.c_str() <<endl;
   int N=rlxd_size();
   int *state = (int*) alloca(N*sizeof(int));
   rlxd_get(state);

   frng=fopen(rng_file.c_str(),"wb");
   error(frng==NULL,1,"main",
         "Unable to open ranlux state file");

   write_rng_state(N,state);
   fclose(frng);
*/
    printf("  time updating = %f s (%f per single operation)\n", time_update, time_update/(params.data.start_measure+params.data.total_measure) );
    printf("  time mesuring = %f s (%f per single operation)\n", time_mes   , time_mes/(params.data.total_measure/ params.data.measure_every_X_updates ));
    printf("  time writing  = %f s (%f per single opertion)\n", time_writing, time_writing/(params.data.total_measure/ params.data.save_config_every_X_updates) );
    printf("total time = %f s\n",time_writing+ time_mes+ time_update );


    fclose(f_G2t);
    fclose(f_mes);
    }
    Kokkos::finalize();
    
    return 0;
}
