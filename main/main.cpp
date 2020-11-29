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

#include <Kokkos_Core.hpp>

static int endian;
std::string rng_file; 
static FILE *frng=NULL ;


 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double *compute_magnetisations( Viewphi::HostMirror phi,  cluster::IO_params params){

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
double  *compute_G2( Viewphi::HostMirror phi, cluster::IO_params params ){
    double  *G2=(double*) calloc(4,sizeof(double));
    double vev;
    double kappa0=params.data.kappa0;
    int V=params.data.V;
    int L0=params.data.L[0];
    int Vs=V/L0;
    double tmp1=0, tmp2=0;
    
    for (int x0=0; x0< L0;x0++){
        for (int xs=0; xs< Vs; xs++){
            int x=xs+ x0*Vs;    
            G2[0]+=phi(0,x);
            G2[1]+=phi(0,x)*(cos(2.*3.1415926535*x0 /(double (L0)) ));
            tmp1+=phi(0,x)*(sin(2.*3.1415926535*x0 /(double (L0)) ));
            G2[2]+=phi(0,x)*(cos(4.*3.1415926535*x0 /(double (L0)) ));
            tmp2=phi(0,x)*(cos(2.*3.1415926535*x0 /(double (L0)) ));
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
    // Create a random number generator pool (64-bit states or 1024-bit state)
    // Both take an 64 bit unsigned integer seed to initialize a Random_XorShift generator 
    // which is used to fill the generators of the pool.
    RandPoolType rand_pool(params.data.seed);
    //Kokkos::Random_XorShift1024_Pool<> rand_pool1024(5374857); 
    
    ViewLatt    hop("hop",V,2*dim_spacetime);
    ViewLatt    even_odd("even_odd",2,V/2);
    ViewLatt    ipt("ipt",V,dim_spacetime);
    
    hopping( params.data.L, hop,even_odd,ipt);    
        
    Viewphi  phi("phi",2,V);
    Viewphi::HostMirror h_phi = Kokkos::create_mirror_view( phi );

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
    // Deep copy divice views to host views.
    Kokkos::deep_copy( h_phi, phi );
    
    std::string mes_file = params.data.outpath + 
                              "/mes_T" + std::to_string(params.data.L[0]) +
                              ".X" + std::to_string(params.data.L[1]) +
                              ".Y" + std::to_string(params.data.L[2]) +
                              ".Z" + std::to_string(params.data.L[3]) +
                              ".msq" + std::to_string(params.data.msq0);
    cout << "Writing magnetization to: " << mes_file << endl;
    FILE *f_mes = fopen(mes_file.c_str(), "wb"); 
    if (f_mes == NULL) {
               printf("Error opening file %s!\n", mes_file.c_str());
               exit(1);
    }    
           
    
    // The update ----------------------------------------------------------------
    for(int ii = 0; ii < params.data.start_measure+params.data.total_measure; ii++) {
        cout << "Starting step   "<< ii <<endl; 
        clock_t begin = clock(); // start time for one update step
        
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
        cout << "Metropolis.acc=" << acc/V << endl ;

        clock_t end = clock(); // end time for one update step
        
        
        
        //Measure every 
        if(ii > params.data.start_measure && ii%params.data.measure_every_X_updates == 0){
            cout << "measuring  " <<endl;
            double *m=compute_magnetisations( h_phi,   params);
            double *G2=compute_G2( h_phi,   params);
            fprintf(f_mes,"%.15g   %.15g   %.15g   %.15g   %.15g  %.15g\n",m[0], m[1], G2[0], G2[1], G2[2], G2[3]);
           // cout << "    phi0 norm=" << m[0]  << "    phi1 norm=" << m[1]  << endl;
            free(m);free(G2);
        }
        // write the configuration to disk
        if(params.data.save_config == "yes" && ii > params.data.start_measure && ii%params.data.save_config_every_X_updates == 0){
            cout << "saving conf  " <<endl;
            std::string conf_file = params.data.outpath + 
                              "/T" + std::to_string(params.data.L[0]) +
                              ".X" + std::to_string(params.data.L[1]) +
                              ".Y" + std::to_string(params.data.L[2]) +
                              ".Z" + std::to_string(params.data.L[3]) +
                              ".kap" + std::to_string(params.data.kappa0) + 
                              ".lam" + std::to_string(params.data.lambda0)+
                              ".rep_" + std::to_string(params.data.replica) + 
                              ".rot_" + params.data.save_config_rotated + 
                              ".conf" + std::to_string(ii);
            cout << "Writing configuration to: " << conf_file << endl;
            FILE *f_conf = fopen(conf_file.c_str(), "wb"); 
            if (f_conf == NULL) {
               printf("Error opening file %s!\n", conf_file.c_str());
               exit(1);
            }
            //for (int x=0; x< V; x++)
            double *p=(double*) &h_phi(0,0);
            fwrite(p, sizeof(double), V, f_conf);
            p=(double*) &h_phi(1,0);
            fwrite(p, sizeof(double), V, f_conf);
            fclose(f_conf);
        }    
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
    Kokkos::finalize();
    return 0;
}
