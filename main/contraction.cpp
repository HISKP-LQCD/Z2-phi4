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
#include "measurements.hpp"

//#include <highfive/H5File.hpp>

#include <Kokkos_Core.hpp>


static int endian;
std::string rng_file; 
static FILE *frng=NULL ;


 
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
    
    cout << "metropolis_local_hits " << params.data.metropolis_local_hits << endl;
    cout << "metropolis_global_hits " << params.data.metropolis_global_hits << endl;
    cout << "metropolis_delta  " << params.data.metropolis_delta << endl;
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
    cout << "   host  execution    space:"<<  &Kokkos::HostSpace::name << endl; 
    
    int layout_value=check_layout();
    // Create a random number generator pool (64-bit states or 1024-bit state)
    // Both take an 64 bit unsigned integer seed to initialize a Random_XorShift generator 
    // which is used to fill the generators of the pool.
    RandPoolType rand_pool(params.data.seed);
    //Kokkos::Random_XorShift1024_Pool<> rand_pool1024(5374857); 
    cout << "random pool initialised"<< endl;
    // we need a random generator on the host for the cluster
    // seed the PRNG (MT19937) for each  lattice size, with seed , CPU only
    std::mt19937_64 host_rand( params.data.seed );
    
  
    
    ViewLatt    hop("hop",V,2*dim_spacetime);
    ViewLatt    even_odd("even_odd",2,V/2);
    ViewLatt    ipt("ipt",V,dim_spacetime);
    
    hopping( params.data.L, hop,even_odd,ipt);    
    cout << "hopping initialised"<< endl; 
        
    Viewphi  phi("phi",2,V);

   
    // Initialize phi on the device
    Kokkos::parallel_for( "init_phi", V, KOKKOS_LAMBDA( size_t x) { 
        // get a random generatro from the pool
        gen_type rgen = rand_pool.get_state();
        phi(0,x)=(rgen.drand()*2.-1.);
        phi(1,x)=(rgen.drand()*2.-1.);
        // Give the state back, which will allow another thread to aquire it
        rand_pool.free_state(rgen);
    });   
    
   
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
    double ave_acc=0;
    // The update ----------------------------------------------------------------
    for(int ii = 0; ii < params.data.start_measure+params.data.total_measure; ii++) {
        double time; 
        //reading
        if(params.data.save_config == "yes" && ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.save_config_every_X_updates == 0){
            Kokkos::Timer timer3;
            
            std::string conf_file = params.data.outpath + 
                              "/T" + std::to_string(params.data.L[0]) +
                              "_L" + std::to_string(params.data.L[1]) +
                              "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                              "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                              "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)  + 
                              "_rep" + std::to_string(params.data.replica) + 
                              "_conf" + std::to_string(ii);
            cout << "reading configuration : " << conf_file << endl;
            FILE *f_conf = fopen(conf_file.c_str(), "r"); 
            if (f_conf == NULL) {
               printf("Error opening file %s!\n", conf_file.c_str());
               exit(1);
            }
            read_viewer(f_conf, layout_value, params , ii , phi ); 
            fclose(f_conf);
            time = timer3.seconds();
            //printf("time writing (%g  s)\n",time);
            time_writing+=time;
        }    
       
        //Measure every 
        if(ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.measure_every_X_updates == 0){
            Kokkos::Timer timer_2;
            double *m=compute_magnetisations( phi,   params);
            compute_G2t( phi,   params,f_G2t);
            fprintf(f_mes,"%.15g   %.15g \n",m[0], m[1]);

            free(m);//free(G2);
           
            time = timer_2.seconds();
            time_mes+=time;


        }
        // write the configuration to disk
      
    }

    

    printf("average acceptance rate= %g\n", ave_acc/(params.data.start_measure+params.data.total_measure));
    
    //printf("  time updating = %f s (%f per single operation)\n", time_update, time_update/(params.data.start_measure+params.data.total_measure) );
    printf("  time mesuring = %f s (%f per single operation)\n", time_mes   , time_mes/(params.data.total_measure/ params.data.measure_every_X_updates ));
    printf("  time reading  = %f s (%f per single opertion)\n", time_writing, time_writing/(params.data.total_measure/ params.data.save_config_every_X_updates) );
    printf("total time = %f s\n",time_writing+ time_mes );


    fclose(f_G2t);
    fclose(f_mes);
    }
    Kokkos::finalize();
    
    return 0;
}
