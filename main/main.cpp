#define CONTROL

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>


#include "lattice.hpp"
#include "mutils.hpp"
#include "IO_params.hpp"
#include "geometry.hpp"
#include "metropolis.hpp"

#include "random.hpp"
#include "utils.hpp" 

static int endian;
std::string rng_file; 
static FILE *frng=NULL ;


 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double *compute_magnetisations(double **phi,  cluster::IO_params params){

  double *m=(double*) calloc(2,sizeof(double));
  for(int x =0; x< params.data.V; x++){
    m[0] += sqrt(phi[0][x]*phi[0][x]);
    m[1] += sqrt(phi[1][x]*phi[1][x]);
  }
  m[0]*=sqrt(2*params.data.kappa0)/((double)params.data.V);
  m[1]*=sqrt(2*params.data.kappa1)/((double)params.data.V);
  return m;
}
 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double  *compute_G2(double **phi, cluster::IO_params params ){
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
            G2[0]+=phi[0][x];
            G2[1]+=phi[0][x]*(cos(2.*3.1415926535*x0 /(double (L0)) ));
            tmp1+=phi[0][x]*(sin(2.*3.1415926535*x0 /(double (L0)) ));
            G2[2]+=phi[0][x]*(cos(4.*3.1415926535*x0 /(double (L0)) ));
            tmp2=phi[0][x]*(cos(2.*3.1415926535*x0 /(double (L0)) ));
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
double create_phi_update(const double delta){
  
  double r[1];
  ranlxd( r,1);
  return (r[0]*2-1)*delta;

} 


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static void write_rng_state(int N, int state[])
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
    int V=params.data.V;
    hopping( params.data.L);
    
    rng_file = params.data.outpath + "/rng";
    init_rng( params);
    
    double **phi=(double**) malloc(sizeof(double*)*2);
    for (int i =0 ; i< 2;i++)
        phi[i]=(double*) malloc(sizeof(double)*V);
    
    for (int x =0 ; x< V;x++){
        phi[0][x]=create_phi_update(1.);
        phi[1][x]=create_phi_update(1.); 
    }
    
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
        clock_t begin = clock(); // start time for one update step
        // metropolis update
        double acc = 0.0;
        for(int global_metro_hits = 0; global_metro_hits < params.data.metropolis_global_hits;         global_metro_hits++)
           acc += metropolis_update(&phi,params );
  
        acc /= params.data.metropolis_global_hits;

        clock_t mid = clock(); // start time for one update step
        
        clock_t end = clock(); // end time for one update step
        //Measure every 
        if(ii > params.data.start_measure && ii%params.data.measure_every_X_updates == 0){
            double *m=compute_magnetisations( phi,   params);
            double *G2=compute_G2( phi,   params);
            fprintf(f_mes,"%.15g   %.15g   %.15g   %.15g   %.15g  %.15g\n",m[0], m[1], G2[0], G2[1], G2[2], G2[3]);
            free(m);free(G2);
        }
        // write the configuration to disk
        if(params.data.save_config == "yes" && ii > params.data.start_measure && ii%params.data.save_config_every_X_updates == 0){
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
            fwrite(phi[0], sizeof(double), V, f_conf);
            fwrite(phi[1], sizeof(double), V, f_conf);
            fclose(f_conf);
        }    
    }
    
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

    return 0;
}
