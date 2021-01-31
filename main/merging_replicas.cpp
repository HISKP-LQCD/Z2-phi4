#define CONTROL

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <complex.h>

#include <cstring> 
#include <string>
#include <fstream>
#include <memory>
//#include "IO_params.hpp"
using namespace std;
std::vector<int> replicas;
std::vector<int> seeds;
std::vector<int> confs;
namespace cluster {

struct LatticeDataContainer { // Just the thing that holds all variables
  // lattice parameter
  int L[4];
  int V;
  // action parameter
  std::string formulation;
  double kappa0;
  double kappa1;
  double lambda0;
  double lambda1;
  double mu;
  double g;
  double msq0;
  double msq1;
  double lambdaC0;
  double lambdaC1;
  double muC;
  double gC;
  
  // metropolis parameter
  int metropolis_local_hits;
  int metropolis_global_hits;
  double metropolis_delta;
  // cluster parameter
  int cluster_hits;
  double cluster_min_size;
  // run parameter
  int seed;
  int level;  
  int append;
  int replica;
  int start_measure;
  int total_measure;
  int measure_every_X_updates;
  std::string save_config;
  std::string save_config_rotated;
  int save_config_every_X_updates;
  std::string outpath;
  
  size_t size;
  int ncorr;
  int header_size;
  
};
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class IO_params {

private:
    
  double comp_kappa(double msq, double lambdaC){
      double k;

      if (lambdaC<1e-12)
	     k=1./(8.+msq);
      else{
         k=-8.-msq+sqrt( (8+msq)*(8+msq)+32*lambdaC );
         k/=(16.*lambdaC);
      }
      return k;
      
  }  

  inline LatticeDataContainer read_infile(int argc, char** argv) {

    int opt = -1;
    int reader = 0;
    char infilename[200];
    char readin[256];
    FILE* infile = NULL;

    LatticeDataContainer data;

    // search for command line option and put filename in "infilename"
    for(int i = 0; i < argc; ++i) {
      if(std::strcmp(argv[i], "-i") == 0) {
        opt = i+1;
        break;
      }
    }
    if(opt < 0) {
      std::cout << "No input file specified, Aborting" << std::endl;
      exit(1);
    } else {
      sprintf(infilename, "%s", argv[opt]);
      std::cout << "Trying input file " << infilename << std::endl;
    }
    // open file for reading
    if ((infile = fopen(infilename, "r")) == NULL ) {
      std::cerr << "Could not open file " << infilename << std::endl;
      std::cerr << "Aborting..." << std::endl;
      exit(-10);
    }
    // lattice size
   
    reader += fscanf(infile, "L = %d %d %d %d \n", &data.L[0], &data.L[1], 
                                                   &data.L[2], &data.L[3]);
    data.V = data.L[0]*data.L[1]*data.L[2]*data.L[3];

    // kappa and lambda
    reader += fscanf(infile, "formulation = %255s\n", readin);
    data.formulation.assign(readin);
    reader += fscanf(infile, "msq0 = %lf\n", &data.msq0);
    reader += fscanf(infile, "msq1 = %lf\n", &data.msq1);
    reader += fscanf(infile, "lambdaC0 = %lf\n", &data.lambdaC0);
    reader += fscanf(infile, "lambdaC1 = %lf\n", &data.lambdaC1);
    reader += fscanf(infile, "muC = %lf\n", &data.muC);
    reader += fscanf(infile, "gC = %lf\n", &data.gC);

    if(data.formulation == "O2"){
      std::cout << "using O2 version of the two scalars lambdaC0=lambdaC1=muC/2: \n" << std::endl;
      if ( data.lambdaC1 !=0 || data.muC !=0 ){
          std::cout    << "if formulation = O2  you need to set lambdaC1= muC2=0 "<< std::endl; 
          exit(0);
      }
      data.lambdaC1= data.lambdaC0;
      data.muC= 2.*data.lambdaC0;

    }
    data.kappa0=comp_kappa(data.msq0, data.lambdaC0);
    data.kappa1=comp_kappa(data.msq1, data.lambdaC1);
    data.lambda0=data.lambdaC0 * 4. * data.kappa0 *data.kappa0;
    data.lambda1=data.lambdaC1 * 4. * data.kappa1 *data.kappa1;
    data.mu=data.muC*(4. *data.kappa0*data.kappa1 );
    data.g=data.gC*(4. * sqrt(data.kappa0 *data.kappa1 * data.kappa1 * data.kappa1) );
    printf("parameters:\n");
    printf( "msq0 = %.6f     -> kappa0   = %.6f\n", data.msq0, data.kappa0);
    printf( "msq1 = %.6f     -> kappa1   = %.6f\n", data.msq1, data.kappa1);
    printf( "lambdaC0 = %.6f -> lambda0  = %.6f\n", data.lambdaC0, data.lambda0);
    printf( "lambdaC1 = %.6f -> lambda1  = %.6f\n", data.lambdaC1, data.lambda1);
    printf( "muC = %.6f      -> mu       = %.6f\n", data.muC, data.mu);
    printf( "gC = %.6f       -> g        = %.6f\n",  data.gC, data.g);

    
    // metropolis 
    reader += fscanf(infile, "metropolis_local_hits = %d\n", 
                             &data.metropolis_local_hits);
    reader += fscanf(infile, "metropolis_global_hits = %d\n", 
                             &data.metropolis_global_hits);
    reader += fscanf(infile, "metropolis_delta = %lf\n", &data.metropolis_delta);
    // cluster
    reader += fscanf(infile, "cluster_hits = %d\n", &data.cluster_hits);
    reader += fscanf(infile, "cluster_min_size = %lf\n", &data.cluster_min_size);
    // configs
    reader += fscanf(infile, "seed = %d\n", &data.seed);
    reader += fscanf(infile, "level = %d\n", &data.level);
    reader += fscanf(infile, "append = %d\n", &data.append);
    
    reader += fscanf(infile, "replica = %d\n", &data.replica);
    reader += fscanf(infile, "start_measure = %d\n", &data.start_measure);
    if(data.append < 0 || data.replica < 0){
      std::cout << "append and replica value must not be negative!" << std::endl;
      exit(0);
    }
    if (data.append ==1 && data.start_measure!=0){
      std::cout << "if append mode you can not wait for termalization" << std::endl;
      exit(0);  
    }
        
    //data.start_measure += data.restart;

    reader += fscanf(infile, "total_measure = %d\n", &data.total_measure);
    reader += fscanf(infile, "measure_every_X_updates = %d\n", 
                             &data.measure_every_X_updates);
    reader += fscanf(infile, "save_config = %255s\n", readin);
    data.save_config.assign(readin);
    reader += fscanf(infile, "save_config_rotated = %255s\n", readin);
    data.save_config_rotated.assign(readin);
    reader += fscanf(infile, "save_config_every_X_updates = %d\n", 
                             &data.save_config_every_X_updates);
    reader += fscanf(infile, "outpath = %255s\n", readin);
    data.outpath.assign(readin);

    // close input file
    fclose(infile);

    return data;
  };

public:
  LatticeDataContainer data; 
  
  //IO_params(int argc, char** argv) : data(read_infile(argc, argv)) {};

}; // end of class definition

} // end of n



int read_nconfs( FILE *stream, cluster::IO_params params){

   long int tmp;
   int s=params.data.header_size;
   

  
   fseek(stream, 0, SEEK_END);
   tmp = ftell(stream);
   tmp-= params.data.header_size ;
   
   s=params.data.size;
   std::cout<< "size="<<s<<std::endl;

   int c= (tmp)/ ( sizeof(int)+(s)*sizeof(double) );

   
   std::cout<< "confs="<<c<<std::endl;
   fseek(stream, params.data.header_size, SEEK_SET);
   
        printf("header size=%d\n",params.data.header_size);
     printf("size=%ld\n",params.data.size);

   
   return c;

}

void read_header(FILE *stream, cluster::IO_params &params ){
   
     fread(&params.data.L[0], sizeof(int), 1, stream); 
     fread(&params.data.L[1], sizeof(int), 1, stream); 
     fread(&params.data.L[2], sizeof(int), 1, stream); 
     fread(&params.data.L[3], sizeof(int), 1, stream); 
     printf("%d %d %d %d\n",params.data.L[0],params.data.L[1],params.data.L[2],params.data.L[3]);
     char string[100];
     fread(string, sizeof(char)*100, 1, stream); 
     params.data.formulation.assign(string,100);
     
     fread(&params.data.msq0, sizeof(double), 1, stream); 
     fread(&params.data.msq1, sizeof(double), 1, stream); 
     fread(&params.data.lambdaC0, sizeof(double), 1, stream); 
     fread(&params.data.lambdaC1, sizeof(double), 1, stream); 
     fread(&params.data.muC, sizeof(double), 1, stream); 
     fread(&params.data.gC, sizeof(double), 1, stream); 

     fread(&params.data.metropolis_local_hits, sizeof(int), 1, stream); 
     fread(&params.data.metropolis_global_hits, sizeof(int), 1, stream); 
     fread(&params.data.metropolis_delta, sizeof(double), 1, stream); 
     
     fread(&params.data.cluster_hits, sizeof(int), 1, stream); 
     fread(&params.data.cluster_min_size, sizeof(double), 1, stream); 

     fread(&params.data.seed, sizeof(int), 1, stream); 
     seeds.emplace_back(params.data.seed);
     fread(&params.data.replica, sizeof(int), 1, stream); 
     replicas.emplace_back(params.data.replica);
     
    
     fread(&params.data.ncorr, sizeof(int), 1, stream); 
     printf("correlators=%d\n",params.data.ncorr);
    
     fread(&params.data.size, sizeof(size_t), 1, stream); 
     printf("size=%ld\n",params.data.size);
     
     params.data.header_size=ftell(stream);
     printf("header size=%d\n",params.data.header_size);
}




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
     
     fwrite(&params.data.ncorr, sizeof(int), 1, f_conf); 
     
     fwrite(&params.data.size, sizeof(size_t), 1, f_conf); 
}



template <typename T>
void error_header(FILE *f_conf, T expected, const char *message){
     T read;
     fread(&read, sizeof(T), 1, f_conf);
     if (read != expected){
         cout <<"error:" << message << "   read=" << read << "  expected="<< expected << endl; 
//         printf("error: %s read=%d   expected %d \n",message,rconf,iconf);
         exit(2);
     }
}
void check_header(FILE *f_conf, cluster::IO_params &params ){

     error_header(f_conf,params.data.L[0],"L0" ); 
     error_header(f_conf,params.data.L[1],"L1" ); 
     error_header(f_conf,params.data.L[2],"L2" ); 
     error_header(f_conf,params.data.L[3],"L3" ); 

     char string[100];
     fread(&string, sizeof(char)*100, 1, f_conf); 
     if (strcmp(params.data.formulation.c_str() ,string ) ){
         printf("error: formulation read=%s   expected %s \n",string,params.data.formulation.c_str());
         exit(2);
     }
     

     error_header(f_conf,params.data.msq0,"msq0" ); 
     error_header(f_conf,params.data.msq1,"msq1" ); 
     error_header(f_conf,params.data.lambdaC0,"lambdaC0" ); 
     error_header(f_conf,params.data.lambdaC1,"lambdaC1" ); 
     error_header(f_conf,params.data.muC,"muC" ); 
     error_header(f_conf,params.data.gC,"gC" ); 
     
     error_header(f_conf,params.data.metropolis_local_hits,"metropolis_local_hits" ); 
     error_header(f_conf,params.data.metropolis_global_hits,"metropolis_global_hits" ); 
     error_header(f_conf,params.data.metropolis_delta,"metropolis_delta" ); 
     error_header(f_conf,params.data.cluster_hits,"cluster_hits" ); 
     error_header(f_conf,params.data.cluster_min_size,"cluster_min_size" ); 
     int tmp;
     //error_header(f_conf,params.data.seed,"seed" ); 
     fread(&tmp, sizeof(int), 1, f_conf);
     for (int &s : seeds){
         if(s==tmp) {
             printf("error two replicas have the same seed\n"); exit(1);
        }
     }
     seeds.emplace_back(tmp);
     
     fread(&tmp, sizeof(int), 1, f_conf);
     for (int &s : replicas){
         if(s==tmp) {
             printf("error you select two identical replicas\n"); exit(1);
        }
         
    }
    replicas.emplace_back(tmp);
     
     //error_header(f_conf,params.data.replica,"replica" ); 
     
     //error_header(f_conf,iconf,"iconf" ); 
     error_header(f_conf,params.data.ncorr,"ncorr" ); 
     error_header(f_conf,params.data.size,"size" );
     
    
}


int main(int argc, char **argv){
   
    if (argc<3) {
        printf("\nusage:   ./meargin_replicas  file1  file2 ...\n\n");
        exit(0);
    }
      
   char namefile[10000];
   cluster::IO_params params;
  
   
   sprintf(namefile,"%s",argv[1]);
   FILE **infiles=NULL;
   infiles=(FILE**) malloc(sizeof(FILE*)*(argc-1));
   infiles[0]=NULL;
   infiles[0]=fopen(namefile,"r+");
   if (infiles[0]==NULL) {printf("can not open contraction file \n"); exit(1);}
   read_header(infiles[0],params); 
   confs.emplace_back( read_nconfs( infiles[0],  params)  ); 
    printf("argc=%d\n",argc);

   
   for (int r=1 ;r < (argc-1);r++){
      infiles[r]=NULL;
      infiles[r]=fopen(argv[1+r],"r+"); 
      if (infiles[r]==NULL) {printf("can not open contraction file \n"); exit(1);}
      check_header(infiles[r],params  )  ;
      confs.emplace_back( read_nconfs( infiles[r],  params)  );    
   }
   
   sprintf(namefile,"%s_merged",argv[1]);
   FILE *outfile = fopen(namefile, "w+"); 
   if (outfile==NULL) {printf("can not open output file \n"); exit(1);}  
   
   write_header_measuraments(outfile,params);
   double *data=(double*) malloc(sizeof(double)*(params.data.size));
   int iii;
   for (int r=0 ;r < (argc-1);r++){
       for(int iconf=0; iconf < confs[r];iconf++){
           fread(&iii,sizeof(int),1,infiles[r]);
           fread(data,sizeof(double),params.data.size, infiles[r]);
           fwrite(&iii,sizeof(int),1,outfile);
           fwrite(data,sizeof(double),params.data.size,outfile);
            
       }
        fclose(infiles[r]);
   }
   fclose(outfile);
  
   free(data);
   
       
}
 
    
