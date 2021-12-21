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

   
    // std::cout<< "confs="<<c<<std::endl;
   fseek(stream, params.data.header_size, SEEK_SET);
   
   //    printf("header size=%d\n",params.data.header_size);
   //    printf("size=%ld\n",params.data.size);

   
   return c;

}

void read_header(FILE *stream, cluster::IO_params &params ){
     int i=0;
     i+=fread(&params.data.L[0], sizeof(int), 1, stream); 
     i+=fread(&params.data.L[1], sizeof(int), 1, stream); 
     i+=fread(&params.data.L[2], sizeof(int), 1, stream); 
     i+=fread(&params.data.L[3], sizeof(int), 1, stream); 
     printf("%d %d %d %d\n",params.data.L[0],params.data.L[1],params.data.L[2],params.data.L[3]);
     char string[100];
     i+=fread(string, sizeof(char)*100, 1, stream); 
     params.data.formulation.assign(string,100);
     
     i+=fread(&params.data.msq0, sizeof(double), 1, stream); 
     i+=fread(&params.data.msq1, sizeof(double), 1, stream); 
     i+=fread(&params.data.lambdaC0, sizeof(double), 1, stream); 
     i+=fread(&params.data.lambdaC1, sizeof(double), 1, stream); 
     i+=fread(&params.data.muC, sizeof(double), 1, stream); 
     i+=fread(&params.data.gC, sizeof(double), 1, stream); 

     i+=fread(&params.data.metropolis_local_hits, sizeof(int), 1, stream); 
     i+=fread(&params.data.metropolis_global_hits, sizeof(int), 1, stream); 
     i+=fread(&params.data.metropolis_delta, sizeof(double), 1, stream); 
     
     i+=fread(&params.data.cluster_hits, sizeof(int), 1, stream); 
     i+=fread(&params.data.cluster_min_size, sizeof(double), 1, stream); 

     i+=fread(&params.data.seed, sizeof(int), 1, stream); 
     for (int &s : seeds){
         //printf("seed=%d\n",s);
         if(s==params.data.seed) {
             printf("error two replicas have the same seed\n"); exit(1);
         }
     }
     seeds.emplace_back(params.data.seed);
     
     i+=fread(&params.data.replica, sizeof(int), 1, stream); 
     for (int &s : replicas){
         //printf("replica=%d\n",s);
         if(s==params.data.replica) {
             printf("error you select two identical replicas\n"); exit(1);
         }
     }
     replicas.emplace_back(params.data.replica);
     
    
     i+=fread(&params.data.ncorr, sizeof(int), 1, stream); 
     printf("correlators=%d\n",params.data.ncorr);
    
     i+=fread(&params.data.size, sizeof(size_t), 1, stream); 
     printf("size=%ld\n",params.data.size);
     
     params.data.header_size=ftell(stream);
     printf("header size=%d\n",params.data.header_size);
     size_t size_check=params.data.ncorr*params.data.L[0];
     if (params.data.size != size_check){
         printf("params.data.size = %ld  !=  params.data.ncorr*params.data.L[0]= %d  *  %d",params.data.size,params.data.ncorr,params.data.L[0] );
         exit(2);
     }
     if (i != 20 ){
         printf("read_header: invilid read header  i=%d instead of %d", i,20);
         exit(2);
     }
}




void write_header_measuraments(FILE *f_conf, cluster::IO_params params ){
     int i=0;
     i+=fwrite(&params.data.L, sizeof(int), 4, f_conf); 

     i+=fwrite(params.data.formulation.c_str(), sizeof(char)*100, 1, f_conf); 

     i+=fwrite(&params.data.msq0, sizeof(double), 1, f_conf); 
     i+=fwrite(&params.data.msq1, sizeof(double), 1, f_conf); 
     i+=fwrite(&params.data.lambdaC0, sizeof(double), 1, f_conf); 
     i+=fwrite(&params.data.lambdaC1, sizeof(double), 1, f_conf); 
     i+=fwrite(&params.data.muC, sizeof(double), 1, f_conf); 
     i+=fwrite(&params.data.gC, sizeof(double), 1, f_conf); 

     i+=fwrite(&params.data.metropolis_local_hits, sizeof(int), 1, f_conf); 
     i+=fwrite(&params.data.metropolis_global_hits, sizeof(int), 1, f_conf); 
     i+=fwrite(&params.data.metropolis_delta, sizeof(double), 1, f_conf); 
     
     i+=fwrite(&params.data.cluster_hits, sizeof(int), 1, f_conf); 
     i+=fwrite(&params.data.cluster_min_size, sizeof(double), 1, f_conf); 

     i+=fwrite(&params.data.seed, sizeof(int), 1, f_conf);
     i+=fwrite(&params.data.replica, sizeof(int), 1, f_conf); 
     
     i+=fwrite(&params.data.ncorr, sizeof(int), 1, f_conf); 
     
      
     i+=fwrite(&params.data.size, sizeof(size_t), 1, f_conf); 
     
         
}



template <typename T>
void error_header(FILE *f_conf, T expected, const char *message){
     T read;
     int i=fread(&read, sizeof(T), 1, f_conf);
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
     int i=fread(&string, sizeof(char)*100, 1, f_conf); 
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
     i+=fread(&tmp, sizeof(int), 1, f_conf);
     for (int &s : seeds){
         if(s==tmp) {
             printf("error two replicas have the same seed\n"); exit(1);
        }
     }
    seeds.emplace_back(tmp);
     
    i+=fread(&tmp, sizeof(int), 1, f_conf);
     for (int &s : replicas){
         if(s==tmp) {
             printf("error you select two identical replicas\n"); exit(1);
        }
         
    }
    replicas.emplace_back(tmp);
     
     //error_header(f_conf,params.data.replica,"replica" ); 
     
     //error_header(f_conf,iconf,"iconf" ); 
     //error_header(f_conf,params.data.ncorr,"ncorr" ); 
     int ncorr;
     i+=fread(&ncorr, sizeof(int), 1, f_conf);
     if (ncorr > params.data.ncorr){
         cout <<"more correlators found, ignoring the extra:   read=" << ncorr << "  expected="<< params.data.ncorr << endl; 
     }
     else if (ncorr < params.data.ncorr){
         cout <<"less  correlators found aborting   read=" << ncorr << "  expected="<< params.data.ncorr << endl; 
         exit(2);
     }
     error_header(f_conf,params.data.size,"size" );
     
    
}

template <typename T>
void compare_2(T compare, T expected, const char *message){
    
    if (compare != expected){
        cout <<"error:" << message << "   read=" << compare << "  expected="<< expected << endl; 
        //         printf("error: %s read=%d   expected %d \n",message,rconf,iconf);
        exit(2);
    }
}

void compare_headers( cluster::IO_params &params,  cluster::IO_params &reference ){
    
    compare_2(params.data.L[0],reference.data.L[0],"L0" ); 
    compare_2(params.data.L[1],reference.data.L[1],"L1" ); 
    compare_2(params.data.L[2],reference.data.L[2],"L2" ); 
    compare_2(params.data.L[3],reference.data.L[3],"L3" ); 
    
    if (strcmp(params.data.formulation.c_str() ,reference.data.formulation.c_str() ) ){
        printf("error: formulation read=%s   expected %s \n",params.data.formulation.c_str(),reference.data.formulation.c_str());
        exit(2);
    }
    
    compare_2(params.data.msq0,reference.data.msq0,"msq0" ); 
    compare_2(params.data.msq1,reference.data.msq1,"msq1" ); 
    compare_2(params.data.lambdaC0,reference.data.lambdaC0,"lambdaC0" ); 
    compare_2(params.data.lambdaC1,reference.data.lambdaC1,"lambdaC1" ); 
    compare_2(params.data.muC,reference.data.muC,"muC" ); 
    compare_2(params.data.gC,reference.data.gC,"gC" ); 
    
    compare_2(params.data.metropolis_local_hits,reference.data.metropolis_local_hits,"metropolis_local_hits" ); 
    compare_2(params.data.metropolis_global_hits,reference.data.metropolis_global_hits,"metropolis_global_hits" ); 
    compare_2(params.data.metropolis_delta,reference.data.metropolis_delta,"metropolis_delta" ); 
    compare_2(params.data.cluster_hits,reference.data.cluster_hits,"cluster_hits" ); 
    compare_2(params.data.cluster_min_size,reference.data.cluster_min_size,"cluster_min_size" ); 
    
    
    //error_header(f_conf,iconf,"iconf" ); 
    //error_header(f_conf,params.data.ncorr,"ncorr" ); 
    int ncorr=params.data.ncorr;
    if (ncorr > reference.data.ncorr){
        cout <<"more correlators found, ignoring the extra:   read=" << ncorr << "  expected="<< reference.data.ncorr << endl; 
    }
    else if (ncorr < reference.data.ncorr){
        cout <<"less  correlators found aborting   read=" << ncorr << "  expected="<< reference.data.ncorr << endl; 
        exit(2);
    }
    //compare_2(params.data.size,reference.data.size,"size" ); 
    
    
    
    
}

int main(int argc, char **argv){
   
    if (argc<3) {
        printf("\nusage:   ./meargin_replicas  file1  file2 ...\n\n");
        exit(0);
    }
      
   char namefile[10000];
   vector<cluster::IO_params> params(argc-1);
   
   printf("argc=%d\n",argc);

   FILE *infiles=NULL;
//    infiles=(FILE*) malloc(sizeof(FILE)*(argc-1));
   printf("considering file:  %s\n",argv[1] );

   sprintf(namefile,"%s_merged",argv[1]);
   printf("writing output %s\n",namefile);
   FILE *outfile = fopen(namefile, "w+"); 
   if (outfile==NULL) {printf("can not open output file:  \n %s\n",namefile); exit(1);}  
   
   sprintf(namefile,"%s",argv[1]);
   infiles=fopen(namefile,"r+");
   if (infiles==NULL) {printf("can not open contraction file: \n %s\n",namefile); exit(1);}
   read_header(infiles,params[0]); 
   confs.emplace_back( read_nconfs( infiles,  params[0])  ); 
//    fclose(infiles);
   
//    for (int r=1 ;r < (argc-1);r++){
      
//       //check_header(infiles[r],params  )  ;
//       //confs.emplace_back( read_nconfs( infiles[r],  params)  );    
//    }
   
   write_header_measuraments(outfile,params[0]);
   
   int iii;
   int fi=0;
   double *data=(double*) malloc(sizeof(double)*(params[0].data.size));

   for (int r=0 ;r < (argc-1);r++){
       if (r>0){
           infiles=NULL;
           infiles=fopen(argv[1+r],"r+");
           printf("considering file:  %s\n",argv[1+r] );
           if (infiles==NULL) {printf("can not open contraction file: \n\n"); exit(1);}
           read_header(infiles,params[r]); 
           compare_headers(params[r],params[0]);
           confs.emplace_back( read_nconfs( infiles,  params[r])  ); 
           }
       
       for(int iconf=0; iconf < confs[r];iconf++){
           fi+=fread(&iii,sizeof(int),1,infiles);
           fi+=fread(data,sizeof(double),params[r].data.size, infiles);
           fwrite(&iii,sizeof(int),1,outfile);
           if ( params[r].data.size == params[0].data.size ){
               fwrite(data,sizeof(double),params[0].data.size,outfile);
           }
           else{
               for(int t=0;t<params[0].data.L[0];t++){
                   for(int c=0;c<params[0].data.ncorr;c++){
                       fwrite(&data[c+t*params[r].data.ncorr],sizeof(double),1,outfile);
                   }
               }
               
           }
            
       }
       fclose(infiles);
   }
   free(data);

   fclose(outfile);
  
   sprintf(namefile,"%s_merged",argv[1]);
   printf("output written in: %s\n",namefile);
   
   
       
}
 
    
