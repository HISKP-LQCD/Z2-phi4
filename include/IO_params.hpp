#ifndef IO_params_H
#define IO_params_H

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>
#include "lattice.hpp"

namespace cluster {

struct LatticeDataContainer { // Just the thing that holds all variables
  // lattice parameter
  int L[dim_spacetime];
  size_t V;
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
  std::string save_config_FT;
  int save_config_every_X_updates;
  std::string outpath;
  
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
      std::cout << "No input file specified, Aborting" << endl;
      exit(1);
    } else {
      sprintf(infilename, "%s", argv[opt]);
      std::cout << "Trying input file " << infilename << endl;
    }
    // open file for reading
    if ((infile = fopen(infilename, "r")) == NULL ) {
      std::cerr << "Could not open file " << infilename << endl;
      std::cerr << "Aborting..." << endl;
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
      cout << "using O2 version of the two scalars lambdaC0=lambdaC1=muC/2: \n" << endl;
      if ( data.lambdaC1 !=0 || data.muC !=0 ){
          cout    << "if formulation = O2  you need to set lambdaC1= muC2=0 "<< endl; 
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
      cout << "append and replica value must not be negative!" << endl;
      exit(0);
    }
    if (data.append ==1 && data.start_measure!=0){
      cout << "if append mode you can not wait for termalization" << endl;
      exit(0);  
    }
        
    //data.start_measure += data.restart;

    reader += fscanf(infile, "total_measure = %d\n", &data.total_measure);
    reader += fscanf(infile, "measure_every_X_updates = %d\n", 
                             &data.measure_every_X_updates);
    reader += fscanf(infile, "save_config = %255s\n", readin);
    data.save_config.assign(readin);
    reader += fscanf(infile, "save_config_FT = %255s\n", readin);
    data.save_config_FT.assign(readin);
    reader += fscanf(infile, "save_config_every_X_updates = %d\n", 
                             &data.save_config_every_X_updates);
    reader += fscanf(infile, "outpath = %255s\n", readin);
    data.outpath.assign(readin);

    // close input file
    fclose(infile);

    return data;
  };

public:
  const LatticeDataContainer data; 
  
  IO_params(int argc, char** argv) : data(read_infile(argc, argv)) {};

}; // end of class definition

} // end of namespace

#endif // IO_params
