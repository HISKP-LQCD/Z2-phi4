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
    std::string save_config_FT_bundle;
    std::string compute_contractions;
    int save_config_every_X_updates;
    std::string outpath;
    std::string checks;

    std::string smearing;
    std::string FT_phin;
    std::string smearing3FT;

    bool compute_E;

  };
  // -----------------------------------------------------------------------------
  // -----------------------------------------------------------------------------
  class IO_params {

  private:

    double comp_kappa(double msq, double lambdaC) {
      double k;

      if (lambdaC < 1e-12)
        k = 1. / (8. + msq);
      else {
        k = -8. - msq + sqrt((8 + msq) * (8 + msq) + 32 * lambdaC);
        k /= (16. * lambdaC);
      }
      return k;

    }
    /*
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

        if (data.formulation.size()>99){
            printf("formulation must be a string of maximum 100 char");
            exit(1);
        }


        if(data.formulation == "O2"){
          cout << "using O2 version of the two scalars lambdaC0=lambdaC1=muC/2: \n" << endl;
          if ( data.lambdaC1 !=0 || data.muC !=0 ){
              cout    << "if formulation = O2  you need to set lambdaC1= muC2=0 "<< endl;
              exit(0);
          }
          data.lambdaC1= data.lambdaC0;
          data.muC= 2.*data.lambdaC0;

        }
        //fill formulation with null char
        for (int i=data.formulation.size();i<99;i++)
            data.formulation=data.formulation+'\0';

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
        reader += fscanf(infile, "compute_contractions = %255s\n", readin);
        data.compute_contractions.assign(readin);
        if (data.compute_contractions!= "yes" &&  data.save_config_FT!="yes"  &&  data.save_config!="yes"){
            printf("error: at least one of the following in the input file must be yes \n");
            printf("save_config    = %s\n",data.save_config.c_str());
            printf("save_config_FT = %s\n",data.save_config_FT.c_str());
            printf("compute_contractions = %s\n",data.compute_contractions.c_str());
            exit(0);
        }
        // unused
        reader += fscanf(infile, "checks =  %255s\n", readin);
        data.checks.assign(readin);
        reader += fscanf(infile, "outpath = %255s\n", readin);
        data.outpath.assign(readin);

        // close input file
        fclose(infile);

        return data;
      };
      */


    template <typename Out>
    void split(const std::string& s, char delim, Out result) {
      std::istringstream iss(s);
      std::string item;
      while (std::getline(iss, item, delim)) {
        *result++ = item;
      }
    }

    std::vector<std::string> split(const std::string& s, char delim) {
      std::vector<std::string> elems;
      split(s, delim, std::back_inserter(elems));
      std::vector<std::string>  r;
      std::string empty = " ";
      for (std::string s1 : elems)
        if (s1.empty() == 0) {
          r.emplace_back(s1);
        }

      return r;
    }

    void read_L(std::fstream& newfile, int* L) {
      int match = 0;
      if (newfile.is_open()) { //checking whether the file is open
        std::string tp;
        while (getline(newfile, tp)) { //read data from file object and put it into string.
          std::vector<std::string> x = split(tp, '=');
          if (x.empty() == 0) {// if not empty line
            if (x.size() != 2) { printf("error infile scan for L\n expected:  param = value  \n found: %s \n", tp.c_str()); exit(-9); }

            std::vector<std::string> x0 = split(x[0], ' ');
            if (x0.size() != 1) { printf("error infile scan for L\n param must be 1 word only \n found: %s ", x[0].c_str()); exit(-9); }
            std::string  name = "L";
            if (x0[0].compare(name) == 0) {

              std::vector<std::string> rl = split(x[1], ' ');
              L[0] = stoi(rl[0]);
              L[1] = stoi(rl[1]);
              L[2] = stoi(rl[2]);
              L[3] = stoi(rl[3]);
              match++;
            }
          }
        }
      }
      else { printf("infile is not open \n"); exit(-10); }

      if (match == 0) { printf("could not find line L = \n"); exit(-10); }
      if (match > 1) { printf("multiple line L = \n"); exit(-10); }
      std::cout << "L = " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << endl;

      //rewind
      newfile.clear();
      newfile.seekg(0);
    }
    void read_par_double(std::fstream& newfile, std::string  name, double& p) {
      int match = 0;
      if (newfile.is_open()) { //checking whether the file is open
        std::string tp;
        while (getline(newfile, tp)) { //read data from file object and put it into string.
          std::vector<std::string> x = split(tp, '=');
          if (x.empty() == 0) {// if not empty line
            if (x.size() != 2) { printf("error infile scan for %s\n expected:  param = value  \n found: %s \n", name.c_str(), tp.c_str()); exit(-9); }

            std::vector<std::string> x0 = split(x[0], ' ');
            if (x0.size() != 1) { printf("error infile scan for %s\n param must be 1 word only \n found: %s ", name.c_str(), x[0].c_str()); exit(-9); }
            if (x0[0].compare(name) == 0) {
              std::vector<std::string> rl = split(x[1], ' ');
              p = stod(rl[0]);
              match++;
            }
          }
        }
      }
      else { printf("infile is not open \n"); exit(-10); }

      if (match == 0) { printf("could not find line %s = \n", name.c_str()); exit(-10); }
      if (match > 1) { printf("multiple line %s = \n", name.c_str()); exit(-10); }
      // std::cout << name << " = "<< p  << endl;

      //rewind
      newfile.clear();
      newfile.seekg(0);
    }
    void read_par_int(std::fstream& newfile, std::string  name, int& p) {
      int match = 0;
      if (newfile.is_open()) { //checking whether the file is open
        std::string tp;
        while (getline(newfile, tp)) { //read data from file object and put it into string.
          std::vector<std::string> x = split(tp, '=');
          if (x.empty() == 0) {// if not empty line
            if (x.size() != 2) { printf("error infile scan for %s\n expected:  param = value  \n found: %s \n", name.c_str(), tp.c_str()); exit(-9); }

            std::vector<std::string> x0 = split(x[0], ' ');
            if (x0.size() != 1) { printf("error infile scan for %s\n param must be 1 word only \n found: %s ", name.c_str(), x[0].c_str()); exit(-9); }
            if (x0[0].compare(name) == 0) {
              std::vector<std::string> rl = split(x[1], ' ');
              p = stoi(rl[0]);
              match++;
            }
          }
        }
      }
      else { printf("infile is not open \n"); exit(-10); }

      if (match == 0) { printf("could not find line %s = \n", name.c_str()); exit(-10); }
      if (match > 1) { printf("multiple line %s = \n", name.c_str()); exit(-10); }
      // std::cout << name << " = "<< p  << endl;
      //rewind
      newfile.clear();
      newfile.seekg(0);
    }
    void read_par_string(std::fstream& newfile, std::string  name, std::string& s, bool required = true) {
      int match = 0;
      if (newfile.is_open()) { //checking whether the file is open
        std::string tp;
        while (getline(newfile, tp)) { //read data from file object and put it into string.
          std::vector<std::string> x = split(tp, '=');
          if (x.empty() == 0) {// if not empty line
            if (x.size() != 2) { printf("error infile scan for %s\n expected:  param = value  \n found: %s \n", name.c_str(), tp.c_str()); exit(-9); }

            std::vector<std::string> x0 = split(x[0], ' ');
            if (x0.size() != 1) { printf("error infile scan for %s\n param must be 1 word only \n found: %s ", name.c_str(), x[0].c_str()); exit(-9); }
            if (x0[0].compare(name) == 0) {
              std::vector<std::string> rl = split(x[1], ' ');
              s = rl[0];
              match++;
            }
          }
        }
      }
      else { printf("infile is not open \n"); exit(-10); }

      if (match == 0) {
        if (required) { std::cout << "could not find line: " << name.c_str() << " =" << endl; exit(-10); }
        else
          std::cout << "could not find param: " << name << "\n default falue: " << name << " = " << s << "" << endl;
      }
      if (match > 1) { printf("multiple line %s = \n", name.c_str()); exit(-10); }
      // std::cout << name << " = "<< s  << endl;
      //rewind
      newfile.clear();
      newfile.seekg(0);
    }
    bool read_par_bool(std::fstream& newfile, std::string  name, bool required = true) {
      std::string s;
      bool r = false;
      int match = 0;
      if (newfile.is_open()) { //checking whether the file is open
        std::string tp;
        while (getline(newfile, tp)) { //read data from file object and put it into string.
          std::vector<std::string> x = split(tp, '=');
          if (x.empty() == 0) {// if not empty line
            if (x.size() != 2) { printf("error infile scan for %s\n expected:  param = value  \n found: %s \n", name.c_str(), tp.c_str()); exit(-9); }

            std::vector<std::string> x0 = split(x[0], ' ');
            if (x0.size() != 1) { printf("error infile scan for %s\n param must be 1 word only \n found: %s ", name.c_str(), x[0].c_str()); exit(-9); }
            if (x0[0].compare(name) == 0) {
              std::vector<std::string> rl = split(x[1], ' ');
              s = rl[0];
              match++;
            }
          }
        }
      }
      else { printf("infile is not open \n"); exit(-10); }
      newfile.clear();
      newfile.seekg(0);

      if (match == 0) {
        if (required) { std::cout << "could not find line: " << name.c_str() << " =" << endl; exit(-10); }
        else
          std::cout << "could not find param: " << name << "\n default falue: " << name << " = " << r << "" << endl;
        return r;
      }
      if (match > 1) { printf("multiple line %s = \n", name.c_str()); exit(-10); }
      // std::cout << name << " = "<< s  << endl;
      //rewind
      if (s == "yes" || s == "y" || s == "on" || s == "Yes" || s == "Y" || s == "On" || s == "YES" || s == "ON") {
        r = true;
      }

      return r;
    }

    inline LatticeDataContainer read_infile(int argc, char** argv) {

      int opt = -1;
      char infilename[200];

      LatticeDataContainer data_in;

      // search for command line option and put filename in "infilename"
      for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "-i") == 0) {
          opt = i + 1;
          break;
        }
      }
      if (opt < 0) {
        std::cout << "No input file specified, Aborting" << endl;
        exit(1);
      }
      else {
        sprintf(infilename, "%s", argv[opt]);
        std::cout << "Trying input file " << infilename << endl;
      }
      std::fstream newfile;

      newfile.open(infilename, std::ios::in);

      // open file for reading
      if (!newfile.is_open()) {
        std::cerr << "Could not open file " << infilename << endl;
        std::cerr << "Aborting..." << endl;
        exit(-10);
      }
      // lattice size
      read_L(newfile, data_in.L);
      //reader += fscanf(infile, "L = %d %d %d %d \n", &data_in.L[0], &data_in.L[1], 
      //                                               &data_in.L[2], &data_in.L[3]);
      data_in.V = data_in.L[0] * data_in.L[1] * data_in.L[2] * data_in.L[3];

      read_par_string(newfile, "formulation", data_in.formulation);

      // kappa and lambda
      read_par_double(newfile, "msq0", data_in.msq0);
      read_par_double(newfile, "msq1", data_in.msq1);
      // read_par_double(newfile,"lambdaC0",data_in.lambdaC0);
      // read_par_double(newfile,"lambdaC1",data_in.lambdaC1);
      // read_par_double(newfile,"muC",data_in.muC);
      printf("ignoring lambdaC0, lambdaC1, muC,  resetting the to zero \n");
      data_in.lambdaC0 = 0;
      data_in.lambdaC1 = 0;
      data_in.muC = 0;
      read_par_double(newfile, "gC", data_in.gC);

      if (data_in.formulation.size() > 99) {
        printf("formulation must be a string of maximum 100 char");
        exit(1);
      }


      if (data_in.formulation == "O2") {
        cout << "using O2 version of the two scalars lambdaC0=lambdaC1=muC/2: \n" << endl;
        if (data_in.lambdaC1 != 0 || data_in.muC != 0) {
          cout << "if formulation = O2  you need to set lambdaC1= muC2=0 " << endl;
          exit(0);
        }
        data_in.lambdaC1 = data_in.lambdaC0;
        data_in.muC = 2. * data_in.lambdaC0;

      }
      //fill formulation with null char    
      for (int i = data_in.formulation.size(); i < 99; i++)
        data_in.formulation = data_in.formulation + '\0';

      data_in.kappa0 = comp_kappa(data_in.msq0, data_in.lambdaC0);
      data_in.kappa1 = comp_kappa(data_in.msq1, data_in.lambdaC1);
      data_in.lambda0 = data_in.lambdaC0 * 4. * data_in.kappa0 * data_in.kappa0;
      data_in.lambda1 = data_in.lambdaC1 * 4. * data_in.kappa1 * data_in.kappa1;
      data_in.mu = data_in.muC * (4. * data_in.kappa0 * data_in.kappa1);
      data_in.g = data_in.gC * (4. * sqrt(data_in.kappa0 * data_in.kappa1 * data_in.kappa1 * data_in.kappa1));
      printf("parameters:\n");
      printf("msq0 = %.6f     -> kappa0   = %.6f\n", data_in.msq0, data_in.kappa0);
      printf("msq1 = %.6f     -> kappa1   = %.6f\n", data_in.msq1, data_in.kappa1);
      // printf( "lambdaC0 = %.6f -> lambda0  = %.6f\n", data_in.lambdaC0, data_in.lambda0);
      // printf( "lambdaC1 = %.6f -> lambda1  = %.6f\n", data_in.lambdaC1, data_in.lambda1);
      // printf( "muC = %.6f      -> mu       = %.6f\n", data_in.muC, data_in.mu);
      printf("gC = %.6f       -> g        = %.6f\n", data_in.gC, data_in.g);

      // metropolis 
      read_par_int(newfile, "metropolis_local_hits", data_in.metropolis_local_hits);
      read_par_int(newfile, "metropolis_global_hits", data_in.metropolis_global_hits);
      read_par_double(newfile, "metropolis_delta", data_in.metropolis_delta);

      // cluster
      // read_par_int(newfile, "cluster_hits", data_in.cluster_hits);
      // read_par_double(newfile, "cluster_min_size", data_in.cluster_min_size);
      data_in.cluster_hits = 0;
      data_in.cluster_min_size = 0;
      // configs
      read_par_int(newfile, "seed", data_in.seed);
      // read_par_int(newfile, "level", data_in.level);
      // read_par_int(newfile, "append", data_in.append);
      data_in.level = 1;
      data_in.append = 0;
      read_par_int(newfile, "replica", data_in.replica);
      read_par_int(newfile, "start_measure", data_in.start_measure);


      if (data_in.append < 0 || data_in.replica < 0) {
        cout << "append and replica value must not be negative!" << endl;
        exit(0);
      }
      if (data_in.append == 1 && data_in.start_measure != 0) {
        cout << "if append mode you can not wait for termalization" << endl;
        exit(0);
      }

      read_par_int(newfile, "total_measure", data_in.total_measure);
      read_par_int(newfile, "measure_every_X_updates", data_in.measure_every_X_updates);
      read_par_string(newfile, "save_config", data_in.save_config);
      read_par_string(newfile, "save_config_FT", data_in.save_config_FT);
      data_in.save_config_FT_bundle = "no";
      read_par_string(newfile, "save_config_FT_bundle", data_in.save_config_FT_bundle, false);
      read_par_string(newfile, "compute_contractions", data_in.compute_contractions);


      if (data_in.compute_contractions != "yes" && data_in.save_config_FT != "yes" && data_in.save_config != "yes") {
        printf("error: at least one of the following in the input file must be yes \n");
        printf("save_config    = %s\n", data_in.save_config.c_str());
        printf("save_config_FT = %s\n", data_in.save_config_FT.c_str());
        printf("compute_contractions = %s\n", data_in.compute_contractions.c_str());
        exit(0);
      }

      data_in.checks = "yes";
      read_par_string(newfile, "checks", data_in.checks, false);
      read_par_string(newfile, "outpath", data_in.outpath);

      data_in.smearing = "yes";
      data_in.FT_phin = "yes";
      read_par_string(newfile, "smearing", data_in.smearing, false);
      read_par_string(newfile, "FT_phin", data_in.FT_phin, false);
      data_in.smearing3FT = "yes";
      read_par_string(newfile, "smearing3FT", data_in.smearing3FT, false);
      data_in.compute_E = read_par_bool(newfile, "compute_E", false);
      
      newfile.close();

      return data_in;
    };


  public:
    const LatticeDataContainer data;

    IO_params(int argc, char** argv) : data(read_infile(argc, argv)) {};

  }; // end of class definition

} // end of namespace

#endif // IO_params
