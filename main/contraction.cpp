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
#include "DFT.hpp"

//#include <highfive/H5File.hpp>

#include <Kokkos_Core.hpp>




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    
    endian=endianness();
    #ifdef DEBUG
        printf("DEBUG mode ON\n");
        test_FT(params);
    #endif
    
    printf("endianness=%d  (0 unknown , 1 little , 2 big)\n",endian);
    if (endian==UNKNOWN_ENDIAN) {printf("UNKNOWN_ENDIAN abort\n"); exit(0);}

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
    
    printf("ignoring line\n compute_contractions = %s\n", params.data.compute_contractions.c_str());
    
    
    
    
    // init_rng( params);
    
    // starting kokkos
    Kokkos::initialize( argc, argv );{
    
  
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
    write_header_measuraments(f_G2t, params ); 
    
    double time_update=0,time_mes=0,time_writing=0;
    double ave_acc=0;
    
    
    // The update ----------------------------------------------------------------
    for(int ii = 0; ii < params.data.start_measure+params.data.total_measure; ii++) {
        double time; 
        
        // bool condition in the infile 
        bool measure =(ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.measure_every_X_updates == 0 );
        bool read_FT=     (measure &&   params.data.save_config_FT == "yes");
        //bool read=        (measure &&   params.data.save_config == "yes");
        
        if(measure){
            Viewphi::HostMirror h_phip;
            
            if(read_FT){
                Kokkos::Timer timer3;
                Viewphi::HostMirror   construct_h_phip("h_phip",2,params.data.L[0]*Vp);
                h_phip=construct_h_phip;
                
                std::string conf_file = params.data.outpath + 
                    "/T" + std::to_string(params.data.L[0]) +
                    "_L" + std::to_string(params.data.L[1]) +
                    "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                    "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                    "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)  + 
                    "_rep" + std::to_string(params.data.replica) + 
                    "_conf_FT" + std::to_string(ii);
                //cout << "reading configuration : " << conf_file << endl;
                FILE *f_conf = fopen(conf_file.c_str(), "r"); 
                if (f_conf == NULL) {
                    printf("Error opening file %s!\n", conf_file.c_str());
                    exit(1);
                }
                read_conf_FT(f_conf, layout_value, params , ii , h_phip ); 
                fclose(f_conf);
                time = timer3.seconds();
                time_writing+=time;
            }
            else{
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
                
                Viewphi::HostMirror   construct_h_phip("h_phip",2,params.data.L[0]*Vp);
                h_phip=construct_h_phip;
                compute_FT(phi, params ,   ii, h_phip);
                
                fclose(f_conf);
                time = timer3.seconds();
                time_writing+=time;
            }
        
        
            //Measure every 
            Kokkos::Timer timer_2;
            double *m=compute_magnetisations( phi,   params);
            fprintf(f_mes,"%.15g   %.15g \n",m[0], m[1]);
            free(m);//free(G2);
            
            compute_G2t( h_phip,   params,f_G2t, ii);
            
            time = timer_2.seconds();
            time_mes+=time;


        }
      
    }

    

    printf("average acceptance rate= %g\n", ave_acc/(params.data.start_measure+params.data.total_measure));
    
    printf("  time mesuring = %f s (%f per single operation)\n", time_mes   , time_mes/(params.data.total_measure/ params.data.measure_every_X_updates ));
    printf("  time reading  = %f s (%f per single opertion)\n", time_writing, time_writing/(params.data.total_measure/ params.data.save_config_every_X_updates) );
    printf("total time = %f s\n",time_writing+ time_mes );


    fclose(f_G2t);
    fclose(f_mes);
    }
    Kokkos::finalize();
    
    return 0;
}
