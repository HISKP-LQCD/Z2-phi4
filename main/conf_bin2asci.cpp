#define CONTROL

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
    // starting kokkos
    Kokkos::initialize( argc, argv );{
        // The update ----------------------------------------------------------------
        for(int ii = 0; ii < params.data.start_measure+params.data.total_measure; ii++) {
            
        
            // bool condition in the infile 
            bool measure =(ii >= params.data.start_measure && (ii-params.data.start_measure)%params.data.measure_every_X_updates == 0 );
            if(measure){
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
                
                cout << "Kokkos started:"<< endl; 
                cout << "   execution space:"<< typeid(Kokkos::DefaultExecutionSpace).name() << endl; 
                cout << "   host  execution    space:"<<  &Kokkos::HostSpace::name << endl; 
                
                int layout_value=check_layout();
                Viewphi  phi("phi",2,V);
                Viewphi::HostMirror  h_phi = Kokkos::create_mirror_view( phi );
                read_viewer(f_conf, layout_value, params , ii , phi );
                fclose(f_conf);
                
                std::string out_file = params.data.outpath + "/ASCI_"
                "T" + std::to_string(params.data.L[0]) +
                "_L" + std::to_string(params.data.L[1]) +
                "_msq0" + std::to_string(params.data.msq0)  +   "_msq1" + std::to_string(params.data.msq1)+
                "_l0" + std::to_string(params.data.lambdaC0)+     "_l1" + std::to_string(params.data.lambdaC1)+
                "_mu" + std::to_string(params.data.muC)   + "_g" + std::to_string(params.data.gC)  + 
                "_rep" + std::to_string(params.data.replica) + 
                "_conf" + std::to_string(ii);
                cout << "writing configuration : " << out_file << endl;
                FILE *f_out = fopen(out_file.c_str(), "w+"); 
                if (f_out == NULL) {
                    printf("Error opening file %s!\n", out_file.c_str());
                    exit(1);
                }
                // Deep copy device views to host views.
                Kokkos::deep_copy( h_phi, phi ); 
                for (size_t x=0; x<params.data.V; x++ ){
                    fprintf(f_out,"%ld  %.12g   %.12g\n",x,phi(0,x),phi(1,x));
                    
                }
                fclose(f_out);
            }
            
        }
        
    }
    Kokkos::finalize();
    
    return 0;
}
