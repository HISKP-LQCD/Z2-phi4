#define write_viewer_C

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include "IO_params.hpp"
#include "write_viewer.hpp"
#include "utils.hpp"

int check_layout(){
    //check the layout of phi    
    Viewphi v_tmp("test_l",2,4);
    Viewphi::HostMirror h_tmp = Kokkos::create_mirror_view( v_tmp );
    for (int c =0 ; c< 2;c++)
        for (int x =0 ; x< 4;x++)
            h_tmp(c,x)=c+x*2;
        

    double *p_tmp=&h_tmp(0,0);
    int count_l=0,count_r=0;
    for (int c =0 ; c< 2;c++){   
        for (int x =0 ; x< 4;x++){
            if ( (x==0 && c==0) || (x==3 && c==1)) {p_tmp++ ;continue;}// the first and the last are always in the same position, so we remove from the test

            if ( *p_tmp == h_tmp(c,x)  ){
//                printf("Layout of the field phi(c=%ld,x=%ld) : c+x*2 : LayoutLeft \n",c,x);
                count_l++;
            }
            else if ( *p_tmp == x+c*4  ){
//                printf("Layout of the field phi(c=%ld,x=%ld) : x+c*V : LayoutRight \n",c,x);
                count_r++;
            }
            else{
                printf("\n\n urecogised Layout\n\n");
                exit(1);
            }
                
            p_tmp++;
        }
    }
    
//    Viewphi w_phi("w_phi",2,V);
    int swap_layout=0;
    if (count_l==6 && count_r==0) {
        printf("Layout of the field phi(c,x) : c+x*2 : LayoutRight \n it need a reordering before writing\n");
        swap_layout=1;
    } 
    else if (count_l==0 && count_r==6) {
        printf("Layout of the field phi(c,x) : x+c*V : LayoutLeft \n nothing to be done to write\n");
    }
    else{
        printf("\n\n urecogised Layout\n\n");
        exit(1);
    }
    return swap_layout;
}

void write_header(FILE *f_conf, cluster::IO_params params ,int iconf){

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
     //fwrite(&params.data.start_measure, sizeof(int), 1, f_conf); 
     //fwrite(&params.data.total_measure, sizeof(int), 1, f_conf); 
     //fwrite(&params.data.measure_every_X_updates, sizeof(int), 1, f_conf); 
     
     fwrite(&iconf, sizeof(int), 1, f_conf); 

}


void write_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , const Viewphi phi  ){
 
     #ifdef TIMER
         Kokkos::Timer timer;
     #endif
     write_header(f_conf, params, iconf);
     #ifdef TIMER
         double time=timer.seconds();
         printf("time to write the header %f\n",time);
     #endif
     size_t V=params.data.V; 
     Viewphi::HostMirror h_phi("write_host",2,V);// = Kokkos::create_mirror_view( phi ); does not create a new one if the device is a CPU
          
     if (layout_value==0){
         // Deep copy device views to host views.
         Kokkos::deep_copy( h_phi, phi );
         
     }
     else if (layout_value==1){
                
         Viewphi w_phi("w_phi",2,V);
         Kokkos::parallel_for( "reordering for writing loop", V, KOKKOS_LAMBDA( size_t x ) {    
             //phi (c,x ) is stored in the divice with the order i=c+x*2
             // I want to save it on the disk with order i1=x+c*V
             // so we need the coordinate c1 and x1 of  i1=c1+x1*2
             for(size_t c=0; c<2;c++){
			 size_t i1=x+c*V;
			 size_t c1=i1%2;
			 size_t x1=i1/2;
			 w_phi(c1,x1)=phi(c,x);
             }
         });
         
         // Deep copy device views to host views.
         Kokkos::deep_copy( h_phi, w_phi );
         
     }
     if(endian==BIG_ENDIAN){
         for(size_t x=0; x<V;x++) 
             for(size_t c=0; c<2;c++)
                 bswap_double(1,&h_phi(c,x));
     }
     #ifdef TIMER
         double time1=timer.seconds()-time;
         printf("time to copy data on the host %f\n",time1);
     #endif
     fwrite(&h_phi(0,0), sizeof(double), 2*V, f_conf); 
     #ifdef TIMER
         double time2=timer.seconds()-time-time1;
         printf("time of fwrite %f\n",time2);
     #endif
     

}

////////////////////////////////////////////////////
// same routine as before but for reading
////////////////////////////////////////////////////

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

void check_header(FILE *f_conf, cluster::IO_params &params ,int iconf){

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
     error_header(f_conf,params.data.seed,"seed" ); 
     error_header(f_conf,params.data.replica,"replica" ); 
     
     error_header(f_conf,iconf,"iconf" ); 
}

void read_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  ,  Viewphi &phi  ){
     #ifdef TIMER
         Kokkos::Timer timer;
     #endif
     check_header(f_conf, params, iconf);
     #ifdef TIMER
         double time=timer.seconds();
         printf("time to check header %f\n",time);
     #endif
     size_t V=params.data.V; 
     Viewphi::HostMirror h_phi = Kokkos::create_mirror_view( phi );
 
     fread(&h_phi(0,0), sizeof(double), 2*V, f_conf); 
     #ifdef TIMER
         double time1=timer.seconds()-time;
         printf("time to read on Host %f\n",time1);
     #endif
     if (layout_value==0){
         // Deep copy host views to device views.
         Kokkos::deep_copy( phi, h_phi );
         //double time =timer.seconds();
         //printf(" deep_copy %f\n", time);
     }
     else if (layout_value==1){
         Viewphi r_phi("r_phi",2,V);
         // Deep copy host views to device views.
         Kokkos::deep_copy( r_phi, h_phi );
         Kokkos::parallel_for( "reordering for writing loop", V, KOKKOS_LAMBDA( size_t x ) {  
             
             //we have the field phi (c,x) on the disk as i=x+c*V
             // we want to load it in to the device as i1=c+x*2
             // we need the coordinate i1=x1+c1*V 
             for(size_t c=0; c<2;c++){
			 size_t i1=c+x*2;
			 size_t c1=i1/V;
			 size_t x1=i1%V;
			 phi(c1,x1)=r_phi(c,x);
             }
         });
     }
     
     #ifdef TIMER
         double time2=timer.seconds()-time-time1;
         printf("time to copy on device %f\n",time2);
     #endif

}


////////////////////////////////////////////////////
// write conf after the FT, only a sublattice is written. 
////////////////////////////////////////////////////

void write_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi::HostMirror h_phip ){
    
    #ifdef TIMER
    Kokkos::Timer timer;
    #endif
    write_header(f_conf, params, iconf);
    #ifdef TIMER
    double time=timer.seconds();
    printf("time to write the header %f\n",time);
    #endif
    size_t V=params.data.L[0]*128; 
    
    if(endian==BIG_ENDIAN){
        for(size_t x=0; x<V;x++) 
            for(size_t c=0; c<2;c++)
                bswap_double(1,&h_phip(c,x));
    }
   
    if (layout_value==1){
        
        Viewphi::HostMirror w_phip("w_phi",2,V);
        for (int x=0;x<V;x++){
            //phi (c,x ) is stored in the divice with the order i=c+x*2
            // I want to save it on the disk with order i1=x+c*V
            // so we need the coordinate c1 and x1 of  i1=c1+x1*2
            for(size_t c=0; c<2;c++){
                size_t i1=x+c*V;
                size_t c1=i1%2;
                size_t x1=i1/2;
                w_phip(c1,x1)=h_phip(c,x);
            }
        }
        
        // Deep copy device views to host views.
        //Kokkos::deep_copy( h_phi, w_phi );
        fwrite(&w_phip(0,0), sizeof(double), 2*V, f_conf); 
        
        
    }
    else{
        fwrite(&h_phip(0,0), sizeof(double), 2*V, f_conf); 
        
    }
    
    if(endian==BIG_ENDIAN){
        for(size_t x=0; x<V;x++) 
            for(size_t c=0; c<2;c++)
                bswap_double(1,&h_phip(c,x));
    }
    #ifdef TIMER
    double time2=timer.seconds()-time;
    printf("time of fwrite %f\n",time2);
    #endif
    
}


void read_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi::HostMirror &h_phip ){
    #ifdef TIMER
    Kokkos::Timer timer;
    #endif
    check_header(f_conf, params, iconf);
    #ifdef TIMER
    double time=timer.seconds();
    printf("time to check header %f\n",time);
    #endif
    size_t V=params.data.L[0]*128; 
    
    
    fread(&h_phip(0,0), sizeof(double), 2*V, f_conf); 
    #ifdef TIMER
    double time1=timer.seconds()-time;
    printf("time to read on Host %f\n",time1);
    #endif
    
    if (layout_value==1){
        Viewphi r_phip("r_phip",2,V);
        
        for (int x=0;x<V;x++){
            //phi (c,x ) is stored in the divice with the order i=c+x*2
            // I want to save it on the disk with order i1=x+c*V
            // so we need the coordinate c1 and x1 of  i1=c1+x1*2
            for(size_t c=0; c<2;c++){
                size_t i1=c+x*2;
                size_t c1=i1/V;
                size_t x1=i1%V;
                r_phip(c1,x1)=h_phip(c,x);
            }
        }
        for (int x=0;x<V;x++)
            for(size_t c=0; c<2;c++)
                h_phip(c,x)=r_phip(c,x);
        
    }
    
    #ifdef TIMER
    double time2=timer.seconds()-time-time1;
    printf("time to copy on device %f\n",time2);
    #endif
    
}

