#define write_viewer_C

#include <Kokkos_Core.hpp>
#include "lattice.hpp"
#include "write_viewer.hpp"

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

void write_viewer(FILE *f_conf,int layout_value, size_t V, const Viewphi &phi  ){
//     Kokkos::Timer timer; 
     Viewphi::HostMirror h_phi = Kokkos::create_mirror_view( phi );
 
     if (layout_value==0){
         // Deep copy device views to host views.
         Kokkos::deep_copy( h_phi, phi );
         //double time =timer.seconds();
         //printf(" deep_copy %f\n", time);
     }
     else if (layout_value==1){
                
         Viewphi w_phi("w_phi",2,V);
         Kokkos::parallel_for( "reordering for writing loop", V, KOKKOS_LAMBDA( size_t x ) {    
            //Kokkos::resize(w_phi, 2,V);  
            //phi (c,x ) is stored in position i=c+x*2
            // I want to save this value in i'=x+c*V
            // the cooordinate of i'=c'+x'*2
             for(size_t c=0; c<2;c++){
			 size_t i1=x+c*V;
			 size_t c1=i1%2;
			 size_t x1=i1/2;
			 w_phi(c1,x1)=phi(c,x);
             }
         });
         //double time =timer.seconds();
         //printf(" swap %f\n", time);
         // Deep copy device views to host views.
         Kokkos::deep_copy( h_phi, w_phi );
         //time =timer.seconds();
         //printf(" deep_copy %f\n", time);
     }
     fwrite(&h_phi(0,0), sizeof(double), 2*V, f_conf); 
     //double time =timer.seconds();
     //printf(" fwrite %f\n", time);

}
