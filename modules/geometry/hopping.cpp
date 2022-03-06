/*********************************************************
 *
 *  File hopping.c
 *
 *  Initialization of the hopping field for a D dimensional
 *  lattice of size V=L**D
 *  The index of a point with coordinates (n_0,n_1,..,n_{d-1})
 *  is i=sum_k n_k L**k
 *  The index of its neighbor in positive direction nu
 *  is hop[i][mu]
 *  In negative direction it is hop[i][D+mu]
 *  The coordinate k=0,...,D-1, of a given point with
 *  index i is then stored in ipt[i][k]
 *
 ********************************************************/

#define HOPPING_C

#include "lattice.hpp"
#include "geometry.hpp"
#include <cstdlib>
#include <Kokkos_Core.hpp>
 /*
 void hopping(int V,int L[D])
 {
     hop=(int**) malloc(sizeof(int*)*V);
     ipt=(int**) malloc(sizeof(int*)*V);
     for (int i =0;i<V;i++){
         hop[i]=(int*) malloc(sizeof(int)*2*D);
         ipt[i]=(int*) malloc(sizeof(int)*D);
     }


     int x, y, Lk;
     int xk, k, dxk ;

    // go through all the points
    for (x=0; x < V ; x++){
       Lk = V;
       y  = x;

       // go through the components k
       for (k=D-1; k >= 0; k--){

          Lk/=L;                        // pow(L,k)
          xk =y/Lk;                     // kth component
          y  =y-xk*Lk;                  // y<-y%Lk

          ipt[x][k]=xk;

          // forward
          if (xk<L-1) dxk = Lk;
          else        dxk = Lk*(1-L);
          hop[x][k] = x + dxk;

          // backward
          if (xk>0)   dxk = -Lk;
          else        dxk = Lk*(L-1);
          hop[x][k+D] = x + dxk;
       }
    }
 } // hopping
 */

 //void hopping(const int *L, ViewLatt &hop, ViewLatt  &ipt ,ViewLatt &even_odd )
void hopping(const int* L, Kokkos::View<size_t**>& hop, ViewLatt& sectors, Kokkos::View<size_t**>& ipt) {
    int L1 = L[1], L2 = L[2], L3 = L[3], L0 = L[0];

    Kokkos::View<size_t**>::HostMirror h_hop = Kokkos::create_mirror_view(hop);
    Kokkos::View<size_t**>::HostMirror h_ipt = Kokkos::create_mirror_view(ipt);
    Kokkos::View<size_t**>::HostMirror h_rbg = Kokkos::create_mirror_view(sectors.rbg);

    //dx is 2 if L is even , if else(L is odd) dx=3 
    int dx = L1 % 2 + 2;
    int dy = L2 % 2 + 2;
    int dz = L3 % 2 + 2;
    int dt = L0 % 2 + 2;
    int dtot = 2;
    if (L0 % 2 == 1 || L1 % 2 == 1 || L2 % 2 == 1 || L3 % 2 == 1) dtot = 3;
    size_t eo = 0;
    // run in such a way that i+=1;
    for (int t = 0; t < L0; t++) {
        for (int z = 0; z < L3; z++) {
            for (int y = 0; y < L2; y++) {
                for (int x = 0; x < L1; x++) {
                    size_t i = x + y * L1 + z * L1 * L2 + t * L3 * L2 * L1;
                    h_hop(i, 0) = x + y * L1 + z * L1 * L2 + ((t + 1) % L0) * L3 * L2 * L1;
                    h_hop(i, 1) = x + y * L1 + ((z + 1) % L3) * L1 * L2 + t * L3 * L2 * L1;
                    h_hop(i, 2) = x + ((y + 1) % L2) * L1 + z * L1 * L2 + t * L3 * L2 * L1;
                    h_hop(i, 3) = ((x + 1) % L1) + y * L1 + z * L1 * L2 + t * L3 * L2 * L1;

                    h_hop(i, 4) = x + y * L1 + z * L1 * L2 + ((t + L0 - 1) % L0) * L3 * L2 * L1;
                    h_hop(i, 5) = x + y * L1 + ((z + L3 - 1) % L3) * L1 * L2 + t * L3 * L2 * L1;
                    h_hop(i, 6) = x + ((y + L2 - 1) % L2) * L1 + z * L1 * L2 + t * L3 * L2 * L1;
                    h_hop(i, 7) = ((x + L1 - 1) % L1) + y * L1 + z * L1 * L2 + t * L3 * L2 * L1;


                    h_ipt(i, 0) = t; h_ipt(i, 1) = x; h_ipt(i, 2) = y; h_ipt(i, 3) = z;
                    int color = ((x % dx) + (y % dy) + (z % dz) + (t % dt)) % dtot;
                    // special case for L = 7, 13, 19, 25, ...
                    // in this case we color as r b g r b g b
                    if ((L[0] - 1) % 6 == 0 && t == L[0] - 1) color = (color + 1) % dtot;
                    if ((L[1] - 1) % 6 == 0 && x == L[1] - 1) color = (color + 1) % dtot;
                    if ((L[2] - 1) % 6 == 0 && y == L[2] - 1) color = (color + 1) % dtot;
                    if ((L[3] - 1) % 6 == 0 && z == L[3] - 1) color = (color + 1) % dtot;
                    // printf("  %ld = %ld  %ld  %ld  %ld   -> %d\n", i, x, y, z, t, color);
                    if (color == 0) {
                        h_rbg(0, sectors.size[0]) = i;
                        sectors.size[0] += 1;
                    }
                    else if (color == 1) {
                        h_rbg(1, sectors.size[1]) = i;
                        sectors.size[1] += 1;
                    }
                    else if (color == 2) {
                        h_rbg(2, sectors.size[2]) = i;
                        sectors.size[2] += 1;
                    }
                    //                    eo=(x+y+z+t)%2;        
                //printf("even_odd(%ld, %ld)=%ld = %ld \n", eo,i/2, i, h_even_odd(eo,i/2) );
                    eo = (eo + 1) % 2;


                }
                eo = (eo + 1) % 2;
            }
            eo = (eo + 1) % 2;
        }
        eo = (eo + 1) % 2;
    }
    printf("sizes of the sectors:\n");
    printf("color=%d size=%d\n", 0, sectors.size[0]);
    printf("color=%d size=%d\n", 1, sectors.size[1]);
    printf("color=%d size=%d\n", 2, sectors.size[2]);
    // Deep copy host views to device views.
    Kokkos::deep_copy(hop, h_hop);
    Kokkos::deep_copy(ipt, h_ipt);
    Kokkos::deep_copy(sectors.rbg, h_rbg);
#ifdef DEBUG
    for (int color = 0; color < 3; color++) {
        Kokkos::parallel_for("check_sectors", sectors.size[color], KOKKOS_LAMBDA(int xx){
            int x = sectors.rbg((color), xx);
            for (int d = 0; d < 8; d++) {
                size_t n = hop(x, d);
                for (int j = 0; j < sectors.size[color]; j++) {
                    if (n == sectors.rbg(color, j)) {
                        printf("x=%d  hopping=%ld is in the same colored sector=%d\n", x, n, color);
                        Kokkos::abort("check sectors");
                    }
                }
            }
        });
    }
#endif


}
