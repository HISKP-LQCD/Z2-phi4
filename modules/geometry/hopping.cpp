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
void hopping(const int *L , ViewLatt &hop, ViewLatt &even_odd,  ViewLatt &ipt )
{
    int x, y,z,t, Lk;
    int xk, k, dxk, i ;
    int L1=L[1],L2=L[2],L3=L[3], L0=L[0];
    int V=L1*L2*L3*L0;
   /* 
    hop=(int**) malloc(sizeof(int*)*V);
    ipt=(int**) malloc(sizeof(int*)*V);
    for (int i =0;i<V;i++){
        hop[i]=(int*) malloc(sizeof(int)*2*dim_spacetime);
        ipt[i]=(int*) malloc(sizeof(int)*dim_spacetime);
    }
    even_odd=(int**) malloc(sizeof(int*)*2);
    for (int i =0;i<2;i++)
        even_odd[i]=(int*) malloc(sizeof(int)*V/2);
    */
    
    int count_e=0;
    int count_o=0;
    int eo=0;
    for(x=0;x<L1;x++)
        for(y=0;y<L2;y++)
            for(z=0;z<L3;z++)
                for(t=0;t<L0;t++){
                    i=x+y*L1+z*L1*L2+t*L3*L2*L1;
                    hop(i,0)=x+y*L1+z*L1*L2+((t+1)%L0)*L3*L2*L1;
                    hop(i,1)=x+y*L1+((z+1)%L3)*L1*L2+t*L3*L2*L1;
                    hop(i,2)=x+((y+1)%L2)*L1+z*L1*L2+t*L3*L2*L1;
                    hop(i,3)=((x+1)%L1)+y*L1+z*L1*L2+t*L3*L2*L1;

                    hop(i,4)=x+y*L1+z*L1*L2+((t+L0-1)%L0)*L3*L2*L1;
                    hop(i,5)=x+y*L1+((z+L3-1)%L3)*L1*L2+t*L3*L2*L1;
                    hop(i,6)=x+((y+L2-1)%L2)*L1+z*L1*L2+t*L3*L2*L1;
                    hop(i,7)=((x+L1-1)%L1)+y*L1+z*L1*L2+t*L3*L2*L1;

                    //hop(i,[8]=i;
                    ipt(i,0)=t;ipt(i,1)=x;ipt(i,2)=y;ipt(i,3)=z;
                    
                    even_odd(eo,i/2)=i;
                    eo=(eo+1) %2;
                    
                    
                }
     
                
   
} 
