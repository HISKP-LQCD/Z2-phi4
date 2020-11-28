#ifndef LATTICE_H
#define LATTICE_H

#include <Kokkos_Core.hpp>


/* Dimension of the lattice */
#define dim_spacetime 4
/* spatial extend of the lattice */
//#define L 4
// #define L 4LU
/* lattice volume, needs to be adjusted according to number of dimensions */
//#define V (L*L*L*L)

#ifdef CONTROL 
#define EXTERN 
#else
#define EXTERN extern
#endif


using namespace std;
// kokkos viever for the field 
typedef Kokkos::View<double**>  Viewphi;
typedef Kokkos::View<int**>  ViewLatt;
/* GLOBAL_VECTORS */
EXTERN int      V;
EXTERN double  rand_max;
//EXTERN int    hop[V][2*D];
//EXTERN int    ipt[V][D];

/* GLOBAL_STRUCTS */
/*ispt_parms_t ispt_parms;
hmd_parms_t hmd_parms;
smd_parms_t smd_parms;
act_parms_t act_parms;
*/
#endif
