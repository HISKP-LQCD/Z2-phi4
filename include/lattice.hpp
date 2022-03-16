#ifndef LATTICE_H
#define LATTICE_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif

/* Dimension of the lattice */
#define dim_spacetime 4
#define Lp 2
#define Vp Lp*Lp*Lp
#define Ncorr 273

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
typedef  double	scalartype;
// kokkos viever for the field 
typedef Kokkos::View<scalartype**>  Viewphi;
typedef Kokkos::View<Kokkos::complex<double>**>  Viewphip;
//typedef Kokkos::View<double**, Kokkos::LayoutLeft>  writingphi;

// typedef Kokkos::View<size_t**>  ViewLatt;
class ViewLatt{
public:
    Kokkos::View<size_t**>  rbg;
    int size[3];

    ViewLatt() {};

    ViewLatt(int N) : rbg("rgb", 3, N) {
        size[0] = 0;
        size[1] = 0;
        size[2] = 0;
        };
} ;

typedef Kokkos::View<Kokkos::complex<double>**>  complexphi;
typedef Kokkos::View<Kokkos::complex<double>***>  manyphi;

typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
typedef typename RandPoolType::generator_type gen_type;
/* GLOBAL_VECTORS */
//EXTERN int      V;
EXTERN double  rand_max;
EXTERN int endian;
EXTERN int Npfileds;
constexpr double PI  =3.141592653589793238463;
constexpr double twoPI  = 2.0*PI;
//EXTERN int    hop[V][2*D];
//EXTERN int    ipt[V][D];

/* GLOBAL_STRUCTS */
/*ispt_parms_t ispt_parms;
hmd_parms_t hmd_parms;
smd_parms_t smd_parms;
act_parms_t act_parms;
*/


#endif
