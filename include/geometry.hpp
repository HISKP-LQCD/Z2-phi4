#ifndef GEOMETRY_H
#define GEOMETRY_H

#ifndef LATTICE_H
#include "lattice.hpp"
#endif

/* HOPPING_C */
void hopping(const int *L , Kokkos::View<size_t**> &hop, ViewLatt &sectors,  Kokkos::View<size_t**> &ipt );

/* LEX2C_C */
//extern int  c2lex(int *c, int d,int l);
//extern int* lex2c(int k, int d,int l);

#endif
