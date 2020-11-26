#ifndef GEOMETRY_H
#define GEOMETRY_H

#ifndef LATTICE_H
#include "lattice.hpp"
#endif

/* HOPPING_C */
void hopping(const int *L );

/* LEX2C_C */
extern int  c2lex(int *c, int d,int l);
extern int* lex2c(int k, int d,int l);

#endif
