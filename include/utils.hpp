
/*******************************************************************************
*
* File utils.h
*
* Copyright (C) 2014 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <limits.h>
#include <float.h>
//#include <gmp.h>

//#define NAME_SIZE 128

#if ((DBL_MANT_DIG!=53)||(DBL_MIN_EXP!=-1021)||(DBL_MAX_EXP!=1024))
#error : Machine is not compliant with the IEEE-754 standard
#endif

#if (SHRT_MAX==0x7fffffff)
typedef short int stdint_t;
typedef unsigned short int stduint_t;
#elif (INT_MAX==0x7fffffff)
typedef int stdint_t;
typedef unsigned int stduint_t;
#elif (LONG_MAX==0x7fffffff)
typedef long int stdint_t;
typedef unsigned long int stduint_t;
#else
#error : There is no four-byte integer type on this machine
#endif

#undef UNKNOWN_ENDIAN
#undef LITTLE_ENDIAN
#undef BIG_ENDIAN

#define UNKNOWN_ENDIAN 0
#define LITTLE_ENDIAN 1
#define BIG_ENDIAN 2

#undef IMAX
#define IMAX(n,m) ((n)+((m)-(n))*((m)>(n)))
/*
typedef struct
{
   int mx,n;
   mpq_t *c;
} cnum_t;

// CNUM_C 
extern cnum_t *alloc_cnum(int nc,int mx);
extern void free_cnum(int nc,cnum_t *cn);
extern void set_cnum(int n,long (*r)[2],cnum_t *cn);
extern void copy_cnum(cnum_t *cn1,cnum_t *cn2);
extern void add_cnum(cnum_t *cn1,cnum_t *cn2);
extern void sub_cnum(cnum_t *cn1,cnum_t *cn2);
extern void mul_cnum(cnum_t *cn1,cnum_t *cn2);
extern int is_zero_cnum(cnum_t *cn);
extern double eval_cnum(double x,cnum_t *cn);
extern int print_cnum(FILE *out,char *x,cnum_t *cn);
*/
/* ENDIAN_C */
extern int endianness(void);
extern void bswap_int(int n,void *a);
extern void bswap_double(int n,void *a);
void bswap_scalartype(int n,void *a, int size);

/* MUTILS_C */
extern int find_opt(int argc,char *argv[],char *opt);
extern int digits(double x,double dx,char *fmt);
extern int fdigits(double x);
extern void check_dir(char* dir);
extern int name_size(char *format,...);
extern long find_section(FILE *stream,char *title);
extern long read_line(FILE *stream,char *tag,char *format,...);
extern int count_tokens(FILE *stream,char *tag);
extern void read_iprms(FILE *stream,char *tag,int n,int *iprms);
extern void read_dprms(FILE *stream,char *tag,int n,double *dprms);

/* UTILS_C */
extern int safe_mod(int x,int y);
extern void *amalloc(size_t size,int p);
extern void afree(void *addr);
//extern void error(int test,int no,char *name,char *format,...);
//extern void error_root(int test,int no,char *name,char *format,...);
extern int error_loc(int test,int no,char *name,char *format,...);
extern void message(char *format,...);

/* BPOW_C */
extern int binary_power(int a, int b);

#endif
