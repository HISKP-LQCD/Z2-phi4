#define LEX2C_C

#include <stdio.h>
#include <stdlib.h>
#include "geometry.hpp"

/**********************************
to compie:
gcc -D DEBUG lex2c.c -o lex2c

**********************************/

int c2lex(int *c, int d,int l)
{
   int m,j;

   m=c[d-1];
   for(j=d-2;j>=0;j--)
   {
    m*=l;
    m+=c[j];
  }

  return m;
}

/******************************/

int* lex2c(int k, int d,int l)
{
   int j,kp;
   int* c=(int*) malloc(sizeof(int)*d);
    
   kp=k;
   for(j=d-1;j>=0;j--)
   {
      c[d-1-j]=kp%l;
      kp/=l;	 
   }

   return c;
}

#ifdef DEBUG

int main(int argc,char** argv)
{
   int  *coordinates,j,k,d,l,i,j0;
   
   if(argc!=3)
   {
      printf("lex2c d l\n");
      error(EXIT_FAILURE);
   }

   sscanf(argv[1],"%d",&d);
   sscanf(argv[2],"%d",&l);

   j0=1;   
   for(i=0;i<d;i++)
	   j0*=l;
	
   for(j=0;j<j0;j++)
   {
      coordinates=lex2c(j,d,l);
		printf("j=%d [",j);

		for(i=0;i<d;i++) printf("%d,",coordinates[i]);
		k=c2lex(coordinates,d,l);
		printf("] =%d\n",k);

		if(j!=k)
      {
			printf("error\n");
		   error(EXIT_FAILURE);
		}

		free(coordinates);
	}

   return EXIT_SUCCESS;
}

#endif
