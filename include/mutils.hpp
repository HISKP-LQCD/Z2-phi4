
/*******************************************************************************
*
* File mutils.h
*
* Copyright (C) 2005, 2007, 2008, 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utility functions used in main programs.
*
* The externally accessible functions are
*
*   int find_opt(int argc,char *argv[],char *opt)
*     On process 0, this program compares the string opt with the arguments
*     argv[1],..,argv[argc-1] and returns the position of the first argument
*     that matches the string. If there is no matching argument, or if the
*     program is called from another process, the return value is 0.
*
*   int digits(double x,double dx,char *fmt)
*     Assuming x is a value with error dx, this program returns the number n
*     of fractional digits to print so that all significant digits plus two
*     more are shown. The print format fmt has to be "e" or "f" depending on
*     whether the number is to be printed using the "%.ne" or "%.nf" format
*     string. In the second case dx has to be in the range 0<dx<1, and
*     (int)(10^n*dx) is then a two-digit integer that represents the error
*     in the last two digits of the printed value.
*
*   int fdigits(double x)
*     Returns the smallest integer n such that the value of x printed with
*     print format %.nf coincides with x up to a relative error at most a
*     few times the machine precision DBL_EPSILON.
*
*   void check_dir(char* dir)
*     This program checks whether the directory dir is accessible. An
*     error occurs if it is not.
*
*   int name_size(char *format,...)
*     On process 0, this program returns the length of the string that
*     would be printed by calling sprintf(*,format,...). The format
*     string can be any combination of literal text and the conversion
*     specifiers %s, %d and %.nf (where n is a positive integer). When
*     called on other processes, the program does nothing and returns
*     the value of NAME_SIZE.
*
*   long find_section(FILE *stream,char *title)
*     This program scans stream for a line starting with the string "[title]"
*     (after any number of blanks). It terminates with an error message if no
*     such line is found or if there are several of them. The program returns
*     the offset of the line from the beginning of the file and positions the
*     file pointer to the next line.
*
*   long read_line(FILE *stream,char *tag,char *format,...)
*     This program scans stream and reads a line of text in a controlled
*     manner, as described in the notes below. The tag can be the empty
*     string "" and must otherwise be an alpha-numeric word that starts
*     with a letter. If it is not empty, the program searches for the tag
*     in the current section. An error occurs if the tag is not found. The
*     program returns the offset of the line from the beginning of the file
*     and positions the file pointer to the next line.
*
*   int count_tokens(FILE *stream,char *tag)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, and returns the number of tokens found on that line after the tag.
*     Tokens are separated by white space (blanks, tabs or newline characters)
*     and comments (text beginning with #) are ignored. On exit, the file
*     pointer is positioned at the next line.
*
*   void read_iprms(FILE *stream,char *tag,int n,int *iprms)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, reads n integer values from that line after the tag and assigns
*     them to the elements of the array iprms. An error occurs if less than
*     n values are found on the line. The values must be separated by white
*     space (blanks, tabs or newline characters). On exit, the file pointer
*     is positioned at the next line.
*
*   void read_dprms(FILE *stream,char *tag,int n,double *dprms)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, reads n double values from that line after the tag and assigns
*     them to the elements of the array iprms. An error occurs if less than
*     n values are found on the line. The values must be separated by white
*     space (blanks, tabs or newline characters). On exit, the file pointer
*     is positioned at the next line.
*
* Notes:
*
* The programs find_section() and read_line() serve to read structured
* input parameter files (such as the *.in files in the directory main).
*
* Parameter lines that can be read by read_line() must be of the form
*
*   tag v1 v2 ...
*
* where v1,v2,... are data values (strings, integers or floating-point
* numbers) separated by blanks. If the tag is empty, the first data value
* may not be a string. Such lines are read by calling
*
*   read_line(tag,format,&var1,&var2,...)
*
* where var1,var2,... are the variables to which the values v1,v2,... are
* to be assigned. The format string must include the associated sequence
* of conversion specifiers %s, %d, %f or %lf without any modifiers. Other
* tokens are not allowed in the format string, except for additional blanks
* and a newline character at the end of the string (none of these have any
* effect).
*
* The programs find_section() and read_line() ignore blank lines and any text
* appearing after the character #. Lines longer than NAME_SIZE-1 characters are
* not permitted. Each section may occur at most once and, within each section,
* a line tag may not appear more than once. The number of characters written
* to the target string variables is at most NAME_SIZE-1. Buffer overflows are
* thus excluded if the target strings are of size NAME_SIZE or larger.
*
*******************************************************************************/
#ifndef mutils_H
#define mutils_H


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
//#include <gmp.h>

#define NAME_SIZE 1000

#if ((DBL_MANT_DIG!=53)||(DBL_MIN_EXP!=-1021)||(DBL_MAX_EXP!=1024))
#error : Machine is not compliant with the IEEE-754 standard
#endif


//static char line[NAME_SIZE+1];
//static char inum[3*sizeof(int)+4];


extern void error_root(int test,int no,char *name,char *format,...);
void error(int test,int no, const char *name,const char *format,...);
int find_opt(int argc,char *argv[],char *opt);
int digits(double x,double dx,char *fmt);
int fdigits(double x);
int name_size(char *format,...);
//static int cmp_text(char *text1,char *text2);
//static char *get_line(FILE *stream);
long find_section(FILE *stream,char *title);
//static void check_tag(char *tag);
//static long find_tag(FILE *stream,char *tag);
long read_line(FILE *stream,char *tag,char *format,...);
long myscanf(int n, char *format,...);
int count_tokens(FILE *stream,char *tag);
void read_iprms(FILE *stream,char *tag,int n,int *iprms);
void read_dprms(FILE *stream,char *tag,int n,double *dprms);
void go_to_line(FILE *stream,int line);
void move_line(FILE *stream,int line);


FILE *open_file(const char * name, const char * option);
void mysprintf(char *str, size_t size, const char *format, ...);
void gdb_hook();

#endif
