/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef HELPER_TYPE_H
#define HELPER_TYPE_H

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

// typedef unsigned long long uinteger_t;
// #define UINTEGER_T_MAX ULL_MAX;
// #define MPI_UINTEGER_T MPI_UNSIGNED_LONG_LONG

#ifndef INTEGER_T
#define INTEGER_T

typedef unsigned int uinteger_t;
#define UINTEGER_T_MAX UINT_MAX
#define MPI_UINTEGER_T MPI_UNSIGNED
#define FT_UINTEGER_T "%u"
#define PT_UINTEGER_T(prev, next) prev FT_UINTEGER_T next

typedef int integer_t;
#define INTEGER_T_MAX INT_MAX
#define MPI_INTEGER_T MPI_INT
#define FT_INTEGER_T "%d" 
#define PT_INTEGER_T(prev, next) prev FT_INTEGER_T next

// #ifdef USE_GPU
#define NCCL_INTEGER_T ncclInt32
#define NCCL_UINTEGER_T ncclUint32 
// #endif

#endif // INTEGER_T


#ifndef USE_DOUBLE
typedef float real;
#else
typedef double real;
#endif

#ifndef USE_DOUBLE
#define MPI_U_REAL MPI_FLOAT
#else
#define MPI_U_REAL MPI_DOUBLE
#endif

#if SIZE_MAX == UCHAR_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "unsupported size_t"
#endif

// const real ZERO = 1e-10;

#endif /* HELPER_TYPE_H */

