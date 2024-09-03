
#ifndef TEST_H
#define TEST_H

#include "../msg_utils/msg_utils.h"
#include "../msg_utils/CrossSpike.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

inline nid_t get_value(int d, int s_p, int s_t,  int d_p, int d_t, int d_n) {
	return 1000000*1 + 100000* d + 10000 * s_p + 1000 * s_t + 100 * d_p + 10 * d_t + d_n;
}

#endif
