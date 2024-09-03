
#define CATCH_CONFIG_RUNNER

#include <stdio.h>
#include <pthread.h>
#include <vector>

#include "catch.hpp"
#include "mpi.h"

#include "../helper/helper_c.h"
#include "../msg_utils/GPUManager.h"

using std::vector;

const int THREAD_NUM = 2;

int proc_rank = 0;
int thread_rank[THREAD_NUM] = {-1};
int device_id[THREAD_NUM] = {-1};

pthread_barrier_t bar;

void * run_thread(void * para)
{
	long long i = reinterpret_cast<long long>(para);
	thread_rank[i] = i;
	gm.set(i);
	gm.check();
	pthread_barrier_wait(&bar);
	device_id[i] = gm.get();
	gm.check();
	return NULL;
}

TEST_CASE("CHECK DEVICE", "") {
	CHECK_THAT(vector<int>(thread_rank, thread_rank + THREAD_NUM), 
			Catch::Equals(vector<int>({0, 1})));

	CHECK_THAT(vector<int>(device_id, device_id + THREAD_NUM), 
			Catch::Equals(vector<int>({0, 1})));
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int proc_num = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	pthread_barrier_init(&bar, NULL, THREAD_NUM);
	pthread_t *thread_ids = malloc_c<pthread_t>(THREAD_NUM);

	for (int i=0; i<THREAD_NUM; i++) {
		int ret = pthread_create(&(thread_ids[i]), NULL, &run_thread, (void*)i);
	}

	for (int i=0; i<THREAD_NUM; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	pthread_barrier_destroy(&bar);

	int result = Catch::Session().run( argc, argv );
	MPI_Finalize();
	return result;
}
