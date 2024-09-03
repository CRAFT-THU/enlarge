/* This header file is writen by qp09
 * usually just for fun
 * Sun December 13 2015
 */
#ifndef HELPER_ARRAY_C_H
#define HELPER_ARRAY_C_H

#include <stdio.h>
#include <math.h>
#include <typeinfo>

// #include "../third_party/json/json.h"
#include "helper_type.h"
#include "helper_c.h"

static bool rand_seed_inited = false;

template<typename T>
T real_rand(T range)
{
	long f = rand();
	return static_cast<T>((static_cast<double>(f/RAND_MAX)*range));
}

template<typename T>
long long get_index(T *array, size_t size, T elem)
{
	for (size_t i=0; i<size; i++) {
		if (array[i] == elem) {
			return i;
		}
	}

	return -1;
}

template<typename T>
long long get_near_index(T *array, size_t size, T elem)
{
	for (size_t i=0; i<size-1; i++) {
		if (array[i+1] > elem) {
			return i;
		}
	}

	//printf("ERROR: Cannot find index %d !!!\n", index);
	return -1;
}

template<typename T>
T get_diff(int *array, size_t size, T elem)
{
	for (size_t i=0; i<size-1; i++) {
		if (array[i+1] > elem) {
			return (elem - array[i]);
		}
	}

	return elem - array[size-1];
}

template<typename T>
T *load_array(const char *name, size_t size)
{
	FILE *f = fopen(name, "rb+");
	if (f == NULL) {
		printf("ERROR: Open file %s failed\n", name);
		return NULL;
	}

	T *res = malloc_c<T>(size);
	fread_c(res, size, f);

	fflush(f);
	fclose(f);

	return res;
}

template<typename T>
int save_array(const char *name, T *array, size_t size)
{
	FILE *f = fopen(name, "wb+");
	if (f == NULL) {
		printf("ERROR: Open file %s failed\n", name);
		return -1;
	}
	fwrite_c(array, size, f);
	fflush(f);
	fclose(f);

	return 0;
}

inline int upzero_else_set_one(int value) {
	if (value > 0) {
		return value;
	}

	return 1;
}

template<typename T>
T *get_rand_array(T range, size_t size) {
	if (!rand_seed_inited) {
		srand(time(NULL));
		rand_seed_inited = true;
	}

	T *res = malloc_c<T>(size);
	for (size_t i=0; i<size; i++) {
		res[i] = static_cast<T>(real_rand(static_cast<double>(range)));
	}

	return res;
}

template<typename T>
T *get_const_array(T value, size_t size)
{
	T *res = malloc_c<T>(size);
	for (size_t i=0; i<size; i++) {
		res[i] = value;
	}

	return res;
}

template<typename T>
T* free_array(T *value)
{
	free(value);
	return NULL;
}

template<typename T>
bool is_equal_array(T const & a, T const & b, size_t size)
{
	for (size_t i=0; i<size; i++) {
		if (fabs(a[i] - b[i]) > 1e-10) {
			return false;
		}
	}
	return true;
}

template<typename T, typename T1>
bool is_equal_array(T const & a, T const & b, size_t size, T1 *shuffle1=NULL, T1 *shuffle2=NULL)
{
	for (size_t i=0; i<size; i++) {
		if (shuffle1 != NULL && shuffle2 !=NULL)  {
			if (fabs(a[shuffle1[i]] - b[shuffle2[i]]) > 1e-10) {
				return false;
			}
		} else if (shuffle1 != NULL) {
			if (fabs(a[shuffle1[i]] - b[i]) > 1e-10) {
				return false;
			}
		} else {
			if (fabs(a[i] - b[i]) > 1e-10) {
				return false;
			}
		}
	}
	return true;
}

inline void log_array_noendl(FILE *f, int *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%d ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, unsigned int *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%u ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, long *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%ld ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, unsigned long *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lu ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, long long *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lld ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, unsigned long long *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%llu ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, float *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lf ", array[i]);
	}
}

inline void log_array_noendl(FILE *f, double *array, size_t size) {
	for (size_t i=0; i<size; i++) {
		fprintf(f, "%lf ", array[i]);
	}
}


template<typename T>
void log_array(FILE *f, T *array, size_t size, const char *msg="")
{
	fprintf(f, "%s ", msg);
	log_array_noendl(f, array, size);
	fprintf(f, "\n");
}

template<typename T>
void print_array(T *array, size_t size, const char *msg="")
{
	fprintf(stdout, "%s ", msg);
	log_array_noendl(stdout, array, size);
	fprintf(stdout, "\n");
}


template<typename T1, typename T2>
T1 *shuffle(T1 *array, T2 *idx, size_t size)
{
	T1 *ret = malloc_c<T1>(size);
	for (size_t i=0; i<size; i++) {
		ret[i] = array[idx[i]];
	}
	
	return ret;
}

template<typename T1, typename T2>
void shuffle(T1 * res, T1 *array, T2 *idx, size_t size)
{
	for (size_t i=0; i<size; i++) {
		res[i] = array[idx[i]];
	}
}

#endif /* HELPER_ARRAY_C_H */
