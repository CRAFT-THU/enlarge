/* This program is writen by qp09.
 * usually just for fun.
 * Mon March 14 2016
 */

#include <iostream>
#include "utils.h"
#include "proc_info.h"

void print_mem(const char *info)
{
	proc_info pinfo;
	get_proc_info(&pinfo);
	printf("%s, MEM used: %lfGB\n", info, static_cast<double>(pinfo.mem_used/1024.0/1024.0));
}

double realRandom(double range)
{
	long f = rand();
	return ((double)f/RAND_MAX)*range;
}

//int id2idx(ID* array, int num, ID id) {
//	for (int i=0; i<num; i++) {
//		if (array[i] == id) {
//			return i;
//		}
//	}
//	printf("ERROR: Cannot find ID!!!\n");
//	return 0;
//}

int getIndex(Type *array, int size, Type type)
{
	for (int i=0; i<size; i++) {
		if (array[i] == type) {
			return i;
		}
	}

	//printf("ERROR: Cannot find type %d !!!\n", type);
	return -1;
}

int getType(int *array, int size, int index)
{
	for (int i=0; i<size; i++) {
		if (array[i+1] > index) {
			return i;
		}
	}

	//printf("ERROR: Cannot find index %d !!!\n", index);
	return -1;
}

int getOffset(int *array, int size, int index)
{
	for (int i=0; i<size; i++) {
		if (array[i+1] > index) {
			return (index - array[i]);
		}
	}

	//printf("ERROR: Cannot find index %d !!!\n", index);
	return -1;
}

Json::Value testValue(Json::Value value, unsigned int idx)
{
	if (value.type() == Json::nullValue) {
		return 0;
	}

	if (value.type() == Json::arrayValue) {
		if (idx < value.size()) {
			return value[idx];
		} else {
			std::cout  << "Not enough parameters:" << value << "@" << idx << std::endl;
		}
	} 

	return value;
}

real *loadArray(const char *filename, int size)
{
	real *res = (real*)malloc(sizeof(real) * size);
	FILE *logFile = fopen(filename, "rb+");
	if (logFile == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
		return res;
	}
	fread(res, sizeof(real), size, logFile);

	fflush(logFile);
	fclose(logFile);

	return res;
}

int saveArray(const char *filename, real *array, int size)
{
	FILE *logFile = fopen(filename, "wb+");
	if (logFile == NULL) {
		printf("ERROR: Open file %s failed\n", filename);
		return -1;
	}
	fwrite(array, sizeof(real), size, logFile);
	fflush(logFile);
	fclose(logFile);

	return 0;
}
