
#ifndef GPUMANAGER_H
#define GPUMANAGER_H

#include <atomic>

using std::atomic;

class GPUManager {
public:
	GPUManager();
	~GPUManager();

	int set(int id);
	int get();

	int lock();
	int unlock();

	void check();

private:
	static thread_local int _id;
	static thread_local atomic<char> _lock;
};

extern thread_local GPUManager gm;

#endif //GPUMANAGER_H
