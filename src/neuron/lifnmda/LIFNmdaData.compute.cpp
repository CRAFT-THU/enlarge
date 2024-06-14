#include <cmath>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "LIFNmdaData.h"

void updateLIFNmda(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time)
{
	LIFNmdaData *data = (LIFNmdaData *)_data;
	int currentIdx = time % (connection->maxDelay+1);
	for (size_t nid=0; nid<num; nid++) {
		size_t gnid = offset + nid; 

		if (data->pRefracStep[nid] <= 0) { // * 正常工作
            real V = data->pV[nid];
            data->pI[nid] += buffer[gnid] * (data->pE[nid] - V) / (1 + data->pM_c[nid] * exp(-0.062 * V));
            data->pV[nid] = (1 - data->pC_m[nid]) * data->pV[nid] + data->pC_m[nid] * (data->pV_tmp[nid] + data->pR[nid] * data->pI[nid]);

			bool fired = data->pV[nid] >= data->pV_thresh[nid];
			data->_fire_count[gnid] += fired;

			if (fired) {
				firedTable[firedTableSizes[currentIdx] + firedTableCap * currentIdx] = gnid;
				firedTableSizes[currentIdx]++;

				// data->pRefracStep[nid] = data->pRefracTime[nid] - 1;
                data->pRefracStep[nid] = data->pRefracTime[nid]; // ! 按需求严格打 time次拍 后再恢复
				data->pV[nid] = data->pV_reset[nid];
			}
		} else { // * 不应期
			data->pRefracStep[nid]--;
		}
		buffer[gnid] = 0;
		// buffer[num + gnid] = 0;
	}
}

