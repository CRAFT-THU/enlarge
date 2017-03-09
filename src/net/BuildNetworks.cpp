
#include <assert.h>

#include "../utils/utils.h"
#include "../utils/TypeFunc.h"
#include "MultiNetwork.h"

//int MultiNetwork::addConnectionInfo(ID nID, int nid, int offset, int *delayStart, int *delayNum, int *pSynapsesIdx, int nodeIdx)
//{
//	int count = 0;
//	map<ID, vector<ID>>::iterator n2siter = network->n2sNetwork.find(nID);
//	if (n2siter == network->n2sNetwork.end()) {
//		for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
//			delayStart[delay_t + network->maxDelaySteps*nid] = count + offset; 
//			delayNum[delay_t + network->maxDelaySteps*nid] = 0;
//		}
//
//		return count;
//	}
//
//	int synapsesNum_t = n2siter->second.size();
//	for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
//		delayStart[delay_t + network->maxDelaySteps*nid] = offset + count;
//		for (int i=0; i<synapsesNum_t; i++) {
//			if (network->id2synapse[n2siter->second.at(i)]->getDelay() == delay_t+1) {
//				map<ID, int>::iterator sid2idxiter = network->sid2idx.find(n2siter->second.at(i));
//				assert(sid2idxiter != network->sid2idx.end());
//				if (sid2node[sid2idxiter->first] == nodeIdx) {
//					int sid = sid2idxiter->second;
//					pSynapsesIdx[offset + count] = sid;
//					count++;
//				}
//			}
//		}
//		delayNum[delay_t + network->maxDelaySteps*nid] = offset + count - delayStart[delay_t + network->maxDelaySteps*nid];
//	}
//
//	return count;
//}

void MultiNetwork::countTypeNum(int nodeNum, bool autoSplited) 
{
	_globalNTypeInfo.resize(nodeNum);
	_globalSTypeInfo.resize(nodeNum);

	vector<PopulationBase*>::iterator piter;
	vector<NeuronBase*>::iterator niter;
	vector<SynapseBase*>::iterator siter;

	for (piter = network->pPopulations.begin(); piter != network->pPopulations.end();  piter++) {
		PopulationBase * p = *piter;
		Type type = p->getType();
		int node = p->getNode();
		if (autoSplited) {
			node = _nID2node[p->getNeuron(0)->getID()];
		}

		if (_globalNTypeInfo[node].find(type) == _globalNTypeInfo[node].end()) {
			_globalNTypeInfo[node][type] = p->getNum();
		} else {
			_globalNTypeInfo[node][type] += p->getNum();
		}
	}

	for (niter = network->pNeurons.begin(); niter != network->pNeurons.end();  niter++) {
		NeuronBase * p = *niter;
		Type type = p->getType();
		int node = p->getNode();
		if (autoSplited) {
			node = _nID2node[(*niter)->getID()];
		}

		if (_globalNTypeInfo[node].find(type) == _globalNTypeInfo[node].end()) {
			_globalNTypeInfo[node][type] = 1;
		} else {
			_globalNTypeInfo[node][type] += 1;
		}
	}

	for (siter = network->pSynapses.begin(); siter != network->pSynapses.end();  siter++) {
		SynapseBase * p = *siter;
		Type type = p->getType();
		int node = p->getNode();
		if (autoSplited) {
			node = _sID2node[(*siter)->getID()];
		}

		if (_globalSTypeInfo[node].find(type) == _globalSTypeInfo[node].end()) {
			_globalSTypeInfo[node][type] = 1;
		} else {
			_globalSTypeInfo[node][type] += 1;
		}
	}
}


DistriNetwork* MultiNetwork::buildNetworks(int nodeNum, bool autoSplited)
{
	this->nodeNum = nodeNum;
	_globalIdx2nID.resize(nodeNum);
	_globalIdx2sID.resize(nodeNum);

	vector<PopulationBase*>::iterator piter;
	vector<NeuronBase*>::iterator niter;
	vector<SynapseBase*>::iterator siter;

	vector<map<ID, int> > nodeNid2idx(nodeNum);

	if (autoSplited) {
		splitNetwork(nodeNum);
	}

	GNetwork *ret = (GNetwork*)malloc(sizeof(GNetwork)*nodeNum);
	assert(ret != NULL);

	crossNodeMap = (CrossNodeMap*)malloc(sizeof(CrossNodeMap)*nodeNum);
	assert(crossNodeMap != NULL);

	for (int i=0; i<nodeNum; i++) {
		crossNodeMap[i].idx2index = NULL;
		crossNodeMap[i].crossNodeMap = NULL;
	}

	crossNodeData = (CrossNodeData*)malloc(sizeof(CrossNodeData)*nodeNum*nodeNum);
	assert(crossNodeData != NULL);

	for (int i=0; i<nodeNum*nodeNum; i++) {
		crossNodeData[i].maxNeuronNum = 0;
		crossNodeData[i].firedNeuronNum = 0;
		crossNodeData[i].firedNeuronIdx = NULL;
	}

	for (int nodeIdx = 0; nodeIdx < nodeNum; nodeIdx++) {
		int nodeNeuronTypeNum = globalNTypeInfo[nodeIdx].size();
		int nodeSynapseTypeNum = globalSTypeInfo[nodeIdx].size();
		void **pAllNeurons = (void**)malloc(sizeof(void*)*nodeNeuronTypeNum);
		assert(pAllNeurons != NULL);
		void **pAllSynapses = (void**)malloc(sizeof(void*)*nodeSynapseTypeNum);
		assert(pAllSynapses != NULL);
		N2SConnection *pAllConnections = (N2SConnection*)malloc(sizeof(N2SConnection));
		assert(pAllConnections != NULL);

		int *pNeuronsNum = (int*)malloc(sizeof(int)*(nodeNeuronTypeNum + 1));
		assert(pNeuronsNum != NULL);
		int *pSynapsesNum = (int*)malloc(sizeof(int)*(nodeSynapseTypeNum + 1));
		assert(pSynapsesNum != NULL);
		pNeuronsNum[0] = 0;
		pSynapsesNum[0] = 0;

		Type *pNTypes = (Type*)malloc(sizeof(Type)*nodeNeuronTypeNum);
		assert(pNTypes != NULL);
		Type *pSTypes = (Type*)malloc(sizeof(Type)*nodeSynapseTypeNum);
		assert(pSTypes != NULL);

		map<int, ID> nodeIdx2nid;
		map<int, ID> nodeIdx2sid;

		int index = 0;
		for (map<Type, int>::const_iterator tmp_iter = globalNTypeInfo[nodeIdx].begin(); tmp_iter != globalNTypeInfo[nodeIdx].end(); tmp_iter++) {
			Type type = tmp_iter->first;
			pNTypes[index] = tmp_iter->first;

			void *pN = createType[type]();
			assert(pN != NULL);
			allocType[type](pN, tmp_iter->second);

			int idx = 0;
			for (piter = network->pPopulations.begin(); piter != network->pPopulations.end();  piter++) {
				PopulationBase * p = *piter;
				int node = p->getNode();
				if (autoSplited) {
					node = nid2node[p->getNeuron(0)->getID()];
				}

				if (node == nodeIdx && p->getType() == type) {
					size_t copied = p->hardCopy(pN, idx, pNeuronsNum[index], network->nid2idx, nodeIdx2nid);
					idx += copied;
				}
			}
			for (niter = network->pNeurons.begin(); niter != network->pNeurons.end();  niter++) {
				NeuronBase * p = *niter;
				int node = p->getNode();
				if (autoSplited) {
					node = nid2node[(*niter)->getID()];
				}
				if (node == nodeIdx && p->getType() == type) {
					size_t copied = p->hardCopy(pN, idx, pNeuronsNum[index], network->nid2idx, nodeIdx2nid);
					idx += copied;
				}
			}

			assert(idx == tmp_iter->second);
			pNeuronsNum[index+1] = idx + pNeuronsNum[index];
			pAllNeurons[index] = pN;
			index++;
		}
		globalIdx2nid[nodeIdx] = nodeIdx2nid;

		index = 0;
		for (map<Type, int>::const_iterator tmp_iter = globalSTypeInfo[nodeIdx].begin(); tmp_iter != globalSTypeInfo[nodeIdx].end(); tmp_iter++) {
			Type type = tmp_iter->first;
			pSTypes[index] = type;

			void *pS = createType[type]();
			assert(pS != NULL);
			allocType[type](pS, tmp_iter->second);

			int idx = 0;
			for (siter = network->pSynapses.begin(); siter != network->pSynapses.end();  siter++) {
				SynapseBase * p = *siter;
				int node = p->getNode();
				if (autoSplited) {
					node = sid2node[(*siter)->getID()];
				}

				if (node == nodeIdx && p->getType() == type) {
					int copied = p->hardCopy(pS, idx, pSynapsesNum[index], network->sid2idx, nodeIdx2sid);
					idx += copied;
				}
			}

			assert(idx == tmp_iter->second); 
			pSynapsesNum[index+1] = idx + pSynapsesNum[index];
			pAllSynapses[index] = pS;
			index++;
		}
		globalIdx2sid[nodeIdx] = nodeIdx2sid;

		int nodeNeuronNum = pNeuronsNum[nodeNeuronTypeNum];
		int nodeSynapseNum = pSynapsesNum[nodeSynapseTypeNum];

		for (map<ID, set<int> >::iterator tmp_iter = crossNodeInfo.begin(); tmp_iter != crossNodeInfo.end(); tmp_iter++) {	
			if (tmp_iter->second.find(nodeIdx) != tmp_iter->second.end()) {
				int size = nodeNid2idx[nodeIdx].size();
				nodeNid2idx[nodeIdx][tmp_iter->first] = nodeNeuronNum + size;
			}
		}

		int crossNodeNeuronNum = nodeNid2idx[nodeIdx].size();


		int *delayNum = (int*)malloc(sizeof(int)*(this->network->maxDelaySteps)*(nodeNeuronNum + crossNodeNeuronNum));
		assert(delayNum != NULL);
		int *delayStart = (int*)malloc(sizeof(int)*(this->network->maxDelaySteps)*(nodeNeuronNum + crossNodeNeuronNum));
		assert(delayStart != NULL);
		int *pSynapsesIdx = (int*)malloc(sizeof(int)*nodeSynapseNum);
		assert(pSynapsesIdx != NULL);

		int synapseIdx = 0;
		for (int nid=0; nid<nodeNeuronNum; nid++) {
			map<int, ID>::iterator iter = nodeIdx2nid.find(nid);
			assert(iter != nodeIdx2nid.end());
			
			map<ID, vector<ID>>::iterator n2siter = network->n2sNetwork.find(iter->second);
			if (n2siter == network->n2sNetwork.end()) {
				for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
					delayStart[delay_t + network->maxDelaySteps*nid] = synapseIdx;
					delayNum[delay_t + network->maxDelaySteps*nid] = 0;
				}
				continue;
			}

			int synapsesNum_t = n2siter->second.size();
			for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
				delayStart[delay_t + network->maxDelaySteps*nid] = synapseIdx;
				for (int i=0; i<synapsesNum_t; i++) {
					if (network->id2synapse[n2siter->second.at(i)]->getDelay() == delay_t+1) {
						map<ID, int>::iterator sid2idxiter = network->sid2idx.find(n2siter->second.at(i));

						if(sid2idxiter != network->sid2idx.end() && sid2node[sid2idxiter->first] == nodeIdx) {
							int sid = sid2idxiter->second;
							assert(synapseIdx < nodeSynapseNum);
							pSynapsesIdx[synapseIdx] = sid;
							synapseIdx++;
						}
					}
				}
				delayNum[delay_t + network->maxDelaySteps*nid] = synapseIdx - delayStart[delay_t + network->maxDelaySteps*nid];
			}
		}

		for (map<ID, int>::const_iterator tmp_iter = nodeNid2idx[nodeIdx].begin(); tmp_iter != nodeNid2idx[nodeIdx].end(); tmp_iter++) {
			ID nID = tmp_iter->first; 
			int nid = tmp_iter->second;
			map<ID, vector<ID>>::iterator n2siter = network->n2sNetwork.find(nID);
			if (n2siter == network->n2sNetwork.end()) {
				for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
					delayStart[delay_t + network->maxDelaySteps*nid] = synapseIdx;
					delayNum[delay_t + network->maxDelaySteps*nid] = 0;
				}
				continue;
			}

			int synapsesNum_t = n2siter->second.size();
			for (int delay_t=0; delay_t < network->maxDelaySteps; delay_t++) {
				delayStart[delay_t + network->maxDelaySteps*nid] = synapseIdx;
				for (int i=0; i<synapsesNum_t; i++) {
					if (network->id2synapse[n2siter->second.at(i)]->getDelay() == delay_t+1) {
						map<ID, int>::iterator sid2idxiter = network->sid2idx.find(n2siter->second.at(i));
						assert(sid2idxiter != network->sid2idx.end());

						if(sid2node[sid2idxiter->first] == nodeIdx) {
							int sid = sid2idxiter->second;
							assert(synapseIdx < nodeSynapseNum);
							pSynapsesIdx[synapseIdx] = sid;
							synapseIdx++;
						}
					}
				}
				delayNum[delay_t + network->maxDelaySteps*nid] = synapseIdx - delayStart[delay_t + network->maxDelaySteps*nid];
			}
		}

		pAllConnections->pSynapsesIdx = pSynapsesIdx;
		pAllConnections->delayStart = delayStart;
		pAllConnections->delayNum = delayNum;

		for (int i=0; i<nodeSynapseTypeNum; i++) {
			int *pSynapsesDst = (int*)malloc(sizeof(int)*(pSynapsesNum[i+1] - pSynapsesNum[i]));
			assert(pSynapsesDst != NULL);
			map<ID, ID>::iterator s2nIter;
			for (s2nIter = network->s2nNetwork.begin(); s2nIter != network->s2nNetwork.end(); s2nIter++) {
				map<ID, int>::iterator iter = network->sid2idx.find(s2nIter->first);
				if ((iter == network->sid2idx.end()) || (i != getType(pSynapsesNum, nodeSynapseTypeNum, iter->second))) {
					continue;
				}
				int idx = getOffset(pSynapsesNum, nodeSynapseTypeNum, iter->second);
				iter = network->nid2idx.find(s2nIter->second);
				assert(iter != network->nid2idx.end());
				pSynapsesDst[idx] = iter->second;
			}

			addTypeConnection[pSTypes[i]](pAllSynapses[i], pSynapsesDst);
		}

		ret[nodeIdx].pNeurons = pAllNeurons;
		ret[nodeIdx].pSynapses = pAllSynapses;
		ret[nodeIdx].pN2SConnection = pAllConnections;

		ret[nodeIdx].nTypeNum = nodeNeuronTypeNum;
		ret[nodeIdx].sTypeNum = nodeSynapseTypeNum;
		ret[nodeIdx].nTypes = pNTypes;
		ret[nodeIdx].sTypes = pSTypes;
		ret[nodeIdx].neuronNums = pNeuronsNum;
		ret[nodeIdx].synapseNums = pSynapsesNum;

		ret[nodeIdx].MAX_DELAY = network->maxDelaySteps;

		crossNodeIdx2Idx.resize(nodeNum);
		for (map<ID, set<int> >::const_iterator tmp_iter = crossNodeInfo.begin(); tmp_iter != crossNodeInfo.end(); tmp_iter++) {
			int node = nid2node[tmp_iter->first];
			int idx = network->nid2idx[tmp_iter->first];
			map<int, vector<int> >::const_iterator tmp_iter2 = crossNodeIdx2Idx[node].find(idx);
			if (tmp_iter2 == crossNodeIdx2Idx[node].end()) {
				crossNodeIdx2Idx[node][idx].resize(nodeNum, -1);
			}

			for (set<int>::const_iterator tmp_iter3 = tmp_iter->second.begin(); tmp_iter3 != tmp_iter->second.end(); tmp_iter3++) {
				crossNodeIdx2Idx[node][idx][*tmp_iter3] = nodeNid2idx[*tmp_iter3][tmp_iter->first];
			}
		}

		crossNodeMap[nodeIdx].idx2index = (int*)malloc(sizeof(int)*nodeNeuronNum);
		assert(crossNodeMap[nodeIdx].idx2index != NULL);
		crossNodeMap[nodeIdx].crossNodeMap = (int*)malloc(sizeof(int)*crossNodeIdx2Idx[nodeIdx].size()*nodeNum);
		assert(crossNodeMap[nodeIdx].crossNodeMap != NULL);
		for (int tmp = 0; tmp < nodeNeuronNum; tmp++) {
			crossNodeMap[nodeIdx].idx2index[tmp] = -1;
		}

		int count_idx = 0;
		for (map<int, vector<int> >::const_iterator tmp_iter = crossNodeIdx2Idx[nodeIdx].begin(); tmp_iter != crossNodeIdx2Idx[nodeIdx].end(); tmp_iter++) {
			crossNodeMap[nodeIdx].idx2index[tmp_iter->first] = count_idx;
			for (int tmp =0; tmp<nodeNum; tmp++) {
				crossNodeMap[nodeIdx].crossNodeMap[count_idx*nodeNum + tmp] = tmp_iter->second[tmp];
			}
		}

		
		for (int tmp = 0; tmp < nodeNum; tmp++) {
			// nodeIdx to tmp 
			int tmp_idx = tmp * nodeNum + nodeIdx;
			crossNodeData[tmp_idx].maxNeuronNum = 0;
			for (map<int, vector<int> >::const_iterator tmp_iter = crossNodeIdx2Idx[nodeIdx].begin(); tmp_iter != crossNodeIdx2Idx[nodeIdx].end(); tmp_iter++) {
				if (tmp_iter->second[tmp] != -1) {
					crossNodeData[tmp_idx].maxNeuronNum++;
				}
			}
			if (crossNodeData[tmp_idx].maxNeuronNum > 0) {
				crossNodeData[tmp_idx].firedNeuronIdx = (int*)malloc(sizeof(int)*crossNodeData[tmp_idx].maxNeuronNum);
				assert(crossNodeData[tmp_idx].firedNeuronIdx != NULL);
			}
		}
	}

	for (int i=0; i<nodeNum; i++) {
		int tmp_idx = i * nodeNum + i;
		for (int j=0; j<nodeNum; j++) {
			if (j != i) {
				crossNodeData[tmp_idx].maxNeuronNum += crossNodeData[i*nodeNum + j].maxNeuronNum;
			}
		}

		if (crossNodeData[tmp_idx].maxNeuronNum > 0) {
			crossNodeData[tmp_idx].firedNeuronIdx = (int*)malloc(sizeof(int)*crossNodeData[tmp_idx].maxNeuronNum);
			assert(crossNodeData[tmp_idx].firedNeuronIdx != NULL);
		}
	}

	return ret;
}
