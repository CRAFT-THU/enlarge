#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "helper.h"
#include "units.h"

using namespace std;
using namespace std::chrono;

const int numNeuronsMax = 200;
const int numLipExcA = 200;
const int numLipExcB = 200;
const int numLipInh = 100;

class ColorDecision {
   public:
    // float scale = 1.0;

    float V_init = -70.0;

    float poissonNoiseFreq = 1000.0;

    float g_noise_E = 2;
    float g_noise_I = 3;

    float g_EE_AMPA = 0.01;
    float g_EI_AMPA = 0.01;

    float g_IE = 0.8;
    float g_II = 0.3;

    float g_EE_NMDA = 0.5;
    float g_EI_NMDA = 0.3;

    float E_syn_AMPA = 0.0;
    float E_syn_NMDA = 0.0;
    float Mg = 1.0;
    float E_syn_GABA = -70.0;

    float tau_AMPA = 2.0;
    float tau_GABA = 5.0;
    float tau_rise_NMDA = 2.0;
    float tau_decay_NMDA = 100;

    float dt = 1.0;

    // 定义神经元群
    vector<LIFNeuron> lipPopExcA;  // LIP脑区兴奋性神经元群A
    vector<LIFNeuron> lipPopExcB;  // LIP脑区兴奋性神经元群B
    vector<LIFNeuron> lipPopInh;   // LIP脑区抑制性神经元群

    // 定义突触, 其中 '==>': AMPA, '-=>': NDMA, '-->': GABA
    vector<ExponentialSynapse> lipExcA2ExcASyn_AMPA;  // A ==> A
    vector<ExponentialSynapse> lipExcA2InhSyn_AMPA;   // A ==> I

    vector<NMDASynapse> lipExcA2ExcASyn_NMDA;  // A -=> A
    vector<NMDASynapse> lipExcA2InhSyn_NMDA;   // A -=> I

    vector<ExponentialSynapse> lipExcB2ExcBSyn_AMPA;  // B ==> B
    vector<ExponentialSynapse> lipExcB2InhSyn_AMPA;   // B ==> I

    vector<NMDASynapse> lipExcB2ExcBSyn_NMDA;  // B -=> B
    vector<NMDASynapse> lipExcB2InhSyn_NMDA;   // B -=> I

    vector<ExponentialSynapse> lipInh2ExcASyn_GABA;  // I --> A
    vector<ExponentialSynapse> lipInh2ExcBSyn_GABA;  // I --> B
    vector<ExponentialSynapse> lipInh2InhSyn_GABA;   // I --> I

    ColorDecision() {
        initializeNeurons();
        createSynapses();

        // cout << lipPopExcA.size() << endl;
        // cout << lipPopExcB.size() << endl;
        // cout << lipPopInh.size() << endl;

        // cout << lipPopExcA[0].getMembranePotential() << endl;
        // cout << lipPopExcB[0].tau_ref << endl;

        // cout << lipExcA2InhSyn_AMPA.size() << endl;
        // cout << lipExcA2InhSyn_AMPA[6].pre->getMembranePotential() << endl;
        // cout << lipExcB2ExcBSyn_NMDA[5].Mg << endl;
        // cout << lipInh2ExcASyn_GABA[0].E_syn << endl;
        // cout << "Init complete!" << endl;
    }

    void initializeNeurons() {
        // 初始化神经元
        for (int i = 0; i < numLipExcA; i++) {
            lipPopExcA.emplace_back(
                LifParams{lifParamInitDefault.tau_m, lifParamInitDefault.V_rest,
                          lifParamInitDefault.V_reset, lifParamInitDefault.V_th,
                          lifParamInitDefault.R,
                          lifParamInitDefault.t_refractory, V_init});
            lipPopExcB.emplace_back(
                LifParams{lifParamInitDefault.tau_m, lifParamInitDefault.V_rest,
                          lifParamInitDefault.V_reset, lifParamInitDefault.V_th,
                          lifParamInitDefault.R,
                          lifParamInitDefault.t_refractory, V_init});
            if (i < 100) {
                lipPopInh.emplace_back(LifParams{
                    lifParamInitDefault.tau_m, lifParamInitDefault.V_rest,
                    lifParamInitDefault.V_reset, lifParamInitDefault.V_th,
                    lifParamInitDefault.R, lifParamInitDefault.t_refractory,
                    V_init});
            }
        }
    }

    void createSynapses() {
        // lipPopExcA自连接, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcA) {
            for (auto& post_neuron : lipPopExcA) {
                lipExcA2ExcASyn_AMPA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_EE_AMPA, E_syn_AMPA,
                              tau_AMPA});  // AMPASynapse 参数
                lipExcA2ExcASyn_NMDA.emplace_back(SynParams{
                    &pre_neuron, &post_neuron, g_EE_NMDA, tau_rise_NMDA,
                    tau_decay_NMDA, E_syn_NMDA, Mg});  // NMDASynapse 参数
            }
        }

        // lipPopExcA -> lipPopInh, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcA) {
            for (auto& post_neuron : lipPopInh) {
                lipExcA2InhSyn_AMPA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_EI_AMPA, E_syn_AMPA,
                              tau_AMPA});  // AMPASynapse 参数
                lipExcA2InhSyn_NMDA.emplace_back(SynParams{
                    &pre_neuron, &post_neuron, g_EI_NMDA, tau_rise_NMDA,
                    tau_decay_NMDA, E_syn_NMDA, Mg});  // NMDASynapse 参数
            }
        }

        // lipPopExcB自连接, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcB) {
            for (auto& post_neuron : lipPopExcB) {
                lipExcB2ExcBSyn_AMPA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_EE_AMPA, E_syn_AMPA,
                              tau_AMPA});  // AMPASynapse 参数
                lipExcB2ExcBSyn_NMDA.emplace_back(SynParams{
                    &pre_neuron, &post_neuron, g_EE_NMDA, tau_rise_NMDA,
                    tau_decay_NMDA, E_syn_NMDA, Mg});  // NMDASynapse 参数
            }
        }

        // lipPopExcB -> lipPopInh, AMPA & NMDA
        for (auto& pre_neuron : lipPopExcB) {
            for (auto& post_neuron : lipPopInh) {
                lipExcB2InhSyn_AMPA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_EI_AMPA, E_syn_AMPA,
                              tau_AMPA});  // AMPASynapse 参数
                lipExcA2ExcASyn_NMDA.emplace_back(SynParams{
                    &pre_neuron, &post_neuron, g_EI_NMDA, tau_rise_NMDA,
                    tau_decay_NMDA, E_syn_NMDA, Mg});  // NMDASynapse 参数
            }
        }

        // lipPopInh -> lipPopExcA, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopExcA) {
                lipInh2ExcASyn_GABA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_IE, E_syn_GABA,
                              tau_GABA});  // GABASynapse 参数
            }
        }

        // lipPopInh -> lipPopExcB, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopExcB) {
                lipInh2ExcBSyn_GABA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_IE, E_syn_GABA,
                              tau_GABA});  // GABASynapse 参数
            }
        }

        // lipPopInh自连接, GABA
        for (auto& pre_neuron : lipPopInh) {
            for (auto& post_neuron : lipPopInh) {
                lipInh2InhSyn_GABA.emplace_back(
                    SynParams{&pre_neuron, &post_neuron, g_II, E_syn_GABA,
                              tau_GABA});  // GABASynapse 参数
            }
        }
    }

    void update(int perStep) {
        // FIXME: Invalid records after t = 0
        log_nrn_volt(lipPopExcA, "lipPopExcA", 0);
        log_nrn_volt(lipPopExcB, "lipPopExcB", 0);
        log_nrn_volt(lipPopInh, "lipPopInh", 0);
        for (int i = 0; i < perStep; i++)  // f=jax.numpy.array(t*fps/1000,int)
        {
            // 突触更新
            // lipPopExcA自连接突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcA2ExcASyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcA2ExcASyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcA -> lipPopInh突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcA2InhSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcA2InhSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcB自连接突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcB2ExcBSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcB2ExcBSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopExcB -> lipPopInh突触更新, AMPA & NMDA
            for (auto& ampaSyn : lipExcB2InhSyn_AMPA) {
                ampaSyn.update(dt);
            }
            for (auto& nmdaSyn : lipExcB2InhSyn_NMDA) {
                nmdaSyn.update(dt);
            }

            // lipPopInh -> lipPopExcA突触更新, GABA
            for (auto& gabaSyn : lipInh2ExcASyn_GABA) {
                gabaSyn.update(dt);
            }

            // lipPopInh -> lipPopExcB突触更新, GABA
            for (auto& gabaSyn : lipInh2ExcBSyn_GABA) {
                gabaSyn.update(dt);
            }

            // lipPopInh自连接突触更新, GABA
            for (auto& gabaSyn : lipInh2InhSyn_GABA) {
                gabaSyn.update(dt);
            }

            // 神经元更新
            for (auto& neuronA : lipPopExcA) {
                neuronA.update(dt);
            }
            for (auto& neuronB : lipPopExcB) {
                neuronB.update(dt);
            }
            for (auto& neuronI : lipPopInh) {
                neuronI.update(dt);
            }
        }
    }
};
int main() {
    cout << "ColorDecision model init!" << endl;
    auto start_init = high_resolution_clock::now();
    ColorDecision cd;
    auto end_init = high_resolution_clock::now();
    auto dur_init = duration_cast<microseconds>(end_init - start_init);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model init: " << dur_init.count() / 1000.0 << " ms"
         << endl;

    int stimFrame = 1000 / 9;
    int perStep = 9;
    auto start_update = high_resolution_clock::now();
    for (int i = 0; i < stimFrame; i++) {
        cd.update(perStep);
    }

    auto end_update = high_resolution_clock::now();
    auto dur_update = duration_cast<microseconds>(end_update - start_update);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model update: " << dur_update.count() / 1000.0
         << " ms" << endl;
}
