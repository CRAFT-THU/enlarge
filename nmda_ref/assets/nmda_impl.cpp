#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

// LIF神经元初始化参数
struct LifParams {
    float V_rest = -70.0;      // rest voltage
    float V_reset = -75.0;     // reset voltage
    float V_th = -50.0;        // threshold voltage
    float tau_m = 7.5;         // time constant
    float t_refractory = 2.0;  // refractory period
    float R = 0.05;            // 电阻
    float V_init = -70.0;      // 初始膜电位
    float current_init = 0.0;  // 初始外部输入的电流
} lifParamInitDefault;

// LIF neuron model class
// 功能：神经元初始化，接收电流，神经元状元更新，神经元状态重置，获取当前发放情况，获取当前神经元膜电位
class LIFNeuron {
   public:
    float tau_m;            // 膜时间常数
    float V_rest;           // 静息电位
    float V_reset;          // 重置电位
    float V_th;             // 阈值电位
    float R;                // 膜电阻
    float V;                // 膜电位
    float tau_ref;          // 不应期时长
    float refractory_time;  // 当前不应期剩余时间
    float input_current;    // 输入电流
    bool spiked;            // 神经元是否发放

    // 默认构造函数
    // 无参数传入时神经元初始化使用以下参数
    LIFNeuron() {
        tau_m = lifParamInitDefault.tau_m;           // 膜时间常数
        V_rest = lifParamInitDefault.V_rest;         // 静息电位
        V_reset = lifParamInitDefault.V_reset;       // 重置电位
        V_th = lifParamInitDefault.V_th;             // 阈值电位
        R = lifParamInitDefault.R;                   // 膜电阻，电导倒数
        tau_ref = lifParamInitDefault.t_refractory;  // 不应期时长
        V = lifParamInitDefault.V_init;  // 膜电位，默认初始化为-70.0mV
        input_current = lifParamInitDefault.current_init;  // 初始外部输入电流

        refractory_time =
            0.0;  // 当前不应期剩余时间，默认初始化为0ms(一开始神经元未处于不应期)
        spiked = false;  // 神经元是否发放
    }

    // 构造函数
    // 自定义神经元的初始化参数
    LIFNeuron(LifParams lifParamInit) {
        tau_m = lifParamInit.tau_m;                 // 膜时间常数
        V_rest = lifParamInit.V_rest;               // 静息电位
        V_reset = lifParamInit.V_reset;             // 重置电位
        V_th = lifParamInit.V_th;                   // 阈值电位
        R = lifParamInit.R;                         // 膜电阻，电导倒数
        tau_ref = lifParamInit.t_refractory;        // 不应期时长
        V = lifParamInit.V_init;                    // 初始化膜电位
        input_current = lifParamInit.current_init;  // 外部输入电流

        refractory_time = 0.0;  // 当前不应期剩余时间，默认为0
        spiked = false;         // 神经元是否发放
    }

    // 神经元电流输入
    // current: 外部输入电流
    void receiveCurrent(float current) { input_current += current; }

    // 更新膜电位和发放
    // dt: 时间步长
    virtual void update(float dt) {
        spiked = false;
        // 判断是否处在不应期
        // 是则电压为V_reset
        if (refractory_time > 0) {
            refractory_time -= dt;
            V = V_reset;
        }
        // 否则更新LIF神经元
        else {
            float total_current =
                input_current;  // 会考虑是否有神经元内部的噪声电流影响，所以额外定义了total_current
            // LIF神经元膜电位更新
            float V_inf = V_rest + R * total_current;
            V += dt * (V_inf - V) / tau_m;

            if (V >= V_th) {
                // cout << "Spike!" << endl;
                spiked = true;
                V = V_reset;                // 膜电位重置
                refractory_time = tau_ref;  // 不应期重置
            }
        }
        input_current = 0;  // 重置输入电流
    }

    // 检测神经元是否发放
    // false:未发放，true:发放
    bool hasFired() { return spiked; }

    // 获取当前神经元膜电位
    float getMembranePotential() { return V; }

    // 重置函数，将神经元的状态置为初始化
    void reset(LifParams lifParamInit) {
        // 重置神经元为初始参数
        tau_m = lifParamInit.tau_m;                 // 膜时间常数
        V_rest = lifParamInit.V_rest;               // 静息电位
        V_reset = lifParamInit.V_reset;             // 重置电位
        V_th = lifParamInit.V_th;                   // 阈值电位
        R = lifParamInit.R;                         // 膜电阻，电导倒数
        tau_ref = lifParamInit.t_refractory;        // 不应期时长
        V = lifParamInit.V_init;                    // 初始化膜电位
        input_current = lifParamInit.current_init;  // 外部输入电流

        refractory_time = 0.0;  // 当前不应期剩余时间，默认为0
        spiked = false;         // 神经元是否发放
    }
};

// 带高斯噪声的LIF神经元模型
// 继承自LIFNeuron class
class LIFNeuron_gaussnoise : public LIFNeuron {
   public:
    float noise_mean;    // 噪声的均值
    float noise_stddev;  // 噪声标准差
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

    // 带噪声参数的构造函数
    // 传入LifParams的对象，以及自定义的均值和方差
    LIFNeuron_gaussnoise(LifParams params, float mean = 0.0f,
                         float stddev = 1.0f)
        : LIFNeuron(params),
          noise_mean(mean),
          noise_stddev(stddev),
          distribution(
              std::normal_distribution<float>(noise_mean, noise_stddev)) {}

    // 更新膜电位和发放
    // dt: 时间步长
    void update(float dt) override {
        spiked = false;
        // 判断是否处在不应期
        // 是则电压为V_reset
        if (refractory_time > 0) {
            refractory_time -= dt;
            V = V_reset;
        }
        // 否则更新LIF神经元
        else {
            float noise = distribution(generator);  // 计算噪声
            float total_current =
                input_current +
                noise;  // 噪声影响电流，所以额外定义了total_current
            // LIF神经元膜电位更新
            float V_inf = V_rest + R * total_current;
            V += dt * (V_inf - V) / tau_m;

            if (V >= V_th) {
                // cout << "Spike!" << endl;
                spiked = true;
                V = V_reset;                // 膜电位重置
                refractory_time = tau_ref;  // 不应期重置
            }
        }
        input_current = 0;  // 重置输入电流
    }
};
// 指数型衰减突触和NMDA突触初始化参数
struct SynParams {
    LIFNeuron* pre = nullptr;   // 前神经元
    LIFNeuron* post = nullptr;  // 后神经元
    float g_max;                // 最大突触电导
    float tau_rise;             // 神经递质浓度上升时间常数
    float tau_decay;            // 电导衰减时间常数
    float E_syn;                // 突触反转电位
    float s = 0.0;              // 当前电导
    float x = 0.0;              // 神经递质浓度
    float Mg = 1.0;             // Mg2+ 浓度
    float I_syn = 0.0;          // 突触电流
} SynParamsInitDefault;

// 指数型衰减突触模型(AMPA, GABA)
// 功能：初始化AMPA/GABA突触参数，突触电流计算，突触电流更新，突触重置
class ExponentialSynapse {
   public:
    LIFNeuron* pre;   // 突触前神经元指针
    LIFNeuron* post;  // 突触后神经元指针
    float g_max;      // 最大突触电导
    float E_syn;      // 突触反转电位
    float tau;        // 电导衰减时间常数
    float s;          // 中间变量
    float I_syn;      // 突触电流

    ExponentialSynapse(SynParams SynParamsInit) {
        pre = SynParamsInit.pre;       // 突触前神经元
        post = SynParamsInit.post;     // 突触后神经元
        g_max = SynParamsInit.g_max;   // 最大突触电导
        E_syn = SynParamsInit.E_syn;   // 突触反转电位
        tau = SynParamsInit.tau_rise;  // 电导衰减时间常数初始化
        s = 0.0;                       // 当前电导
        I_syn = SynParamsInit.I_syn;   // 电流初始化为0
    }

    // 计算突触电流
    void calCurrent(float dt) {
        // 更新s和g的值
        s -= s / tau * dt;  // 根据 tau 更新 s，电导的指数衰减

        if (pre->hasFired()) {
            s += 1.0;  // 突触前神经元fire时s++
        }
        // s += pre->hasFired();

        // 计算突触电流
        float g_exp = g_max * s;
        I_syn = g_exp * (E_syn - post->getMembranePotential());
    }

    // 突触后神经元电流更新
    void update(float dt) {
        calCurrent(dt);  // FIXME: 传递前记得先更新 I_syn
        post->receiveCurrent(I_syn);
    }

    // 重置函数，将指数型衰减突触的状态置为初始化
    void reset(SynParams SynParamsInit) {
        pre = SynParamsInit.pre;       // 突触前神经元
        post = SynParamsInit.post;     // 突触后神经元
        g_max = SynParamsInit.g_max;   // 最大突触电导
        E_syn = SynParamsInit.E_syn;   // 突触反转电位
        tau = SynParamsInit.tau_rise;  // 电导衰减时间常数初始化
        s = 0.0;                       // 当前电导
        I_syn = SynParamsInit.I_syn;   // 电流初始化为0
    }
};

// 待补充：引用的文献等...
// NMDA突触模型
// 功能：初始化NMDA突触参数，突触电流计算，突触电流更新，突触重置
class NMDASynapse {
   public:
    LIFNeuron* pre;   // 前神经元
    LIFNeuron* post;  // 后神经元
    float g_max;      // 最大突触电导
    float tau_rise;   // 神经递质浓度上升时间常数
    float tau_decay;  // 电导衰减时间常数
    float E_syn;      // 突触反转电位
    float s;          // 当前电导
    float x;          // 神经递质浓度
    float Mg;         // Mg2+ 浓度
    float I_syn;      // 突触电流

    // 构造函数初始化
    NMDASynapse(SynParams SynParamsInit) {
        pre = SynParamsInit.pre;            // 突触前神经元初始化
        post = SynParamsInit.post;          // 突触后神经元初始化
        g_max = SynParamsInit.g_max;        // 最大突触电导
        tau_rise = SynParamsInit.tau_rise;  // 神经递质浓度上升时间常数
        tau_decay = SynParamsInit.tau_decay;  // 电导衰减时间常数
        E_syn = SynParamsInit.E_syn;          // 突触反转电位
        s = 0;                                // 当前电导初始化为0
        x = 0;                                // 神经递质浓度初始化为0
        Mg = SynParamsInit.Mg;                // Mg2+浓度
        I_syn = SynParamsInit.I_syn;          // 突触电流初始化
    }

    // 突触电流计算
    void calCurrent(float dt) {
        // 更新 x 和 g 的值
        s += dt * (-s / tau_decay + 0.5 * x * (1 - s));  // 根据 τ_decay 更新 g
        x -= dt * x / tau_rise;  // 根据 τ_rise 更新 x

        if (pre->hasFired()) {
            x += 1.0;  // 突触前神经元发放动作电位时增加神经递质浓度
        }
        // x += pre->hasFired();

        // 计算突触电流
        float g_NMDA = g_max * s;
        float V_post = post->getMembranePotential();
        I_syn =
            g_NMDA * (E_syn - V_post) / (1 + Mg * exp(-0.062 * V_post) / 3.57);
    }

    // 突触后神经元电流更新
    void update(float dt) {
        calCurrent(dt);
        post->receiveCurrent(I_syn);
    }

    // NMDA突触重置函数
    void reset(SynParams SynParamsInit) {
        pre = SynParamsInit.pre;            // 突触前神经元初始化
        post = SynParamsInit.post;          // 突触后神经元初始化
        g_max = SynParamsInit.g_max;        // 最大突触电导
        tau_rise = SynParamsInit.tau_rise;  // 神经递质浓度上升时间常数
        tau_decay = SynParamsInit.tau_decay;  // 电导衰减时间常数
        E_syn = SynParamsInit.E_syn;          // 突触反转电位
        s = 0;                                // 当前电导初始化为0
        x = 0;                                // 神经递质浓度初始化为0
        Mg = SynParamsInit.Mg;                // Mg2+浓度
        I_syn = SynParamsInit.I_syn;          // 突触电流初始化
    }

   private:
};

// // demo：神经元和突触的初始化例程
// int main()
// {
//     // 初始化神经元参数结构体
//     LifParams lifParamInit;
//     lifParamInit.tau_m = 7.5; // 时间常数
//     lifParamInit.V_rest = -70.0; // 静息电位
//     lifParamInit.V_reset = -75.0; // 重置电位
//     lifParamInit.V_init = -65.0; // 初始膜电位
//     lifParamInit.R = 1 / 10.0;  // 神经元膜电阻，为电导倒数
//     lifParamInit.t_refractory = 2.0; // 神经元不应期时间
//     lifParamInit.current_init = 0.0; // 神经元输入电流初始化

//     // 根据参数结构体定义一个LIF神经元对象
//     // 此处两个神经元假设都是兴奋性神经元
//     LIFNeuron lifNode_pre(lifParamInit); // 突触前神经元
//     LIFNeuron lifNode_post(lifParamInit); // 突触后神经元

//     // 初始化AMPA突触参数结构体
//     SynParams synAMPAParamInit;
//     synAMPAParamInit.pre = &lifNode_pre; // 突触前神经元
//     synAMPAParamInit.post = &lifNode_post; // 突触后神经元
//     synAMPAParamInit.g_max = 0.01;    // 最大突触电导，权重参数
//     synAMPAParamInit.tau_rise = 2.0;  // AMPA突触电导衰减时间常数
//     synAMPAParamInit.E_syn = 0.0; // AMPA突触反转电位
//     synAMPAParamInit.I_syn = 0.0; // AMPA突触电流初始化

//     // 初始化NMDA突触参数结构体
//     SynParams synNMDAParamInit;
//     synNMDAParamInit.pre = &lifNode_pre; // 突触前神经元
//     synNMDAParamInit.post = &lifNode_post; // 突触后神经元
//     synNMDAParamInit.g_max = 0.5;    // 最大突触电导，权重参数
//     synNMDAParamInit.tau_rise = 2.0; // NMDA突触神经递质浓度上升时间常数
//     synNMDAParamInit.tau_decay = 100.0;  // NMDA突触电导衰减时间常数
//     synNMDAParamInit.E_syn = 0.0;  // AMPA突触反转电位
//     synNMDAParamInit.Mg = 1.0;  // Mg2+浓度
//     synNMDAParamInit.I_syn = 0.0; // NDMA突触电流初始化

//     // 根据突触结构体定义AMPA突触
//     ExponentialSynapse syn_AMPA(synAMPAParamInit);

//     // 根据突触结构体定义NDMA突触
//     NMDASynapse syn_NMDA(synNMDAParamInit);

//     // 定义带噪声的LIFNeuron
//     float mean = 0.0f; //高斯噪声均值
//     float stddev = 1.0f; //高斯噪声标准差
//     LIFNeuron_gaussnoise lifNode_noise(lifParamInit, mean, stddev);

//     return 0;
// }

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
