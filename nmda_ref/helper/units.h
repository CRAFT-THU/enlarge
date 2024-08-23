#if !defined(__UNITS__)
#define __UNITS__

#include <random>
#include <vector>

/* structs */

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

/* initializers */

std::vector<LIFNeuron> create_lif(size_t num = 1) {
    LifParams lifParamInit;
    lifParamInit.tau_m = 7.5;         // 时间常数
    lifParamInit.V_rest = -70.0;      // 静息电位
    lifParamInit.V_reset = -75.0;     // 重置电位
    lifParamInit.V_init = -65.0;      // 初始膜电位
    lifParamInit.R = 1 / 10.0;        // 神经元膜电阻，为电导倒数
    lifParamInit.t_refractory = 2.0;  // 神经元不应期时间
    lifParamInit.current_init = 0.0;  // 神经元输入电流初始化

    std::vector<LIFNeuron> nrnVec(num, LIFNeuron(lifParamInit));
    return nrnVec;
}

// * all to all connect
std::vector<ExponentialSynapse> connect_exp(std::vector<LIFNeuron>& pre,
                                            std::vector<LIFNeuron>& post) {
    SynParams synAMPAParamInit;
    synAMPAParamInit.g_max = 0.01;    // 最大突触电导，权重参数
    synAMPAParamInit.tau_rise = 2.0;  // AMPA突触电导衰减时间常数
    synAMPAParamInit.E_syn = 0.0;     // AMPA突触反转电位
    synAMPAParamInit.I_syn = 0.0;     // AMPA突触电流初始化

    std::vector<ExponentialSynapse> synVec(
        pre.size() * post.size(), ExponentialSynapse(synAMPAParamInit));
    for (size_t i = 0; i < pre.size(); i++)
        for (size_t j = 0; j < post.size(); j++) {
            synVec[i * post.size() + j].pre = &pre[i];
            synVec[i * post.size() + j].post = &post[j];
        }
    return synVec;
}

// * all to all connect
std::vector<NMDASynapse> connect_nmda(std::vector<LIFNeuron>& pre,
                                      std::vector<LIFNeuron>& post) {
    SynParams synNMDAParamInit;
    synNMDAParamInit.g_max = 0.5;  // 最大突触电导，权重参数
    synNMDAParamInit.tau_rise = 2.0;  // NMDA突触神经递质浓度上升时间常数
    synNMDAParamInit.tau_decay = 100.0;  // NMDA突触电导衰减时间常数
    synNMDAParamInit.E_syn = 0.0;        // NMDA突触反转电位
    synNMDAParamInit.Mg = 1.0;           // Mg2+浓度
    synNMDAParamInit.I_syn = 0.0;        // NDMA突触电流初始化

    std::vector<NMDASynapse> synVec(pre.size() * post.size(),
                                    NMDASynapse(synNMDAParamInit));
    for (size_t i = 0; i < pre.size(); i++)
        for (size_t j = 0; j < post.size(); j++) {
            synVec[i * post.size() + j].pre = &pre[i];
            synVec[i * post.size() + j].post = &post[j];
        }
    return synVec;
}

#endif  // __UNITS__
