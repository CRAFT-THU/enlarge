#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <numeric>

using namespace std;
using namespace std::chrono;

//线性整流函数
float relu(float x) {
    return x > 0 ? x : 0;
}

float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

//生成连接矩阵（向量）
vector<float> random_matrix(int shape, float avg = 0.05, float var = 0.5) //shape = channels*rols*columns
{
    vector<float> m;
    int i;
    if (var == 0) {
        for (i = 0; i < shape; ++i) {
            m.push_back(avg);
        }
    }
    else {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(avg, avg * var);

        for (i = 0; i < shape; ++i) {
            m.push_back(relu(d(gen)));
        }
    }

    return m;
}

// 创建ON/OFF Grid（突触前神经元列表）
std::vector<float> sliceMatrix(const std::vector<float>& inputVec, int rows, int cols, int index) {
    std::vector<float> slices;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (index == (i % 2) * 2 + (j % 2)) {
                slices.push_back(inputVec[i * cols + j]);
            }
            else {
                slices.push_back(0);
            }
        }
    }

    return slices;
}

// 函数：im2col（生成突触前神经元列表）步骤是每次从50*50的随机矩阵中，取出11*11大小的向量，步长为2, 共产生20*20这么多个向量。
// im2col 函数会返回一个向量的向量，其中每个向量的大小与原始图像大小相同。我们在每个对应小块的位置填入原始矩阵的值，其余部分置为 0。
vector<vector<float>> im2col(vector<float>& input, int input_size, int block_size, int stride) {
    vector<vector<float>> result;

    for (int i = 0; i <= input_size - block_size; i += stride) { //滑窗步数
        for (int j = 0; j <= input_size - block_size; j += stride) {
            vector<float> block(input_size * input_size, 0); //创建和输入size相同的0向量
            for (int m = 0; m < block_size; ++m) { //卷积核内元素个数
                for (int n = 0; n < block_size; ++n) {
                    int row = i + m;
                    int col = j + n;
                    block[row * input_size + col] = input[row * input_size + col]; //将卷积核对应位置的元素幅值给block
                }
            }
            result.push_back(block);
        }
    }

    return result;
}

// 创建随机矩阵函数
std::vector<std::vector<float>> createRandomMatrix(int rows, int cols, int onesPerCol, float avg = 0.1, float var = 0.5) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0.0));

    // 使用随机设备初始化随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<> d(avg, avg * var);  // 正态分布

    for (int j = 0; j < cols; ++j) {
        std::vector<int> indices(rows);
        // 初始化索引
        for (int i = 0; i < rows; ++i) {
            indices[i] = i;
        }
        // 打乱索引
        std::shuffle(indices.begin(), indices.end(), g);
        // 选取前 onesPerCol 个索引
        for (int i = 0; i < onesPerCol; ++i) {
            matrix[indices[i]][j] = relu(d(g));
        }
    }

    return matrix;
}

// 读取CSV文件中一行的函数
std::vector<float> readCSVRow(const std::string& filename, int rowIndex) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    int currentRow = 0;
    while (std::getline(file, line)) {
        if (currentRow == rowIndex) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            file.close();
            return row;
        }
        ++currentRow;
    }

    file.close();
    throw std::runtime_error("Row index out of range");
}


// LIF神经元初始化参数
struct LifParams {
    float V_rest = -70.0;   // rest voltage
    float V_reset = -75.0;   // reset voltage
    float V_th = -50.0;  // threshold voltage
    float tau_m = 7.5;  // time constant
    float t_refractory = 2.0;  // refractory period
    float R = 0.05; // 电阻
    float V_init = -70.0; // 初始膜电位
    float current_init = 0.0; // 初始外部输入的电流
} lifParamInitDefault;

// LIF neuron model class
// 功能：神经元初始化，接收电流，神经元状元更新，神经元状态重置，获取当前发放情况，获取当前神经元膜电位
class LIFNeuron {
public:
    float tau_m;  // 膜时间常数
    float V_rest; // 静息电位
    float V_reset; // 重置电位
    float V_th;   // 阈值电位
    float R;      // 膜电阻
    float V;      // 膜电位
    float tau_ref; // 不应期时长
    float refractory_time; // 当前不应期剩余时间
    float input_current; // 输入电流
    bool spiked;         // 神经元是否发放

    // 默认构造函数
    // 无参数传入时神经元初始化使用以下参数
    LIFNeuron()
    {
        tau_m = lifParamInitDefault.tau_m;     // 膜时间常数
        V_rest = lifParamInitDefault.V_rest;   // 静息电位
        V_reset = lifParamInitDefault.V_reset; // 重置电位
        V_th = lifParamInitDefault.V_th;       // 阈值电位
        R = lifParamInitDefault.R;             // 膜电阻，电导倒数
        tau_ref = lifParamInitDefault.t_refractory; // 不应期时长
        V = lifParamInitDefault.V_init;             // 膜电位，默认初始化为-70.0mV
        input_current = lifParamInitDefault.current_init; // 初始外部输入电流

        refractory_time = 0.0;          // 当前不应期剩余时间，默认初始化为0ms(一开始神经元未处于不应期)
        spiked = false;                 // 神经元是否发放
    }

    // 构造函数
    // 自定义神经元的初始化参数
    LIFNeuron(LifParams lifParamInit)
    {
        tau_m = lifParamInit.tau_m;         // 膜时间常数
        V_rest = lifParamInit.V_rest;       // 静息电位
        V_reset = lifParamInit.V_reset;     // 重置电位
        V_th = lifParamInit.V_th;           // 阈值电位
        R = lifParamInit.R;                 // 膜电阻，电导倒数
        tau_ref = lifParamInit.t_refractory;     // 不应期时长
        V = lifParamInit.V_init;            // 初始化膜电位
        input_current = lifParamInit.current_init;  // 外部输入电流

        refractory_time = 0.0;  // 当前不应期剩余时间，默认为0
        spiked = false;         // 神经元是否发放

    }

    // 神经元电流输入
    // current: 外部输入电流
    void receiveCurrent(float current) {
        input_current += current;
    }

    // 更新膜电位和发放
    // dt: 时间步长
    virtual void update(float dt)
    {
        spiked = false;
        // 判断是否处在不应期
        // 是则电压为V_reset
        if (refractory_time > 0)
        {
            refractory_time -= dt;
            V = V_reset;
        }
        // 否则更新LIF神经元
        else
        {
            float total_current = input_current; // 会考虑是否有神经元内部的噪声电流影响，所以额外定义了total_current
            // LIF神经元膜电位更新
            float V_inf = V_rest + R * total_current;
            V += dt * (V_inf - V) / tau_m;

            if (V >= V_th) {
                // cout << "Spike!" << endl;
                spiked = true;
                V = V_reset;    // 膜电位重置
                refractory_time = tau_ref; // 不应期重置
            }
        }
        input_current = 0; // 重置输入电流
    }

    // 检测神经元是否发放
    // false:未发放，true:发放
    bool hasFired() {
        return spiked;
    }

    // 获取当前神经元膜电位
    float getMembranePotential() {
        return V;
    }

    // 重置函数，将神经元的状态置为初始化
    void reset(LifParams lifParamInit)
    {
        // 重置神经元为初始参数
        tau_m = lifParamInit.tau_m;         // 膜时间常数
        V_rest = lifParamInit.V_rest;       // 静息电位
        V_reset = lifParamInit.V_reset;     // 重置电位
        V_th = lifParamInit.V_th;           // 阈值电位
        R = lifParamInit.R;                 // 膜电阻，电导倒数
        tau_ref = lifParamInit.t_refractory;     // 不应期时长
        V = lifParamInit.V_init;            // 初始化膜电位
        input_current = lifParamInit.current_init;  // 外部输入电流

        refractory_time = 0.0;  // 当前不应期剩余时间，默认为0
        spiked = false;         // 神经元是否发放

    }
};

// 带OU噪声的LIF神经元模型
// 继承自LIFNeuron class
class LIFNeuron_OUnoise : public LIFNeuron
{
public:
    float noise_mean;       // 噪声的均值
    float noise_stddev;     // 噪声标准差
    float tau_ou;
    float dt;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

    // 带噪声参数的构造函数
    // 传入LifParams的对象，以及自定义的均值和方差
    LIFNeuron_OUnoise(LifParams params, float mean = 400.0f, float stddev = 100.0f, float tau_ou = 10.0f, float dt = 0.2)
        : LIFNeuron(params), noise_mean(mean), noise_stddev(stddev), tau_ou(tau_ou), dt(dt),
        distribution(std::normal_distribution<float>(noise_mean, noise_stddev))
    {
        _sig = std::sqrt(2 / tau_ou) * noise_stddev;
        std::random_device rd;
        generator.seed(rd());
    }

    // 更新膜电位和发放
    // dt: 时间步长
    void update(float dt) override
    {
        spiked = false;
        // 判断是否处在不应期
        // 是则电压为V_reset
        if (refractory_time > (dt / 2))
        {
            refractory_time -= dt;
            V = V_reset;
        }
        // 否则更新LIF神经元
        else
        {
            noise = distribution(generator);  // 计算噪声
            temp = noise - noise_mean;
            noise -= (dt / tau_ou) * temp;
            std::normal_distribution<double> dist(0.0, std::sqrt(dt) * _sig);
            temp = dist(generator);
            noise += temp;
            float total_current = input_current + noise; // 噪声影响电流，所以额外定义了total_current
            // LIF神经元膜电位更新
            float V_inf = V_rest + R * total_current;
            V += dt * (V_inf - V) / tau_m;

            if (V >= V_th) {
                // cout << "Spike!" << endl;
                spiked = true;
                V = V_reset;    // 膜电位重置
                refractory_time = tau_ref; // 不应期重置
            }
        }
        input_current = 0; // 重置输入电流
    }
private:
    float noise;
    float temp;
    float _sig;
};

// 指数型衰减突触和NMDA突触初始化参数
struct SynParams
{
    LIFNeuron* pre = nullptr;   // 前神经元
    LIFNeuron* post = nullptr;  // 后神经元
    float g_max;     // 最大突触电导
    float tau_rise;  // 神经递质浓度上升时间常数
    float tau_decay; // 电导衰减时间常数
    float E_syn;     // 突触反转电位
    float s = 0.0;         // 当前电导
    float x = 0.0;         // 神经递质浓度
    float Mg = 1.0;        // Mg2+ 浓度
    float I_syn = 0.0;     // 突触电流
} SynParamsInitDefault;


// 指数型衰减突触模型(AMPA, GABA)
// 功能：初始化AMPA/GABA突触参数，突触电流计算，突触电流更新，突触重置
class ExponentialSynapse {
public:
    LIFNeuron* pre; // 突触前神经元指针
    LIFNeuron* post; // 突触后神经元指针
    float g_max; // 最大突触电导
    float E_syn; // 突触反转电位
    float tau; // 电导衰减时间常数
    float s; // 中间变量
    float I_syn; // 突触电流

    ExponentialSynapse(SynParams SynParamsInit)
    {
        pre = SynParamsInit.pre;      // 突触前神经元
        post = SynParamsInit.post;    // 突触后神经元
        g_max = SynParamsInit.g_max;  // 最大突触电导
        E_syn = SynParamsInit.E_syn;  // 突触反转电位
        tau = SynParamsInit.tau_rise; // 电导衰减时间常数初始化
        s = 0.0; // 当前电导
        I_syn = SynParamsInit.I_syn; // 电流初始化为0
    }

    // 计算突触电流
    void update(float dt)
    {
        // 更新s和g的值
        s -= s / tau * dt; // 根据 tau 更新 s，电导的指数衰减

        if (pre->hasFired()) {
            s += 1.0; // 突触前神经元fire时s++
        }
        // s += pre->hasFired();

        // 计算突触电流
        float g_exp = g_max * s;
        I_syn = g_exp * (E_syn - post->getMembranePotential());

        // 突触后神经元电流更新
        post->receiveCurrent(I_syn);
    }


    // 重置函数，将指数型衰减突触的状态置为初始化
    void reset(SynParams SynParamsInit)
    {
        pre = SynParamsInit.pre;      // 突触前神经元
        post = SynParamsInit.post;    // 突触后神经元
        g_max = SynParamsInit.g_max;  // 最大突触电导
        E_syn = SynParamsInit.E_syn;  // 突触反转电位
        tau = SynParamsInit.tau_rise; // 电导衰减时间常数初始化
        s = 0.0; // 当前电导
        I_syn = SynParamsInit.I_syn; // 电流初始化为0

    }
};

// 待补充：引用的文献等...
// NMDA突触模型
// 功能：初始化NMDA突触参数，突触电流计算，突触电流更新，突触重置
class NMDASynapse {
public:
    LIFNeuron* pre;   // 前神经元
    LIFNeuron* post;  // 后神经元
    float g_max;     // 最大突触电导
    float tau_rise;  // 神经递质浓度上升时间常数
    float tau_decay; // 电导衰减时间常数
    float E_syn;     // 突触反转电位
    float s;         // 当前电导
    float x;         // 神经递质浓度
    float Mg;        // Mg2+ 浓度
    float I_syn;     // 突触电流

    // 构造函数初始化
    NMDASynapse(SynParams SynParamsInit)
    {
        pre = SynParamsInit.pre;    // 突触前神经元初始化
        post = SynParamsInit.post;  // 突触后神经元初始化
        g_max = SynParamsInit.g_max;  // 最大突触电导
        tau_rise = SynParamsInit.tau_rise; // 神经递质浓度上升时间常数
        tau_decay = SynParamsInit.tau_decay; // 电导衰减时间常数
        E_syn = SynParamsInit.E_syn; // 突触反转电位
        s = 0; // 当前电导初始化为0
        x = 0; // 神经递质浓度初始化为0
        Mg = SynParamsInit.Mg; // Mg2+浓度
        I_syn = SynParamsInit.I_syn; // 突触电流初始化
    }

    // 突触电流计算
    void update(float dt)
    {
        // 更新 x 和 g 的值
        s += dt * (-s / tau_decay + 0.5 * x * (1 - s)); // 根据 τ_decay 更新 g
        x -= dt * x / tau_rise; // 根据 τ_rise 更新 x

        if (pre->hasFired()) {
            x += 1.0; // 突触前神经元发放动作电位时增加神经递质浓度
        }
        // x += pre->hasFired();

        // 计算突触电流
        float g_NMDA = g_max * s;
        float V_post = post->getMembranePotential();
        I_syn = g_NMDA * (E_syn - V_post) / (1 + Mg * exp(-0.062 * V_post) / 3.57);

        // 突触后神经元电流更新
        post->receiveCurrent(I_syn);
    }

    // NMDA突触重置函数
    void reset(SynParams SynParamsInit)
    {
        pre = SynParamsInit.pre;    // 突触前神经元初始化
        post = SynParamsInit.post;  // 突触后神经元初始化
        g_max = SynParamsInit.g_max;  // 最大突触电导
        tau_rise = SynParamsInit.tau_rise; // 神经递质浓度上升时间常数
        tau_decay = SynParamsInit.tau_decay; // 电导衰减时间常数
        E_syn = SynParamsInit.E_syn; // 突触反转电位
        s = 0; // 当前电导初始化为0
        x = 0; // 神经递质浓度初始化为0
        Mg = SynParamsInit.Mg; // Mg2+浓度
        I_syn = SynParamsInit.I_syn; // 突触电流初始化
    }

private:

};

const int numLGNExc = 10000;
const int numV1Exc = 2500;
const int numMTExc = 400;
const int numLipExc = 400;
const int numLipInh = 400;

class LGN_V1_MT_LIP
{
public:

    float dt = 0.2;

    // 定义神经元群
    vector<LIFNeuron_OUnoise> lgnPopExcOn;   // LGN脑区兴奋性神经元群On
    vector<LIFNeuron_OUnoise> lgnPopExcOff;   // LGN脑区兴奋性神经元群Off
    vector<LIFNeuron_OUnoise> v1PopExcM1;   // V1脑区兴奋性神经元群M1
    vector<LIFNeuron_OUnoise> v1PopExcM2;   // V1脑区兴奋性神经元群M2
    vector<LIFNeuron_OUnoise> mtPopExcL;    // MT脑区抑制性神经元群L
    vector<LIFNeuron_OUnoise> mtPopExcR;    // MT脑区抑制性神经元群R
    vector<LIFNeuron_OUnoise> lipPopExcA;   // LIP脑区兴奋性神经元群A
    vector<LIFNeuron_OUnoise> lipPopExcB;   // LIP脑区兴奋性神经元群B
    vector<LIFNeuron_OUnoise> lipPopInh;    // LIP脑区抑制性神经元群

    // 定义突触, 其中 '==>': AMPA, '-=>': NDMA, '-->': GABA
    vector<ExponentialSynapse> lgnExcOn2v1ExcM1_AMPA;  // On ==> M1
    vector<ExponentialSynapse> lgnExcOn2v1ExcM2_AMPA;  // On ==> M2
    vector<ExponentialSynapse> lgnExcOff2v1ExcM1_AMPA;  // Off ==> M1
    vector<ExponentialSynapse> lgnExcOff2v1ExcM2_AMPA;  // Off ==> M2
    vector<ExponentialSynapse> v1ExcM12mtExcL_AMPA;   // M1 ==> L
    vector<ExponentialSynapse> v1ExcM12mtExcR_AMPA;   // M1 ==> R
    vector<ExponentialSynapse> v1ExcM22mtExcL_AMPA;   // M2 ==> L
    vector<ExponentialSynapse> v1ExcM22mtExcR_AMPA;   // M2 ==> R
    vector<ExponentialSynapse> mtExcL2lipExcA_AMPA;  // L ==> A
    vector<ExponentialSynapse> mtExcR2lipExcB_AMPA;  // R ==> B

    vector<ExponentialSynapse> lipExcA2ExcASyn_AMPA;  // A ==> A
    vector<ExponentialSynapse> lipExcA2InhSyn_AMPA;   // A ==> I
    vector<NMDASynapse> lipExcA2ExcASyn_NMDA;  // A -=> A
    vector<NMDASynapse> lipExcA2InhSyn_NMDA;   // A -=> I
    vector<ExponentialSynapse> lipExcB2ExcBSyn_AMPA;  // B ==> B
    vector<ExponentialSynapse> lipExcB2InhSyn_AMPA;   // B ==> I
    vector<NMDASynapse> lipExcB2ExcBSyn_NMDA;  // B -=> B
    vector<NMDASynapse> lipExcB2InhSyn_NMDA;   // B -=> I
    vector<ExponentialSynapse> lipInh2ExcASyn_GABA;   // I --> A
    vector<ExponentialSynapse> lipInh2ExcBSyn_GABA;   // I --> B
    vector<ExponentialSynapse> lipInh2InhSyn_GABA;    // I --> I

    vector<int> frL;
    vector<int> frR;
    vector<int> frA;
    vector<int> frB;

    LGN_V1_MT_LIP()
    {
        initializeNeurons();
        createSynapses();
    }

    void initializeNeurons()
    {
        // 初始化神经元参数结构体
        LifParams lifParamInit;
        // common params
        lifParamInit.V_th = -50.0; // 静息电位
        lifParamInit.V_reset = -55.0; // 重置电位
        lifParamInit.V_rest = -70.0; // 静息电位
        float tau_ou = 10.0;
        float mu = 400.0;
        float sigma = 100.0;
        // Excitatory: ~OU(530, 150)
        lifParamInit.V_init = -70.0; // 初始膜电位-70
        lifParamInit.tau_m = 20.0; // tau = RC = Cm/gl = 500/25
        lifParamInit.R = 1 / 25.0;  // 神经元膜电阻，为电导倒数
        lifParamInit.t_refractory = 2.0; // 神经元不应期时间
        lifParamInit.current_init = 0.0; // 神经元输入电流初始化

        // 初始化神经元
        for (int i = 0; i < numLGNExc; i++)
        {
            lgnPopExcOn.emplace_back(lifParamInit);
            lgnPopExcOff.emplace_back(lifParamInit);
        }
        for (int i = 0; i < numV1Exc; i++)
        {
            v1PopExcM1.emplace_back(lifParamInit);
            v1PopExcM2.emplace_back(lifParamInit);
        }
        for (int i = 0; i < numMTExc; i++)
        {
            mtPopExcL.emplace_back(lifParamInit);
            mtPopExcR.emplace_back(lifParamInit);
        }
        for (int i = 0; i < numLipExc; i++)
        {
            lipPopExcA.emplace_back(lifParamInit);
            lipPopExcB.emplace_back(lifParamInit);
        }

        // Inhibitory: ~OU(410, 70)
        lifParamInit.tau_m = 10.0; // tau = RC = Cm/gl = 200/20
        lifParamInit.R = 1 / 20.0;  // 神经元膜电阻，为电导倒数
        lifParamInit.t_refractory = 1.0; // 神经元不应期时间
        for (int i = 0; i < numLipInh; i++)
        {
            lipPopInh.emplace_back(lifParamInit);
        }
    }

    void createSynapses()
    {
        // 初始化AMPA突触参数结构体
        SynParams synAMPAParamInit;

        synAMPAParamInit.g_max = 1;    // 最大突触电导，权重参数
        synAMPAParamInit.tau_rise = 2.0;  // AMPA突触电导衰减时间常数
        synAMPAParamInit.E_syn = 0.0; // AMPA突触反转电位
        synAMPAParamInit.I_syn = 0.0; // AMPA突触电流初始化

        //lgn突触前神经元列表
        vector<float> conn_lgnOn = random_matrix(100 * 100, 60, 0);
        vector<float> conn_lgnOff = random_matrix(100 * 100, 60, 0);

        // 获取四种切片
        auto on1 = sliceMatrix(conn_lgnOn, 100, 100, 0); // 奇数行奇数列
        auto off1 = sliceMatrix(conn_lgnOff, 100, 100, 1);  // 奇数行偶数列
        auto on2 = sliceMatrix(conn_lgnOn, 100, 100, 3);  // 偶数行偶数列
        auto off2 = sliceMatrix(conn_lgnOff, 100, 100, 2);   // 偶数行奇数列

        int nid = 0;
        int sid = 0;
        // lgnPopExcOn -> v1PopExcM1, AMPA
        for (auto& pre_neuron : lgnPopExcOn) {
            //for (auto& post_neuron : v1PopExcM1) {
            if (on1[nid] != 0) {
                synAMPAParamInit.pre = &pre_neuron; // 突触前神经元
                synAMPAParamInit.post = &v1PopExcM1[sid]; // 突触后神经元
                synAMPAParamInit.g_max = on1[nid];
                //cout << "sid" << sid << "g_max " << on1[nid] << endl;
                lgnExcOn2v1ExcM1_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                sid += 1;
            }
            nid += 1;
        }
        //cout << "sid " << sid << endl;
        nid = 0;
        sid = 0;
        // lgnPopExcOn -> v1PopExcM2, AMPA
        for (auto& pre_neuron : lgnPopExcOn) {
            //for (auto& post_neuron : v1PopExcM2) {
            if (on2[nid] != 0) {
                synAMPAParamInit.pre = &pre_neuron; // 突触前神经元
                synAMPAParamInit.post = &v1PopExcM2[sid]; // 突触后神经元
                synAMPAParamInit.g_max = on2[nid];
                lgnExcOn2v1ExcM2_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                sid += 1;
            }
            nid += 1;
        }
        nid = 0;
        sid = 0;
        // lgnPopExcOff -> v1PopExcM1, AMPA
        for (auto& pre_neuron : lgnPopExcOff) {
            //for (auto& post_neuron : v1PopExcM1) {
            if (off1[nid] != 0) {
                synAMPAParamInit.pre = &pre_neuron; // 突触前神经元
                synAMPAParamInit.post = &v1PopExcM1[sid]; // 突触后神经元
                synAMPAParamInit.g_max = off1[nid];
                lgnExcOff2v1ExcM1_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                sid += 1;
            }
            nid += 1;
        }
        nid = 0;
        sid = 0;
        // lgnPopExcOff -> v1PopExcM2, AMPA
        for (auto& pre_neuron : lgnPopExcOff) {
            //for (auto& post_neuron : v1PopExcM2) {
            if (off2[nid] != 0) {
                synAMPAParamInit.pre = &pre_neuron; // 突触前神经元
                synAMPAParamInit.post = &v1PopExcM2[sid]; // 突触后神经元
                synAMPAParamInit.g_max = off2[nid];
                lgnExcOff2v1ExcM2_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                sid += 1;
            }
            nid += 1;
        }

        //V1突触前神经元列表
        int sizeV1 = 50;
        vector<float> conn_v1M1 = random_matrix(sizeV1 * sizeV1, 2.0, 0.5);
        vector<float> conn_v1M2 = random_matrix(sizeV1 * sizeV1, 2.0, 0.5);

        // 计算im2col
        int block_size = 11;
        int stride = 2;
        vector<vector<float>> convM1 = im2col(conn_v1M1, sizeV1, block_size, stride);
        vector<vector<float>> convM2 = im2col(conn_v1M2, sizeV1, block_size, stride);

        // v1PopExcM1 -> mtPopExcL, AMPA
        for (size_t prenum = 0; prenum < convM1[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < convM1.size(); postnum++) {
                if (convM1[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &v1PopExcM1[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &mtPopExcL[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = convM1[postnum][prenum];
                    v1ExcM12mtExcL_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                    sid += 1;
                }
            }
        }
        // v1PopExcM2 -> mtPopExcR, AMPA
        for (size_t prenum = 0; prenum < convM2[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < convM2.size(); postnum++) {
                if (convM2[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &v1PopExcM2[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &mtPopExcR[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = convM2[postnum][prenum];
                    v1ExcM22mtExcR_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
            }
        }
        //MT突触前神经元列表
        vector<vector<float>> conn_mt2lipL = createRandomMatrix(400, 400, 200);
        vector<vector<float>> conn_mt2lipR = createRandomMatrix(400, 400, 200);

        // mtPopExcL -> lipPopExcA, AMPA
        for (size_t prenum = 0; prenum < conn_mt2lipL[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_mt2lipL.size(); postnum++) {
                if (conn_mt2lipL[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &mtPopExcL[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_mt2lipL[postnum][prenum];
                    mtExcL2lipExcA_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
            }
        }
        // mtPopExcR -> lipPopExcB, AMPA
        for (size_t prenum = 0; prenum < conn_mt2lipR[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_mt2lipR.size(); postnum++) {
                if (conn_mt2lipR[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &mtPopExcR[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_mt2lipR[postnum][prenum];
                    mtExcR2lipExcB_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
            }
        }

        // 初始化NMDA突触参数结构体
        SynParams synNMDAParamInit;

        synNMDAParamInit.g_max = 1;    // 最大突触电导，权重参数
        synNMDAParamInit.tau_rise = 2.0; // NMDA突触神经递质浓度上升时间常数
        synNMDAParamInit.tau_decay = 100.0;  // NMDA突触电导衰减时间常数
        synNMDAParamInit.E_syn = 0.0;  // AMPA突触反转电位
        synNMDAParamInit.Mg = 1.0;  // Mg2+浓度
        synNMDAParamInit.I_syn = 0.0; // NDMA突触电流初始化

        //lipPopExcA自循环连接
        float gA_E2E = 0.05 * 1.3;
        vector<vector<float>> conn_lipAE2E = createRandomMatrix(400, 400, 400, gA_E2E);
        float gA_E4E = 0.165 * 1.3;
        vector<vector<float>> conn_lipAE4E = createRandomMatrix(400, 400, 400, gA_E4E);

        // lipPopExcA自连接, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipAE2E[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipAE2E.size(); postnum++) {
                if (conn_lipAE2E[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipAE2E[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipAE4E[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipAE4E[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }

        //lipPopExcA组间连接
        float gA_E2X = 0.05 * 0.7;
        vector<vector<float>> conn_lipAE2X = createRandomMatrix(400, 400, 400, gA_E2X);
        float gA_E4X = 0.165 * 0.7;
        vector<vector<float>> conn_lipAE4X = createRandomMatrix(400, 400, 400, gA_E4X);

        // lipPopExcA组间连接, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipAE2X[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipAE2X.size(); postnum++) {
                if (conn_lipAE2X[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipAE2X[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipAE4X[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipAE4X[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }

        //lipPopExcA2Inh连接
        float gA_E2I = 0.04;
        vector<vector<float>> conn_lipAE2I = createRandomMatrix(400, 400, 400, gA_E2I);
        float gA_E4I = 0.13;
        vector<vector<float>> conn_lipAE4I = createRandomMatrix(400, 400, 400, gA_E4I);

        // lipPopExcA -> lipPopInh, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipAE2I[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipAE2I.size(); postnum++) {
                if (conn_lipAE2I[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopInh[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipAE2I[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipAE4I[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopInh[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipAE4I[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }


        //lipPopExcB自循环连接
        float gB_E2E = 0.05 * 1.3;
        vector<vector<float>> conn_lipBE2E = createRandomMatrix(400, 400, 400, gB_E2E);
        float gB_E4E = 0.165 * 1.3;
        vector<vector<float>> conn_lipBE4E = createRandomMatrix(400, 400, 400, gB_E4E);

        // lipPopExcB自连接, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipBE2E[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipBE2E.size(); postnum++) {
                if (conn_lipBE2E[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipBE2E[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipBE4E[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipBE4E[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }

        //lipPopExcB组间连接
        float gB_E2X = 0.05 * 0.7;
        vector<vector<float>> conn_lipBE2X = createRandomMatrix(400, 400, 400, gB_E2X);
        float gB_E4X = 0.165 * 0.7;
        vector<vector<float>> conn_lipBE4X = createRandomMatrix(400, 400, 400, gB_E4X);

        // lipPopExcB组间连接, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipBE2X[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipBE2X.size(); postnum++) {
                if (conn_lipBE2X[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipBE2X[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipBE4X[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipBE4X[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }

        //lipPopExcB2Inh连接
        float gB_E2I = 0.04;
        vector<vector<float>> conn_lipBE2I = createRandomMatrix(400, 400, 400, gB_E2I);
        float gB_E4I = 0.13;
        vector<vector<float>> conn_lipBE4I = createRandomMatrix(400, 400, 400, gB_E4I);

        // lipPopExcB -> lipPopInh, AMPA & NMDA
        for (size_t prenum = 0; prenum < conn_lipBE2I[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipBE2I.size(); postnum++) {
                if (conn_lipBE2I[postnum][prenum] != 0) {
                    synAMPAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synAMPAParamInit.post = &lipPopInh[postnum]; // 突触后神经元
                    synAMPAParamInit.g_max = conn_lipBE2I[postnum][prenum];
                    lipExcA2ExcASyn_AMPA.emplace_back(synAMPAParamInit); // AMPASynapse 参数
                }
                if (conn_lipBE4I[postnum][prenum] != 0) {
                    synNMDAParamInit.pre = &lipPopExcA[prenum]; // 突触前神经元
                    synNMDAParamInit.post = &lipPopInh[postnum]; // 突触后神经元
                    synNMDAParamInit.g_max = conn_lipBE4I[postnum][prenum];
                    lipExcA2ExcASyn_NMDA.emplace_back(synNMDAParamInit); // NMDASynapse 参数
                }
            }
        }


        // 初始化GABA突触参数结构体
        SynParams synGABAParamInit;
        synGABAParamInit.g_max = 1;    // 最大突触电导，权重参数
        synGABAParamInit.tau_rise = 5.0;  // AMPA突触电导衰减时间常数
        synGABAParamInit.E_syn = -70.0; // AMPA突触反转电位
        synGABAParamInit.I_syn = 0.0; // AMPA突触电流初始化

        //lipPopInh2ExcA连接
        float g_I2EA = 1.3;
        vector<vector<float>> conn_lipI2EA = createRandomMatrix(400, 400, 400, g_I2EA);

        // lipPopInh -> lipPopExcA, GABA
        for (size_t prenum = 0; prenum < conn_lipI2EA[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipI2EA.size(); postnum++) {
                if (conn_lipI2EA[postnum][prenum] != 0) {
                    synGABAParamInit.pre = &lipPopInh[prenum]; // 突触前神经元
                    synGABAParamInit.post = &lipPopExcA[postnum]; // 突触后神经元
                    synGABAParamInit.g_max = conn_lipI2EA[postnum][prenum];
                    lipInh2ExcASyn_GABA.emplace_back(synGABAParamInit); // GABASynapse 参数
                }
            }
        }

        //lipPopInh2ExcB连接
        float g_I2EB = 1.3;
        vector<vector<float>> conn_lipI2EB = createRandomMatrix(400, 400, 400, g_I2EB);

        // lipPopInh -> lipPopExcB, GABA
        for (size_t prenum = 0; prenum < conn_lipI2EB[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipI2EB.size(); postnum++) {
                if (conn_lipI2EB[postnum][prenum] != 0) {
                    synGABAParamInit.pre = &lipPopInh[prenum]; // 突触前神经元
                    synGABAParamInit.post = &lipPopExcB[postnum]; // 突触后神经元
                    synGABAParamInit.g_max = conn_lipI2EB[postnum][prenum];
                    lipInh2ExcASyn_GABA.emplace_back(synGABAParamInit); // GABASynapse 参数
                }
            }
        }

        //lipPopInh自连接
        float g_I2I = 1.0;
        vector<vector<float>> conn_lipI2I = createRandomMatrix(400, 400, 400, g_I2I);

        // lipPopInh自连接, GABA
        for (size_t prenum = 0; prenum < conn_lipI2I[0].size(); prenum++) {
            for (size_t postnum = 0; postnum < conn_lipI2I.size(); postnum++) {
                if (conn_lipI2I[postnum][prenum] != 0) {
                    synGABAParamInit.pre = &lipPopInh[prenum]; // 突触前神经元
                    synGABAParamInit.post = &lipPopInh[postnum]; // 突触后神经元
                    synGABAParamInit.g_max = conn_lipI2I[postnum][prenum];
                    lipInh2ExcASyn_GABA.emplace_back(synGABAParamInit); // GABASynapse 参数
                }
            }
        }

    }

    void update(int perStep, bool input = false, int tt = 0)
    {
        std::string filename1 = "curr_On.csv";
        std::string filename2 = "curr_Off.csv";
        frL.resize(perStep);
        frR.resize(perStep);
        frA.resize(perStep);
        frB.resize(perStep);
        std::fill(frL.begin(), frL.end(), 0);
        std::fill(frR.begin(), frR.end(), 0);
        std::fill(frA.begin(), frA.end(), 0);
        std::fill(frB.begin(), frB.end(), 0);
        for (int i = 0; i < perStep; i++) //f=jax.numpy.array(t*fps/1000,int)
        {
            std::vector<float> curr2On;
            std::vector<float> curr2Off;
            std::vector<float> zerocurr(numLGNExc, 0.0);
            if (input == true) {
                //输入电流
                curr2On = readCSVRow(filename1, tt);
                curr2Off = readCSVRow(filename2, tt);
            }
            else {
                curr2On = zerocurr;
                curr2Off = zerocurr;
            }

            // LGN神经元更新
            int nid = 0;
            for (auto& neuronOn : lgnPopExcOn) {
                neuronOn.receiveCurrent(curr2On[nid]);
                neuronOn.update(dt);
                nid += 1;
            }
            nid = 0;
            for (auto& neuronOff : lgnPopExcOff) {
                neuronOff.receiveCurrent(curr2Off[nid]);
                neuronOff.update(dt);
                nid += 1;
            }

            // LGN突触更新
            // lgnExcOn2v1ExcM1_AMPA突触更新, AMPA
            for (auto& ampaSyn : lgnExcOn2v1ExcM1_AMPA) {
                ampaSyn.update(dt);
            }
            // lgnExcOn2v1ExcM2_AMPA突触更新, AMPA
            for (auto& ampaSyn : lgnExcOn2v1ExcM2_AMPA) {
                ampaSyn.update(dt);
            }
            // lgnExcOff2v1ExcM1_AMPA突触更新, AMPA
            for (auto& ampaSyn : lgnExcOff2v1ExcM1_AMPA) {
                ampaSyn.update(dt);
            }
            // lgnExcOff2v1ExcM2_AMPA突触更新, AMPA
            for (auto& ampaSyn : lgnExcOff2v1ExcM2_AMPA) {
                ampaSyn.update(dt);
            }

            //V1神经元更新
            for (auto& neuronM1 : v1PopExcM1) {
                neuronM1.input_current = relu(neuronM1.input_current + (-50) * 60);
                neuronM1.update(dt);
            }
            for (auto& neuronM2 : v1PopExcM2) {
                neuronM2.input_current = relu(neuronM2.input_current + (-50) * 60);
                neuronM2.update(dt);
            }

            // v1ExcM12mtExcL_AMPA突触更新, AMPA
            for (auto& ampaSyn : v1ExcM12mtExcL_AMPA) {
                ampaSyn.update(dt);
            }
            // v1ExcM12mtExcR_AMPA突触更新, AMPA
            for (auto& ampaSyn : v1ExcM12mtExcR_AMPA) {
                ampaSyn.update(dt);
            }
            // v1ExcM22mtExcL_AMPA突触更新, AMPA
            for (auto& ampaSyn : v1ExcM22mtExcL_AMPA) {
                ampaSyn.update(dt);
            }
            // v1ExcM22mtExcR_AMPA突触更新, AMPA
            for (auto& ampaSyn : v1ExcM22mtExcR_AMPA) {
                ampaSyn.update(dt);
            }

            //MT神经元更新
            for (auto& neuronL : mtPopExcL) {
                neuronL.update(dt);
                if (neuronL.hasFired()) {
                    frL[i] += 1;
                }
            }
            for (auto& neuronR : mtPopExcR) {
                neuronR.update(dt);
                if (neuronR.hasFired()) {
                    frR[i] += 1;
                }
            }

            // mtExcL2lipExcA_AMPA突触更新, AMPA
            for (auto& ampaSyn : mtExcL2lipExcA_AMPA) {
                ampaSyn.update(dt);
            }
            // mtExcR2lipExcB_AMPA突触更新, AMPA
            for (auto& ampaSyn : mtExcR2lipExcB_AMPA) {
                ampaSyn.update(dt);
            }

            for (auto& neuronI : lipPopInh) {
                neuronI.update(dt);
            }
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

            // float curr2A = 0;
            // float curr2B = 0;
            
            //LIP神经元更新
            for (auto& neuronA : lipPopExcA) {
                neuronA.input_current = neuronA.input_current + 150;
                // curr2A += neuronA.input_current;
                neuronA.update(dt);
                if (neuronA.hasFired()) {
                    frA[i] += 1;
                }
            }
            for (auto& neuronB : lipPopExcB) {
                neuronB.input_current = neuronB.input_current + 150;
                // curr2B += neuronB.input_current;
                neuronB.update(dt);
                if (neuronB.hasFired()) {
                    frB[i] += 1;
                }
            }
            // cout << "curr2A : " << curr2A << endl;
            // cout << "curr2B : " << curr2B << endl;

            for (auto& neuronI : lipPopInh) {
                neuronI.update(dt);
            }
        }
    }
};

vector<float> cal_result(LGN_V1_MT_LIP lgn_v1_mt_lip_net, int simStep, vector<float> fr) {
    for (int i = 0; i < simStep; ++i) {
        fr[0] += lgn_v1_mt_lip_net.frA[i];
    }
    for (int i = 0; i < simStep; ++i) {
        fr[1] += lgn_v1_mt_lip_net.frB[i];
    }
    return fr;
}

int main()
{
    cout << "Decision model init!" << endl;
    auto start_init = high_resolution_clock::now();
    LGN_V1_MT_LIP lgn_v1_mt_lip_net;
    auto end_init = high_resolution_clock::now();
    auto dur_init = duration_cast<microseconds>(end_init - start_init);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model init: " << dur_init.count() / 1000.0 << " ms" << endl;

    int perStep = 10;
    int prepareStep = 100;
    int simStep = 100;
    auto start_update = high_resolution_clock::now();
    vector<float> fr(2, 0);
    lgn_v1_mt_lip_net.update(prepareStep, false, 0);
    for (int tt = 0; tt < simStep; tt++) {
        lgn_v1_mt_lip_net.update(perStep, true, tt);
        fr = cal_result(lgn_v1_mt_lip_net, perStep, fr);
        cout << "step: " << (tt+1)*10 << endl;
        cout << "frA:" << fr[0] << " frB:" << fr[1] << endl;
        if (fr[0] > 192 || fr[1] > 192) {
            break; // 当a或b大于150时，退出循环
        }
    }
    if (fr[0] > fr[1]) {
        cout << "predict: " << 'L' << endl;
    }
    else if (fr[0] < fr[1]) {
        cout << "predict: " << 'R' << endl;
    }
    else {
        cout << "predict: " << 'N' << endl;
    }
    //cout << "final_fr_L: " << fr[0] << " final_fr_R:" << fr[1] << endl;

    auto end_update = high_resolution_clock::now();
    auto dur_update = duration_cast<microseconds>(end_update - start_update);
    // 输出耗时时间（以毫秒为单位）
    cout << "Cost time of model update: " << dur_update.count() / 1000.0 << " ms" << endl;
    
}