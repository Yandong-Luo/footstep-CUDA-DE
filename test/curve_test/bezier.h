#ifndef CPP_BEZIER_CURVE_H
#define CPP_BEZIER_CURVE_H

#include <vector>
#include <cmath>
#include <iostream>

// 与CUDA代码保持一致的常量定义
#define BEZIER_SIZE 7  // 6阶贝塞尔曲线有7个控制点
#define N 30  // 步长

// 初始状态和目标状态
const float init_state[5] = {0.29357406, 0.29125562, -0.01193462, -0.01774755, 1.58432257}; // x, y, dx, dy, theta
const float target_state[5] = {1.5, 2.8, 0, 0, 0}; // x, y, dx, dy, theta

// 简单的2D向量结构
struct float2 {
    float x, y;
    
    float2() : x(0.0f), y(0.0f) {}
    float2(float _x, float _y) : x(_x), y(_y) {}
};

// 状态向量结构（类似CUDA代码中的footstep::StateVector）
struct StateVector {
    float data[5];  // x, y, dx, dy, theta
};

// C++版本的贝塞尔曲线实现
class BezierCurve {
public:
    BezierCurve() {
        // 初始化二项式系数
        prepareBinomial();
        // 准备查表权重
        prepareWeightLookup();
    }
    
    // 直接计算贝塞尔曲线上的点
    float2 getPositionDirect(const std::vector<float>& paramsX, const std::vector<float>& paramsY, float t) const {
        std::vector<float> tPowers(BEZIER_SIZE);
        std::vector<float> oneMinusTpowers(BEZIER_SIZE);
        
        tPowers[0] = oneMinusTpowers[0] = 1.0f;
        for (int i = 1; i < BEZIER_SIZE; ++i) {
            tPowers[i] = tPowers[i - 1] * t;
            oneMinusTpowers[i] = oneMinusTpowers[i - 1] * (1.0f - t);
        }
        
        float2 result;
        result.x = result.y = 0.0f;
        
        for (int i = 0; i < BEZIER_SIZE; ++i) {
            float bernsteinT = binomialCoeff_[i] * tPowers[i] * oneMinusTpowers[BEZIER_SIZE - 1 - i];
            result.x += bernsteinT * paramsX[i];
            result.y += bernsteinT * paramsY[i];
        }
        
        return result;
    }
    
    // 直接计算贝塞尔曲线上的速度
    float2 getVelocityDirect(const std::vector<float>& paramsX, const std::vector<float>& paramsY, float t) const {
        std::vector<float> tPowers(BEZIER_SIZE);
        std::vector<float> oneMinusTpowers(BEZIER_SIZE);
        
        tPowers[0] = oneMinusTpowers[0] = 1.0f;
        for (int i = 1; i < BEZIER_SIZE - 1; ++i) {
            tPowers[i] = tPowers[i - 1] * t;
            oneMinusTpowers[i] = oneMinusTpowers[i - 1] * (1.0f - t);
        }
        
        float2 velocity;
        velocity.x = velocity.y = 0.0f;
        int n = BEZIER_SIZE - 1;
        
        for (int i = 0; i < n; ++i) {
            float bernsteinT = n * binomialCoeff_[i] * tPowers[i] * oneMinusTpowers[n - 1 - i];
            velocity.x += bernsteinT * (paramsX[i + 1] - paramsX[i]);
            velocity.y += bernsteinT * (paramsY[i + 1] - paramsY[i]);
        }
        
        return velocity;
    }
    
    // 直接计算极坐标到笛卡尔坐标的完整状态
    void getStateDirect(StateVector* state, const std::vector<float>& paramsX, const std::vector<float>& paramsY, float t) const {
        float2 position = getPositionDirect(paramsX, paramsY, t);
        float2 velocity = getVelocityDirect(paramsX, paramsY, t);
        
        // 从极坐标转换到笛卡尔坐标
        float radius = position.x;
        float theta = position.y;
        float vR = velocity.x;
        float vTheta = velocity.y;
        
        state->data[0] = radius * cos(theta);  // x
        state->data[1] = radius * sin(theta);  // y
        state->data[2] = vR * cos(theta) - radius * vTheta * sin(theta);  // dx
        state->data[3] = vR * sin(theta) + radius * vTheta * cos(theta);  // dy
        state->data[4] = theta;  // theta
    }
    
    // 使用查表获取贝塞尔曲线位置
    float2 getPositionLookup(const std::vector<float>& paramsX, const std::vector<float>& paramsY, int step) const {
        float2 result;
        result.x = result.y = 0.0f;
        
        for (int i = 0; i < BEZIER_SIZE; ++i) {
            float bernsteinT = bernsteinWeights_[step][i];
            result.x += bernsteinT * paramsX[i];
            result.y += bernsteinT * paramsY[i];
        }
        
        return result;
    }
    
    // 使用查表获取贝塞尔曲线速度
    float2 getVelocityLookup(const std::vector<float>& paramsX, const std::vector<float>& paramsY, int step) const {
        float2 velocity;
        velocity.x = velocity.y = 0.0f;
        
        for (int i = 0; i < BEZIER_SIZE - 1; ++i) {
            float bernsteinDerivT = bernsteinDerivWeights_[step][i];
            velocity.x += bernsteinDerivT * (paramsX[i + 1] - paramsX[i]);
            velocity.y += bernsteinDerivT * (paramsY[i + 1] - paramsY[i]);
        }
        
        return velocity;
    }
    
    // 使用查表获取完整状态
    void getStateLookup(StateVector* state, const std::vector<float>& paramsX, const std::vector<float>& paramsY, int step) const {
        float2 position = getPositionLookup(paramsX, paramsY, step);
        float2 velocity = getVelocityLookup(paramsX, paramsY, step);
        
        // 从极坐标转换到笛卡尔坐标
        float radius = position.x;
        float theta = position.y;
        float vR = velocity.x;
        float vTheta = velocity.y;
        
        state->data[0] = radius * cos(theta);  // x
        state->data[1] = radius * sin(theta);  // y
        state->data[2] = vR * cos(theta) - radius * vTheta * sin(theta);  // dx
        state->data[3] = vR * sin(theta) + radius * vTheta * cos(theta);  // dy
        state->data[4] = theta;  // theta
    }
    
    // 获取二项式系数
    const std::vector<float>& getBinomialCoeff() const {
        return binomialCoeff_;
    }
    
    // 获取Bernstein权重
    const std::vector<std::vector<float>>& getBernsteinWeights() const {
        return bernsteinWeights_;
    }
    
    // 获取Bernstein导数权重
    const std::vector<std::vector<float>>& getBernsteinDerivWeights() const {
        return bernsteinDerivWeights_;
    }

private:
    std::vector<float> binomialCoeff_;  // 二项式系数
    std::vector<std::vector<float>> bernsteinWeights_;  // 预计算的Bernstein多项式权重
    std::vector<std::vector<float>> bernsteinDerivWeights_;  // 预计算的Bernstein导数权重
    
    // 使用Pascal三角计算二项式系数
    void prepareBinomial() {
        binomialCoeff_.resize(BEZIER_SIZE);
        std::vector<std::vector<float>> tmpBinomialCoeff(BEZIER_SIZE, std::vector<float>(BEZIER_SIZE + 1));
        
        tmpBinomialCoeff[0][0] = tmpBinomialCoeff[1][1] = 1;
        tmpBinomialCoeff[0][1] = 0;
        tmpBinomialCoeff[1][0] = 1;
        
        // Pascal's Triangle
        for (int i = 2; i < BEZIER_SIZE; ++i) {
            tmpBinomialCoeff[i][0] = tmpBinomialCoeff[i][i] = 1;
            for (int j = 1; j < i; ++j) {
                tmpBinomialCoeff[i][j] = tmpBinomialCoeff[i - 1][j - 1] + tmpBinomialCoeff[i - 1][j];
            }
        }
        
        // 保存系数
        for (int i = 0; i < BEZIER_SIZE; ++i) {
            binomialCoeff_[i] = tmpBinomialCoeff[BEZIER_SIZE - 1][i];
        }
    }
    
    // 预计算Bernstein权重
    void prepareWeightLookup() {
        bernsteinWeights_.resize(N + 1);
        bernsteinDerivWeights_.resize(N + 1);
        
        for (int i = 0; i <= N; ++i) {
            float t = static_cast<float>(i) / N;
            bernsteinWeights_[i].resize(BEZIER_SIZE);
            bernsteinDerivWeights_[i].resize(BEZIER_SIZE);
            
            std::vector<float> tPowers(BEZIER_SIZE);
            std::vector<float> oneMinusTpowers(BEZIER_SIZE);
            
            tPowers[0] = oneMinusTpowers[0] = 1.0f;
            for (int j = 1; j < BEZIER_SIZE; ++j) {
                tPowers[j] = tPowers[j - 1] * t;
                oneMinusTpowers[j] = oneMinusTpowers[j - 1] * (1.0f - t);
            }
            
            for (int j = 0; j < BEZIER_SIZE; ++j) {
                bernsteinWeights_[i][j] = binomialCoeff_[j] * tPowers[j] * oneMinusTpowers[BEZIER_SIZE - 1 - j];
            }
            
            // 为速度计算预计算
            for (int j = 0; j < BEZIER_SIZE - 1; ++j) {
                bernsteinDerivWeights_[i][j] = (BEZIER_SIZE - 1) * (bernsteinWeights_[i][j + 1] - bernsteinWeights_[i][j]);
            }
            bernsteinDerivWeights_[i][BEZIER_SIZE - 1] = 0;
        }
    }
};

// 从笛卡尔坐标转换到极坐标
float2 cartesianToPolar(float x, float y) {
    float2 result;
    result.x = sqrt(x*x + y*y);  // radius
    result.y = atan2(y, x);      // theta
    return result;
}

// 计算贝塞尔控制点
void calculateBezierControlPoints(std::vector<float>& paramsX, std::vector<float>& paramsY) {
    // 将初始和目标状态转换为极坐标
    float2 initialPos = cartesianToPolar(init_state[0], init_state[1]);
    float2 targetPos = cartesianToPolar(target_state[0], target_state[1]);
    
    // 设置第一个和最后一个控制点匹配端点
    paramsX[0] = initialPos.x;
    paramsY[0] = initialPos.y;
    paramsX[BEZIER_SIZE - 1] = targetPos.x;
    paramsY[BEZIER_SIZE - 1] = targetPos.y;
    
    // 计算初始速度的极坐标表示
    float init_speed = sqrt(init_state[2]*init_state[2] + init_state[3]*init_state[3]);
    float init_angle = atan2(init_state[3], init_state[2]);
    
    // 调整第二个控制点以匹配初始速度
    float dr = init_speed * cos(init_angle - initialPos.y);
    float dtheta = init_speed * sin(init_angle - initialPos.y) / initialPos.x;
    
    paramsX[1] = paramsX[0] + dr / (BEZIER_SIZE - 1) * 3.0;
    paramsY[1] = paramsY[0] + dtheta / (BEZIER_SIZE - 1) * 3.0;
    
    // 调整倒数第二个控制点以匹配最终速度（本例中为零）
    paramsX[BEZIER_SIZE - 2] = paramsX[BEZIER_SIZE - 1];
    paramsY[BEZIER_SIZE - 2] = paramsY[BEZIER_SIZE - 1];
    
    // 将其余控制点均匀分布
    for (int i = 2; i < BEZIER_SIZE - 2; ++i) {
        float t = static_cast<float>(i) / (BEZIER_SIZE - 1);
        paramsX[i] = initialPos.x * (1.0f - t) + targetPos.x * t;
        paramsY[i] = initialPos.y * (1.0f - t) + targetPos.y * t;
    }
}

#endif // CPP_BEZIER_CURVE_H