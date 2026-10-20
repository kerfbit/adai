// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-10-19

#ifdef ADAI_ENABLE_GPU

#include "GPUOptimizer.hpp"
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include "Logger.hpp"

namespace {

using adai::gpu::GPUMatrix;
using adai::gpu::GPUOptimizerRule;

std::vector<float> flatten(const Matrix& m) {
    std::vector<float> v;
    v.reserve(static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols));
    for (const auto& row : m.data)
        v.insert(v.end(), row.begin(), row.end());
    return v;
}

void unflatten(const std::vector<float>& v, Matrix& m) {
    size_t idx = 0;
    for (auto& row : m.data)
        for (auto& x : row)
            x = v[idx++];
}

float host_sq_norm(const Matrix& m) {
    double sum = 0.0;
    for (const auto& row : m.data)
        for (float x : row)
            sum += static_cast<double>(x) * static_cast<double>(x);
    return static_cast<float>(sum);
}

bool uses_first_moment(OptimizerType t) {
    return t != OptimizerType::SGD;
}

bool uses_second_moment(OptimizerType t) {
    return t == OptimizerType::ADAM || t == OptimizerType::ADAMW;
}

GPUOptimizerRule rule_of(OptimizerType t) {
    switch (t) {
        case OptimizerType::SGD:
            return GPUOptimizerRule::SGD;
        case OptimizerType::SGD_MOMENTUM:
            return GPUOptimizerRule::SGD_MOMENTUM;
        case OptimizerType::ADAM:
            return GPUOptimizerRule::ADAM;
        case OptimizerType::ADAMW:
            return GPUOptimizerRule::ADAMW;
    }
    throw std::invalid_argument("GPUOptimizer: unknown optimizer type");
}

}  // namespace

GPUOptimizer::GPUOptimizer(Optimizer& host, const std::vector<adai::gpu::GPUParamRef>& params)
    : host_(host) {
    std::unordered_map<const Matrix*, const adai::gpu::GPUParamRef*> by_host;
    for (const auto& p : params)
        by_host[p.host] = &p;

    const size_t n = host_.parameter_groups.size();
    layer_of_group_.assign(n, -1);
    sq_norms_.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const ParameterGroup& g = host_.parameter_groups[i];
        const auto it = by_host.find(g.weights);
        if (it == by_host.end()) {
            host_groups_.push_back(i);
            continue;
        }
        const adai::gpu::GPUParamRef& ref = *it->second;
        const int size = g.weights->rows * g.weights->cols;
        if (ref.weights->size() != size || ref.grads->size() != size) {
            throw std::logic_error("GPUOptimizer: device mirror shape differs from its host weight");
        }
        DeviceGroup d{i, ref.weights, ref.grads, nullptr, nullptr, size, ref.layer};
        if (uses_first_moment(host_.type))
            d.m = std::make_unique<GPUMatrix>(g.weights->rows, g.weights->cols);
        if (uses_second_moment(host_.type))
            d.v = std::make_unique<GPUMatrix>(g.weights->rows, g.weights->cols);
        layer_of_group_[i] = ref.layer;
        device_.push_back(std::move(d));
    }
    if (!device_.empty())
        sq_slots_ = std::make_unique<GPUMatrix>(1, static_cast<int>(device_.size()));
    upload_state();
    adai::Logger::info("GPU optimizer: {} parameter group(s) stepped on the device, {} on the host",
                       device_.size(), host_groups_.size());
}

void GPUOptimizer::upload_state() {
    for (auto& d : device_) {
        const ParameterGroup& g = host_.parameter_groups[d.index];
        // The CPU optimizer allocates its moments at registration (zeroed) and fills them from a
        // restored training state (TD-269); start the device moments from them.
        if (d.m) {
            if (g.momentum.rows == g.weights->rows && g.momentum.cols == g.weights->cols) {
                const auto flat = flatten(g.momentum);
                d.m->upload(flat.data(), d.size);
            } else {
                d.m->zero();
            }
        }
        if (d.v) {
            if (g.velocity.rows == g.weights->rows && g.velocity.cols == g.weights->cols) {
                const auto flat = flatten(g.velocity);
                d.v->upload(flat.data(), d.size);
            } else {
                d.v->zero();
            }
        }
    }
}

void GPUOptimizer::download_state() {
    std::vector<float> tmp;
    for (auto& d : device_) {
        ParameterGroup& g = host_.parameter_groups[d.index];
        tmp.resize(static_cast<size_t>(d.size));
        if (d.m) {
            if (g.momentum.rows != g.weights->rows || g.momentum.cols != g.weights->cols)
                g.momentum = Matrix(g.weights->rows, g.weights->cols);
            d.m->download(tmp.data(), d.size);
            unflatten(tmp, g.momentum);
        }
        if (d.v) {
            if (g.velocity.rows != g.weights->rows || g.velocity.cols != g.weights->cols)
                g.velocity = Matrix(g.weights->rows, g.weights->cols);
            d.v->download(tmp.data(), d.size);
            unflatten(tmp, g.velocity);
        }
    }
}

const std::vector<float>& GPUOptimizer::gradient_sq_norms() {
    if (!device_.empty()) {
        sq_slots_->zero();
        float* slots = sq_slots_->device_ptr();
        for (size_t k = 0; k < device_.size(); ++k)
            adai::gpu::matrix_sum_squares_accumulate_gpu(device_[k].grads->device_ptr(),
                                                         device_[k].size, slots + k);
        // The only transfer of the step: one float per device group.
        std::vector<float> partials(device_.size());
        sq_slots_->download(partials.data(), static_cast<int>(device_.size()));
        for (size_t k = 0; k < device_.size(); ++k)
            sq_norms_[device_[k].index] = partials[k];
    }
    for (size_t i : host_groups_)
        sq_norms_[i] = host_sq_norm(*host_.parameter_groups[i].gradients);
    return sq_norms_;
}

adai::gpu::GPUOptimizerUpdate GPUOptimizer::update_params(const ParameterGroup& group,
                                                          float grad_scale) const {
    adai::gpu::GPUOptimizerUpdate u;
    u.rule = rule_of(host_.type);
    u.learning_rate = host_.learning_rate;
    u.momentum_beta = host_.momentum_beta;
    u.beta1 = host_.beta1;
    u.beta2 = host_.beta2;
    u.epsilon = host_.epsilon;
    u.weight_decay = host_.weight_decay;
    u.bias_correction1 = 1.0f - std::pow(host_.beta1, static_cast<float>(group.step));
    u.bias_correction2 = 1.0f - std::pow(host_.beta2, static_cast<float>(group.step));
    u.grad_scale = grad_scale;
    return u;
}

void GPUOptimizer::step(float grad_scale) {
    host_.global_step++;
    const bool skip_zero = host_.weight_decay == 0.0f;

    for (auto& d : device_) {
        ParameterGroup& g = host_.parameter_groups[d.index];
        if (skip_zero && sq_norms_[d.index] == 0.0f)
            continue;  // as Optimizer::step(): no gradient and no decay, so leave it alone
        // Adam/AdamW count their own steps for bias correction (Optimizer::step_adam/adamw).
        if (uses_second_moment(host_.type))
            g.step++;
        adai::gpu::optimizer_update_gpu(d.weights->device_ptr(), d.grads->device_ptr(),
                                        d.m ? d.m->device_ptr() : nullptr,
                                        d.v ? d.v->device_ptr() : nullptr, d.size,
                                        update_params(g, grad_scale));
    }

    for (size_t i : host_groups_) {
        ParameterGroup& g = host_.parameter_groups[i];
        if (skip_zero && sq_norms_[i] == 0.0f)
            continue;
        if (grad_scale != 1.0f) {
            for (auto& row : g.gradients->data)
                for (auto& x : row)
                    x *= grad_scale;
        }
        switch (host_.type) {
            case OptimizerType::SGD:
                host_.step_sgd(g);
                break;
            case OptimizerType::SGD_MOMENTUM:
                host_.step_sgd_momentum(g);
                break;
            case OptimizerType::ADAM:
                host_.step_adam(g);
                break;
            case OptimizerType::ADAMW:
                host_.step_adamw(g);
                break;
        }
    }
}

float GPUOptimizer::weight_norm() {
    double total = 0.0;
    if (!device_.empty()) {
        sq_slots_->zero();
        float* slots = sq_slots_->device_ptr();
        for (size_t k = 0; k < device_.size(); ++k)
            adai::gpu::matrix_sum_squares_accumulate_gpu(device_[k].weights->device_ptr(),
                                                         device_[k].size, slots + k);
        std::vector<float> partials(device_.size());
        sq_slots_->download(partials.data(), static_cast<int>(device_.size()));
        for (float p : partials)
            total += p;
    }
    for (size_t i : host_groups_)
        total += host_sq_norm(*host_.parameter_groups[i].weights);
    return static_cast<float>(std::sqrt(total));
}

void GPUOptimizer::zero_host_grads() {
    for (size_t i : host_groups_)
        host_.parameter_groups[i].gradients->fill(0.0f);
}

float GPUOptimizer::layer_gradient_norm(int layer) const {
    double total = 0.0;
    for (size_t i = 0; i < sq_norms_.size(); ++i)
        if (layer_of_group_[i] == layer)
            total += sq_norms_[i];
    return static_cast<float>(std::sqrt(total));
}

#endif  // ADAI_ENABLE_GPU
