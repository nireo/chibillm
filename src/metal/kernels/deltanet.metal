#include "common.metalh"

// DeltaNet primitives use f32 activations and state with native checkpoint
// weight types. Each dispatch processes consecutive tokens from one sequence.
inline float
deltanet_sigmoid(float x)
{
    const float e = exp(-abs(x));
    return x >= 0.0F ? 1.0F / (1.0F + e) : e / (1.0F + e);
}

kernel void
causal_conv1d_silu(device const float* input [[buffer(0)]],
                   device const bf16_storage* weight [[buffer(1)]],
                   device float* history [[buffer(2)]],
                   device float* output [[buffer(3)]],
                   constant uint* geometry [[buffer(4)]],
                   uint channel [[thread_position_in_grid]])
{
    const uint rows = geometry[0], channels = geometry[1], kernel_size = geometry[2];
    if (channel >= channels)
        return;
    const ulong base = ulong(channel) * kernel_size;
    for (uint t = 0; t < rows; ++t) {
        const ulong index = ulong(t) * channels + channel;
        const float current = input[index];
        float sum = current * load_bf16(weight[base + kernel_size - 1]);
        // One thread owns a channel, so shifting its raw history is race-free.
        for (uint j = 0; j + 1 < kernel_size; ++j) {
            const float previous = history[base + j + 1];
            sum += previous * load_bf16(weight[base + j]);
            history[base + j] = previous;
        }
        history[base + kernel_size - 1] = current;
        output[index] = sum * deltanet_sigmoid(sum);
    }
}

kernel void
gated_delta_rule(device const float* qkv [[buffer(0)]],
                 device const float* a [[buffer(1)]],
                 device const float* b [[buffer(2)]],
                 device const float* A_log [[buffer(3)]],
                 device const bf16_storage* dt_bias [[buffer(4)]],
                 device float* state [[buffer(5)]],
                 device float* output [[buffer(6)]],
                 constant uint* geometry [[buffer(7)]],
                 constant float& epsilon [[buffer(8)]],
                 uint column [[thread_position_in_grid]])
{
    const uint rows = geometry[0], key_heads = geometry[1], value_heads = geometry[2];
    const uint key_dim = geometry[3], value_dim = geometry[4];
    const uint value_width = value_heads * value_dim;
    if (column >= value_width)
        return;
    const uint head = column / value_dim, value = column % value_dim;
    const uint key_head = head / (value_heads / key_heads);
    const uint key_width = key_heads * key_dim, packed_width = 2 * key_width + value_width;
    const ulong state_base = ulong(head) * key_dim * value_dim + value;
    const float decay_rate = exp(A_log[head]);
    const float bias = load_bf16(dt_bias[head]);
    // Threads own independent state columns; value columns are contiguous in
    // memory. The sequential scan is also a correctness baseline for prefill.
    for (uint t = 0; t < rows; ++t) {
        const ulong q_base = ulong(t) * packed_width + key_head * key_dim;
        const ulong k_base = q_base + key_width;
        float q_square = 0.0F, k_square = 0.0F;
        for (uint j = 0; j < key_dim; ++j) {
            q_square += qkv[q_base + j] * qkv[q_base + j];
            k_square += qkv[k_base + j] * qkv[k_base + j];
        }
        const float q_scale = rsqrt(q_square + epsilon) * rsqrt(float(key_dim));
        const float k_scale = rsqrt(k_square + epsilon);
        const ulong gate_index = ulong(t) * value_heads + head;
        const float x = a[gate_index] + bias;
        // Metal has no log1p; preserve the small tail when 1 + tail rounds to 1.
        const float tail = exp(-abs(x));
        const float log_term = tail < 1e-4F ? tail * (1.0F - 0.5F * tail) : log(1.0F + tail);
        const float softplus = max(x, 0.0F) + log_term;
        const float decay = exp(-decay_rate * softplus);
        const float beta = deltanet_sigmoid(b[gate_index]);
        float prediction = 0.0F;
        for (uint j = 0; j < key_dim; ++j) {
            const ulong index = state_base + ulong(j) * value_dim;
            state[index] *= decay;
            prediction += state[index] * (qkv[k_base + j] * k_scale);
        }
        const float v = qkv[ulong(t) * packed_width + 2 * key_width + column];
        const float delta = (v - prediction) * beta;
        float result = 0.0F;
        for (uint j = 0; j < key_dim; ++j) {
            const ulong index = state_base + ulong(j) * value_dim;
            const float updated = state[index] + qkv[k_base + j] * k_scale * delta;
            state[index] = updated;
            result += updated * (qkv[q_base + j] * q_scale);
        }
        output[ulong(t) * value_width + column] = result;
    }
}

// A bounded 32-token block. Q/K normalization is shared by all value columns
// and grouped value heads. The remaining kernels implement
// D_i = beta_i (V_i - exp(G_i) K_i S_0)
//       - sum_{j<i} beta_i exp(G_i-G_j) (K_i K_j^T) D_j,
// O_i = exp(G_i) Q_i S_0 + sum_{j<=i} exp(G_i-G_j) (Q_i K_j^T) D_j.
// Decays use sums over the actual interval, never division by exp(G_j), so
// underflow / a complete state reset cannot introduce 0/0 or inf-inf.
constant uint delta_block = 32;

kernel void
delta_prepare(device const float* qkv [[buffer(0)]],
              device const float* a [[buffer(1)]],
              device const float* b [[buffer(2)]],
              device const float* logs [[buffer(3)]],
              device const bf16_storage* bias [[buffer(4)]],
              device float* qk [[buffer(9)]],
              device float* gates [[buffer(10)]],
              constant uint* g [[buffer(7)]],
              constant float& epsilon [[buffer(8)]],
              uint2 id [[thread_position_in_grid]])
{
    uint t = id.x, h = id.y;
    uint n = g[0], kh = g[1], vh = g[2], kd = g[3], vd = g[4];
    if (t >= n || h >= vh)
        return;
    uint kw = kh * kd, packed = 2 * kw + vh * vd;
    if (h < kh) {
        float qs = epsilon, ks = epsilon;
        for (uint k = 0; k < kd; ++k) {
            float q = qkv[ulong(t) * packed + h * kd + k];
            float v = qkv[ulong(t) * packed + kw + h * kd + k];
            qs += q * q;
            ks += v * v;
        }
        float qscale = rsqrt(qs) * rsqrt(float(kd)), kscale = rsqrt(ks);
        for (uint k = 0; k < kd; ++k) {
            qk[ulong(t) * 2 * kw + h * kd + k] = qkv[ulong(t) * packed + h * kd + k] * qscale;
            qk[ulong(t) * 2 * kw + kw + h * kd + k] =
                qkv[ulong(t) * packed + kw + h * kd + k] * kscale;
        }
    }
    float x = a[ulong(t) * vh + h] + load_bf16(bias[h]);
    float tail = exp(-abs(x));
    float softplus = max(x, 0.0F) + (tail < 1e-4F ? tail * (1.0F - 0.5F * tail) : log(1.0F + tail));
    ulong base = (ulong(h) * delta_block + t) * 3;
    gates[base] = -exp(logs[h]) * softplus;
    gates[base + 1] = deltanet_sigmoid(b[ulong(t) * vh + h]);
}

kernel void
delta_products(device const float* qk [[buffer(9)]],
               device float* gates [[buffer(10)]],
               device float* products [[buffer(11)]],
               constant uint* g [[buffer(7)]],
               uint3 id [[thread_position_in_grid]])
{
    uint i = id.x, j = id.y, h = id.z;
    uint n = g[0], kh = g[1], vh = g[2], kd = g[3];
    if (i >= n || j > i || h >= vh)
        return;
    uint kw = kh * kd, head = h / (vh / kh);
    ulong gi = (ulong(h) * delta_block + i) * 3;
    if (j == 0) {
        float prefix = 0;
        for (uint t = 0; t <= i; ++t)
            prefix += gates[(ulong(h) * delta_block + t) * 3];
        gates[gi + 2] = exp(prefix);
    }
    float interval = 0;
    for (uint t = j + 1; t <= i; ++t)
        interval += gates[(ulong(h) * delta_block + t) * 3];
    float decay = exp(interval), kk = 0, qk_dot = 0;
    for (uint k = 0; k < kd; ++k) {
        float kj = qk[ulong(j) * 2 * kw + kw + head * kd + k];
        kk += qk[ulong(i) * 2 * kw + kw + head * kd + k] * kj;
        qk_dot += qk[ulong(i) * 2 * kw + head * kd + k] * kj;
    }
    ulong p = ((ulong(h) * delta_block + i) * delta_block + j) * 3;
    products[p] = kk * decay * gates[gi + 1];
    products[p + 1] = qk_dot * decay;
    products[p + 2] = decay;
}

// Parallel matrix products K S_0 and Q S_0; state remains read-only here.
kernel void
delta_project(device const float* qkv [[buffer(0)]],
              device const float* state [[buffer(5)]],
              device const float* qk [[buffer(9)]],
              device const float* gates [[buffer(10)]],
              device float* work [[buffer(12)]],
              constant uint* g [[buffer(7)]],
              uint2 id [[thread_position_in_grid]])
{
    uint c = id.x, t = id.y;
    uint n = g[0], kh = g[1], vh = g[2], kd = g[3], vd = g[4];
    if (c >= vh * vd || t >= n)
        return;
    uint h = c / vd, v = c % vd, kw = kh * kd, head = h / (vh / kh);
    float ks = 0, qs = 0;
    for (uint k = 0; k < kd; ++k) {
        float s = state[(ulong(h) * kd + k) * vd + v];
        ks += qk[ulong(t) * 2 * kw + kw + head * kd + k] * s;
        qs += qk[ulong(t) * 2 * kw + head * kd + k] * s;
    }
    ulong gate = (ulong(h) * delta_block + t) * 3;
    ulong w = ((ulong(h) * delta_block + t) * vd + v) * 3;
    float value = qkv[ulong(t) * (2 * kw + vh * vd) + 2 * kw + c];
    work[w] = gates[gate + 1] * (value - gates[gate + 2] * ks);
    work[w + 1] = gates[gate + 2] * qs;
}

// Only the small unit-lower-triangular solve remains sequential, independently
// for each value column. All key-dimension products are outside this scan.
kernel void
delta_solve(device const float* products [[buffer(11)]],
            device float* work [[buffer(12)]],
            constant uint* g [[buffer(7)]],
            uint c [[thread_position_in_grid]])
{
    uint n = g[0], vh = g[2], vd = g[4];
    if (c >= vh * vd)
        return;
    uint h = c / vd, v = c % vd;
    float d[32];
    for (uint i = 0; i < n; ++i) {
        ulong w = ((ulong(h) * delta_block + i) * vd + v) * 3;
        float value = work[w];
        for (uint j = 0; j < i; ++j)
            value -= products[((ulong(h) * delta_block + i) * delta_block + j) * 3] * d[j];
        d[i] = value;
        work[w + 2] = value;
    }
}

// Parallel output matrix product and one state update per block. Output uses
// saved Q S_0, so it never races the state writes in this dispatch.
kernel void
delta_finish(device float* state [[buffer(5)]],
             device float* output [[buffer(6)]],
             device const float* qk [[buffer(9)]],
             device const float* gates [[buffer(10)]],
             device const float* products [[buffer(11)]],
             device const float* work [[buffer(12)]],
             constant uint* g [[buffer(7)]],
             uint2 id [[thread_position_in_grid]])
{
    uint c = id.x, i = id.y;
    uint n = g[0], kh = g[1], vh = g[2], kd = g[3], vd = g[4];
    if (c >= vh * vd)
        return;
    uint h = c / vd, v = c % vd, kw = kh * kd, head = h / (vh / kh);
    if (i < n) {
        float o = work[((ulong(h) * delta_block + i) * vd + v) * 3 + 1];
        for (uint j = 0; j <= i; ++j)
            o += products[((ulong(h) * delta_block + i) * delta_block + j) * 3 + 1]
                * work[((ulong(h) * delta_block + j) * vd + v) * 3 + 2];
        output[ulong(i) * vh * vd + c] = o;
    }
    if (i < kd) {
        ulong s = (ulong(h) * kd + i) * vd + v;
        float updated = gates[(ulong(h) * delta_block + n - 1) * 3 + 2] * state[s];
        for (uint j = 0; j < n; ++j)
            updated += qk[ulong(j) * 2 * kw + kw + head * kd + i]
                * products[((ulong(h) * delta_block + n - 1) * delta_block + j) * 3 + 2]
                * work[((ulong(h) * delta_block + j) * vd + v) * 3 + 2];
        state[s] = updated;
    }
}

kernel void
rms_norm_gated(device const float* input [[buffer(0)]],
               device const float* gate [[buffer(1)]],
               device const float* weight [[buffer(2)]],
               device float* output [[buffer(3)]],
               constant uint* geometry [[buffer(4)]],
               constant float& epsilon [[buffer(5)]],
               uint group [[thread_position_in_grid]])
{
    const uint groups = geometry[0], width = geometry[1];
    if (group >= groups)
        return;
    const ulong base = ulong(group) * width;
    float square = 0.0F;
    for (uint j = 0; j < width; ++j)
        square += input[base + j] * input[base + j];
    const float scale = rsqrt(square / float(width) + epsilon);
    for (uint j = 0; j < width; ++j) {
        const float z = gate[base + j];
        output[base + j] = input[base + j] * scale * weight[j] * z * deltanet_sigmoid(z);
    }
}
