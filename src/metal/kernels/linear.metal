#include "common.metalh"

#if defined(CHIBILLM_ENABLE_TENSOROPS) && __METAL_VERSION__ >= 400 && defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Metal 4 TensorOps kernel for A [rows, input_features] multiplied by the
// transpose of row-major BF16 weights [output_features, input_features]. The
// inline tensor views keep the existing MTLBuffers and command submission path;
// Apple10/M5 executes the matmul operation on each GPU core's neural accelerator.
kernel void
linear_bf16_tensorops(device float* input [[buffer(0)]],
                      device bfloat* weight [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& rows [[buffer(3)]],
                      constant uint& input_features [[buffer(4)]],
                      constant uint& output_features [[buffer(5)]],
                      uint2 threadgroup_position [[threadgroup_position_in_grid]])
{
    constexpr int tile_size = 64;
    const int output_origin = int(threadgroup_position.x) * tile_size;
    const int row_origin = int(threadgroup_position.y) * tile_size;

    auto input_tensor = tensor(input, dextents<int, 2>(int(input_features), int(rows)));
    auto weight_tensor =
        tensor(weight, dextents<int, 2>(int(input_features), int(output_features)));
    auto output_tensor = tensor(output, dextents<int, 2>(int(output_features), int(rows)));

    // Weight storage is [output, input], hence transpose_right=true.
    constexpr auto descriptor =
        matmul2d_descriptor(tile_size, tile_size, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    auto input_slice = input_tensor.slice(0, row_origin);
    auto weight_slice = weight_tensor.slice(0, output_origin);
    auto output_slice = output_tensor.slice(output_origin, row_origin);
    auto product = operation.get_destination_cooperative_tensor<decltype(input_slice),
                                                                decltype(weight_slice), float>();
    operation.run(input_slice, weight_slice, product);
    product.store(output_slice);
}
#endif

template <typename Matrix>
static inline void
linear_add_decode_impl(device const float* input,
                       Matrix weight,
                       device const float* residual,
                       device float* output,
                       constant uint& input_features,
                       constant uint& output_features,
                       constant uint& outputs_per_threadgroup,
                       constant uint& simd_width,
                       uint lane,
                       uint simdgroup,
                       uint3 threadgroup_position)
{
    const uint output_feature = threadgroup_position.x * outputs_per_threadgroup + simdgroup;
    if (output_feature >= output_features) {
        return;
    }

    const float accumulator = weight.dot(input + ulong(threadgroup_position.y) * input_features,
                                         output_feature, lane, simd_width);
    if (lane == 0) {
        const ulong index = ulong(threadgroup_position.y) * output_features + output_feature;
        output[index] = residual[index] + accumulator;
    }
}

kernel void
linear_add_bf16_decode(device const float* input [[buffer(0)]],
                       device const bf16_storage* weight [[buffer(1)]],
                       device const float* residual [[buffer(2)]],
                       device float* output [[buffer(3)]],
                       constant uint& input_features [[buffer(4)]],
                       constant uint& output_features [[buffer(5)]],
                       constant uint& outputs_per_threadgroup [[buffer(6)]],
                       constant uint& simd_width [[buffer(7)]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simdgroup [[simdgroup_index_in_threadgroup]],
                       uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    linear_add_decode_impl(input, bf16_matrix { weight, input_features }, residual, output,
                           input_features, output_features, outputs_per_threadgroup, simd_width,
                           lane, simdgroup, threadgroup_position);
}

kernel void
linear_add_q4_decode(device const float* input [[buffer(0)]],
                     device const uint* weight [[buffer(1)]],
                     device const float* residual [[buffer(2)]],
                     device float* output [[buffer(3)]],
                     constant uint& input_features [[buffer(4)]],
                     constant uint& output_features [[buffer(5)]],
                     constant uint& outputs_per_threadgroup [[buffer(6)]],
                     constant uint& simd_width [[buffer(7)]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simdgroup [[simdgroup_index_in_threadgroup]],
                     device const bf16_storage* scales [[buffer(8)]],
                     device const bf16_storage* offsets [[buffer(9)]],
                     uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    linear_add_decode_impl(input, q4_matrix { weight, scales, offsets, input_features }, residual,
                           output, input_features, output_features, outputs_per_threadgroup,
                           simd_width, lane, simdgroup, threadgroup_position);
}

template <typename Matrix>
static inline void
linear_split_impl(device const float* input,
                  Matrix weight,
                  device float* output_a,
                  device float* output_b,
                  device float* output_c,
                  constant uint& rows,
                  constant uint& input_features,
                  constant uint& width_a,
                  constant uint& width_b,
                  constant uint& width_c,
                  uint2 position)
{
    const uint total_width = width_a + width_b + width_c;
    if (position.x >= total_width || position.y >= rows) {
        return;
    }

    const ulong row = position.y;
    const ulong weight_column = position.x;
    const ulong k_count = input_features;

    device float* output;
    ulong row_length;
    ulong local_column;
    if (weight_column < width_a) {
        output = output_a;
        row_length = width_a;
        local_column = weight_column;
    } else if (weight_column < width_a + width_b) {
        output = output_b;
        row_length = width_b;
        local_column = weight_column - width_a;
    } else {
        output = output_c;
        row_length = width_c;
        local_column = weight_column - width_a - width_b;
    }

    float accumulator = 0.0F;
    for (ulong k = 0; k < k_count; ++k) {
        const float value = weight.load(weight_column, uint(k));
        accumulator += input[row * k_count + k] * value;
    }

    output[row * row_length + local_column] = accumulator;
}

kernel void
linear_split_bf16(device const float* input [[buffer(0)]],
                  device const bf16_storage* weight [[buffer(1)]],
                  device float* output_a [[buffer(2)]],
                  device float* output_b [[buffer(3)]],
                  device float* output_c [[buffer(4)]],
                  constant uint& rows [[buffer(5)]],
                  constant uint& input_features [[buffer(6)]],
                  constant uint& width_a [[buffer(7)]],
                  constant uint& width_b [[buffer(8)]],
                  constant uint& width_c [[buffer(9)]],
                  uint2 position [[thread_position_in_grid]])
{
    linear_split_impl(input, bf16_matrix { weight, input_features }, output_a, output_b, output_c,
                      rows, input_features, width_a, width_b, width_c, position);
}

template <typename Matrix>
static inline void
linear_split_decode_impl(device const float* input,
                         Matrix weight,
                         device float* output_a,
                         device float* output_b,
                         device float* output_c,
                         constant uint& input_features,
                         constant uint& width_a,
                         constant uint& width_b,
                         constant uint& width_c,
                         constant uint& outputs_per_threadgroup,
                         constant uint& simd_width,
                         uint lane,
                         uint simdgroup,
                         uint3 threadgroup_position)
{
    const uint weight_row = threadgroup_position.x * outputs_per_threadgroup + simdgroup;
    const uint total_width = width_a + width_b + width_c;
    if (weight_row >= total_width) {
        return;
    }

    const float accumulator = weight.dot(input + ulong(threadgroup_position.y) * input_features,
                                         weight_row, lane, simd_width);
    if (lane != 0) {
        return;
    }
    if (weight_row < width_a) {
        output_a[ulong(threadgroup_position.y) * width_a + weight_row] = accumulator;
    } else if (weight_row < width_a + width_b) {
        output_b[ulong(threadgroup_position.y) * width_b + weight_row - width_a] = accumulator;
    } else {
        output_c[ulong(threadgroup_position.y) * width_c + weight_row - width_a - width_b] =
            accumulator;
    }
}

kernel void
linear_split_bf16_decode(device const float* input [[buffer(0)]],
                         device const bf16_storage* weight [[buffer(1)]],
                         device float* output_a [[buffer(2)]],
                         device float* output_b [[buffer(3)]],
                         device float* output_c [[buffer(4)]],
                         constant uint& input_features [[buffer(5)]],
                         constant uint& width_a [[buffer(6)]],
                         constant uint& width_b [[buffer(7)]],
                         constant uint& width_c [[buffer(8)]],
                         constant uint& outputs_per_threadgroup [[buffer(9)]],
                         constant uint& simd_width [[buffer(10)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup [[simdgroup_index_in_threadgroup]],
                         uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    linear_split_decode_impl(input, bf16_matrix { weight, input_features }, output_a, output_b,
                             output_c, input_features, width_a, width_b, width_c,
                             outputs_per_threadgroup, simd_width, lane, simdgroup,
                             threadgroup_position);
}

kernel void
linear_split_q4_decode(device const float* input [[buffer(0)]],
                       device const uint* weight [[buffer(1)]],
                       device float* output_a [[buffer(2)]],
                       device float* output_b [[buffer(3)]],
                       device float* output_c [[buffer(4)]],
                       constant uint& input_features [[buffer(5)]],
                       constant uint& width_a [[buffer(6)]],
                       constant uint& width_b [[buffer(7)]],
                       constant uint& width_c [[buffer(8)]],
                       constant uint& outputs_per_threadgroup [[buffer(9)]],
                       constant uint& simd_width [[buffer(10)]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simdgroup [[simdgroup_index_in_threadgroup]],
                       device const bf16_storage* scales [[buffer(11)]],
                       device const bf16_storage* offsets [[buffer(12)]],
                       uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    linear_split_decode_impl(input, q4_matrix { weight, scales, offsets, input_features }, output_a,
                             output_b, output_c, input_features, width_a, width_b, width_c,
                             outputs_per_threadgroup, simd_width, lane, simdgroup,
                             threadgroup_position);
}

kernel void
expand_q4_bf16(device const uint* packed [[buffer(0)]],
               device const bf16_storage* scales [[buffer(1)]],
               device const bf16_storage* offsets [[buffer(2)]],
               device bf16_storage* output [[buffer(3)]],
               constant uint& columns [[buffer(4)]],
               uint2 position [[thread_position_in_grid]])
{
    if (position.x < columns) {
        const q4_matrix weight { packed, scales, offsets, columns };
        output[ulong(position.y) * columns + position.x] =
            store_bf16(weight.load(position.y, position.x));
    }
}
