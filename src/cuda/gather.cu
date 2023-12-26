#include "atomic.h"
#include "cuda_common.h"
#include "gather.h"

#define CACHE_LINE_SIZE 128

namespace offgs {
namespace cuda {
template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernel(
    const DType* const host_array, const DType* const device_array,
    const int64_t num_feat, const IdType* const in_index,
    const IdType* const out_index, const int64_t length, DType* const out,
    const MapIdType* map_ptr, const int64_t map_len) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    assert(in_row >= 0 && in_row < map_len);
    MapIdType in_pos;
    const DType* array;
    if (map_ptr[in_row] > 0) {
      in_pos = map_ptr[in_row] - 1;
      array = host_array;
    } else if (map_ptr[in_row] < 0) {
      in_pos = -map_ptr[in_row] - 1;
      array = device_array;
    }
    const IdType out_row = out_index[out_row_index];
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

/**
 *  This is a cross-device access version of IndexSelectMultiKernel.
 *  Since the memory access over PCIe is more sensitive to the
 *  data access aligment (cacheline), we need a separate version here.
 */
template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const host_array, const DType* const device_array,
    const int64_t num_feat, const IdType* const in_index,
    const IdType* const out_index, const int64_t length, DType* const out,
    const MapIdType* map_ptr, const int64_t map_len) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    assert(in_row >= 0 && in_row < map_len);
    MapIdType in_pos;
    const DType* array;
    if (map_ptr[in_row] > 0) {
      in_pos = map_ptr[in_row] - 1;
      array = host_array;
    } else if (map_ptr[in_row] < 0) {
      in_pos = -map_ptr[in_row] - 1;
      array = device_array;
    }
    const int64_t idx_offset =
        ((uint64_t)(&array[in_pos * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const IdType out_row = out_index[out_row_index];
    while (col < num_feat) {
      if (col >= 0)
        out[out_row * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

void GatherInGPU(torch::Tensor& out, const torch::Tensor& out_idx,
                 const torch::Tensor& in_cpu, const torch::Tensor& in_gpu,
                 const torch::Tensor& in_idx, const torch::Tensor& map_table) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(out_idx.numel() == in_idx.numel());
  auto num_idx = in_idx.numel();
  assert(out.sizes()[1] == in_cpu.sizes()[1]);
  auto feature_dim = out.sizes()[1];

  auto out_dataptr = out.data_ptr<float>();
  auto in_cpu_dataptr = in_cpu.data_ptr<float>();
  auto in_gpu_dataptr = in_gpu.data_ptr<float>();
  auto map_ptr = map_table.data_ptr<int32_t>();
  auto in_idx_ptr = in_idx.data_ptr<int64_t>();
  auto out_idx_ptr = out_idx.data_ptr<int64_t>();

  dim3 block(256, 1);
  while (static_cast<int64_t>(block.x) >= 2 * feature_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_idx + block.y - 1) / block.y);
  if (feature_dim * sizeof(float) < 2 * CACHE_LINE_SIZE) {
    CUDA_KERNEL_CALL((IndexSelectMultiKernel<float, int64_t, int32_t>), grid,
                     block, 0, stream, in_cpu_dataptr, in_gpu_dataptr,
                     feature_dim, in_idx_ptr, out_idx_ptr, num_idx, out_dataptr,
                     map_ptr, map_table.numel());
  } else {
    CUDA_KERNEL_CALL((IndexSelectMultiKernelAligned<float, int64_t, int32_t>),
                     grid, block, 0, stream, in_cpu_dataptr, in_gpu_dataptr,
                     feature_dim, in_idx_ptr, out_idx_ptr, num_idx, out_dataptr,
                     map_ptr, map_table.numel());
  }
}
}  // namespace cuda
}  // namespace offgs