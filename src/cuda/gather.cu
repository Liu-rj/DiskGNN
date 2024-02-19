#include "atomic.h"
#include "cuda_common.h"
#include "gather.h"

#define CACHE_LINE_SIZE 128

namespace offgs {
namespace cuda {
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(const DType* const host_array,
                                       const DType* const device_array,
                                       const int64_t num_feat,
                                       const IdType* const in_index,
                                       const IdType* const out_index,
                                       const int64_t length, DType* const out,
                                       int64_t boundary, int64_t cache_len) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    assert(in_row >= 0 && in_row < cache_len);
    IdType in_pos = (in_row < boundary) ? in_row : in_row - boundary;
    const DType* array = (in_row < boundary) ? device_array : host_array;
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
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const host_array, const DType* const device_array,
    const int64_t num_feat, const IdType* const in_index,
    const IdType* const out_index, const int64_t length, DType* const out,
    int64_t boundary, int64_t cache_len) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    assert(in_row >= 0 && in_row < cache_len);
    IdType in_pos = (in_row < boundary) ? in_row : in_row - boundary;
    const DType* array = (in_row < boundary) ? device_array : host_array;
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
                 const torch::Tensor& in_idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(out_idx.numel() == in_idx.numel());
  auto num_idx = in_idx.numel();
  assert(out.sizes()[1] == in_cpu.sizes()[1]);
  auto feature_dim = out.sizes()[1];

  auto out_dataptr = out.data_ptr<float>();
  auto in_cpu_dataptr = in_cpu.data_ptr<float>();
  auto in_gpu_dataptr = in_gpu.data_ptr<float>();
  auto in_idx_ptr = in_idx.data_ptr<int64_t>();
  auto out_idx_ptr = out_idx.data_ptr<int64_t>();
  auto boundary = in_gpu.size(0);
  auto cache_len = in_gpu.size(0) + in_cpu.size(0);

  dim3 block(256, 1);
  while (static_cast<int64_t>(block.x) >= 2 * feature_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_idx + block.y - 1) / block.y);
  if (feature_dim * sizeof(float) < 2 * CACHE_LINE_SIZE) {
    CUDA_KERNEL_CALL((IndexSelectMultiKernel<float, int64_t>), grid, block, 0,
                     stream, in_cpu_dataptr, in_gpu_dataptr, feature_dim,
                     in_idx_ptr, out_idx_ptr, num_idx, out_dataptr, boundary,
                     cache_len);
  } else {
    CUDA_KERNEL_CALL((IndexSelectMultiKernelAligned<float, int64_t>), grid,
                     block, 0, stream, in_cpu_dataptr, in_gpu_dataptr,
                     feature_dim, in_idx_ptr, out_idx_ptr, num_idx, out_dataptr,
                     boundary, cache_len);
  }
}

template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernelMegaBatch(
    const DType* const host_array, const DType* const device_array,
    const DType* const host_cold_array, const int64_t num_feat,
    const IdType* const in_index, const IdType* const global_index,
    const int64_t length, DType* const out, const MapIdType* hot_map_ptr,
    const MapIdType* cold_map_ptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t local_in_row = in_index[out_row_index];
    const int64_t global_in_row = global_index[local_in_row];
    MapIdType in_pos;
    const DType* array;
    if (hot_map_ptr[global_in_row] > 0) {
      in_pos = hot_map_ptr[global_in_row] - 1;
      array = host_array;
    } else if (hot_map_ptr[global_in_row] < 0) {
      in_pos = -hot_map_ptr[global_in_row] - 1;
      array = device_array;
    } else {
      in_pos = cold_map_ptr[local_in_row];
      array = host_cold_array;
      assert(in_pos != -1);
    }
    while (col < num_feat) {
      out[out_row_index * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernelAlignedMegaBatch(
    const DType* const host_array, const DType* const device_array,
    const DType* const host_cold_array, const int64_t num_feat,
    const IdType* const in_index, const IdType* const global_index,
    const int64_t length, DType* const out, const MapIdType* hot_map_ptr,
    const MapIdType* cold_map_ptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t local_in_row = in_index[out_row_index];
    const int64_t global_in_row = global_index[local_in_row];
    MapIdType in_pos;
    const DType* array;
    if (hot_map_ptr[global_in_row] > 0) {
      in_pos = hot_map_ptr[global_in_row] - 1;
      array = host_array;
    } else if (hot_map_ptr[global_in_row] < 0) {
      in_pos = -hot_map_ptr[global_in_row] - 1;
      array = device_array;
    } else {
      in_pos = cold_map_ptr[local_in_row];
      array = host_cold_array;
      assert(in_pos != -1);
    }
    const int64_t idx_offset =
        ((uint64_t)(&array[in_pos * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row_index * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

void GatherInGPU_MegaBatch(torch::Tensor& out, const torch::Tensor& in_idx,
                           const torch::Tensor& global_idx,
                           const torch::Tensor& in_cpu,
                           const torch::Tensor& in_gpu,
                           const torch::Tensor& in_cold,
                           const torch::Tensor& hot_map_table,
                           const torch::Tensor& cold_map_table) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(in_idx.numel() == out.sizes()[0]);
  auto num_idx = in_idx.numel();
  assert(out.sizes()[1] == in_cpu.sizes()[1]);
  auto feature_dim = out.sizes()[1];

  auto out_dataptr = out.data_ptr<float>();
  auto in_cpu_dataptr = in_cpu.data_ptr<float>();
  auto in_gpu_dataptr = in_gpu.data_ptr<float>();
  auto in_cold_dataptr = in_cold.data_ptr<float>();
  auto hot_map_ptr = hot_map_table.data_ptr<int32_t>();
  auto cold_map_ptr = cold_map_table.data_ptr<int32_t>();
  auto in_idx_ptr = in_idx.data_ptr<int64_t>();
  auto global_idx_ptr = global_idx.data_ptr<int64_t>();

  dim3 block(256, 1);
  while (static_cast<int64_t>(block.x) >= 2 * feature_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_idx + block.y - 1) / block.y);
  if (feature_dim * sizeof(float) < 2 * CACHE_LINE_SIZE) {
    CUDA_KERNEL_CALL((IndexSelectMultiKernelMegaBatch<float, int64_t, int32_t>),
                     grid, block, 0, stream, in_cpu_dataptr, in_gpu_dataptr,
                     in_cold_dataptr, feature_dim, in_idx_ptr, global_idx_ptr,
                     num_idx, out_dataptr, hot_map_ptr, cold_map_ptr);
  } else {
    CUDA_KERNEL_CALL(
        (IndexSelectMultiKernelAlignedMegaBatch<float, int64_t, int32_t>), grid,
        block, 0, stream, in_cpu_dataptr, in_gpu_dataptr, in_cold_dataptr,
        feature_dim, in_idx_ptr, global_idx_ptr, num_idx, out_dataptr,
        hot_map_ptr, cold_map_ptr);
  }
}

template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernelMergeMiniBatch(
    const DType* const host_array, const DType* const device_array,
    const DType* const host_cold_array, const int64_t num_feat,
    const IdType* const in_index, const IdType* const inv_index,
    const int64_t length, DType* const out, const MapIdType* hot_map_ptr,
    const MapIdType* cold_map_ptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    MapIdType in_pos;
    const DType* array;
    if (hot_map_ptr[in_row] > 0) {
      in_pos = hot_map_ptr[in_row] - 1;
      array = host_array;
    } else if (hot_map_ptr[in_row] < 0) {
      in_pos = -hot_map_ptr[in_row] - 1;
      array = device_array;
    } else {
      in_pos = cold_map_ptr[inv_index[out_row_index]];
      array = host_cold_array;
      assert(in_pos != -1);
    }
    while (col < num_feat) {
      out[out_row_index * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

template <typename DType, typename IdType, typename MapIdType>
__global__ void IndexSelectMultiKernelAlignedMergeMiniBatch(
    const DType* const host_array, const DType* const device_array,
    const DType* const host_cold_array, const int64_t num_feat,
    const IdType* const in_index, const IdType* const inv_index,
    const int64_t length, DType* const out, const MapIdType* hot_map_ptr,
    const MapIdType* cold_map_ptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = in_index[out_row_index];
    MapIdType in_pos;
    const DType* array;
    if (hot_map_ptr[in_row] > 0) {
      in_pos = hot_map_ptr[in_row] - 1;
      array = host_array;
    } else if (hot_map_ptr[in_row] < 0) {
      in_pos = -hot_map_ptr[in_row] - 1;
      array = device_array;
    } else {
      in_pos = cold_map_ptr[inv_index[out_row_index]];
      array = host_cold_array;
      assert(in_pos != -1);
    }
    const int64_t idx_offset =
        ((uint64_t)(&array[in_pos * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row_index * num_feat + col] = array[in_pos * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

void GatherInGPU_MergeMiniBatch(torch::Tensor& out, const torch::Tensor& in_idx,
                                const torch::Tensor& unique_inv_idx,
                                const torch::Tensor& in_cpu,
                                const torch::Tensor& in_gpu,
                                const torch::Tensor& in_cold,
                                const torch::Tensor& hot_map_table,
                                const torch::Tensor& cold_map_table) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(in_idx.numel() == out.sizes()[0]);
  auto num_idx = in_idx.numel();
  assert(out.sizes()[1] == in_cpu.sizes()[1]);
  auto feature_dim = out.sizes()[1];

  auto out_dataptr = out.data_ptr<float>();
  auto in_cpu_dataptr = in_cpu.data_ptr<float>();
  auto in_gpu_dataptr = in_gpu.data_ptr<float>();
  auto in_cold_dataptr = in_cold.data_ptr<float>();
  auto hot_map_ptr = hot_map_table.data_ptr<int32_t>();
  auto cold_map_ptr = cold_map_table.data_ptr<int32_t>();
  auto in_idx_ptr = in_idx.data_ptr<int64_t>();
  auto inv_idx_ptr = unique_inv_idx.data_ptr<int64_t>();

  dim3 block(256, 1);
  while (static_cast<int64_t>(block.x) >= 2 * feature_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_idx + block.y - 1) / block.y);
  if (feature_dim * sizeof(float) < 2 * CACHE_LINE_SIZE) {
    CUDA_KERNEL_CALL(
        (IndexSelectMultiKernelMergeMiniBatch<float, int64_t, int32_t>), grid,
        block, 0, stream, in_cpu_dataptr, in_gpu_dataptr, in_cold_dataptr,
        feature_dim, in_idx_ptr, inv_idx_ptr, num_idx, out_dataptr, hot_map_ptr,
        cold_map_ptr);
  } else {
    CUDA_KERNEL_CALL(
        (IndexSelectMultiKernelAlignedMergeMiniBatch<float, int64_t, int32_t>),
        grid, block, 0, stream, in_cpu_dataptr, in_gpu_dataptr, in_cold_dataptr,
        feature_dim, in_idx_ptr, inv_idx_ptr, num_idx, out_dataptr, hot_map_ptr,
        cold_map_ptr);
  }
}
}  // namespace cuda
}  // namespace offgs