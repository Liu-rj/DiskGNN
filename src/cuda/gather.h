#ifndef OFFGS_CUDA_GATHER_H_
#define OFFGS_CUDA_GATHER_H_

#include <torch/torch.h>

namespace offgs {
namespace cuda {
void GatherInGPU(torch::Tensor& out, const torch::Tensor& out_idx,
                 const torch::Tensor& in_cpu, const torch::Tensor& in_gpu,
                 const torch::Tensor& in_idx, const torch::Tensor& map_table);

void GatherInGPU_MegaBatch(torch::Tensor& out, const torch::Tensor& in_idx,
                           const torch::Tensor& global_idx,
                           const torch::Tensor& in_cpu,
                           const torch::Tensor& in_gpu,
                           const torch::Tensor& in_cold,
                           const torch::Tensor& hot_map_table,
                           const torch::Tensor& cold_map_table);

void GatherInGPU_MergeMiniBatch(torch::Tensor& out, const torch::Tensor& in_idx,
                                const torch::Tensor& unique_inv_idx,
                                const torch::Tensor& in_cpu,
                                const torch::Tensor& in_gpu,
                                const torch::Tensor& in_cold,
                                const torch::Tensor& hot_map_table,
                                const torch::Tensor& cold_map_table);
}  // namespace cuda
}  // namespace offgs

#endif